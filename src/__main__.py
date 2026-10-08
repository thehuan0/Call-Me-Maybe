"""Command-line entry point: prompts in, structured function calls out."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, cast

from pydantic import ValidationError

from .constrained_decoding import set_trace
from .function_calling import default_call, generate_function_call
from .llm_protocol import LanguageModel
from .models import (
    SUPPORTED_PARAMETER_TYPES,
    FunctionDefinition,
    OutputEntry,
    TestPrompt,
)
from .vocabulary import (
    build_id_to_token,
    build_numeric_token_ids,
    build_string_content_token_ids,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="JSON file describing the available functions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON file containing the natural-language prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where the resulting JSON file will be written.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Hugging Face model id. It must use a GPT-2 style byte-level "
        f"BPE vocabulary, like the Qwen family (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print every decoding step on stderr: how many tokens were "
        "legal, what the model wanted, and what the constraint allowed.",
    )
    return parser.parse_args(argv)


def load_json_array(path: Path) -> List[Any]:
    """Read a JSON array from ``path``.

    Raises:
        SystemExit: If the file is missing, unreadable, invalid or not
            an array.
    """
    try:
        with path.open("r", encoding="utf-8-sig") as input_file:
            data = json.load(input_file)
    except FileNotFoundError:
        raise SystemExit(f"Error: file not found: {path}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"Error: {path} is not valid JSON ({error}).")
    except UnicodeDecodeError as error:
        raise SystemExit(f"Error: {path} is not valid UTF-8 text ({error}).")
    except OSError as error:
        raise SystemExit(f"Error: cannot read {path}: {error}")
    if not isinstance(data, list):
        raise SystemExit(f"Error: {path} must contain a JSON array.")
    return data


def load_functions(path: Path) -> Dict[str, FunctionDefinition]:
    """Load the function definitions, keyed by name in file order.

    Raises:
        SystemExit: On an invalid entry, a duplicate name or no function.
    """
    functions: Dict[str, FunctionDefinition] = {}
    for entry in load_json_array(path):
        try:
            function = FunctionDefinition.model_validate(entry)
        except ValidationError as error:
            raise SystemExit(
                f"Error: invalid function definition in {path}: {error}"
            )
        if function.name in functions:
            raise SystemExit(
                f"Error: duplicate function name {function.name!r} "
                f"in {path}."
            )
        for param_name, schema in function.parameters.items():
            if schema.type not in SUPPORTED_PARAMETER_TYPES:
                print(
                    f"Warning: {function.name}.{param_name} has unsupported "
                    f"type {schema.type!r}; it will be generated as text.",
                    file=sys.stderr,
                )
        functions[function.name] = function
    if not functions:
        raise SystemExit(f"Error: {path} does not define any functions.")
    return functions


def load_prompts(path: Path) -> List[TestPrompt]:
    """Load the prompts to translate, in file order.

    Raises:
        SystemExit: If an entry is invalid.
    """
    prompts: List[TestPrompt] = []
    for entry in load_json_array(path):
        try:
            prompts.append(TestPrompt.model_validate(entry))
        except ValidationError as error:
            raise SystemExit(
                f"Error: invalid prompt entry in {path}: {error}"
            )
    return prompts


def load_model(model_name: str) -> LanguageModel:
    """Load the ``llm_sdk`` model.

    Raises:
        SystemExit: If ``llm_sdk`` cannot be imported or the model fails
            to load.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as error:
        raise SystemExit(
            f"Error: could not import llm_sdk ({error}). "
            "Make sure it is installed (see the README's Instructions)."
        )
    try:
        # llm_sdk is not type-checked; we assert it fits the protocol.
        return cast(LanguageModel, Small_LLM_Model(model_name))
    except Exception as error:  # model loading can fail for many reasons
        raise SystemExit(
            f"Error: could not load model '{model_name}': {error}"
        )


def write_results(path: Path, results: List[OutputEntry]) -> None:
    """Write the results file, creating its directory if needed.

    Raises:
        SystemExit: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(
                [entry.model_dump() for entry in results],
                output_file,
                indent=2,
            )
    except OSError as error:
        raise SystemExit(f"Error: cannot write {path}: {error}")


def main(argv: List[str] | None = None) -> int:
    """Run the whole pipeline and return the exit code."""
    args = parse_args(argv)

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    if not prompts:
        print(f"Warning: {args.input} contains no prompts.", file=sys.stderr)
    set_trace(args.trace)
    model = load_model(args.model)

    try:
        vocab_path = model.get_path_to_vocab_file()
        id_to_token = build_id_to_token(vocab_path)
    except Exception as error:
        raise SystemExit(f"Error: could not read the vocabulary: {error}")

    numeric_token_ids = build_numeric_token_ids(id_to_token)
    string_content_token_ids = build_string_content_token_ids(id_to_token)

    started = time.monotonic()
    results: List[OutputEntry] = []
    for test_prompt in prompts:
        if args.trace:
            print(f"[trace] ===== {test_prompt.prompt!r}", file=sys.stderr)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                name, parameters = generate_function_call(
                    model,
                    id_to_token,
                    numeric_token_ids,
                    string_content_token_ids,
                    test_prompt.prompt,
                    functions,
                )
            for warning in caught:
                print(
                    f"Warning: prompt {test_prompt.prompt!r}: "
                    f"{warning.message}",
                    file=sys.stderr,
                )
        except Exception as error:
            # One failing prompt must not stop the run or shift the
            # position of later answers.
            print(
                f"Warning: prompt {test_prompt.prompt!r} failed ({error}); "
                "writing a placeholder call.",
                file=sys.stderr,
            )
            name, parameters = default_call(functions)
        results.append(
            OutputEntry(
                prompt=test_prompt.prompt, name=name, parameters=parameters
            )
        )

    write_results(args.output, results)
    print(
        f"Wrote {len(results)} of {len(prompts)} function call(s) "
        f"to {args.output} in {time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.")
