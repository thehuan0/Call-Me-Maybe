"""Turn one request into a ``(name, parameters)`` function call."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

from .constrained_decoding import (
    extend_with_text,
    generate_enum_value,
    generate_numeric_value,
    generate_string_value,
    string_token_budget,
)
from .llm_protocol import LanguageModel
from .models import EnumValue, FunctionDefinition, ParameterSchema


def build_prompt_context(
    prompt: str, functions: List[FunctionDefinition]
) -> str:
    """Build the text shown to the model.

    The instructions and the two examples are generic (their functions are
    not in any real function set), so they teach the shape of an answer
    without biasing which function is picked. The text ends with the
    opening of the JSON answer, so generation starts at the function name.
    """
    lines = [
        "You translate a user's request into exactly one function call.",
        "Pick the single function below that best matches the request,",
        "then fill in its arguments using information from the request.",
        "Copy text values exactly as written in the request. If an argument",
        "is a pattern (a regular expression), write a regex describing the",
        "kind of text to match, not a literal word from the request.",
        "",
        "Available functions:",
    ]
    for function in functions:
        params = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in function.parameters.items()
        )
        lines.append(f"- {function.name}({params}): {function.description}")
    lines += [
        "",
        "Example: given a function repeat(text: string, times: number)",
        "and the request \"say 'hi' 3 times\", the correct answer is",
        '{"name": "repeat", "parameters": {"text": "hi", "times": 3}}.',
        "",
        "Example: given a function mask(text: string, pattern: string,",
        "symbol: string) and the request",
        "\"replace all spaces in 'a b' with _\", the correct answer is",
        '{"name": "mask", "parameters": {"text": "a b", "pattern": "\\\\s",'
        ' "symbol": "_"}}.',
        "",
        f"Request: {json.dumps(prompt, ensure_ascii=False)}",
        "",
        "Answer with a JSON object only, no other text.",
        '{"name": "',
    ]
    return "\n".join(lines)


def _generate_enumerated_value(
    model: LanguageModel,
    input_ids: List[int],
    id_to_token: Dict[int, str],
    prefix: str,
    values: Sequence[EnumValue],
    is_last: bool,
) -> Tuple[EnumValue, List[int]]:
    """Generate a parameter restricted to ``values``.

    Strings go between quotes; numbers and booleans are bare literals
    followed by ``,`` or ``}``. Returns the value with its original type.
    """
    if all(isinstance(value, str) for value in values):
        rendered = {json.dumps(v, ensure_ascii=False)[1:-1]: v for v in values}
        input_ids = extend_with_text(model, input_ids, prefix + '"')
        text, input_ids = generate_enum_value(
            model, input_ids, list(rendered), id_to_token, '"'
        )
        input_ids = extend_with_text(model, input_ids, '"')
    else:
        rendered = {json.dumps(v, ensure_ascii=False): v for v in values}
        input_ids = extend_with_text(model, input_ids, prefix)
        text, input_ids = generate_enum_value(
            model, input_ids, list(rendered), id_to_token,
            "}" if is_last else ",",
        )
    return rendered[text], input_ids


def _generate_number_value(
    model: LanguageModel,
    input_ids: List[int],
    id_to_token: Dict[int, str],
    numeric_token_ids: Dict[int, str],
    prefix: str,
    schema: ParameterSchema,
    is_last: bool,
) -> Tuple[float, List[int]]:
    """Generate a ``number`` (float) or ``integer`` (int) parameter."""
    input_ids = extend_with_text(model, input_ids, prefix)
    raw_value, input_ids = generate_numeric_value(
        model,
        input_ids,
        numeric_token_ids,
        id_to_token,
        "}" if is_last else ",",
        allow_decimal=schema.type == "number",
    )
    if schema.type == "number":
        return float(raw_value), input_ids
    return int(raw_value), input_ids


def generate_function_call(
    model: LanguageModel,
    id_to_token: Dict[int, str],
    numeric_token_ids: Dict[int, str],
    string_content_token_ids: Dict[int, str],
    prompt: str,
    functions: Dict[str, FunctionDefinition],
) -> Tuple[str, Dict[str, object]]:
    """Resolve one prompt into ``(function_name, parameters)``.

    The JSON punctuation is appended as fixed text. The model only decides
    the function name and each argument value. Types without a dedicated
    grammar are generated as free text.

    Raises:
        ValueError: If constrained generation cannot complete a value.
    """
    context = build_prompt_context(prompt, list(functions.values()))
    input_ids: List[int] = model.encode(context)[0].tolist()
    max_string_tokens = string_token_budget(
        len(model.encode(prompt)[0].tolist())
    )

    name, input_ids = generate_enum_value(
        model, input_ids, list(functions.keys()), id_to_token
    )
    schema = functions[name]
    input_ids = extend_with_text(model, input_ids, '", "parameters": {')

    parameters: Dict[str, object] = {}
    param_names = list(schema.parameters.keys())
    for index, param_name in enumerate(param_names):
        param_schema = schema.parameters[param_name]
        is_last = index == len(param_names) - 1
        prefix = f'"{param_name}": ' if index == 0 else f', "{param_name}": '

        enum_values: Sequence[EnumValue] = param_schema.enum or (
            [True, False] if param_schema.type == "boolean" else []
        )
        if enum_values:
            parameters[param_name], input_ids = _generate_enumerated_value(
                model, input_ids, id_to_token, prefix, enum_values, is_last
            )
        elif param_schema.type in ("number", "integer"):
            parameters[param_name], input_ids = _generate_number_value(
                model, input_ids, id_to_token, numeric_token_ids, prefix,
                param_schema, is_last,
            )
        else:
            input_ids = extend_with_text(model, input_ids, prefix + '"')
            text, input_ids = generate_string_value(
                model, input_ids, string_content_token_ids, id_to_token,
                max_string_tokens, prompt,
            )
            input_ids = extend_with_text(model, input_ids, '"')
            parameters[param_name] = text

    return name, parameters


def default_call(
    functions: Dict[str, FunctionDefinition],
) -> Tuple[str, Dict[str, object]]:
    """Build a schema-valid placeholder call for when generation fails.

    Uses the first function with a neutral value per parameter, so the
    output stays aligned with the prompts.
    """
    function = next(iter(functions.values()))
    parameters: Dict[str, object] = {}
    for name, schema in function.parameters.items():
        if schema.enum:
            parameters[name] = schema.enum[0]
        elif schema.type == "number":
            parameters[name] = 0.0
        elif schema.type == "integer":
            parameters[name] = 0
        elif schema.type == "boolean":
            parameters[name] = False
        else:
            parameters[name] = ""
    return function.name, parameters
