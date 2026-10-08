*This project has been created as part of the 42 curriculum by jperez-s.*

## Description

**call me maybe** turns a natural-language request such as *"What is the sum
of 2 and 3?"* into a structured function call:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It uses a small model (`Qwen/Qwen3-0.6B`) that would often produce invalid
JSON on its own. Instead of hoping, the program uses **constrained
decoding**: at every step it gets the model's logits, discards every token
that would break the JSON or the function's schema, and lets the model pick
among the rest. The output is always valid, schema-compliant JSON, while the
model still decides *which* function to call and *which* values to use.

## Instructions

**Requirements:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and the
`llm_sdk` package included in this repository.

```bash
make install   # uv sync; the first run also downloads the Qwen weights
make run       # uv run python -m src
```

By default it reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. Override with:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Optional flags:

```bash
--trace                  # print every decoding step on stderr
--model Qwen/Qwen3-1.7B  # any GPT-2 style byte-level BPE model
```

Other targets: `make debug` (pdb), `make lint` (flake8 + mypy),
`make lint-strict` (mypy --strict), `make clean`.

## Algorithm explanation

The target output is split into two kinds of pieces:

1. **Fixed syntax** (`{"name": "`, `", "parameters": {`, parameter keys,
   closing braces). The schema fully determines it, so it is tokenised and
   appended with no model call.
2. **Model-chosen values** (the function name and each parameter value),
   generated one token at a time: get logits, compute the legal token ids,
   set all other logits to `-inf`, take the arg-max (`select_next_token`),
   repeat until the value is complete.

How "legal" is defined for each value type:

- **Enum-like** (function name, `enum` parameters, booleans): only tokens
  that extend at least one still-matching candidate are allowed
  (`generate_enum_value`). Once one candidate remains, the rest is appended
  directly. If a candidate is a prefix of another (`fn_add` / `fn_add_numbers`),
  the model may also choose to stop.
- **Numbers**: a *prefix* regex keeps the text a valid unfinished number (no
  leading zeros, one decimal point). Once a *complete* regex matches, tokens
  starting with the next delimiter (`,` or `}`) are allowed so the model can
  end the value. `integer` never allows a `.`.
- **Strings**: any token with no raw quote or control character and only
  complete escapes is allowed. A token starting with `"` ends the string. The
  text is then decoded with `json.loads`, so a regex written as `\\d+`
  becomes `\d+`.

"Stop" is detected by *starts with* the delimiter, because BPE vocabularies
merge punctuation (`",`, `}}`).

**Reading the vocabulary.** `src/vocabulary.py` reads `vocab.json` once at
start-up and rebuilds the real text of every token, undoing the GPT-2
byte-to-unicode mapping. This avoids calling `model.decode` for ~150k tokens
at every step.

## Design decisions

- **Fixed syntax is appended, not generated:** fewer model calls, and the
  model only decides what is really a decision.
- **Strings are generated as JSON text, then decoded:** regexes, quotes and
  newlines stay reachable and the output stays valid. A lone backslash is
  excluded so the allowed set never changes between steps.
- **The prompt says "pattern means regex"**, with a generic worked example,
  after the model once copied a word into a regex argument.
- **Greedy decoding:** the same input always gives the same output.
- **Numbers follow the schema:** `number` is a float (`2.0`), `integer` an int.
- **Unknown parameter types fall back to free text** instead of crashing.
- **One JSON array is written at the end**, not streamed per prompt.


## Performance analysis

- **Validity:** guaranteed by construction, not by luck.
- **Speed:** vocabulary tables are built once, enum selection uses a
  text-to-ids index, and the model is skipped when only one token is legal.
  The SDK has no KV cache, so every token costs a full forward pass. On a
  CPU-only machine the first run took 440 s; length caps, repetition
  banning and skipped calls brought it to about 2 minutes. A visible GPU is
  used automatically and is the biggest speed-up to about 30s
- **Accuracy:** constrained decoding guarantees the shape, not the choice.
  Which function and values are picked depends on the model and on
  `build_prompt_context` (function list, descriptions, two worked examples).

## Challenges faced

- **Knowing when a value ends:** the model chooses between "extend" and
  "start the text that follows", and picking the latter is a stop signal.
- **Invalid numbers (`007`, `1.2.3`):** separate prefix and complete regexes.
- **Merged closing tokens:** a model preferring `"}}` could not end a value
  until stop matching used "starts with".
- **Loops:** one prompt produced `([0-9]+)\\s([0-9]+)\\s...` forever. Fixed
  with a length cap, an 8-token repeat ban (allowed when the request itself
  repeats that text), and keeping the truncated valid value.
- **Stray leading space** (`" *"` instead of `"*"`): leading spaces are stripped.
- **An optimisation that hurt accuracy:** forcing the shared `fn_` prefix made
  the model pick `fn_get_square_root` for "Greet shrek". Accuracy is checked
  after every optimisation, not only speed.
- **Failed prompts shift answers:** results are matched to prompts by
  position, so a failed prompt gets a schema-valid placeholder and a warning.

## Testing strategy

There is no automated test suite. The program was checked in four ways:

- **Provided prompts:** every run on `function_calling_tests.json` was
  reviewed by hand. Each result was checked for the right function, the
  right argument values, and the right types (`2.0` for `number`, `2` for
  `integer`).
- **Output validity:** the results file is loaded with `json.load` and every
  entry is checked against its function definition: known name, all
  parameters present, correct types. Validity is guaranteed by
  construction, so this mainly catches bugs in the generators.
- **Edge cases:** extra prompts for negative and decimal numbers, regex
  arguments (`\d+`), strings with quotes, backslashes or spaces, empty
  strings, and requests that fit no function well.
- **Failure paths:** missing or invalid input files, duplicate function
  names, and an unsupported parameter type, each checked for a clear error
  message instead of a traceback.

`--trace` was the main debugging tool. Many `MASKED` lines on a value mean
the model is struggling with the prompt, and `forced` lines confirm
that single-option steps skip the model. Code quality is checked with
`make lint` (flake8 and mypy) and `make lint-strict`.

## Bonus features

- **`--model`:** any GPT-2 style byte-level BPE model (SentencePiece models
  such as Llama would need a different vocabulary reader).
  Example of usage 
  ```bash
  uv run python -m src --model HuggingFaceTB/SmolLM2-1.7B-Instruct
  ```
- **`--trace`:** per token, shows how many tokens were legal, what the model
  wanted, and what the constraint allowed (`kept`, `MASKED`, `forced`). Fixed
  text is shown as `fixed text appended`. Output goes to stderr.
- **Error recovery:** a failing prompt never stops the run, and over-long
  strings or numbers are cut at the last valid point.

## Example usage

```bash
$ make run
Wrote 11 of 11 function call(s) to data/output/function_calling_results.json in <elapsed>s

$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  ...
]
```

Exact values and timing depend on the model and machine; the shape and
validity of the JSON do not.

## Resources

- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Hugging Face: How to generate text](https://huggingface.co/blog/how-to-generate)
- [OpenAI GPT-2 source](https://github.com/openai/gpt-2): origin of the
  byte-level tokenizer mapping re-implemented in `src/vocabulary.py`.
- Grammar-constrained decoding libraries ([Guidance](https://github.com/guidance-ai/guidance),
  [Outlines](https://github.com/dottxt-ai/outlines),
  [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)): not
  used here, but the inspiration for masking logits token by token.

### How AI was used

- AI helped design the architecture (what the model generates versus what is
appended as fixed text)
- structure this README
- rewrite PEP 257 appropiate docstrings.

🥭