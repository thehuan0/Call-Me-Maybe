"""Token-by-token generation restricted to schema-valid continuations."""

from __future__ import annotations

import json
import re
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .llm_protocol import LanguageModel

# "Prefix" regexes also accept unfinished numbers ("-", "12.").
_NUMBER_PREFIX_RE = re.compile(r"^-?$|^-?(0|[1-9][0-9]*)(\.[0-9]*)?$")
_NUMBER_COMPLETE_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")
_INTEGER_PREFIX_RE = re.compile(r"^-?$|^-?(0|[1-9][0-9]*)$")
_INTEGER_COMPLETE_RE = re.compile(r"^-?(0|[1-9][0-9]*)$")

MAX_NUMBER_TOKENS = 32

# A string value is bounded by the request length plus a margin, which
# stops a looping model from wasting forward passes.
MIN_STRING_TOKENS = 32
STRING_TOKEN_MARGIN = 16

# Banned repeat length inside one string. Long on purpose, so copies such
# as "1234 1234" (digits are one token each) are not broken.
NO_REPEAT_NGRAM = 8

_trace_enabled = False


def set_trace(enabled: bool) -> None:
    """Turn the decoding trace (printed on stderr) on or off."""
    global _trace_enabled
    _trace_enabled = enabled


def _trace_step(
    id_to_token: Dict[int, str],
    allowed: Set[int],
    chosen: int,
    logits: Optional[Sequence[float]],
) -> None:
    """Print one decoding step; ``logits`` is None when forced."""
    if not _trace_enabled:
        return
    chosen_text = repr(id_to_token.get(chosen, "?"))
    if logits is None:
        detail = f"forced (no model call) -> {chosen_text}"
    else:
        free = int(np.argmax(np.asarray(logits)))
        free_text = repr(id_to_token.get(free, "?"))
        verdict = "kept" if free == chosen else "MASKED, next best legal"
        detail = (
            f"model alone: {free_text:<12} constrained: "
            f"{chosen_text:<12} {verdict}"
        )
    print(f"[trace] {len(allowed):>6} legal | {detail}", file=sys.stderr)


def string_token_budget(request_tokens: int) -> int:
    """Return the max tokens a string value may use for this request."""
    return max(MIN_STRING_TOKENS, request_tokens + STRING_TOKEN_MARGIN)


def select_next_token(
    logits: Sequence[float], allowed_ids: Iterable[int]
) -> int:
    """Return the highest-scoring token among ``allowed_ids``.

    Every other logit is set to ``-inf`` before taking the arg-max.

    Raises:
        ValueError: If ``allowed_ids`` is empty.
    """
    allowed_array = np.fromiter(allowed_ids, dtype=np.int64)
    if allowed_array.size == 0:
        raise ValueError("no allowed token to choose from")
    logits_array = np.asarray(logits, dtype=np.float64)
    masked = np.full_like(logits_array, -np.inf)
    masked[allowed_array] = logits_array[allowed_array]
    return int(np.argmax(masked))


def tokens_starting_with(
    id_to_token: Dict[int, str], prefix: str
) -> List[int]:
    """Return the ids of tokens starting with ``prefix`` (none if empty).

    BPE merges punctuation (``",``, ``}}``), so "end of value" is detected
    by prefix, not by the bare delimiter token.
    """
    if not prefix:
        return []
    return [tid for tid, tok in id_to_token.items() if tok.startswith(prefix)]


def extend_with_text(
    model: LanguageModel, input_ids: List[int], text: str
) -> List[int]:
    """Append fixed text (no model decision) to the token sequence."""
    if not text:
        return input_ids
    if _trace_enabled:
        print(f"[trace] fixed text appended, no model call: {text!r}",
              file=sys.stderr)
    new_ids: List[int] = model.encode(text)[0].tolist()
    return input_ids + new_ids


def _choose(
    model: LanguageModel,
    input_ids: List[int],
    allowed: Set[int],
    id_to_token: Dict[int, str],
) -> int:
    """Pick an allowed token, skipping the model if only one is legal."""
    if len(allowed) == 1:
        only = next(iter(allowed))
        _trace_step(id_to_token, allowed, only, None)
        return only
    logits = model.get_logits_from_input_ids(input_ids)
    chosen = select_next_token(logits, allowed)
    _trace_step(id_to_token, allowed, chosen, logits)
    return chosen


def generate_enum_value(
    model: LanguageModel,
    input_ids: List[int],
    candidates: List[str],
    id_to_token: Dict[int, str],
    terminator: str = '"',
) -> Tuple[str, List[int]]:
    """Generate exactly one of ``candidates``.

    The model picks each token among those that continue a matching
    candidate. When one candidate is left, the rest is appended directly.
    If a candidate is a prefix of another, tokens starting with
    ``terminator`` are allowed too, so the shorter one can be chosen; the
    terminator itself is not consumed.

    Raises:
        ValueError: If there are no candidates or none can continue.
    """
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise ValueError("no candidate values to choose from")
    if len(unique) == 1:
        return unique[0], extend_with_text(model, input_ids, unique[0])

    buffer = ""
    by_text: Dict[str, List[int]] = {}
    for token_id, token_text in id_to_token.items():
        if token_text:
            by_text.setdefault(token_text, []).append(token_id)
    stop_ids = set(tokens_starting_with(id_to_token, terminator))

    # Every token adds at least one character, so this bounds the loop.
    for _ in range(max(len(c) for c in unique) + 1):
        remaining = [c for c in unique if c.startswith(buffer)]
        if not remaining:
            raise ValueError(f"no candidate matches prefix {buffer!r}")
        if len(remaining) == 1:
            winner = remaining[0]
            rest = winner[len(buffer):]
            return winner, extend_with_text(model, input_ids, rest)

        growing = [c for c in remaining if c != buffer]
        is_complete = len(growing) < len(remaining)
        continue_ids: Set[int] = set()
        for candidate in growing:
            rest = candidate[len(buffer):]
            for end in range(1, len(rest) + 1):
                continue_ids.update(by_text.get(rest[:end], ()))
        stop_now = stop_ids if is_complete else set()
        allowed = continue_ids | stop_now
        if not allowed:
            raise ValueError(f"no valid continuation after {buffer!r}")

        next_id = _choose(model, input_ids, allowed, id_to_token)
        if next_id in stop_now and next_id not in continue_ids:
            return buffer, input_ids
        buffer += id_to_token[next_id]
        input_ids = input_ids + [next_id]

    raise ValueError(f"enum value generation did not terminate: {buffer!r}")


def generate_numeric_value(
    model: LanguageModel,
    input_ids: List[int],
    numeric_token_ids: Dict[int, str],
    id_to_token: Dict[int, str],
    terminator_char: str,
    allow_decimal: bool,
) -> Tuple[str, List[int]]:
    """Generate a JSON number as text (an integer if not ``allow_decimal``).

    Only tokens keeping the text a valid, possibly unfinished number are
    allowed. Once it is complete, tokens starting with ``terminator_char``
    may end it (that token is not consumed). After ``MAX_NUMBER_TOKENS``
    the valid digits so far are kept, with a ``RuntimeWarning``.

    Raises:
        ValueError: If no continuation exists or the number is unfinished.
    """
    prefix_re = _NUMBER_PREFIX_RE if allow_decimal else _INTEGER_PREFIX_RE
    complete_re = (
        _NUMBER_COMPLETE_RE if allow_decimal else _INTEGER_COMPLETE_RE
    )
    stop_ids = set(tokens_starting_with(id_to_token, terminator_char))

    buffer = ""
    for _ in range(MAX_NUMBER_TOKENS):
        continue_ids = {
            token_id
            for token_id, token_text in numeric_token_ids.items()
            if prefix_re.match(buffer + token_text)
        }
        stop_now = stop_ids if complete_re.match(buffer) else set()
        allowed = continue_ids | stop_now
        if not allowed:
            raise ValueError(f"no valid continuation for number {buffer!r}")

        next_id = _choose(model, input_ids, allowed, id_to_token)
        if next_id in stop_now and next_id not in continue_ids:
            break
        buffer += numeric_token_ids[next_id]
        input_ids = input_ids + [next_id]
    else:
        warnings.warn(
            f"number hit the {MAX_NUMBER_TOKENS}-token limit and was cut "
            f"short: {buffer!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        buffer = buffer.rstrip(".")

    if not complete_re.match(buffer):
        raise ValueError(f"incomplete number generated: {buffer!r}")
    return buffer, input_ids


def generate_string_value(
    model: LanguageModel,
    input_ids: List[int],
    string_content_token_ids: Dict[int, str],
    id_to_token: Dict[int, str],
    max_tokens: int,
    source_text: str,
) -> Tuple[str, List[int]]:
    """Generate the content of a JSON string.

    The model writes the text as it appears in JSON (a backslash is
    ``\\``); a token starting with ``"`` ends it, and ``json.loads``
    decodes the result. Leading spaces are stripped. A token that would
    repeat an ``NO_REPEAT_NGRAM``-token pattern is banned unless that text
    is in ``source_text`` (the model is copying the request, not looping).
    At ``max_tokens`` the text so far is kept, with a ``RuntimeWarning``.
    """
    stop_ids = set(tokens_starting_with(id_to_token, '"'))
    allowed = set(string_content_token_ids) | stop_ids

    escaped_source = json.dumps(source_text, ensure_ascii=False)[1:-1]
    buffer = ""
    generated: List[int] = []
    followers: Dict[Tuple[int, ...], Set[int]] = {}
    context_size = NO_REPEAT_NGRAM - 1
    for _ in range(max_tokens):
        key = tuple(generated[-context_size:])
        has_context = len(generated) >= context_size
        banned = {
            token_id
            for token_id in (followers.get(key, ()) if has_context else ())
            if "".join(
                string_content_token_ids[t] for t in (*key, token_id)
            ) not in escaped_source
        }
        next_id = _choose(
            model, input_ids, (allowed - banned) or allowed, id_to_token
        )
        if next_id in stop_ids and next_id not in string_content_token_ids:
            break
        if has_context:
            followers.setdefault(key, set()).add(next_id)
        generated.append(next_id)
        buffer += string_content_token_ids[next_id]
        input_ids = input_ids + [next_id]
    else:
        warnings.warn(
            f"string value hit the {max_tokens}-token limit and was cut "
            f"short: {buffer!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    value: str = json.loads(f'"{buffer}"').lstrip(" ")
    return value, input_ids
