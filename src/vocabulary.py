"""Token lookup tables built once from the model's ``vocab.json``."""

from __future__ import annotations

import json
import re
from typing import Dict

# Text safe inside a JSON string: no raw quote, backslash or control
# character, and only complete escapes.
_JSON_STRING_BODY_RE = re.compile(
    r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*'
)


def _bytes_to_unicode() -> Dict[int, str]:
    """Rebuild GPT-2's byte -> printable-character table.

    Byte-level BPE vocabularies (GPT-2, Qwen) store every byte as a
    visible character, so a space appears as ``Ġ``.
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_points = list(byte_values)
    next_point = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_points.append(256 + next_point)
            next_point += 1
    return dict(zip(byte_values, (chr(p) for p in unicode_points)))


def build_id_to_token(vocab_path: str) -> Dict[int, str]:
    """Map every token id to its real text.

    Tokens that are only a fragment of a multi-byte character map to
    ``""`` and are never used.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not a JSON token -> id mapping.
    """
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        raw_vocab = json.load(vocab_file)
    if not isinstance(raw_vocab, dict):
        raise ValueError("vocab file does not contain a token -> id map")

    byte_decoder = {char: byte for byte, char in _bytes_to_unicode().items()}
    id_to_token: Dict[int, str] = {}
    for token, token_id in raw_vocab.items():
        try:
            raw_bytes = bytes(byte_decoder[char] for char in token)
        except KeyError:
            # Special tokens (e.g. "<|im_start|>") are plain text.
            text = token
        else:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
        id_to_token[int(token_id)] = text
    return id_to_token


def build_numeric_token_ids(id_to_token: Dict[int, str]) -> Dict[int, str]:
    """Select the tokens made only of digits, ``-`` and ``.``."""
    allowed_chars = set("-0123456789.")
    return {
        token_id: token_text
        for token_id, token_text in id_to_token.items()
        if token_text and set(token_text) <= allowed_chars
    }


def build_string_content_token_ids(
    id_to_token: Dict[int, str],
) -> Dict[int, str]:
    """Select the tokens that are safe inside a JSON string.

    A lone backslash is excluded, so the set never changes between steps.
    Regexes like ``\\d+`` stay reachable as the escaped ``\\\\d+``.
    """
    return {
        token_id: token_text
        for token_id, token_text in id_to_token.items()
        if token_text and _JSON_STRING_BODY_RE.fullmatch(token_text)
    }
