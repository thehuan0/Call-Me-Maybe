"""Typing protocol for the part of ``Small_LLM_Model`` this project uses."""

from __future__ import annotations

from typing import Any, List, Protocol


class LanguageModel(Protocol):
    """The three public SDK methods we rely on."""

    def encode(self, text: str) -> Any:
        """Tokenise ``text`` into a ``(1, n_tokens)`` tensor."""
        ...

    def get_logits_from_input_ids(
        self, input_ids: List[int]
    ) -> List[float]:
        """Return the next-token logits, one per vocabulary entry."""
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the path of the model's ``vocab.json``."""
        ...
