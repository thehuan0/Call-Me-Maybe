"""Pydantic models for the input and output JSON shapes."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Union

from pydantic import BaseModel, Field

# Types with a dedicated grammar; any other type is generated as free text.
SUPPORTED_PARAMETER_TYPES: FrozenSet[str] = frozenset(
    {"string", "number", "integer", "boolean"}
)

EnumValue = Union[str, int, float, bool]


class ParameterSchema(BaseModel):
    """One function argument: its JSON type and optional allowed values."""

    type: str
    enum: Optional[List[EnumValue]] = None


class FunctionDefinition(BaseModel):
    """A function from ``functions_definition.json``.

    ``parameters`` keeps file order, which is the generation order.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterSchema] = Field(default_factory=dict)


class TestPrompt(BaseModel):
    """One request from ``function_calling_tests.json``."""

    prompt: str


class OutputEntry(BaseModel):
    """One resolved function call, as written to the results file."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
