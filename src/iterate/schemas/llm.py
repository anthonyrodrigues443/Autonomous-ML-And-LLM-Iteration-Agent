"""LLM interaction contracts.

Provider-agnostic message, tool, and response types that flow between the agent
and any LLM backend. The concrete client (``iterate.llm.openai_compatible``)
translates these to/from a vendor wire format; everything upstream — Proposer,
Researcher, discovery agent — speaks only these types, never a vendor SDK.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolSpec(BaseModel):
    """A tool the agent exposes to the model (an OpenAI-style function schema)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema describing the tool's arguments


class ToolCall(BaseModel):
    """A tool invocation the model requested in its reply."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One conversation turn.

    ``content`` is optional because an assistant turn may carry only
    ``tool_calls``; ``tool_call_id`` is set on a ``tool``-role result turn.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class Usage(BaseModel):
    """Token accounting for one model call — feeds cost tracking."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """A normalized reply from any LLM backend.

    ``content`` is ``None`` when the model replied purely with ``tool_calls``.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


__all__ = [
    "ChatResponse",
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
