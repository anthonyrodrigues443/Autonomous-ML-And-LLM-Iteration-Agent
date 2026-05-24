"""The LLM backend contract.

Every backend (OpenAI-compatible, Anthropic, ...) implements ``LLMClient``, so the
agent swaps providers/models by config alone. The orchestrator, proposer, and
researcher depend only on this protocol — never on a vendor SDK.

It's a Protocol (structural typing) rather than an ABC: a backend just has to match
the shape, not inherit a base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from iterate.schemas.llm import ChatResponse, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """What every LLM backend must provide."""

    @property
    def model(self) -> str:
        """Model identifier this client targets (for logging + cost attribution)."""
        ...

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a conversation (optionally exposing tools) and return one reply.

        The reply may carry text, ``tool_calls``, or both. Implementations MUST NOT
        execute tools — they only surface the calls the model requested; running
        them is the caller's job.
        """
        ...
