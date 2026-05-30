"""LLM client factory — pick the right adapter for a named backend.

The CLI's ``--backend`` flag dispatches through here. Both `OllamaClient` and
`OpenAICompatibleClient` implement the `LLMClient` protocol, so nothing
downstream notices which one came back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iterate.llm.ollama_client import OllamaClient
from iterate.llm.openai_compatible import OpenAICompatibleClient

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient


# Names that route to the OpenAI-compatible client. Adding more cloud aliases (groq,
# together, deepseek, …) makes for friendlier CLI flags without code-path changes.
_OPENAI_COMPATIBLE_ALIASES = frozenset(
    {"openai-compatible", "openai", "groq", "together", "deepseek", "vllm"}
)

_KNOWN = ("ollama", "openai-compatible")


class UnknownBackendError(ValueError):
    """Raised when ``--backend`` names a backend we don't recognize."""


def build_client(
    name: str = "ollama",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Build an `LLMClient` for a named backend.

    - ``"ollama"`` (default) → `OllamaClient` (native ``/api/chat`` with ``think=False``).
    - ``"openai-compatible"`` (or any of the cloud aliases) → `OpenAICompatibleClient`.

    Any of `model`/`base_url`/`api_key` left ``None`` falls through to the client's
    own defaults (which read the central `Settings` / `.env`). API-key validation
    is the CLI layer's job — this factory just dispatches.
    """
    if name == "ollama":
        return OllamaClient(host=base_url, model=model)
    if name in _OPENAI_COMPATIBLE_ALIASES:
        return OpenAICompatibleClient(base_url=base_url, model=model, api_key=api_key)
    raise UnknownBackendError(f"unknown backend {name!r}; choose one of {_KNOWN}")


__all__ = ["UnknownBackendError", "build_client"]
