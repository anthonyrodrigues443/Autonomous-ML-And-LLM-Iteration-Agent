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


# Cloud aliases → their OpenAI-compatible base URL, so `--backend groq` (or a saved
# config) needs only a model + key, never a hand-typed --base-url. An explicit
# --base-url still overrides. "openai-compatible"/"vllm" have no canonical URL (the
# user supplies one), so they're not mapped.
_ALIAS_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

# Names that route to the OpenAI-compatible client.
_OPENAI_COMPATIBLE_ALIASES = frozenset(
    {"openai-compatible", "openai", "groq", "together", "deepseek", "vllm"}
)

_KNOWN = ("ollama", "openai-compatible")


class UnknownBackendError(ValueError):
    """Raised when ``--backend`` names a backend we don't recognize."""


def resolve_base_url(name: str, base_url: str | None) -> str | None:
    """Explicit ``base_url`` wins; otherwise a known cloud alias supplies its own."""
    return base_url if base_url is not None else _ALIAS_BASE_URLS.get(name)


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
        return OpenAICompatibleClient(
            base_url=resolve_base_url(name, base_url), model=model, api_key=api_key
        )
    raise UnknownBackendError(f"unknown backend {name!r}; choose one of {_KNOWN}")


__all__ = ["UnknownBackendError", "build_client", "resolve_base_url"]
