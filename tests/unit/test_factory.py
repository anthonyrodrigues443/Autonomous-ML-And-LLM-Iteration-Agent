"""Tests for the LLM backend factory."""

from __future__ import annotations

import pytest

from iterate.llm.factory import UnknownBackendError, build_client
from iterate.llm.ollama_client import OllamaClient
from iterate.llm.openai_compatible import OpenAICompatibleClient


def test_default_backend_is_ollama() -> None:
    assert isinstance(build_client(), OllamaClient)


def test_explicit_ollama_backend() -> None:
    assert isinstance(build_client("ollama"), OllamaClient)


def test_openai_compatible_backend() -> None:
    client = build_client("openai-compatible", api_key="sk-fake")
    assert isinstance(client, OpenAICompatibleClient)


def test_cloud_aliases_route_to_openai_compatible() -> None:
    for alias in ("groq", "together", "deepseek", "openai", "vllm"):
        assert isinstance(build_client(alias, api_key="sk-fake"), OpenAICompatibleClient)


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnknownBackendError, match="unknown backend"):
        build_client("not-a-real-backend")


def test_model_override_threads_through_to_ollama() -> None:
    client = build_client("ollama", model="qwen3:8b")
    assert client.model == "qwen3:8b"
