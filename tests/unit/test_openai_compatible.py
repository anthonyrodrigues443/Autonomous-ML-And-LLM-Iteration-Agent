"""Tests for the OpenAI-compatible LLM client.

Offline tests mock the SDK call to prove translation both ways — our types →
OpenAI request, and OpenAI response → ``ChatResponse`` — deterministically, with
no network. A live smoke test exercises a real Ollama call and skips cleanly when
the backend is down (version skew, not pulled, offline, etc.).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError, NotFoundError

from iterate.llm.openai_compatible import OpenAICompatibleClient
from iterate.schemas.llm import Message, ToolCall, ToolSpec


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(model="test-model")


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: tuple[int, int, int] | None = (5, 1, 6),
    model: str = "test-model",
    finish: str = "stop",
) -> Any:
    """A duck-typed stand-in for an OpenAI ChatCompletion (version-robust)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish)
    token_usage = (
        SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2])
        if usage is not None
        else None
    )
    return SimpleNamespace(choices=[choice], usage=token_usage, model=model)


def _tool_call(name: str, arguments: str, *, kind: str = "function") -> Any:
    return SimpleNamespace(id="c1", type=kind, function=SimpleNamespace(name=name, arguments=arguments))


def _patch_create(monkeypatch: pytest.MonkeyPatch, client: OpenAICompatibleClient, completion: Any) -> dict[str, Any]:
    """Replace the SDK call with a stub; return a dict that captures its kwargs."""
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return completion

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    return captured


# ─── outgoing: our types → OpenAI request shape ────────────────────


def test_chat_translates_messages_tools_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    captured = _patch_create(monkeypatch, client, _completion(content="ok"))

    client.chat(
        [Message(role="user", content="hi")],
        tools=[ToolSpec(name="add", description="Add two ints.", parameters={"type": "object"})],
        temperature=0.0,
        max_tokens=64,
    )

    assert captured["model"] == "test-model"
    assert captured["messages"][0] == {"role": "user", "content": "hi"}
    assert captured["tools"][0] == {
        "type": "function",
        "function": {"name": "add", "description": "Add two ints.", "parameters": {"type": "object"}},
    }
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 64


def test_chat_omits_params_the_caller_did_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    captured = _patch_create(monkeypatch, client, _completion(content="ok"))

    client.chat([Message(role="user", content="hi")])

    assert "tools" not in captured
    assert "temperature" not in captured
    assert "max_tokens" not in captured


def test_chat_serializes_outgoing_tool_calls_to_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    captured = _patch_create(monkeypatch, client, _completion(content="ok"))

    client.chat(
        [Message(role="assistant", content=None, tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2})])]
    )

    sent = captured["messages"][0]["tool_calls"][0]
    assert sent["id"] == "c1"
    assert sent["type"] == "function"
    assert sent["function"]["arguments"] == '{"a": 2}'  # dict -> JSON string on the wire


# ─── incoming: OpenAI response → ChatResponse ──────────────────────


def test_chat_parses_plain_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    _patch_create(monkeypatch, client, _completion(content="ok"))

    resp = client.chat([Message(role="user", content="hi")])

    assert resp.content == "ok"
    assert resp.model == "test-model"
    assert resp.finish_reason == "stop"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens) == (5, 1, 6)
    assert not resp.has_tool_calls


def test_chat_parses_tool_calls_into_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    completion = _completion(content=None, tool_calls=[_tool_call("add", '{"a": 2, "b": 2}')], finish="tool_calls")
    _patch_create(monkeypatch, client, completion)

    resp = client.chat([Message(role="user", content="add 2 and 2")])

    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].arguments == {"a": 2, "b": 2}  # parsed from JSON string


def test_chat_ignores_non_function_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    _patch_create(monkeypatch, client, _completion(tool_calls=[_tool_call("thing", "{}", kind="custom")]))

    resp = client.chat([Message(role="user", content="x")])

    assert resp.tool_calls == []


def test_chat_handles_empty_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    _patch_create(monkeypatch, client, _completion(tool_calls=[_tool_call("ping", "")]))

    resp = client.chat([Message(role="user", content="x")])

    assert resp.tool_calls[0].arguments == {}


def test_chat_defaults_usage_to_zero_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    _patch_create(monkeypatch, client, _completion(content="ok", usage=None))

    resp = client.chat([Message(role="user", content="hi")])

    assert resp.usage.total_tokens == 0


# ─── live smoke: real backend, skips when unavailable ──────────────


@pytest.mark.integration
def test_live_ollama_smoke() -> None:
    client = OpenAICompatibleClient()
    try:
        resp = client.chat(
            [Message(role="user", content="Reply with exactly: ok")],
            temperature=0,
            max_tokens=10,
        )
    except (APIConnectionError, APITimeoutError, InternalServerError, NotFoundError) as exc:
        pytest.skip(f"Ollama backend unavailable: {exc}")

    assert resp.model
    assert resp.content is not None or resp.has_tool_calls
