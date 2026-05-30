"""Tests for OllamaClient — native /api/chat translation, with httpx mocked."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iterate.llm.base import LLMClient
from iterate.llm.ollama_client import OllamaClient
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    import pytest

_TOOL = ToolSpec(name="propose_candidate", description="Propose.", parameters={"type": "object"})

_NATIVE_TOOL_REPLY = {
    "model": "qwen3:14b",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc",
                "function": {
                    "name": "propose_candidate",
                    "arguments": {"model": "xgboost.XGBClassifier", "params": {"max_depth": 4}},
                },
            }
        ],
    },
    "done_reason": "stop",
    "prompt_eval_count": 389,
    "eval_count": 116,
}


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


def _patch_post(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any], captured: dict[str, Any]
) -> None:
    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _FakeResponse(reply)

    monkeypatch.setattr("iterate.llm.ollama_client.httpx.post", fake_post)


def test_satisfies_the_llm_protocol() -> None:
    assert isinstance(OllamaClient(), LLMClient)


def test_disables_thinking_by_default_and_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, _NATIVE_TOOL_REPLY, captured)

    resp = OllamaClient().chat([Message(role="user", content="hi")], tools=[_TOOL])

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["tools"][0]["function"]["name"] == "propose_candidate"
    assert resp.has_tool_calls
    call = resp.tool_calls[0]
    assert call.name == "propose_candidate"
    assert call.arguments == {"model": "xgboost.XGBClassifier", "params": {"max_depth": 4}}
    assert call.id == "call_abc"
    assert resp.finish_reason == "tool_calls"


def test_usage_maps_from_native_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _NATIVE_TOOL_REPLY, {})
    resp = OllamaClient().chat([Message(role="user", content="hi")])
    assert resp.usage.prompt_tokens == 389
    assert resp.usage.completion_tokens == 116
    assert resp.usage.total_tokens == 505


def test_think_true_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, _NATIVE_TOOL_REPLY, captured)
    OllamaClient(think=True).chat([Message(role="user", content="hi")])
    assert captured["payload"]["think"] is True


def test_temperature_and_max_tokens_map_to_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, _NATIVE_TOOL_REPLY, captured)
    OllamaClient().chat([Message(role="user", content="hi")], temperature=0.2, max_tokens=256)
    assert captured["payload"]["options"]["temperature"] == 0.2
    assert captured["payload"]["options"]["num_predict"] == 256


def test_plain_content_no_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {
        "model": "qwen3:14b",
        "message": {"role": "assistant", "content": "hello"},
        "done_reason": "stop",
    }
    _patch_post(monkeypatch, reply, {})
    resp = OllamaClient().chat([Message(role="user", content="hi")])
    assert resp.content == "hello"
    assert not resp.has_tool_calls
    assert resp.finish_reason == "stop"
