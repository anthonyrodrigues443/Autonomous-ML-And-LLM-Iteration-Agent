"""LLMClient over Ollama's NATIVE `/api/chat` endpoint.

Ollama gets its own adapter — separate from `OpenAICompatibleClient` — because its
OpenAI-compatible `/v1` layer cannot disable qwen3's thinking mode. Only the native
endpoint honors `think: false`, and that's the difference between a ~20s and a ~128s
tool call (thinking-on is also less reliable at emitting the call at all). This
client speaks the native wire format and implements the same `LLMClient` protocol,
so the agent swaps to it by config like any other backend.

Verified 2026-05-29: `/v1` ignores `think:false`, the `/no_think` soft prompt, and
`chat_template_kwargs` alike; the native endpoint is the only lever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from iterate.config import get_settings
from iterate.schemas.llm import ChatResponse, ToolCall, Usage

if TYPE_CHECKING:
    from iterate.schemas.llm import Message, ToolSpec

# Transient connection/timeout failures worth retrying; HTTP 4xx/5xx are not.
_RETRYABLE = (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)


class OllamaClient:
    """Calls Ollama's native `/api/chat`. Disables thinking by default (`think=False`).

    Any constructor argument left ``None`` falls back to the central ``Settings``
    (env / .env), so deployment overrides happen via secrets, not code.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        think: bool = False,
        num_ctx: int | None = None,
    ) -> None:
        settings = get_settings()
        self._host = (host if host is not None else settings.ollama_host).rstrip("/")
        self._model = model if model is not None else settings.iterate_model
        self._timeout = timeout if timeout is not None else settings.ollama_timeout
        self._think = think
        self._num_ctx = num_ctx if num_ctx is not None else settings.ollama_num_ctx

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "think": self._think,
            "stream": False,
            "messages": [self._to_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [self._to_tool(t) for t in tools]
        # Always pin num_ctx: Ollama's 4096 default front-truncates long sessions,
        # silently dropping the system prompt and tool schema.
        options: dict[str, Any] = {"num_ctx": self._num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload["options"] = options
        return self._to_chat_response(self._post(payload))

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    @staticmethod
    def _to_message(m: Message) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            # Native format: arguments is an object, not a JSON string.
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
            ]
        if m.role == "tool" and m.name is not None:
            msg["tool_name"] = m.name
        return msg

    @staticmethod
    def _to_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }

    @staticmethod
    def _to_chat_response(data: dict[str, Any]) -> ChatResponse:
        message = data.get("message") or {}
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{uuid4().hex[:8]}"),
                    name=str(fn.get("name") or ""),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatResponse(
            content=message.get("content") or None,
            tool_calls=tool_calls,
            usage=usage,
            model=str(data.get("model") or ""),
            finish_reason="tool_calls" if tool_calls else data.get("done_reason"),
        )


__all__ = ["OllamaClient"]
