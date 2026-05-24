"""LLMClient over any OpenAI-compatible endpoint.

One client for Ollama (default), Groq, Together, Deepseek, OpenAI, vLLM, and
friends — they all speak the OpenAI chat-completions wire format, so swapping
backends is just ``base_url`` + ``model`` + ``api_key``. This class is the
translation layer: it maps our provider-agnostic types to the OpenAI request
shape and the response back to a normalized ``ChatResponse``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from iterate.config import get_settings
from iterate.schemas.llm import ChatResponse, ToolCall, Usage

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from iterate.schemas.llm import Message, ToolSpec

# Transient failures worth retrying; bad requests / auth errors are not.
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class OpenAICompatibleClient:
    """Calls an OpenAI-compatible chat endpoint. Defaults come from config (local Ollama).

    Any constructor argument left ``None`` falls back to the central ``Settings``
    (env / .env), so deployment overrides happen via secrets, not code.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model if model is not None else settings.iterate_model
        self._client = OpenAI(
            base_url=base_url if base_url is not None else settings.iterate_backend_url,
            api_key=api_key if api_key is not None else settings.iterate_backend_api_key,
            timeout=timeout if timeout is not None else settings.iterate_backend_timeout,
        )

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
        oai_messages = cast(
            "list[ChatCompletionMessageParam]",
            [self._to_openai_message(m) for m in messages],
        )
        # Only forward params the caller actually set, so backend defaults apply otherwise.
        params: dict[str, Any] = {}
        if tools:
            params["tools"] = [self._to_openai_tool(t) for t in tools]
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return self._to_chat_response(self._create(oai_messages, params))

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )
    def _create(
        self,
        messages: list[ChatCompletionMessageParam],
        params: dict[str, Any],
    ) -> ChatCompletion:
        # **params yields an ambiguous (streaming?) overload → Any; we never stream.
        return cast(
            "ChatCompletion",
            self._client.chat.completions.create(model=self._model, messages=messages, **params),
        )

    @staticmethod
    def _to_openai_message(m: Message) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name is not None:
            msg["name"] = m.name
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        return msg

    @staticmethod
    def _to_openai_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }

    @staticmethod
    def _to_chat_response(completion: ChatCompletion) -> ChatResponse:
        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            if tc.type != "function":  # ignore non-function tool calls (e.g. custom)
                continue
            raw = tc.function.arguments
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(raw) if raw else {},
                )
            )
        usage = Usage()
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            model=completion.model,
            finish_reason=choice.finish_reason,
        )


__all__ = ["OpenAICompatibleClient"]
