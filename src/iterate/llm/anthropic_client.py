"""LLMClient over Anthropic's Messages API, for Claude as the model under test.

Imported only when a prompt run names the anthropic provider: the SDK is an optional
extra (`pip install 'iterate-ai[anthropic]'`), and every other provider runs without it.

The client reads nothing from the environment, on the host and in a cell alike, so the
two call one place the same way. The SDK reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
ANTHROPIC_BASE_URL, profiles on disk and ANTHROPIC_CUSTOM_HEADERS by itself; the key
and the address are always passed, and the custom headers are taken off once the
client is built. A copy of the client (`with_options`) reads the environment again, so
iterate makes none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anthropic
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from iterate.llm.claude_models import rules_for
from iterate.schemas.llm import ChatResponse, ToolCall, Usage

if TYPE_CHECKING:
    from iterate.schemas.llm import Message, ToolSpec

PUBLIC_ADDRESS = "https://api.anthropic.com"


_BACKOFF = wait_exponential(min=1, max=10)
_LONGEST_WAIT = 60.0


def _worth_retrying(exc: BaseException) -> bool:
    # Read off the status, not the class: the SDK names 529 OverloadedError, apart from
    # its InternalServerError.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429 and _spend_limit(exc):
            return False
        return status in (408, 409, 429) or status >= 500
    return isinstance(exc, anthropic.APIConnectionError)


def _spend_limit(exc: BaseException) -> bool:
    """A 429 for the spend limit the organisation has reached, which no wait lifts."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    code = details.get("error_code") if isinstance(details, dict) else None
    return code == "enforced_spend_limit_reached"


def _wait(state: RetryCallState) -> float:
    """As long as the provider asked, up to a minute, else a growing backoff."""
    exc = state.outcome.exception() if state.outcome is not None else None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    try:
        asked = float(headers.get("retry-after")) if headers is not None else None
    except (TypeError, ValueError):
        asked = None
    if asked is not None and asked >= 0:
        return min(asked, _LONGEST_WAIT)
    return float(_BACKOFF(state))


def keep_the_environment_out(client: anthropic.Anthropic) -> anthropic.Anthropic:
    """The client without the headers the SDK read from ANTHROPIC_CUSTOM_HEADERS, which
    a cell never holds."""
    client._custom_headers = {}
    return client


def build_sdk_client(
    *, base_url: str, api_key: str, timeout: float, max_retries: int = 0
) -> anthropic.Anthropic:
    """The SDK's client, given everything it would otherwise look for."""
    return keep_the_environment_out(
        anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
    )


class AnthropicClient:
    """Calls Claude through the Messages API. One request per chat, no streaming."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = PUBLIC_ADDRESS,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._rules = rules_for(model)
        # The SDK's own retries are off: tenacity's 3 tries run inside each of the
        # runtime's 3, and a record left unanswered is asked once more, so a steady 429
        # costs a record 18 requests at most.
        self._client = build_sdk_client(base_url=base_url, api_key=api_key, timeout=timeout)

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
        system = "\n\n".join(m.content or "" for m in messages if m.role == "system")
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens if max_tokens is not None else 1024,
            "messages": [self._to_message(m) for m in messages if m.role != "system"],
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            # Forced where the model takes it: left to choose, Claude often writes a line
            # before the call, and an answer allowed 64 tokens can end in that line.
            params["tool_choice"] = (
                {"type": "tool", "name": tools[0].name, "disable_parallel_tool_use": True}
                if len(tools) == 1 and self._rules.force_tool
                else {"type": "auto", "disable_parallel_tool_use": True}
            )
        if self._rules.disable_thinking:
            params["thinking"] = {"type": "disabled"}
        if temperature is not None and self._rules.temperature:
            params["extra_body"] = {"temperature": temperature}
        return self._to_chat_response(self._create(params))

    @retry(
        retry=retry_if_exception(_worth_retrying),
        stop=stop_after_attempt(3),
        wait=_wait,
        reraise=True,
    )
    def _create(self, params: dict[str, Any]) -> Any:
        return self._client.messages.create(**params)

    @staticmethod
    def _to_message(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }
                ],
            }
        if m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = (
                [{"type": "text", "text": m.content}] if m.content else []
            )
            blocks += [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            ]
            return {"role": "assistant", "content": blocks}
        text = m.content or ""
        if m.role == "user" and not text.strip():
            # The API refuses a turn with no text, on every try, and a record left
            # unanswered fails the pass: an empty record is shown as empty.
            text = "(empty)"
        return {"role": m.role, "content": text}

    def _to_chat_response(self, message: Any) -> ChatResponse:
        stop = getattr(message, "stop_reason", None)
        blocks = list(getattr(message, "content", None) or [])
        calls = [
            ToolCall(id=str(b.id), name=str(b.name), arguments=dict(b.input or {}))
            for b in blocks
            if getattr(b, "type", None) == "tool_use"
        ]
        text = "".join(str(b.text) for b in blocks if getattr(b, "type", None) == "text")
        # A refusal and a reply cut off at the cap are the model's answers, scored as
        # unusable: were they errors, a prompt that provoked them would leave its hard
        # records out of the score. A refusal's words may name a label, and a sentence
        # or a tool call cut off mid-way is not the answer the model was heading to.
        if stop in ("refusal", "max_tokens"):
            calls, text = [], ""
        usage = getattr(message, "usage", None)
        prompt_tokens = sum(
            int(getattr(usage, name, 0) or 0)
            for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return ChatResponse(
            content=text or None,
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=str(getattr(message, "model", None) or self._model),
            finish_reason=str(stop) if stop is not None else None,
        )


def served(model: str | None, *, base_url: str, api_key: str, timeout: float) -> list[str] | None:
    """The names this key is served under, as far as one free call tells: the name
    asked for and its full ID when the model is served, nothing when it is not (404),
    and None when no model was asked for. A bad key raises, with status 401."""
    client = build_sdk_client(base_url=base_url, api_key=api_key, timeout=timeout)
    if model is None:
        client.models.list(limit=1)
        return None
    try:
        found = client.models.retrieve(model)
    except anthropic.NotFoundError:
        # A 404 names a model not served only where the list itself is found.
        client.models.list(limit=1)
        return []
    return [model, str(getattr(found, "id", model))]


__all__ = [
    "PUBLIC_ADDRESS",
    "AnthropicClient",
    "build_sdk_client",
    "keep_the_environment_out",
    "served",
]
