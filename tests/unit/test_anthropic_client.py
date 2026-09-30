"""Claude as the model under test: the request each model family takes, the answer read
back, the retries, and a client that reads nothing from the environment.

The SDK is faked at its methods, never at HTTP: the 0.x line runs on httpx and the 1.x
line CI installs runs on httpx2, and a transport built for one is refused by the other.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

from iterate.llm import anthropic_client, claude_models  # noqa: E402
from iterate.llm.anthropic_client import AnthropicClient  # noqa: E402
from iterate.schemas.llm import Message, ToolSpec  # noqa: E402

pytestmark = pytest.mark.unit

# As the decorator built it, before a fixture takes the waits out.
_WAIT_AS_BUILT = AnthropicClient._create.retry.wait  # type: ignore[attr-defined]

_TOOL = ToolSpec(
    name="answer",
    description="The answer",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string", "enum": ["ironic", "not ironic"]}},
        "required": ["value"],
    },
)
_ASK = [
    Message(role="system", content="Say whether the tweet is ironic."),
    Message(role="user", content="love waiting 3 hours at the dentist"),
]


def _reply(
    *blocks: Any,
    stop: str = "tool_use",
    tokens: tuple[int, int] = (500, 20),
    cache: tuple[int | None, int | None] = (None, None),
) -> Any:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        model="claude-haiku-4-5-20251001",
        usage=SimpleNamespace(
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            cache_creation_input_tokens=cache[0],
            cache_read_input_tokens=cache[1],
        ),
    )


def _tool_use(value: str) -> Any:
    return SimpleNamespace(type="tool_use", id="toolu_1", name="answer", input={"value": value})


def _client(
    monkeypatch: pytest.MonkeyPatch, model: str, *replies: Any
) -> tuple[AnthropicClient, list[dict[str, Any]]]:
    client = AnthropicClient(model=model, api_key="sk-ant-test-key")
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def create(**params: Any) -> Any:
        sent.append(params)
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(client._client.messages, "create", create)
    return client, sent


@pytest.fixture(autouse=True)
def _no_wait_between_tries(monkeypatch: pytest.MonkeyPatch) -> None:
    import tenacity

    monkeypatch.setattr(AnthropicClient._create.retry, "wait", tenacity.wait_none())  # type: ignore[attr-defined]


def _status(code: int) -> Exception:
    """An SDK error carrying a status, built without the response object the SDK's own
    constructor wants (httpx on 0.x, httpx2 on 1.x)."""
    error = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
    error.status_code = code
    return error


# ─── the request each family takes ────────────────────────────────────────────


def test_haiku_is_forced_to_the_tool_at_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, sent = _client(monkeypatch, "claude-haiku-4-5", _reply(_tool_use("ironic")))
    client.chat(_ASK, tools=[_TOOL], temperature=0.0, max_tokens=64)
    (params,) = sent
    assert params["model"] == "claude-haiku-4-5"
    assert params["system"] == "Say whether the tweet is ironic."
    assert params["messages"] == [
        {"role": "user", "content": "love waiting 3 hours at the dentist"}
    ]
    assert params["tools"] == [
        {"name": "answer", "description": "The answer", "input_schema": _TOOL.parameters}
    ]
    assert params["tool_choice"] == {
        "type": "tool",
        "name": "answer",
        "disable_parallel_tool_use": True,
    }
    # A named argument is a TypeError on the SDK's 1.x line; extra_body works on both.
    assert params["extra_body"] == {"temperature": 0.0}
    assert "temperature" not in params
    assert "thinking" not in params
    assert params["max_tokens"] == 64


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5"])
def test_a_model_that_rejects_temperature_is_sent_none_and_thinks_no_more(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    client, sent = _client(monkeypatch, model, _reply(_tool_use("ironic")))
    client.chat(_ASK, tools=[_TOOL], temperature=0.0, max_tokens=64)
    (params,) = sent
    assert "extra_body" not in params
    assert "temperature" not in params
    assert params["thinking"] == {"type": "disabled"}
    assert params["tool_choice"]["type"] == "tool"


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-next-9"]
)
def test_a_model_that_cannot_be_forced_or_stop_thinking_is_asked_nothing_it_refuses(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    client, sent = _client(monkeypatch, model, _reply(_tool_use("ironic")))
    client.chat(_ASK, tools=[_TOOL], temperature=0.0, max_tokens=64)
    (params,) = sent
    assert params["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert not {"thinking", "temperature", "extra_body"} & set(params)


def test_a_prompt_run_calls_only_the_families_it_knows_can_answer_in_64_tokens() -> None:
    assert claude_models.RUNS == ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
    for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-opus-5"):
        assert claude_models.refused(model) is None
    for model in (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
        "claude-mythos-preview",
    ):
        assert "thinks before it answers" in str(claude_models.refused(model))
    # Thinking off by default, but what each takes was not checked: asked the wrong
    # way, every record would fail.
    for model in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-3-5-haiku-20241022"):
        assert "not a Claude model iterate knows how to ask yet" in str(
            claude_models.refused(model)
        )
    assert claude_models.rules_for("claude-opus-5").disable_thinking
    assert not claude_models.rules_for("claude-opus-5-5").force_tool
    assert claude_models.rules_for("claude-haiku-4-5-20251001").temperature


def test_a_conversation_with_tool_turns_takes_the_messages_api_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iterate.schemas.llm import ToolCall

    client, sent = _client(monkeypatch, "claude-haiku-4-5", _reply(_tool_use("ironic")))
    turns = [
        *_ASK,
        Message(
            role="assistant",
            content="checking",
            tool_calls=[ToolCall(id="toolu_0", name="answer", arguments={"value": "x"})],
        ),
        Message(role="tool", tool_call_id="toolu_0", content="not a label"),
    ]
    client.chat(turns, tools=[_TOOL, _TOOL.model_copy(update={"name": "other"})])
    (params,) = sent
    assert params["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_0", "name": "answer", "input": {"value": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0", "content": "not a label"}
            ],
        },
    ]
    assert params["tool_choice"]["type"] == "auto"
    assert params["max_tokens"] == 1024


def test_a_record_with_no_text_is_sent_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, "claude-haiku-4-5", _reply(_tool_use("ironic")))
    client.chat([_ASK[0], Message(role="user", content="  ")], tools=[_TOOL], max_tokens=64)
    assert sent[0]["messages"] == [{"role": "user", "content": "(empty)"}]


# ─── the answer read back ─────────────────────────────────────────────────────


def test_the_tool_call_and_every_token_billed_come_back(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = _reply(
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="It reads as irony."),
        _tool_use("ironic"),
        tokens=(480, 31),
        cache=(12, 8),
    )
    client, _ = _client(monkeypatch, "claude-haiku-4-5", reply)
    answer = client.chat(_ASK, tools=[_TOOL], temperature=0.0, max_tokens=64)
    assert [call.arguments for call in answer.tool_calls] == [{"value": "ironic"}]
    assert answer.content == "It reads as irony."
    assert (answer.usage.prompt_tokens, answer.usage.completion_tokens) == (500, 31)
    assert answer.usage.total_tokens == 531
    assert (answer.model, answer.finish_reason) == ("claude-haiku-4-5-20251001", "tool_use")


def test_a_refusal_comes_back_as_no_answer_whatever_it_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said = _reply(SimpleNamespace(type="text", text="I won't say if it is ironic"), stop="refusal")
    client, _ = _client(monkeypatch, "claude-sonnet-5", said)
    answer = client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert (answer.tool_calls, answer.content, answer.finish_reason) == ([], None, "refusal")


def test_a_reply_cut_off_at_the_cap_is_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking_only = _reply(SimpleNamespace(type="thinking", thinking=""), stop="max_tokens")
    client, _ = _client(monkeypatch, "claude-opus-5-5", thinking_only)
    answer = client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert (answer.tool_calls, answer.content, answer.finish_reason) == ([], None, "max_tokens")

    cut_call = _reply(_tool_use("iro"), stop="max_tokens")
    client, _ = _client(monkeypatch, "claude-haiku-4-5", cut_call)
    assert client.chat(_ASK, tools=[_TOOL], max_tokens=64).tool_calls == []

    cut_text = _reply(SimpleNamespace(type="text", text="ironic, because"), stop="max_tokens")
    client, _ = _client(monkeypatch, "claude-haiku-4-5", cut_text)
    answer = client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    # "reads as ironic, but" would be read as ironic.
    assert (answer.tool_calls, answer.content) == ([], None)


# ─── retries ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [429, 529, 500, 408, 409])
def test_a_busy_or_failing_provider_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    client, sent = _client(
        monkeypatch, "claude-haiku-4-5", _status(code), _status(code), _reply(_tool_use("ironic"))
    )
    answer = client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert len(sent) == 3
    assert answer.tool_calls[0].arguments == {"value": "ironic"}


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_request_the_provider_rejects_is_not_tried_again(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    client, sent = _client(monkeypatch, "claude-haiku-4-5", _status(code))
    with pytest.raises(anthropic.APIStatusError):
        client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert len(sent) == 1


def test_three_tries_and_then_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, "claude-haiku-4-5", *[_status(429)] * 3)
    with pytest.raises(anthropic.APIStatusError):
        client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert len(sent) == 3


def test_a_dropped_connection_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    dropped = anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)
    client, sent = _client(
        monkeypatch, "claude-haiku-4-5", dropped, dropped, _reply(_tool_use("ironic"))
    )
    assert client.chat(_ASK, tools=[_TOOL], max_tokens=64).tool_calls
    assert len(sent) == 3


def test_the_retry_waits_what_the_provider_asks_for(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = _status(429)
    busy.response = SimpleNamespace(headers={"retry-after": "7"})  # type: ignore[attr-defined]
    client, _ = _client(monkeypatch, "claude-haiku-4-5", busy, _reply(_tool_use("ironic")))
    waited: list[float] = []
    retrying = AnthropicClient._create.retry  # type: ignore[attr-defined]
    monkeypatch.setattr(retrying, "wait", _WAIT_AS_BUILT)
    monkeypatch.setattr(retrying, "sleep", waited.append)
    client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert waited == [7.0]


def test_a_spend_limit_is_not_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    capped = _status(429)
    capped.body = {  # type: ignore[attr-defined]
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "details": {"error_code": "enforced_spend_limit_reached"},
        },
    }
    client, sent = _client(monkeypatch, "claude-haiku-4-5", capped)
    with pytest.raises(anthropic.APIStatusError):
        client.chat(_ASK, tools=[_TOOL], max_tokens=64)
    assert len(sent) == 1


def test_the_wait_is_the_one_the_provider_asks_for_up_to_a_minute() -> None:
    import tenacity

    def after(headers: dict[str, str] | None) -> float:
        error = _status(429)
        error.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
        state = tenacity.RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]
        state.attempt_number = 1
        state.set_exception((type(error), error, None))
        return anthropic_client._wait(state)

    assert after({"retry-after": "7"}) == 7.0
    assert after({"retry-after": "3600"}) == 60.0
    assert 0 < after({"retry-after": "soon"}) <= 10
    assert 0 < after(None) <= 10


def test_the_sdks_own_retries_are_off_so_one_layer_counts() -> None:
    client = AnthropicClient(model="claude-haiku-4-5", api_key="sk-ant-test-key")
    assert client._client.max_retries == 0


# ─── nothing from the environment ─────────────────────────────────────────────


def test_the_client_reads_nothing_the_environment_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-from-the-environment")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Gateway-Auth: secret\nX-Team: a")
    sdk = AnthropicClient(model="claude-haiku-4-5", api_key="sk-ant-given")._client
    assert str(sdk.base_url).rstrip("/") == "https://api.anthropic.com"
    assert (sdk.api_key, sdk.auth_token) == ("sk-ant-given", None)
    assert dict(sdk._custom_headers) == {}


def test_a_gateway_is_sent_what_a_cell_would_send_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cell never holds ANTHROPIC_CUSTOM_HEADERS, so the host must not send them
    either: a baseline scored through a gateway that a cell cannot pass would make every
    code candidate fail where the host's passes scored."""
    from iterate.adapters.compute.kernel import is_secret

    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Gateway-Auth: secret")
    assert is_secret("ANTHROPIC_CUSTOM_HEADERS")
    sdk = AnthropicClient(
        model="claude-haiku-4-5", api_key="k", base_url="https://gateway.test/anthropic"
    )._client
    assert str(sdk.base_url).rstrip("/") == "https://gateway.test/anthropic"
    assert dict(sdk._custom_headers) == {}


# ─── which models the key is served ───────────────────────────────────────────


def _models_answering(
    monkeypatch: pytest.MonkeyPatch, answer: Any, *, address_missing: bool = False
) -> None:
    real = anthropic_client.build_sdk_client

    def built(**kwargs: Any) -> Any:
        sdk = real(**kwargs)

        def retrieve(model: str) -> Any:
            if isinstance(answer, BaseException):
                raise answer
            return answer

        def listing(**kwargs: Any) -> Any:
            if isinstance(answer, BaseException) and address_missing:
                raise answer
            return None

        monkeypatch.setattr(sdk.models, "retrieve", retrieve)
        monkeypatch.setattr(sdk.models, "list", listing)
        return sdk

    monkeypatch.setattr(anthropic_client, "build_sdk_client", built)


def test_a_model_served_is_named_with_its_full_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _models_answering(monkeypatch, SimpleNamespace(id="claude-haiku-4-5-20251001"))
    served = anthropic_client.served(
        "claude-haiku-4-5", base_url="https://api.anthropic.com", api_key="k", timeout=5
    )
    assert served == ["claude-haiku-4-5", "claude-haiku-4-5-20251001"]


def test_a_model_not_served_is_none_of_the_names(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = anthropic.NotFoundError.__new__(anthropic.NotFoundError)
    missing.status_code = 404
    _models_answering(monkeypatch, missing)
    assert (
        anthropic_client.served(
            "claude-haiku-9", base_url="https://api.anthropic.com", api_key="k", timeout=5
        )
        == []
    )


def test_an_address_that_lists_nothing_is_not_taken_for_a_model_not_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = anthropic.NotFoundError.__new__(anthropic.NotFoundError)
    missing.status_code = 404
    _models_answering(monkeypatch, missing, address_missing=True)
    with pytest.raises(anthropic.NotFoundError):
        anthropic_client.served(
            "claude-haiku-4-5", base_url="https://gateway.test/wrong", api_key="k", timeout=5
        )


def test_a_refused_key_is_raised_with_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _models_answering(monkeypatch, _status(401))
    with pytest.raises(anthropic.APIStatusError) as raised:
        anthropic_client.served(
            "claude-haiku-4-5", base_url="https://api.anthropic.com", api_key="k", timeout=5
        )
    assert raised.value.status_code == 401


# ─── built by the factory, on the host and in a cell ──────────────────────────


def test_a_cell_builds_claude_with_the_one_key_it_was_handed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iterate.core.prompt_runtime import make_ask
    from iterate.llm import factory

    monkeypatch.setenv(factory.TARGET_KEY_ENV, "sk-ant-handed")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-other")
    built: list[Any] = []
    real = factory.build_target_client

    def spy(*args: Any, **kwargs: Any) -> Any:
        client = real(*args, **kwargs)
        built.append(client)
        monkeypatch.setattr(
            client._client.messages, "create", lambda **params: _reply(_tool_use("ironic"))
        )
        return client

    monkeypatch.setattr(factory, "build_target_client", spy)
    ask = make_ask(
        columns=["text"],
        labels=["ironic", "not ironic"],
        backend="anthropic",
        model="claude-haiku-4-5",
        base_url="https://api.anthropic.com",
        cache_path=None,
    )
    from iterate.core.prompting import Prompt

    answers = ask(Prompt(system="s", user_template="{text}"), [{"text": "a"}])
    assert answers == ["ironic"]
    (client,) = built
    assert isinstance(client, AnthropicClient)
    assert client._client.api_key == "sk-ant-handed"
    assert str(client._client.base_url).rstrip("/") == "https://api.anthropic.com"


def test_claude_with_no_key_is_refused_where_it_is_built() -> None:
    from iterate.llm import factory

    with pytest.raises(factory.ProviderError, match="ANTHROPIC_API_KEY"):
        factory.build_target_client(
            "anthropic", model="claude-haiku-4-5", base_url=None, api_key=None
        )


def test_the_model_named_is_checked_against_what_the_key_is_served() -> None:
    from iterate.llm import factory

    found = factory.Provider(
        name="anthropic", base_url="https://api.anthropic.com", api_key="sk-ant-k"
    )

    def serves(*names: str) -> Any:
        return lambda provider, timeout, model: list(names)

    assert (
        factory.refused_key(found, model="claude-haiku-4-5", lister=serves("claude-haiku-4-5"))
        is None
    )
    assert factory.refused_key(found, model="claude-haiku-4-6", lister=serves()) == (
        "anthropic does not serve claude-haiku-4-6 to this key. Check the name in its model list"
    )
    # A gateway the user runs in front of Claude may name its models otherwise.
    gateway = dataclasses.replace(found, base_url="https://llm.gateway.test")
    assert factory.refused_key(gateway, model="haiku", lister=serves()) is None


def test_nothing_imports_the_anthropic_library_until_claude_is_called() -> None:
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['anthropic'] = None\n"
        "import iterate.cli, iterate.llm.factory, iterate.core.prompt_runtime, "
        "iterate.targets.prompt, iterate.llm.claude_models\n"
        "print('ok')"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.stdout.strip() == "ok", done.stderr


def test_an_address_ending_in_v1_is_refused_before_the_library_doubles_it() -> None:
    from iterate.llm import factory

    found = factory.Provider(
        name="anthropic", base_url="https://api.anthropic.com/v1", api_key="sk-ant-k"
    )
    assert factory.not_callable(found, given_as="--target-base-url x") == (
        "--target-base-url x ends in /v1, which Anthropic's library adds itself. Drop the /v1"
    )
    assert (
        factory.not_callable(dataclasses.replace(found, base_url="https://api.anthropic.com"))
        is None
    )
