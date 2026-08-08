"""`ask()`: the answer tool, coercion, caching, and never letting one row kill a run."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from iterate.core.prompt_runtime import (
    UNPARSEABLE,
    AnswerCache,
    AskStats,
    answer_tool,
    ask,
    coerce,
)
from iterate.core.prompting import Prompt
from iterate.schemas.llm import ChatResponse, ToolCall, Usage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iterate.schemas.llm import Message, ToolSpec

pytestmark = pytest.mark.unit

PROMPT = Prompt(system="say toxic or not toxic", user_template="{text}")
LABELS = ["toxic", "not toxic"]
ROWS = [{"text": "you are awful"}, {"text": "have a nice day"}]


class FakeClient:
    """Replays scripted replies and records what it was asked."""

    def __init__(self, replies: Sequence[Any], *, model: str = "fake-12b") -> None:
        self._replies = list(replies)
        self._model = model
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []
        self._lock = threading.Lock()

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
        with self._lock:
            self.calls.append(messages)
            self.tools_seen.append(tools)
            reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _tool_reply(value: str) -> ChatResponse:
    return ChatResponse(
        model="fake-12b",
        tool_calls=[ToolCall(id="1", name="answer", arguments={"value": value})],
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def _text_reply(text: str) -> ChatResponse:
    return ChatResponse(model="fake-12b", content=text, usage=Usage(prompt_tokens=10))


def test_the_answer_tool_constrains_the_label_set() -> None:
    """Asking a model to reply with one of three labels is a request; a tool whose
    only argument is an enum makes anything else impossible."""
    spec = answer_tool(LABELS)

    assert spec.parameters["properties"]["value"]["enum"] == LABELS


def test_free_text_targets_get_an_unconstrained_tool() -> None:
    assert "enum" not in answer_tool(None).parameters["properties"]["value"]


def test_a_tool_call_answer_is_used_directly() -> None:
    client = FakeClient([_tool_reply("toxic")])

    answers = ask(PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS)

    assert answers == ["toxic", "toxic"]


def test_prose_around_a_label_still_resolves() -> None:
    assert coerce("I think this is toxic, honestly", LABELS) == "toxic"


def test_a_reply_naming_two_labels_is_unparseable() -> None:
    """Guessing which one it meant is how a wrong answer becomes right by accident."""
    assert coerce("could be toxic or not toxic", LABELS) == UNPARSEABLE


def test_an_exact_match_wins_over_a_substring() -> None:
    assert coerce("not toxic", LABELS) == "not toxic"


def test_case_and_whitespace_do_not_matter() -> None:
    assert coerce("  TOXIC \n", LABELS) == "toxic"


def test_an_empty_reply_is_unparseable() -> None:
    assert coerce("", LABELS) == UNPARSEABLE
    assert coerce(None, LABELS) == UNPARSEABLE


def test_free_text_targets_keep_whatever_came_back() -> None:
    assert coerce("  a summary  ", None) == "a summary"


def test_an_unusable_row_is_recorded_not_raised() -> None:
    """A prompt that provokes unusable output IS a worse prompt, so it counts as
    wrong and gets counted — but one bad row must not cost the experiment."""
    client = FakeClient([_text_reply("hmm, hard to say either way")])
    stats = AskStats()

    answers = ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        retries=0,
        stats=stats,
    )

    assert answers == [UNPARSEABLE, UNPARSEABLE]
    assert stats.unparseable == 2


def test_a_raising_backend_does_not_kill_the_pass() -> None:
    client = FakeClient([RuntimeError("connection refused")])
    stats = AskStats()

    answers = ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        retries=0,
        stats=stats,
    )

    assert answers == [UNPARSEABLE, UNPARSEABLE]
    assert any("connection refused" in error for error in stats.errors)


def test_tokens_are_counted_even_when_a_retry_was_needed() -> None:
    """Cost that only counts successes is not cost."""
    client = FakeClient([_text_reply("no idea"), _tool_reply("toxic")])
    stats = AskStats()

    ask(
        PROMPT,
        ROWS[:1],
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        retries=1,
        stats=stats,
    )

    assert stats.prompt_tokens == 20


def test_answers_stay_in_row_order_under_concurrency() -> None:
    replies = [_tool_reply("toxic"), _tool_reply("not toxic")]
    rows = [{"text": f"row {i}"} for i in range(2)]

    class Ordered(FakeClient):
        def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
            content = messages[-1].content or ""
            return replies[0] if content.endswith("0") else replies[1]

    answers = ask(
        PROMPT,
        rows,
        client_factory=lambda: Ordered([]),
        columns=["text"],
        labels=LABELS,
        max_workers=2,
    )

    assert answers == ["toxic", "not toxic"]


def test_a_repeated_prompt_and_row_is_served_from_cache(tmp_path: Path) -> None:
    client = FakeClient([_tool_reply("toxic")])
    cache = AnswerCache(tmp_path / "answers.db")
    stats = AskStats()

    ask(PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS, cache=cache)
    ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        cache=cache,
        stats=stats,
    )

    assert stats.calls == 0
    assert stats.cached == 2


def test_the_cache_survives_a_new_process(tmp_path: Path) -> None:
    """Cross-experiment reuse is the point: the baseline prompt gets re-run."""
    path = tmp_path / "answers.db"
    client = FakeClient([_tool_reply("toxic")])
    ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        cache=AnswerCache(path),
    )

    stats = AskStats()
    ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        cache=AnswerCache(path),
        stats=stats,
    )

    assert stats.cached == 2


def test_a_failure_is_never_cached(tmp_path: Path) -> None:
    """Caching a transient network blip would make it permanent for the whole run."""
    cache = AnswerCache(tmp_path / "answers.db")
    ask(
        PROMPT,
        ROWS[:1],
        client_factory=lambda: FakeClient([RuntimeError("boom")]),
        columns=["text"],
        labels=LABELS,
        cache=cache,
        retries=0,
    )

    stats = AskStats()
    answers = ask(
        PROMPT,
        ROWS[:1],
        client_factory=lambda: FakeClient([_tool_reply("toxic")]),
        columns=["text"],
        labels=LABELS,
        cache=cache,
        stats=stats,
    )

    assert answers == ["toxic"]
    assert stats.calls == 1


def test_a_broken_cache_path_degrades_to_memory(tmp_path: Path) -> None:
    """A broken cache must never stop a run."""
    cache = AnswerCache(tmp_path)  # a directory, not a file

    cache.put("k", "v")

    assert cache.get("k") == "v"


def test_no_rows_means_no_calls() -> None:
    client = FakeClient([_tool_reply("toxic")])

    assert ask(PROMPT, [], client_factory=lambda: client, columns=["text"], labels=LABELS) == []
    assert client.calls == []


def test_the_model_under_test_resolves_its_key_the_same_way_the_agent_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--target-backend groq` must pick up GROQ_API_KEY, which is already in the
    environment for the driving model. One table, not two that agree until one
    changes."""
    from iterate.config import get_settings
    from iterate.core.prompt_runtime import make_ask

    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-the-environment")
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def spy(name: str, **kwargs: object) -> FakeClient:
        seen.update({"backend": name, **kwargs})
        return FakeClient([_tool_reply("toxic")])

    monkeypatch.setattr("iterate.llm.factory.build_client", spy)
    try:
        make_ask(columns=["text"], labels=LABELS, backend="groq", model="llama-70b")(PROMPT, ROWS)
    finally:
        get_settings.cache_clear()

    assert seen["api_key"] == "gsk-from-the-environment"
    assert seen["backend"] == "groq"


def test_an_explicit_target_key_overrides_the_shared_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iterate.core.prompt_runtime import make_ask

    monkeypatch.setenv("ITERATE_TARGET_API_KEY", "a-different-key")
    seen: dict[str, object] = {}

    def spy(name: str, **kwargs: object) -> FakeClient:
        seen.update(kwargs)
        return FakeClient([_tool_reply("toxic")])

    monkeypatch.setattr("iterate.llm.factory.build_client", spy)
    make_ask(columns=["text"], labels=LABELS, backend="groq", model="llama-70b")(PROMPT, ROWS)

    assert seen["api_key"] == "a-different-key"


def test_ollama_gets_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its settings default is the placeholder string "ollama", not a real key."""
    from iterate.core.prompt_runtime import make_ask

    monkeypatch.delenv("ITERATE_TARGET_API_KEY", raising=False)
    seen: dict[str, object] = {}

    def spy(name: str, **kwargs: object) -> FakeClient:
        seen.update(kwargs)
        return FakeClient([_tool_reply("toxic")])

    monkeypatch.setattr("iterate.llm.factory.build_client", spy)
    make_ask(columns=["text"], labels=LABELS, backend="ollama", model="gemma4:12b")(PROMPT, ROWS)

    assert seen["api_key"] is None
