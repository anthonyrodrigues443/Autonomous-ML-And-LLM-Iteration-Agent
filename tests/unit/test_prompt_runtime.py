"""`ask()`: the answer tool, coercion, caching, one unusable row never killing a run, and
a row the provider never answered never scored."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from iterate.core.prompt_runtime import (
    NO_REPLY,
    UNPARSEABLE,
    AnswerCache,
    AskStats,
    NoReplyError,
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
        self.caps_seen: list[int | None] = []
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
            self.caps_seen.append(max_tokens)
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


def test_a_record_call_is_capped_so_one_runaway_row_cannot_eat_a_cell() -> None:
    """One record that never stops generating held seven queued workers and killed
    a 600s cell, twice, on a live run. A label is under 20 tokens; the cap is a
    bound a well-behaved model never meets."""
    client = FakeClient([_tool_reply("toxic")])

    ask(PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS)

    assert client.caps_seen == [64, 64]


def test_a_numeric_record_call_gets_the_same_small_cap() -> None:
    client = FakeClient([_tool_reply("3")])

    ask(
        PROMPT,
        ROWS,
        client_factory=lambda: client,
        columns=["text"],
        numeric_range=(0, 5),
    )

    assert client.caps_seen == [64, 64]


def test_free_text_is_bounded_not_sized() -> None:
    """A summary can legitimately run long, so free text gets room. It still gets a
    ceiling, because the failure is a model that does not stop, not one that is
    verbose."""
    client = FakeClient([_text_reply("a short summary")])

    ask(PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=None)

    assert client.caps_seen == [512, 512]


def test_a_cut_off_reply_still_resolves_when_the_label_landed() -> None:
    """What the cap produces on a rambling model: the label, then silence."""
    assert coerce("Looking at the tone here, this one is toxic. Now the next tw", LABELS) == "toxic"


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


class ByRow:
    """Replies scripted per record, each record's in the order they are asked for, so
    the pass's threads cannot shuffle them."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self._script = {text: list(replies) for text, replies in script.items()}
        self.asked: list[str] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return "fake-12b"

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        text = str(messages[-1].content)
        with self._lock:
            self.asked.append(text)
            reply = self._script[text].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_more_than_one_record_in_five_never_answered_fails_the_pass() -> None:
    client = ByRow(
        {
            "you are awful": [_tool_reply("toxic")],
            "have a nice day": [RuntimeError("connection refused")] * 6,
        }
    )
    stats = AskStats()

    with pytest.raises(NoReplyError) as caught:
        ask(
            PROMPT,
            ROWS,
            client_factory=lambda: client,
            columns=["text"],
            labels=LABELS,
            stats=stats,
        )

    assert str(caught.value) == (
        "the provider did not answer 1 of 2 records, each asked twice "
        "(RuntimeError: connection refused); more than one in five is too many to score"
    )
    assert (stats.no_reply, stats.unparseable) == (1, 0)
    # Three tries in the pass, three more when it is asked again.
    assert client.asked.count("have a nice day") == 6
    assert client.asked.count("you are awful") == 1
    assert stats.summary() == "2 calls, 0 cached, 1 not answered by the provider"


def test_a_record_never_answered_is_left_out_and_never_cached(tmp_path: Path) -> None:
    """One in ten is not the model's answer: it comes back marked, is not scored and is
    asked again by the next pass."""
    rows = [{"text": f"comment {i}"} for i in range(10)]
    script: dict[str, list[Any]] = {r["text"]: [_tool_reply("toxic")] for r in rows}
    script["comment 3"] = [RuntimeError("503")] * 6
    client = ByRow(script)
    cache = AnswerCache(tmp_path / "answers.db")
    stats = AskStats()

    answers = ask(
        PROMPT,
        rows,
        client_factory=lambda: client,
        columns=["text"],
        labels=LABELS,
        cache=cache,
        stats=stats,
    )

    assert answers[3] == NO_REPLY
    assert answers.count("toxic") == 9
    assert (stats.no_reply, stats.unparseable) == (1, 0)
    back = ByRow({"comment 3": [_tool_reply("not toxic")]})
    again = ask(
        PROMPT, rows, client_factory=lambda: back, columns=["text"], labels=LABELS, cache=cache
    )
    assert again[3] == "not toxic"
    assert back.asked == ["comment 3"]


def test_a_record_missed_in_a_blip_is_answered_when_asked_again() -> None:
    client = ByRow(
        {
            "you are awful": [_tool_reply("toxic")],
            "have a nice day": [RuntimeError("503")] * 3 + [_tool_reply("not toxic")],
        }
    )
    stats = AskStats()

    answers = ask(
        PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS, stats=stats
    )

    assert answers == ["toxic", "not toxic"]
    assert (stats.calls, stats.no_reply) == (2, 0)


def test_a_tool_call_the_provider_turns_down_is_the_models_answer() -> None:
    """groq answers a malformed tool call with a 400 and Ollama with a 500. The model
    answered, badly: as no reply, it would fail every pass on the same record."""

    class RejectedError(Exception):
        def __init__(self, said: str, body: str = "") -> None:
            super().__init__(said)
            self.response = type("R", (), {"text": body})()

    client = ByRow(
        {
            "you are awful": [RejectedError("Error code: 400 - {'code': 'tool_use_failed'}")] * 3,
            "have a nice day": [RejectedError("Server error '500'", "error parsing tool call")] * 3,
        }
    )
    stats = AskStats()

    answers = ask(
        PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS, stats=stats
    )

    assert answers == [UNPARSEABLE, UNPARSEABLE]
    assert (stats.no_reply, stats.unparseable) == (0, 2)


def test_a_record_with_any_reply_is_the_models_answer() -> None:
    """The model had its chance: a try the provider failed between two the model
    answered changes nothing."""
    client = ByRow(
        {
            "you are awful": [RuntimeError("503"), _text_reply("hmm"), RuntimeError("503")],
            "have a nice day": [RuntimeError("429"), _tool_reply("not toxic")],
        }
    )
    stats = AskStats()

    answers = ask(
        PROMPT, ROWS, client_factory=lambda: client, columns=["text"], labels=LABELS, stats=stats
    )

    assert answers == [UNPARSEABLE, "not toxic"]
    assert (stats.no_reply, stats.unparseable) == (0, 1)


def test_only_the_records_the_provider_never_answered_are_asked_again(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.sqlite")
    outage = ByRow(
        {
            "you are awful": [_tool_reply("toxic")],
            "have a nice day": [RuntimeError("overloaded")] * 6,
        }
    )
    with pytest.raises(NoReplyError, match="did not answer 1 of 2 records"):
        ask(
            PROMPT,
            ROWS,
            client_factory=lambda: outage,
            columns=["text"],
            labels=LABELS,
            cache=cache,
        )

    back = ByRow({"have a nice day": [_tool_reply("not toxic")]})
    answers = ask(
        PROMPT, ROWS, client_factory=lambda: back, columns=["text"], labels=LABELS, cache=cache
    )

    assert answers == ["toxic", "not toxic"]
    assert back.asked == ["have a nice day"]


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
    with pytest.raises(NoReplyError):
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


def test_the_model_under_test_finds_its_providers_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--target-backend groq` picks up GROQ_API_KEY, which is how a delivered notebook
    run by hand finds its key."""
    from iterate.config import get_settings
    from iterate.core.prompt_runtime import make_ask

    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-the-environment")
    monkeypatch.delenv("ITERATE_TARGET_API_KEY", raising=False)
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


def test_the_exported_target_key_beats_the_providers_own(
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


def test_the_key_the_host_settled_is_the_one_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    from iterate.core.prompt_runtime import make_ask

    monkeypatch.setenv("ITERATE_TARGET_API_KEY", "from-the-environment")
    seen: dict[str, object] = {}

    def spy(name: str, **kwargs: object) -> FakeClient:
        seen.update(kwargs)
        return FakeClient([_tool_reply("toxic")])

    monkeypatch.setattr("iterate.llm.factory.build_client", spy)
    make_ask(
        columns=["text"], labels=LABELS, backend="groq", model="llama-70b", api_key="gsk-settled"
    )(PROMPT, ROWS)

    assert seen["api_key"] == "gsk-settled"
    assert seen["base_url"] == "https://api.groq.com/openai/v1"


def test_a_target_with_no_key_never_takes_the_harness_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generic slot holds the harness's key. A groq target with no key of its own
    fails and says so; it is never sent that one."""
    from iterate.config import get_settings
    from iterate.core.prompt_runtime import make_ask
    from iterate.llm.factory import ProviderError

    monkeypatch.chdir(tmp_path)  # away from a project's .env, which holds keys of its own
    monkeypatch.setenv("ITERATE_BACKEND_API_KEY", "the-harness-key")
    for name in ("GROQ_API_KEY", "ITERATE_TARGET_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    built: list[object] = []
    monkeypatch.setattr(
        "iterate.llm.factory.build_client", lambda name, **kw: built.append(kw) or FakeClient([])
    )
    try:
        with pytest.raises(ProviderError, match="groq has no key"):
            make_ask(columns=["text"], labels=LABELS, backend="groq", model="m")(PROMPT, ROWS)
    finally:
        get_settings.cache_clear()
    assert built == []


def test_two_providers_serving_one_model_name_do_not_share_answers(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.db")
    first, second = FakeClient([_tool_reply("toxic")]), FakeClient([_tool_reply("not toxic")])
    shared = {"columns": ["text"], "labels": LABELS, "cache": cache}

    one = ask(PROMPT, ROWS[:1], client_factory=lambda: first, cache_scope="groq|a", **shared)
    two = ask(PROMPT, ROWS[:1], client_factory=lambda: second, cache_scope="together|b", **shared)
    again = AskStats()
    ask(
        PROMPT,
        ROWS[:1],
        client_factory=lambda: second,
        cache_scope="groq|a",
        stats=again,
        **shared,
    )

    assert (one, two) == (["toxic"], ["not toxic"])
    assert (again.calls, again.cached) == (0, 1)


def test_the_ask_a_cell_builds_is_scoped_to_its_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.core.prompt_runtime import make_ask

    replies = iter(["toxic", "not toxic"])
    monkeypatch.setattr(
        "iterate.llm.factory.build_client",
        lambda name, **kw: FakeClient([_tool_reply(next(replies))], model="llama-70b"),
    )
    common = {"columns": ["text"], "labels": LABELS, "model": "llama-70b", "api_key": "k"}
    path = tmp_path / "answers.db"
    groq = make_ask(backend="groq", cache_path=path, scoped=True, **common)(PROMPT, ROWS[:1])
    together = make_ask(backend="together", cache_path=path, scoped=True, **common)(
        PROMPT, ROWS[:1]
    )

    assert (groq, together) == (["toxic"], ["not toxic"])


def test_a_pass_served_whole_from_the_cache_needs_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivered notebook run again beside its cache calls no one, so it is not asked
    for a key it would not use. The first answer that is missing asks for it."""
    from iterate.config import get_settings
    from iterate.core.prompt_runtime import make_ask
    from iterate.llm.factory import ProviderError

    monkeypatch.chdir(tmp_path)  # away from a project's .env, which holds keys of its own
    for name in ("GROQ_API_KEY", "ITERATE_TARGET_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    common = {"columns": ["text"], "labels": LABELS, "backend": "groq", "model": "llama-70b"}
    path = tmp_path / "answers.db"
    monkeypatch.setattr(
        "iterate.llm.factory.build_client",
        lambda name, **kw: FakeClient([_tool_reply("toxic")], model="llama-70b"),
    )
    try:
        make_ask(cache_path=path, scoped=True, api_key="gsk", **common)(PROMPT, ROWS[:1])
        built: list[object] = []
        monkeypatch.setattr("iterate.llm.factory.build_client", lambda name, **kw: built.append(kw))
        stats = AskStats()
        again = make_ask(cache_path=path, scoped=True, **common)(PROMPT, ROWS[:1], stats=stats)
        with pytest.raises(ProviderError, match="groq has no key here"):
            make_ask(cache_path=path, scoped=True, **common)(PROMPT, ROWS)
    finally:
        get_settings.cache_clear()
    assert (again, stats.cached, stats.calls, built) == (["toxic"], 1, 0, [])


def test_a_notebook_from_before_v07_finds_the_answers_it_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its opening cell was written before there was a scope to pass. The key it looks
    under is the one main filed this record under on 2026-09-27."""
    import sqlite3

    from iterate.core.prompt_runtime import make_ask

    path = tmp_path / "answers.db"
    AnswerCache(path).put(
        "26aec45b8f568c22091e8f3fcea00f98d18f6026b9bba4a74cd35da24a842d6b", "toxic"
    )
    monkeypatch.setattr(
        "iterate.llm.factory.build_client",
        lambda name, **kw: FakeClient([_tool_reply("not toxic")], model="llama-70b"),
    )
    common = {"columns": ["text"], "labels": LABELS, "model": "llama-70b", "api_key": "k"}
    old, new = AskStats(), AskStats()

    before = make_ask(backend="groq", cache_path=path, **common)(PROMPT, ROWS[:1], stats=old)
    after = make_ask(backend="groq", cache_path=path, scoped=True, **common)(
        PROMPT, ROWS[:1], stats=new
    )

    assert (before, old.cached, old.calls) == (["toxic"], 1, 0)
    assert (after, new.cached, new.calls) == (["not toxic"], 0, 1)
    assert sqlite3.connect(path).execute("SELECT count(*) FROM answers").fetchone() == (2,)


def test_a_contained_label_does_not_swallow_a_genuine_ambiguity() -> None:
    """The regression my own longest-match fix introduced, and the reason coercion
    scans by POSITION rather than comparing labels to each other.

    "toxic" is a substring of "not toxic", so comparing labels made a reply naming
    both look like a single confident answer. A confident wrong answer is worse than
    an unparseable one because it is invisible.
    """
    assert coerce("could be toxic or not toxic", ["toxic", "not toxic"]) == UNPARSEABLE
    assert coerce("not toxic", ["toxic", "not toxic"]) == "not toxic"
    assert coerce("I think this is toxic", ["toxic", "not toxic"]) == "toxic"
