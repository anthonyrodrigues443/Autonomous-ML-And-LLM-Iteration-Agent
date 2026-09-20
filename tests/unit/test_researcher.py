"""Tests for the Researcher specialist.

The load-bearing property is provenance: a citation this agent reports must be one
a source actually returned. Everything else here degrades gracefully, but an
invented DOI in a tool advertising "literature-aware proposals" would be the worst
bug the project could ship, so most of these tests attack that one guarantee.

No network: the paper sources are fakes, and the LLM is a scripted fake.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from iterate.adapters.research import Paper
from iterate.core.researcher import Findings, Researcher, Suggestion, credited
from iterate.schemas.llm import ChatResponse, ToolCall

_PAPERS = [
    Paper("TabNet", "doi:10.1609/aaai.v35i8.16826", "attentive tabular learning", 2021, 1586, "openalex"),
    Paper("CatBoost", "doi:10.5555/catboost", "ordered target statistics", 2018, 900, "openalex"),
    Paper("SMOTE variants", "arXiv:1106.1813", "oversampling for imbalance", 2011, 0, "arxiv"),
]


class _FakeSource:
    name = "fake"

    def __init__(self, papers: list[Paper] | None = None, *, boom: bool = False) -> None:
        self._papers = papers if papers is not None else list(_PAPERS)
        self._boom = boom
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 5) -> list[Paper]:
        self.queries.append(query)
        if self._boom:
            raise RuntimeError("network down")
        return self._papers[:limit]


class _FakeLLM:
    """Replies in order. A plain string means 'no tool call' (the retry path)."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else "nothing"
        if isinstance(reply, str):
            return ChatResponse(model="fake", content=reply, tool_calls=[])
        name, args = reply
        return ChatResponse(
            model="fake",
            content="",
            tool_calls=[ToolCall(id="call-1", name=name, arguments=args)],
        )


def _queries(*qs: str) -> tuple[str, dict[str, Any]]:
    return ("plan_queries", {"queries": list(qs)})


def _suggest(*items: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("suggest_techniques", {"suggestions": list(items)})


def _researcher(llm: _FakeLLM, source: _FakeSource | None = None) -> Researcher:
    return Researcher(
        llm, metric="f1", direction="maximize", sources=[source or _FakeSource()]
    )


def test_a_full_pass_returns_grounded_suggestions() -> None:
    llm = _FakeLLM([
        _queries("tabular imbalance boosting", "target encoding cardinality"),
        _suggest({"technique": "target-encode high-cardinality columns", "rationale": "16 categoricals", "paper": 2}),
    ])
    findings = _researcher(llm).research(profile="Rows: 5634. 16 categorical.")
    assert len(findings.suggestions) == 1
    assert findings.suggestions[0].citation == "doi:10.5555/catboost"
    assert findings.queries == ["tabular imbalance boosting", "target encoding cardinality"]
    assert bool(findings) is True


# ─── the citation guarantee ──────────────────────────────────────────────────


def test_every_citation_traces_to_a_fetched_paper() -> None:
    llm = _FakeLLM([
        _queries("q"),
        _suggest(
            {"technique": "a", "rationale": "r", "paper": 1},
            {"technique": "b", "rationale": "r", "paper": 3},
        ),
    ])
    findings = _researcher(llm).research(profile="p")
    fetched = {p.identifier for p in _PAPERS}
    assert findings.citations
    assert all(c in fetched for c in findings.citations)


def test_an_out_of_range_index_drops_the_suggestion() -> None:
    """The model cannot cite a paper it was not shown, so a bad index yields no
    suggestion rather than a suggestion with an empty or invented citation."""
    llm = _FakeLLM([_queries("q"), _suggest({"technique": "a", "rationale": "r", "paper": 99})])
    assert _researcher(llm).research(profile="p").suggestions == []


def test_a_non_numeric_index_drops_the_suggestion() -> None:
    llm = _FakeLLM([
        _queries("q"),
        _suggest({"technique": "a", "rationale": "r", "paper": "doi:10.1234/made-up"}),
    ])
    assert _researcher(llm).research(profile="p").suggestions == []


def test_the_model_cannot_smuggle_a_citation_through_another_field() -> None:
    """Even if the model writes a DOI into the technique text, the CITATION is
    still resolved from the index — the identifier never comes from model text."""
    llm = _FakeLLM([
        _queries("q"),
        _suggest({"technique": "use doi:10.9999/fake", "rationale": "r", "paper": 1}),
    ])
    findings = _researcher(llm).research(profile="p")
    assert findings.suggestions[0].citation == _PAPERS[0].identifier
    assert "10.9999" not in findings.suggestions[0].citation


# ─── degradation ─────────────────────────────────────────────────────────────


def test_no_queries_means_no_search_and_no_findings() -> None:
    source = _FakeSource()
    llm = _FakeLLM(["prose", "prose"])  # never emits the tool, even after the nudge
    assert _researcher(llm, source).research(profile="p").suggestions == []
    assert source.queries == []


def test_a_dead_network_yields_empty_findings_not_an_exception() -> None:
    llm = _FakeLLM([_queries("q")])
    findings = _researcher(llm, _FakeSource(boom=True)).research(profile="p")
    assert findings.suggestions == []
    assert findings.queries == ["q"]


def test_no_papers_found_skips_the_second_call_entirely() -> None:
    llm = _FakeLLM([_queries("q"), _suggest({"technique": "a", "rationale": "r", "paper": 1})])
    findings = _researcher(llm, _FakeSource([])).research(profile="p")
    assert findings.suggestions == []
    assert llm.calls == 1  # the suggestion call is never made


def test_the_retry_nudge_recovers_a_missing_tool_call() -> None:
    llm = _FakeLLM([
        "I think you should try boosting.",
        _queries("q"),
        _suggest({"technique": "a", "rationale": "r", "paper": 1}),
    ])
    assert _researcher(llm).research(profile="p").suggestions


def test_malformed_suggestion_rows_are_skipped_individually() -> None:
    llm = _FakeLLM([
        _queries("q"),
        ("suggest_techniques", {"suggestions": ["not-a-dict", {"technique": "", "paper": 1}, {"technique": "ok", "rationale": "r", "paper": 1}]}),
    ])
    findings = _researcher(llm).research(profile="p")
    assert [s.technique for s in findings.suggestions] == ["ok"]


def test_papers_are_deduped_across_overlapping_queries() -> None:
    """Queries attack one problem from several angles, so they overlap by design."""
    llm = _FakeLLM([
        _queries("q1", "q2", "q3"),
        _suggest({"technique": "a", "rationale": "r", "paper": 1}),
    ])
    findings = _researcher(llm).research(profile="p")
    assert findings.papers_seen == len(_PAPERS)


def test_render_is_one_line_per_suggestion() -> None:
    findings = Findings(
        suggestions=[
            Suggestion("target encoding", "16 categoricals", "doi:1"),
            Suggestion("class weights", "27% positive", "doi:2"),
        ]
    )
    assert len(findings.render().splitlines()) == 2
    assert "<doi:1>" in findings.render()


# ─── one suggestion prompt per family ────────────────────────────────────────


class _RecordingLLM(_FakeLLM):
    def __init__(self) -> None:
        super().__init__(
            [_queries("q one", "q two"), _suggest({"technique": "t", "rationale": "r", "paper": 1})]
        )
        self.sent: list[tuple[list[Any], list[Any]]] = []

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.sent.append((list(messages), list(tools or [])))
        return super().chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)


def _suggestion_call(family: str) -> tuple[str, str]:
    """(every message, the tool) one suggestion call sends, as text."""
    llm = _RecordingLLM()
    Researcher(
        llm, metric="f1", direction="maximize", family=family, sources=[_FakeSource()]
    ).research(profile="120 rows, 8 columns", tried=["one-hot encoding"])
    messages, tools = llm.sent[1]
    return (
        json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False),
        json.dumps([[t.name, t.description, t.parameters] for t in tools], sort_keys=True),
    )


@pytest.mark.parametrize(
    "table_only", ["tabular-applicable", "columns this dataset does not have", "THIS dataset"]
)
def test_a_prompt_run_is_not_asked_for_table_techniques(table_only: str) -> None:
    messages, _ = _suggestion_call("prompt")
    assert table_only not in messages


@pytest.mark.parametrize(
    "table_only", ["target-encode", "modelling move", "class balance or row count"]
)
def test_a_prompt_runs_suggest_tool_does_not_describe_a_modelling_move(table_only: str) -> None:
    _, tool = _suggestion_call("prompt")
    assert table_only not in tool


def test_a_prompt_run_is_asked_for_changes_to_the_prompt() -> None:
    messages, tool = _suggestion_call("prompt")
    assert "only the PROMPT can change" in messages
    assert "optimizing 'f1' (maximize)" in messages
    assert "an edit a prompt writer could" in tool
    assert '"suggest_techniques"' in tool


# Digests of what main 6485be3 sends. Only a deliberate rewording of the table or
# image pair may change them.
_TABLE_CALL_ON_MAIN = "7a7976476f0f298724187c0fec556809dd2f53a63e902c1bd901ceebb9b254b2"
_IMAGE_CALL_ON_MAIN = "4c6f808e1d025ce72dca66c2bc39ec55a59c60d1debce0f88a02c005d7e3d57c"


@pytest.mark.parametrize(
    ("family", "sent_by_main"),
    [
        ("tabular", _TABLE_CALL_ON_MAIN),
        ("vision", _IMAGE_CALL_ON_MAIN),
        ("a-family-with-no-pair", _TABLE_CALL_ON_MAIN),
    ],
)
def test_table_and_image_runs_send_the_suggestion_call_main_sent(
    family: str, sent_by_main: str
) -> None:
    messages, tool = _suggestion_call(family)
    assert hashlib.sha256((messages + tool).encode()).hexdigest() == sent_by_main


# ─── crediting: under-attribution is the safe failure ────────────────────────


def test_a_brief_that_takes_up_a_suggestion_is_credited() -> None:
    findings = Findings(suggestions=[Suggestion("target-encode high-cardinality columns", "r", "doi:1")])
    brief = "next: categorical-encoding: target-encode the high-cardinality columns like PaymentMethod."
    assert credited(findings, brief) == ["doi:1"]


def test_a_brief_that_ignores_the_research_is_not_credited() -> None:
    """A pass the supervisor read and ignored must not stamp a citation — an
    unearned citation is no better than an invented one."""
    findings = Findings(suggestions=[Suggestion("target-encode high-cardinality columns", "r", "doi:1")])
    brief = "next: imbalance-or-threshold: train with class_weight balanced."
    assert credited(findings, brief) == []


def test_crediting_needs_more_than_one_shared_word() -> None:
    findings = Findings(suggestions=[Suggestion("gradient boosting ensembles", "r", "doi:1")])
    assert credited(findings, "next: use gradient descent tuning.") == []


def test_no_findings_credits_nothing() -> None:
    assert credited(None, "next: anything") == []
    assert credited(Findings(), "next: anything") == []


def test_the_research_cache_records_the_query_that_produced_it(tmp_path: Path) -> None:
    """The filename is a hash, so before this the question was written down nowhere.
    Auditing "why did it search for that" meant inferring the question from the
    answers — for a tool whose pitch is literature awareness."""
    import json

    from iterate.adapters.research.papers import Paper, _Cache

    cache = _Cache(tmp_path)
    paper = Paper(
        title="T", identifier="arXiv:1", abstract="a", year=2024, cited_by=3, source="arxiv"
    )

    cache.put("arxiv", "few-shot example selection", 5, [paper])
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())

    assert saved["query"] == "few-shot example selection"
    assert saved["source"] == "arxiv"
    assert saved["fetched_at"]
    assert len(saved["papers"]) == 1
    assert cache.get("arxiv", "few-shot example selection", 5) == [paper]


def test_a_pre_v05_cache_file_is_still_readable(tmp_path: Path) -> None:
    """A format change must not silently invalidate every cached search and send a
    run back to the network for results it already has."""
    import hashlib
    import json

    from iterate.adapters.research.papers import _Cache

    key = hashlib.sha256(b"arxiv|5|legacy").hexdigest()[:20]
    (tmp_path / f"arxiv-{key}.json").write_text(
        json.dumps(
            [{"title": "T", "identifier": "i", "abstract": "a", "year": 2024,
              "cited_by": 1, "source": "arxiv"}]
        )
    )

    assert _Cache(tmp_path).get("arxiv", "legacy", 5) is not None
