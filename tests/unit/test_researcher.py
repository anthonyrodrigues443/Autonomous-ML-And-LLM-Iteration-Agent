"""Tests for the Researcher specialist.

The load-bearing property is provenance: a citation this agent reports must be one
a source actually returned. Everything else here degrades gracefully, but an
invented DOI in a tool advertising "literature-aware proposals" would be the worst
bug the project could ship, so most of these tests attack that one guarantee.

No network: the paper sources are fakes, and the LLM is a scripted fake.
"""

from __future__ import annotations

from typing import Any

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
