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
from iterate.core import vision_levers as vl
from iterate.core.researcher import (
    CATALOG_CITATION,
    Findings,
    Researcher,
    Suggestion,
    _catalog_listing,
    credited,
)
from iterate.schemas.llm import ChatResponse, ToolCall
from iterate.targets import layers as arch

pytestmark = pytest.mark.unit

_PAPERS = [
    Paper(
        "TabNet",
        "doi:10.1609/aaai.v35i8.16826",
        "attentive tabular learning",
        2021,
        1586,
        "openalex",
    ),
    Paper("CatBoost", "doi:10.5555/catboost", "ordered target statistics", 2018, 900, "openalex"),
    Paper("SMOTE variants", "arXiv:1106.1813", "oversampling for imbalance", 2011, 0, "arxiv"),
]


class _FakeSource:
    name = "fake"

    def __init__(self, papers: list[Paper] | None = None, *, boom: bool = False) -> None:
        self._papers = papers if papers is not None else list(_PAPERS)
        self._boom = boom
        self.queries: list[str] = []
        self.limits: list[int] = []

    def search(self, query: str, *, limit: int = 5) -> list[Paper]:
        self.queries.append(query)
        self.limits.append(limit)
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
    return Researcher(llm, metric="f1", direction="maximize", sources=[source or _FakeSource()])


def test_a_full_pass_returns_grounded_suggestions() -> None:
    llm = _FakeLLM(
        [
            _queries("tabular imbalance boosting", "target encoding cardinality"),
            _suggest(
                {
                    "technique": "target-encode high-cardinality columns",
                    "rationale": "16 categoricals",
                    "paper": 2,
                }
            ),
        ]
    )
    findings = _researcher(llm).research(profile="Rows: 5634. 16 categorical.")
    assert len(findings.suggestions) == 1
    assert findings.suggestions[0].citation == "doi:10.5555/catboost"
    assert findings.queries == ["tabular imbalance boosting", "target encoding cardinality"]
    assert bool(findings) is True


# ─── the citation guarantee ──────────────────────────────────────────────────


def test_every_citation_traces_to_a_fetched_paper() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(
                {"technique": "a", "rationale": "r", "paper": 1},
                {"technique": "b", "rationale": "r", "paper": 3},
            ),
        ]
    )
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
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest({"technique": "a", "rationale": "r", "paper": "doi:10.1234/made-up"}),
        ]
    )
    assert _researcher(llm).research(profile="p").suggestions == []


def test_the_model_cannot_smuggle_a_citation_through_another_field() -> None:
    """Even if the model writes a DOI into the technique text, the CITATION is
    still resolved from the index — the identifier never comes from model text."""
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest({"technique": "use doi:10.9999/fake", "rationale": "r", "paper": 1}),
        ]
    )
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
    llm = _FakeLLM(
        [
            "I think you should try boosting.",
            _queries("q"),
            _suggest({"technique": "a", "rationale": "r", "paper": 1}),
        ]
    )
    assert _researcher(llm).research(profile="p").suggestions


def test_malformed_suggestion_rows_are_skipped_individually() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            (
                "suggest_techniques",
                {
                    "suggestions": [
                        "not-a-dict",
                        {"technique": "", "paper": 1},
                        {"technique": "ok", "rationale": "r", "paper": 1},
                    ]
                },
            ),
        ]
    )
    findings = _researcher(llm).research(profile="p")
    assert [s.technique for s in findings.suggestions] == ["ok"]


def test_papers_are_deduped_across_overlapping_queries() -> None:
    """Queries attack one problem from several angles, so they overlap by design."""
    llm = _FakeLLM(
        [
            _queries("q1", "q2", "q3"),
            _suggest({"technique": "a", "rationale": "r", "paper": 1}),
        ]
    )
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


def _suggestion_call(family: str, **research: Any) -> tuple[str, str]:
    """(every message, the tool) one suggestion call sends, as text."""
    llm = _RecordingLLM()
    Researcher(
        llm, metric="f1", direction="maximize", family=family, sources=[_FakeSource()]
    ).research(profile="120 rows, 8 columns", tried=["one-hot encoding"], **research)
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
# image pair may change them. The image digest was recomputed once, for the v0.6 hold's
# PR E, which teaches the image pair to suggest a layer stack; the table digest is
# untouched since 6485be3 and is the proof that rewording did not reach the table pair.
_TABLE_CALL_ON_MAIN = "7a7976476f0f298724187c0fec556809dd2f53a63e902c1bd901ceebb9b254b2"
_IMAGE_CALL_ON_MAIN = "a300345dc2b798eccb706d6f039a60718d3acb54258ee34a92366efd4dfa985b"


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
    findings = Findings(
        suggestions=[Suggestion("target-encode high-cardinality columns", "r", "doi:1")]
    )
    brief = (
        "next: categorical-encoding: target-encode the high-cardinality columns like PaymentMethod."
    )
    assert credited(findings, brief) == ["doi:1"]


def test_a_brief_that_ignores_the_research_is_not_credited() -> None:
    """A pass the supervisor read and ignored must not stamp a citation — an
    unearned citation is no better than an invented one."""
    findings = Findings(
        suggestions=[Suggestion("target-encode high-cardinality columns", "r", "doi:1")]
    )
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
            [
                {
                    "title": "T",
                    "identifier": "i",
                    "abstract": "a",
                    "year": 2024,
                    "cited_by": 1,
                    "source": "arxiv",
                }
            ]
        )
    )

    assert _Cache(tmp_path).get("arxiv", "legacy", 5) is not None


# ─── an image suggestion may not invent a network (v0.6 hold, PR E) ──────────


_STACK_PAPERS = [
    Paper(
        "A small CNN for land cover",
        "doi:10.1000/stack",
        "Our network stacks a convolution of 32 filters and one of 64 filters, each "
        "followed by pooling, then dropout of 0.3 and a fully connected layer of 256 "
        "units, trained from scratch on 27000 tiles.",
        2022,
        40,
        "openalex",
    )
]
# The same claim with no width anywhere, which is how an abstract normally reads. The
# numbers it does carry are the ones a 12B invents, so membership alone would pass them.
_NO_WIDTH_PAPERS = [
    Paper(
        "A small CNN for land cover",
        "doi:10.1000/nowidth",
        "Our network stacks several convolutional blocks with pooling and dropout, "
        "followed by a fully connected layer, on 64x64 tiles across 10 land cover "
        "classes, with a dropout rate of 0.3, batches of 32 and 256 epochs.",
        2022,
        40,
        "openalex",
    )
]


def _vision_suggestion(technique: str, papers: list[Paper] | None = None) -> list[Suggestion]:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest({"technique": technique, "rationale": "r", "paper": 1}),
        ]
    )
    researcher = Researcher(
        llm,
        metric="accuracy",
        direction="maximize",
        family="vision",
        sources=[_FakeSource(list(papers if papers is not None else _STACK_PAPERS))],
    )
    return researcher.research(profile="27000 images, 10 classes").suggestions


@pytest.mark.parametrize(
    ("technique", "kept"),
    [
        ("train conv(32) pool conv(64) pool dropout(0.3) linear(256) from zero", True),
        # One width the abstract never states: the stack is all this said, so it goes.
        ("train conv(32) pool conv(128) pool dropout(0.3) linear(256) from zero", False),
        # No stack written out at all: this check does not touch it.
        ("fine-tune timm efficientnet_b0 on all layers at 128 px", True),
    ],
)
def test_an_image_stack_survives_only_when_the_abstract_states_every_number(
    technique: str, kept: bool
) -> None:
    """The lever ladder opens a layer class on a finding, and a 12B copies the example it
    is shown. A network it made up, with a real paper against it, is the failure to stop."""
    assert [s.technique for s in _vision_suggestion(technique)] == ([technique] if kept else [])


@pytest.mark.parametrize(
    ("technique", "left"),
    [
        (
            "use timm vit_small_patch16_224 with a linear(768) dropout(0.2) classifier",
            "use timm vit_small_patch16_224 with a linear dropout classifier",
        ),
        (
            "fine-tune timm resnet50 on all layers, with a head of linear(512) dropout(0.5)",
            "fine-tune timm resnet50 on all layers, with a head of linear dropout",
        ),
    ],
)
def test_an_unstated_head_costs_the_stack_and_not_the_model_the_paper_names(
    technique: str, left: str
) -> None:
    """What the abstract check guards is the STACK. A paper that names a real model and
    sketches a head is a backbone or own-model finding, and dropping it whole would throw
    the model name and its citation away for the sake of two numbers nobody can use."""
    kept = _vision_suggestion(technique)
    assert [s.technique for s in kept] == [left]
    assert arch.found_strict(kept[0].technique) is None


def test_an_abstract_that_states_no_width_at_all_states_no_stack() -> None:
    """Measured on gemma4:12b: given an abstract with no width in it, the model invents a
    stack anyway, out of the example the prompt shows it, and the numbers it invents are
    the ones an image abstract carries for other reasons. Set membership alone kept that
    invented stack 5 times out of 5, so the abstract has to state widths at all."""
    invented = "conv(32) pool conv(64) dropout(0.3) linear(256) from zero"
    assert _vision_suggestion(invented, list(_NO_WIDTH_PAPERS)) == []
    assert [s.technique for s in _vision_suggestion(invented)] == [invented]


def test_a_table_run_is_never_checked_against_the_abstract() -> None:
    """Image family only: "conv(32)" in a tabular technique is not a network this harness
    would ever build, and the check must not quietly drop a table suggestion."""
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(
                {"technique": "conv(32) pool conv(999) linear(7)", "rationale": "r", "paper": 1}
            ),
        ]
    )
    findings = Researcher(
        llm, metric="f1", direction="maximize", sources=[_FakeSource(list(_STACK_PAPERS))]
    ).research(profile="p")
    assert len(findings.suggestions) == 1


# ─── v0.7 Day 4: the Researcher and the Pricer go back and forth ─────────────


_MONEY_WORDS = ("$", "budget", "cost", "price", "dollar", "serving", "month", "afford", "cheap")


class _Recorder(_FakeLLM):
    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.sent: list[list[Any]] = []

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.sent.append(list(messages))
        return super().chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)


def _catalog(*names: Any) -> tuple[str, dict[str, Any]]:
    return ("name_models", {"models": list(names)})


def _image_researcher(llm: _FakeLLM, source: _FakeSource | None = None) -> Researcher:
    return Researcher(
        llm,
        metric="accuracy",
        direction="maximize",
        family="vision",
        sources=[source or _FakeSource()],
    )


_NO_MODEL = {"technique": "random crops and flips", "rationale": "r", "paper": 1}


def test_the_ruled_out_names_reach_every_call_and_no_money_word_does() -> None:
    """The Researcher is told which networks are out and nothing about why: the price is
    the wall's to judge, and a money word in the question tilts the search."""
    llm = _Recorder([_queries("q"), _suggest(_NO_MODEL), _catalog("efficientnet_b0")])
    _image_researcher(llm).research(
        profile="27000 images, 10 classes, 64x64 px",
        tried=["resnet18 64px all layers, 3 epochs"],
        ruled_out=["resnet18", "convnext_tiny"],
        round=2,
    )
    assert len(llm.sent) == 3  # the queries, the papers, the catalog
    assert "The library catalog" in llm.sent[2][-1].content
    for messages in llm.sent:
        text = "\n".join(m.content for m in messages)
        assert "ruled out this run, do not suggest these: resnet18, convnext_tiny" in text
        assert [w for w in _MONEY_WORDS if w in text.lower()] == []


def test_a_network_fit_builds_from_zero_is_never_named_as_ruled_out() -> None:
    llm = _Recorder([_queries("q"), _suggest(_NO_MODEL)])
    _image_researcher(llm).research(profile="p", ruled_out=["layers_net", "resnet50"])
    sent = llm.sent[1][-1].content
    assert "do not suggest these: resnet50\n" in sent
    assert "layers_net" not in sent


@pytest.mark.parametrize(
    ("family", "sent_by_main"),
    [("tabular", _TABLE_CALL_ON_MAIN), ("vision", _IMAGE_CALL_ON_MAIN)],
)
def test_no_ruled_out_names_on_the_first_round_send_the_bytes_main_sent(
    family: str, sent_by_main: str
) -> None:
    messages, tool = _suggestion_call(family, ruled_out=(), round=1)
    assert hashlib.sha256((messages + tool).encode()).hexdigest() == sent_by_main


@pytest.mark.parametrize(("round_", "limit"), [(1, 4), (2, 8), (3, 12)])
def test_a_later_round_fetches_more_papers_a_query(round_: int, limit: int) -> None:
    source = _FakeSource()
    llm = _FakeLLM(
        [_queries("q1", "q2"), _suggest({"technique": "a", "rationale": "r", "paper": 1})]
    )
    _researcher(llm, source).research(profile="p", round=round_)
    assert source.limits == [limit, limit]


def test_a_bigger_fetch_misses_the_cache_a_smaller_one_filled(tmp_path: Path) -> None:
    """The search is cached by (source, limit, query), so the same query at a bigger
    limit goes back to the source instead of handing round one's papers back."""
    from iterate.adapters.research.papers import _Cache

    cache = _Cache(tmp_path)
    cache.put("openalex", "transfer learning land cover", 4, [_PAPERS[0]])
    assert cache.get("openalex", "transfer learning land cover", 4) == [_PAPERS[0]]
    assert cache.get("openalex", "transfer learning land cover", 8) is None


_RANKED = [
    Paper(f"Title {chr(65 + i)}", f"doi:10.1/{i}", "abstract", 2020, 1000 - i, "openalex")
    for i in range(12)
]


def test_a_later_round_shows_the_papers_no_earlier_pass_showed_first() -> None:
    """Ranked by citations alone, the bigger fetch would put round one's most-cited papers
    straight back on top."""
    llm = _Recorder(
        [
            _queries("q"),
            _suggest({"technique": "a", "rationale": "r", "paper": 1}),
            _queries("q"),
            _suggest({"technique": "b", "rationale": "r", "paper": 1}),
        ]
    )
    researcher = _researcher(llm, _FakeSource(list(_RANKED)))
    assert researcher.research(profile="p").citations == ["doi:10.1/0"]
    assert researcher.research(profile="p", round=2).citations == ["doi:10.1/4"]
    listing = llm.sent[3][-1].content
    assert listing.index("Title E") < listing.index("Title A")


def test_the_first_round_ranks_by_citations_however_often_it_runs() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest({"technique": "a", "rationale": "r", "paper": 1}),
            _queries("q"),
            _suggest({"technique": "b", "rationale": "r", "paper": 1}),
        ]
    )
    researcher = _researcher(llm, _FakeSource(list(_RANKED)))
    researcher.research(profile="p")
    assert researcher.research(profile="p").citations == ["doi:10.1/0"]


def test_a_dry_later_round_keeps_only_catalog_names_not_ruled_out_or_tried() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(_NO_MODEL),
            _catalog(
                "timm/efficientnet_b0",
                "not_a_real_net",
                "resnet18",
                "convnext_large",
                "hf_hub:timm/mobilenetv3_small_100.lamb_in1k",
                "test_vit",
                "deit_tiny_patch16_224",
            ),
        ]
    )
    findings = _image_researcher(llm).research(
        profile="27000 images, 10 classes",
        tried=["resnet18 64px all layers, 3 epochs"],
        ruled_out=["convnext_large"],
        round=2,
    )
    assert findings.suggestions[-1].technique == "random crops and flips"  # catalog lines first
    kept = [s for s in findings.suggestions if s.citation == CATALOG_CITATION]
    assert [vl.models_named(s.technique) for s in kept] == [
        ["efficientnet_b0"],
        ["mobilenetv3_small_100"],
        ["deit_tiny_patch16_224"],
    ]


def test_a_paper_that_names_a_new_network_is_enough() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(
                {
                    "technique": "fine-tune timm efficientnet_b0 on all layers",
                    "rationale": "r",
                    "paper": 1,
                }
            ),
            _catalog("mobilenetv3_small_100"),
        ]
    )
    findings = _image_researcher(llm).research(profile="p", ruled_out=["convnext_large"], round=2)
    assert llm.calls == 2
    assert CATALOG_CITATION not in findings.citations


@pytest.mark.parametrize("named", ["convnext_large", "efficientnet_b0"])
def test_a_paper_naming_only_a_ruled_out_or_tried_network_is_dry(named: str) -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(
                {"technique": f"fine-tune timm {named} on all layers", "rationale": "r", "paper": 1}
            ),
            _catalog("mobilenetv3_small_100"),
        ]
    )
    findings = _image_researcher(llm).research(
        profile="p",
        tried=["own code: efficientnet_b0, 64px, 3 epochs"],
        ruled_out=["convnext_large"],
        round=2,
    )
    assert findings.citations[0] == CATALOG_CITATION


@pytest.mark.parametrize(("family", "round_"), [("vision", 1), ("tabular", 2), ("prompt", 2)])
def test_the_catalog_is_asked_only_on_a_later_image_round(family: str, round_: int) -> None:
    llm = _FakeLLM([_queries("q"), _suggest(_NO_MODEL), _catalog("efficientnet_b0")])
    findings = Researcher(
        llm, metric="accuracy", direction="maximize", family=family, sources=[_FakeSource()]
    ).research(profile="p", round=round_)
    assert llm.calls == 2
    assert CATALOG_CITATION not in findings.citations


@pytest.mark.parametrize(
    ("replies", "source"),
    [
        ([_queries("q"), _catalog("efficientnet_b0")], _FakeSource([])),
        (["prose", "prose", _catalog("efficientnet_b0")], _FakeSource()),
    ],
)
def test_a_later_round_with_no_papers_at_all_still_asks_the_catalog(
    replies: list[Any], source: _FakeSource
) -> None:
    """The case this pass exists for: on this machine the papers seldom name a network."""
    findings = _image_researcher(_FakeLLM(replies), source).research(profile="p", round=2)
    assert findings.citations == [CATALOG_CITATION]


def test_a_catalog_finding_cites_the_catalog_in_the_harness_words() -> None:
    """The model wrote no finding, so none of its words ride on the line the lever ladder
    reads, and nothing on the line reads as a paper."""
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(_NO_MODEL),
            (
                "name_models",
                {"models": [{"model": "efficientnet_b0", "why": "beats resnet50 at 224 px"}]},
            ),
        ]
    )
    findings = _image_researcher(llm).research(profile="p", round=2)
    line = findings.render().splitlines()[0]
    assert line.endswith(f"<{CATALOG_CITATION}>")
    assert vl.models_named(line) == ["efficientnet_b0"]
    assert "224" not in line
    assert "resnet50" not in line
    assert not findings.citations[0].startswith(("doi:", "arXiv:"))


def test_a_catalog_pass_that_fails_leaves_the_paper_findings() -> None:
    llm = _FakeLLM([_queries("q"), _suggest(_NO_MODEL), "prose", "prose"])
    findings = _image_researcher(llm).research(profile="p", round=2)
    assert [s.technique for s in findings.suggestions] == ["random crops and flips"]


def test_the_catalog_is_shown_one_family_a_line_without_excluded_names() -> None:
    """All 1,101 names are 20 kB of a 12B's context; a line a family is a fifth of it."""
    listing = _catalog_listing({"efficientnet_b0", "resnet18"})
    lines = listing.splitlines()
    names = [n for line in lines for n in line.split(", ")]
    assert len(listing) < 6000
    assert all(len(line.split(", ")) <= 2 for line in lines)
    assert all(n in vl.CATALOG for n in names)
    assert "efficientnet_b0" not in names
    assert "resnet18" not in names
    assert [n for n in names if n.startswith("test_")] == []


def test_merge_keeps_what_came_first_and_adds_only_what_is_new() -> None:
    first = Findings(
        suggestions=[Suggestion("a", "r", "doi:1"), Suggestion("b", "r", "doi:2")],
        queries=["q1"],
        papers_seen=4,
    )
    later = Findings(
        suggestions=[
            Suggestion("A", "other words", "doi:1"),
            Suggestion("c", "r", CATALOG_CITATION),
            Suggestion("c", "r", CATALOG_CITATION),
        ],
        queries=["q1", "q2"],
        papers_seen=8,
    )
    merged = first.merge(later)
    assert [s.technique for s in merged.suggestions] == ["a", "b", "c"]
    assert merged.queries == ["q1", "q2"]
    assert merged.papers_seen == 12
    assert first.merge(Findings()) == first


def test_a_catalog_citation_is_credited_only_to_the_brief_that_took_the_name_up() -> None:
    findings = Findings(
        suggestions=[
            Suggestion("fine-tune timm efficientnet_b0 on all layers", "r", "doi:1"),
            Suggestion(
                "fine-tune mobilenetv3_small_100, pretrained, all layers",
                "from the library catalog, not a paper",
                CATALOG_CITATION,
            ),
        ]
    )
    brief = (
        "next: own-model: write torch code for mobilenetv3_small_100 with pretrained "
        "weights, all layers, at 64 px"
    )
    assert credited(findings, brief, networks=True) == [CATALOG_CITATION]


def test_a_network_finding_is_not_credited_to_a_brief_that_names_another_or_none() -> None:
    """Every depth or own-model brief shares two words with "fine-tune ... on all layers",
    so the words alone would stamp the paper on work it never touched."""
    findings = Findings(
        suggestions=[Suggestion("fine-tune timm efficientnet_b0 on all layers", "r", "doi:1")]
    )
    assert (
        credited(
            findings,
            "next: fine-tune-depth: keep the recipe and fine-tune all layers",
            networks=True,
        )
        == []
    )
    assert (
        credited(
            findings,
            "next: own-model: fine-tune deit_tiny_patch16_224 on all layers",
            networks=True,
        )
        == []
    )
    assert credited(
        findings, "next: own-model: fine-tune efficientnet_b0 on all layers", networks=True
    ) == ["doi:1"]


def test_a_table_or_prompt_technique_keeps_the_credit_it_had_before_day_4() -> None:
    """The network rule is for image runs. A slash in a table or prompt technique is not a
    model id, and its credit is read by the words as it always was."""
    for technique, brief in [
        (
            "k-fold/out-of-fold target encoding for high-cardinality categoricals",
            "next: categorical-encoding: out-of-fold target encoding of the high-cardinality "
            "categoricals",
        ),
        (
            "add chain-of-thought/self-consistency to the instructions",
            "next: add chain-of-thought reasoning to the instructions",
        ),
    ]:
        findings = Findings(suggestions=[Suggestion(technique, "r", "doi:1")])
        assert credited(findings, brief) == ["doi:1"]
        assert credited(findings, brief, networks=True) == ["doi:1"]


def test_the_catalog_never_spends_a_slot_on_a_fit_backbone_or_an_unsized_network() -> None:
    """A fit() backbone opens no own-model entry and a network timm never sized can only
    be refused as not priced: either would spend a round on nothing the wall can pass."""
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(_NO_MODEL),
            _catalog("resnet50", "convnext_tiny", "vit_small_patch16_dinov3", "efficientnet_b3"),
        ]
    )
    findings = _image_researcher(llm).research(
        profile="27000 images, 10 classes", tried=[], ruled_out=["convnext_large"], round=2
    )
    kept = [s for s in findings.suggestions if s.citation == CATALOG_CITATION]
    assert [vl.models_named(s.technique) for s in kept] == [["efficientnet_b3"]]
    listed = {n.strip() for line in _catalog_listing(set()).splitlines() for n in line.split(",")}
    assert listed.isdisjoint(vl.FIT_BACKBONES)


def test_a_later_round_is_told_the_models_already_tried_and_never_keeps_one() -> None:
    llm = _Recorder(
        [_queries("q"), _suggest(_NO_MODEL), _catalog("efficientnet_b3", "efficientnet_b0")]
    )
    findings = _image_researcher(llm).research(  # type: ignore[arg-type]
        profile="27000 images, 10 classes",
        tried=["resnet18 64px all layers, 3 epochs"],
        ruled_out=["convnext_large"],
        round=2,
        tried_models={"efficientnet_b3"},
    )
    kept = [s for s in findings.suggestions if s.citation == CATALOG_CITATION]
    assert [vl.models_named(s.technique) for s in kept] == [["efficientnet_b0"]]
    text = "\n".join(m.content for m in llm.sent[-1])
    assert "do not suggest these: convnext_large, efficientnet_b3" in text


def test_a_paper_naming_only_what_the_wall_cannot_price_still_gets_the_catalog() -> None:
    llm = _FakeLLM(
        [
            _queries("q"),
            _suggest(
                {"technique": "fine-tune google/vit-base-patch16-224", "rationale": "r", "paper": 1}
            ),
            _catalog("efficientnet_b0"),
        ]
    )
    findings = _image_researcher(llm).research(
        profile="27000 images, 10 classes", tried=[], ruled_out=[], round=2
    )
    assert findings.citations[0] == CATALOG_CITATION


def test_each_version_of_a_family_is_its_own_catalog_line() -> None:
    listing = _catalog_listing(set())
    families = {line.split(",")[0].strip()[:12] for line in listing.splitlines()}
    assert any(f.startswith("mobilenetv2") for f in families)
    assert any(f.startswith("mobilenetv3") for f in families)
