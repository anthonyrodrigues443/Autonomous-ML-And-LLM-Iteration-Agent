"""The Researcher: literature grounding for the next experiment.

``plan_queries`` writes the search queries, the harness searches OpenAlex and arXiv,
``suggest_techniques`` picks papers by INDEX in the list it was shown, so it cannot
emit an identifier that was not fetched. Never raises; a failure returns empty
findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iterate.adapters.research import ArxivClient, OpenAlexClient, search_all
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iterate.adapters.research import Paper, PaperSource
    from iterate.llm.base import LLMClient

log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["researcher"]
_MAX_QUERIES = 3
_MAX_SUGGESTIONS = 3
_PAPERS_PER_QUERY = 4
_PAPERS_SHOWN = 10


@dataclass(frozen=True)
class Suggestion:
    """One technique worth an experiment, and the paper that supports it."""

    technique: str
    rationale: str
    citation: str  # resolved by the harness from the model's paper INDEX


@dataclass(frozen=True)
class Setup:
    """How the run should be measured, chosen from the same papers the technique
    suggestions came from. Validated by the caller before anything is built."""

    metric: str
    starting_model: str = ""
    why: str = ""


@dataclass(frozen=True)
class Findings:
    """What one research pass produced. Empty is a valid, common outcome."""

    suggestions: list[Suggestion] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    papers_seen: int = 0
    setup: Setup | None = None  # only on the pre-loop pass, when no metric was given

    def __bool__(self) -> bool:
        return bool(self.suggestions)

    @property
    def citations(self) -> list[str]:
        return [s.citation for s in self.suggestions]

    def render(self) -> str:
        """The lean block handed to the supervisor. One line per suggestion, so a
        research pass costs the planning prompt three lines, not three pages."""
        return "\n".join(
            f"- {s.technique} — {s.rationale} <{s.citation}>" for s in self.suggestions
        )


def _queries_tool() -> ToolSpec:
    spec = _PROMPTS["queries_tool"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": spec["fields"]["queries"],
                }
            },
            "required": ["queries"],
        },
    )


def _suggest_tool(key: str = "suggest_tool") -> ToolSpec:
    spec = _PROMPTS[key]
    fields = spec["fields"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "technique": {"type": "string", "description": fields["technique"]},
                            "rationale": {"type": "string", "description": fields["rationale"]},
                            "paper": {"type": "integer", "description": fields["paper"]},
                        },
                        "required": ["technique", "rationale", "paper"],
                    },
                }
            },
            "required": ["suggestions"],
        },
    )


def _setup_tool() -> ToolSpec:
    spec = _PROMPTS["setup_tool"]
    fields = spec["fields"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": fields["metric"]},
                "starting_model": {"type": "string", "description": fields["starting_model"]},
                "why": {"type": "string", "description": fields["why"]},
            },
            "required": ["metric"],
        },
    )


def _vision_setup_tool(allowed: Sequence[str]) -> ToolSpec:
    """The image metric choice, as an enum: the model can only name a metric that
    scores these labels."""
    spec = _PROMPTS["vision_setup_tool"]
    fields = spec["fields"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": sorted(allowed),
                    "description": fields["metric"],
                },
                "why": {"type": "string", "description": fields["why"]},
            },
            "required": ["metric", "why"],
        },
    )


CHOOSE_SETUP = _setup_tool()
PLAN_QUERIES = _queries_tool()
SUGGEST_TECHNIQUES = _suggest_tool()
VISION_SUGGEST = _suggest_tool("vision_suggest_tool")
PROMPT_SUGGEST = _suggest_tool("prompt_suggest_tool")
_SUGGEST_BY_FAMILY = {
    "vision": ("vision_suggest_system", VISION_SUGGEST),
    "prompt": ("prompt_suggest_system", PROMPT_SUGGEST),
}


class Researcher:
    """Grounds the next experiment in retrievable literature."""

    def __init__(
        self,
        client: LLMClient,
        *,
        metric: str = "",
        direction: str = "",
        family: str = "tabular",
        sources: Sequence[PaperSource] | None = None,
        cache_dir: Any | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._metric = metric
        self._direction = direction
        self._family = family
        self._sources: Sequence[PaperSource] = (
            sources
            if sources is not None
            else [OpenAlexClient(cache_dir=cache_dir), ArxivClient(cache_dir=cache_dir)]
        )
        self._temperature = temperature
        self._max_tokens = max_tokens

    def research(
        self,
        *,
        profile: str,
        tried: Sequence[str] = (),
        choose_setup: bool = False,
        allowed_metrics: Sequence[str] = (),
        suggest: bool = True,
        allow_without_papers: bool = False,
    ) -> Findings:
        """One research pass. Never raises; empty findings mean the run proceeds
        without literature grounding, exactly like a failed digest."""
        tried_text = ", ".join(tried) or "nothing yet"
        try:
            queries = self._plan_queries(profile, tried_text)
        except Exception as exc:
            # INFO, not DEBUG: research silently producing nothing looks identical
            # to research finding nothing, and only one of those is a problem.
            log.info("researcher: query planning failed (%s: %s)", type(exc).__name__, exc)
            queries = []
        # An image run chooses its metric even when the search comes back empty: a failed
        # search must not quietly decide how the run is measured. A table or prompt run
        # takes the deterministic default there, as it did before images.
        ungrounded = choose_setup and allow_without_papers
        if not queries and not ungrounded:
            return Findings()

        papers: list[Paper] = []
        for query in queries:
            papers.extend(search_all(self._sources, query, limit=_PAPERS_PER_QUERY))
        # search_all dedupes per call; a second pass dedupes ACROSS queries, which
        # overlap by design since they attack one problem from several angles.
        unique: dict[str, Paper] = {}
        for paper in papers:
            unique.setdefault(paper.identifier, paper)
        shortlist = sorted(unique.values(), key=lambda p: -p.cited_by)[:_PAPERS_SHOWN]
        if not shortlist and not ungrounded:
            return Findings(queries=queries)

        setup: Setup | None = None
        if choose_setup:
            # Runs BEFORE the technique judgement and over the same papers, so the
            # metric choice compounds off the full research rather than off a
            # thinner second pass. Separate call, not extra fields: a 12B asked for
            # techniques, a metric and a model in one emit starts dropping fields.
            try:
                setup = self._choose_setup(profile, shortlist, allowed=allowed_metrics)
            except Exception as exc:
                log.info("researcher: setup choice failed (%s: %s)", type(exc).__name__, exc)

        suggestions: list[Suggestion] = []
        if suggest and shortlist:
            try:
                suggestions = self._suggest(profile, tried_text, shortlist)
            except Exception as exc:
                log.info("researcher: suggestion failed (%s: %s)", type(exc).__name__, exc)
        return Findings(
            suggestions=suggestions,
            queries=queries,
            papers_seen=len(shortlist),
            setup=setup,
        )

    def _plan_queries(self, profile: str, tried: str) -> list[str]:
        # An image run with no metric yet is searching for how a problem like this one
        # is measured; every other pass is searching for what raises a metric it has.
        setup_pass = self._family == "vision" and not self._metric
        system = {"prompt": "prompt_queries_system", "vision": "vision_queries_system"}.get(
            self._family, "queries_system"
        )
        messages = [
            Message(
                role="system",
                content=_PROMPTS["vision_setup_queries_system" if setup_pass else system].format(
                    metric=self._metric, direction=self._direction
                ),
            ),
            Message(
                role="user",
                content=_PROMPTS["setup_queries_user" if setup_pass else "queries_user"].format(
                    metric=self._metric, profile=profile.strip(), tried=tried
                ),
            ),
        ]
        args = self._call(messages, PLAN_QUERIES)
        raw = args.get("queries") if args else None
        if not isinstance(raw, list):
            return []
        queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()]
        return queries[:_MAX_QUERIES]

    def _suggest(self, profile: str, tried: str, papers: list[Paper]) -> list[Suggestion]:
        listing = "\n".join(f"{i + 1}. {p.brief()}\n   {p.abstract}" for i, p in enumerate(papers))
        system, tool = _SUGGEST_BY_FAMILY.get(self._family, ("suggest_system", SUGGEST_TECHNIQUES))
        messages = [
            Message(
                role="system",
                content=_PROMPTS[system].format(metric=self._metric, direction=self._direction),
            ),
            Message(
                role="user",
                content=_PROMPTS["suggest_user"].format(
                    metric=self._metric, profile=profile.strip(), tried=tried, papers=listing
                ),
            ),
        ]
        args = self._call(messages, tool)
        raw = args.get("suggestions") if args else None
        if not isinstance(raw, list):
            return []

        out: list[Suggestion] = []
        for item in raw[:_MAX_SUGGESTIONS]:
            if not isinstance(item, dict):
                continue
            technique = str(item.get("technique") or "").strip()
            rationale = str(item.get("rationale") or "").strip()
            paper = _resolve(item.get("paper"), papers)
            # A suggestion whose index does not resolve is DROPPED, not kept with a
            # blank citation: an ungrounded suggestion is exactly what this
            # specialist exists to rule out.
            if not technique or paper is None:
                continue
            out.append(
                Suggestion(technique=technique, rationale=rationale, citation=paper.identifier)
            )
        return out

    def _choose_setup(
        self, profile: str, papers: list[Paper], *, allowed: Sequence[str]
    ) -> Setup | None:
        listing = (
            "\n".join(f"{i + 1}. {p.brief()}\n   {p.abstract}" for i, p in enumerate(papers))
            or _PROMPTS["no_papers"]
        )
        metrics = ", ".join(sorted(allowed))
        vision = self._family == "vision"
        messages = [
            Message(
                role="system", content=_PROMPTS["vision_setup_system" if vision else "setup_system"]
            ),
            Message(
                role="user",
                content=_PROMPTS["vision_setup_user" if vision else "setup_user"]
                .replace("{profile}", profile.strip())
                .replace("{metrics}", metrics)
                .replace("{papers}", listing),
            ),
        ]
        args = self._call(messages, _vision_setup_tool(allowed) if vision else CHOOSE_SETUP)
        if not args:
            return None
        metric = str(args.get("metric") or "").strip()
        if not metric:
            return None
        return Setup(
            metric=metric,
            starting_model=str(args.get("starting_model") or "").strip(),
            why=str(args.get("why") or "").strip(),
        )

    def _call(self, messages: list[Message], tool: ToolSpec) -> dict[str, Any] | None:
        """One structured call with a single retry nudge, mirroring the Summarizer."""
        for attempt in range(2):
            response = self._client.chat(
                messages,
                tools=[tool],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next((c for c in response.tool_calls if c.name == tool.name), None)
            if call is not None:
                return dict(call.arguments)
            if attempt == 0:
                messages = [*messages, Message(role="user", content=_PROMPTS["retry_nudge"])]
        return None


def _resolve(index: Any, papers: list[Paper]) -> Paper | None:
    """Map the model's 1-based paper number onto a fetched paper.

    This function is the citation guarantee. The model hands over an integer; if it
    is not a valid position in the list it was actually shown, nothing is cited.
    """
    try:
        position = int(index)
    except (TypeError, ValueError):
        return None
    if 1 <= position <= len(papers):
        return papers[position - 1]
    return None


__all__ = ["Findings", "Researcher", "Setup", "Suggestion", "credited"]


_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "into", "from", "that", "this", "using", "use", "data"}
)


def _content_words(text: str) -> set[str]:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return {w for w in cleaned.split() if len(w) > 3 and w not in _STOPWORDS}


def credited(findings: Findings | None, brief: str) -> list[str]:
    """Citations for the suggestions this brief actually took up.

    Deliberately conservative. Stamping every citation from the pass onto every
    later candidate would claim a paper informed work it never touched, and an
    unearned citation is no better than an invented one — the whole point of this
    specialist is that its provenance can be trusted. So a suggestion is credited
    only when the brief and the technique share at least two content words, or the
    technique phrase appears outright. Under-attribution is the safe failure.
    """
    if findings is None or not brief.strip():
        return []
    brief_words = _content_words(brief)
    lowered = brief.lower()
    out: list[str] = []
    for suggestion in findings.suggestions:
        technique = suggestion.technique.strip()
        if not technique:
            continue
        overlap = _content_words(technique) & brief_words
        matched = technique.lower() in lowered or len(overlap) >= 2
        if matched and suggestion.citation not in out:
            out.append(suggestion.citation)
    return out
