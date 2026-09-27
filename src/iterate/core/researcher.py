"""The Researcher: literature grounding for the next experiment.

``plan_queries`` writes the search queries, the harness searches OpenAlex and arXiv,
``suggest_techniques`` picks papers by INDEX in the list it was shown, so it cannot
emit an identifier that was not fetched. Never raises; a failure returns empty
findings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from iterate.adapters.research import ArxivClient, OpenAlexClient, search_all
from iterate.core import vision_levers as vl
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec
from iterate.targets import layers as arch

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from iterate.adapters.research import Paper, PaperSource
    from iterate.llm.base import LLMClient

log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["researcher"]
_MAX_QUERIES = 3
_MAX_SUGGESTIONS = 3
_PAPERS_PER_QUERY = 4
_PAPERS_SHOWN = 10
# Two networks a family is 4 to 5 kB of catalog, where all 1,101 names are 20 kB of a
# 12B's context.
_CATALOG_PER_FAMILY = 2
_FAMILY = re.compile(r"[a-z]+?v\d|[a-z]+")
# timm's own test networks: toys for its test suite, never worth fine-tuning.
_TOY = "test_"
CATALOG_CITATION = "catalog:timm_models.txt"


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

    def merge(self, other: Findings) -> Findings:
        """This pass and a later one as one: what was found first keeps its place, and a
        later suggestion joins only when its technique and citation are new."""
        seen = {(s.technique.casefold(), s.citation) for s in self.suggestions}
        added: list[Suggestion] = []
        for s in other.suggestions:
            key = (s.technique.casefold(), s.citation)
            if key not in seen:
                seen.add(key)
                added.append(s)
        return Findings(
            suggestions=[*self.suggestions, *added],
            queries=[*self.queries, *(q for q in other.queries if q not in self.queries)],
            papers_seen=self.papers_seen + other.papers_seen,
            setup=self.setup or other.setup,
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


def _catalog_tool() -> ToolSpec:
    spec = _PROMPTS["catalog_tool"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": spec["fields"]["models"],
                }
            },
            "required": ["models"],
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
NAME_MODELS = _catalog_tool()
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
        self._shown: set[str] = set()

    def research(
        self,
        *,
        profile: str,
        tried: Sequence[str] = (),
        choose_setup: bool = False,
        allowed_metrics: Sequence[str] = (),
        suggest: bool = True,
        allow_without_papers: bool = False,
        ruled_out: Sequence[str] = (),
        round: int = 1,
        tried_models: Collection[str] = (),
    ) -> Findings:
        """One research pass. Never raises; empty findings mean the run proceeds
        without literature grounding, exactly like a failed digest.

        ``ruled_out`` names the networks this run may not use, and the question says no
        more than that: what they would cost is the wall's business, never the search's.
        A later ``round`` fetches more papers a query and shows the ones no earlier pass
        showed first; on an image run whose papers then name no network a library loads,
        it names up to three from the catalog the package ships. ``tried_models`` are the
        models this run already spent an experiment on, a failed one included, which a
        later round tells the model and never keeps."""
        tried_text = ", ".join(tried) or "nothing yet"
        # Only a network a library loads is worth naming here: a stack fit() builds from
        # zero is the harness's own, and anything else is not a model at all.
        named_out = [name for name in ruled_out if _is_network(_bare(name))]
        later = round >= 2
        if later:
            named_out += [
                name for name in sorted(tried_models) if _bare(name) not in map(_bare, named_out)
            ]
        if named_out:
            tried_text += _PROMPTS["ruled_out_label"].format(names=", ".join(named_out))
        found = self._from_papers(
            profile,
            tried_text,
            choose_setup=choose_setup,
            allowed_metrics=allowed_metrics,
            suggest=suggest,
            allow_without_papers=allow_without_papers,
            limit=_PAPERS_PER_QUERY * max(round, 1),
            unseen_first=later,
        )
        if not (later and suggest and self._family == "vision"):
            return found
        exclude = {_bare(name) for name in [*ruled_out, *tried_models]} | _words(tried_text)
        # A paper that names only networks the wall cannot price, or has refused, leaves
        # the round as dry as one that names none, so the catalog still fills it.
        if any((_loadable(s.technique) & _priceable()) - exclude for s in found.suggestions):
            return found
        try:
            named = self._name_models(profile, tried_text, exclude=exclude)
        except Exception as exc:
            log.info("researcher: the catalog pass failed (%s: %s)", type(exc).__name__, exc)
            return found
        if named:
            log.info(
                "researcher: no paper named a network a library loads; from the catalog: %s",
                "; ".join(s.technique for s in named),
            )
        # First, because the Supervisor reads findings cut short and these are the lines
        # the round was spent for.
        return replace(found, suggestions=[*named, *found.suggestions])

    def _from_papers(
        self,
        profile: str,
        tried_text: str,
        *,
        choose_setup: bool,
        allowed_metrics: Sequence[str],
        suggest: bool,
        allow_without_papers: bool,
        limit: int,
        unseen_first: bool,
    ) -> Findings:
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
            papers.extend(search_all(self._sources, query, limit=limit))
        # search_all dedupes per call; a second pass dedupes ACROSS queries, which
        # overlap by design since they attack one problem from several angles.
        unique: dict[str, Paper] = {}
        for paper in papers:
            unique.setdefault(paper.identifier, paper)
        # Ranked by citations alone, a bigger fetch hands the most-cited papers of the
        # first pass straight back, so a later round puts the ones never shown first.
        shortlist = sorted(
            unique.values(),
            key=lambda p: (unseen_first and p.identifier in self._shown, -p.cited_by),
        )[:_PAPERS_SHOWN]
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
            self._shown.update(p.identifier for p in shortlist)
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
            if self._family == "vision" and not _stack_is_stated(technique, paper.abstract):
                kept = _without_the_stack(technique)
                log.info(
                    "researcher: dropped a stack the abstract does not state (%s)%s",
                    technique,
                    "" if kept is None else f", kept the model it names as {kept!r}",
                )
                if kept is None:
                    continue
                technique = kept
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

    def _name_models(
        self, profile: str, tried: str, *, exclude: Collection[str] = ()
    ) -> list[Suggestion]:
        """Up to three pretrained networks named from the catalog the package ships. A
        name the catalog does not list, or one excluded, is dropped. The words of what is
        kept are the harness's and it cites the catalog: the model wrote no finding, so a
        name or a size in a reason of its own must not reach the lever line."""
        listing = _catalog_listing(exclude)
        if not listing:
            return []
        messages = [
            Message(
                role="system",
                content=_PROMPTS["catalog_system"].format(
                    metric=self._metric, direction=self._direction
                ),
            ),
            Message(
                role="user",
                content=_PROMPTS["catalog_user"].format(
                    profile=profile.strip(), tried=tried, catalog=listing
                ),
            ),
        ]
        args = self._call(messages, NAME_MODELS)
        raw = args.get("models") if args else None
        if not isinstance(raw, list):
            return []
        out: list[Suggestion] = []
        for item in raw:
            name = _bare(str((item.get("model") if isinstance(item, dict) else item) or ""))
            if (
                name not in vl.CATALOG
                or vl.models_named(name) != [name]
                or name not in _priceable()
                or name in exclude
                or name.startswith(_TOY)
            ):
                continue
            technique = _PROMPTS["catalog_technique"].format(name=name)
            if any(s.technique == technique for s in out):
                continue
            out.append(
                Suggestion(
                    technique=technique,
                    rationale=_PROMPTS["catalog_rationale"],
                    citation=CATALOG_CITATION,
                )
            )
            if len(out) == _MAX_SUGGESTIONS:
                break
        return out

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


_NUMBER = re.compile(r"\d*\.\d+|\d+")
# A width an abstract really states sits next to the word for what it is. Without one of
# these anywhere, every number in the abstract is an input size, a batch, an epoch count
# or a class count, and matching an invented width against those passes by coincidence:
# measured on gemma4:12b, a width-free abstract carrying 64 px, batches of 32 and 10
# classes kept an invented conv(32) pool conv(64) linear(10) five times out of five.
_WIDTH_WORD = re.compile(r"filters|channels|units|neurons|feature maps|kernels|hidden", re.I)
# The bracketed part of a layer, for taking a stack out of a suggestion that says more.
_LAYER_CALL = re.compile(r"\b(conv|pool|dropout|linear)\s*\(\s*[\d.,\s]*\)", re.I)


def _stack_is_stated(technique: str, abstract: str) -> bool:
    """Whether an image suggestion that writes a layer stack may become a finding: the
    abstract the suggestion cites has to state widths at all, and every number in the
    stack has to appear in it.

    A 12B copies the example it is shown, so a stack it invented would arrive with a real
    paper pinned to it, and the lever ladder opens on findings. A suggestion that writes
    no stack is not touched. Abstracts seldom list widths, so this drops more than it
    keeps, which is the safe direction."""
    spec = arch.found_strict(technique)
    if spec is None:
        return True
    if not _WIDTH_WORD.search(abstract):
        return False
    stated = {float(m.group()) for m in _NUMBER.finditer(abstract)}
    return all(float(value) in stated for layer in spec for value in layer[1:])


def _without_the_stack(technique: str) -> str | None:
    """The suggestion with the widths taken out of its layers, when it names a network as
    well, or None when the stack is all it said.

    What may not survive an unstated stack is the STACK: a paper that names a real model
    and sketches a head is a legitimate own-model or backbone finding, and dropping it
    whole for the head's sake loses the model name and the citation with it."""
    lower = technique.lower().replace("-", "_")
    if not vl.models_named(technique) and not any(b in lower for b in vl.FIT_BACKBONES):
        return None
    stripped = _LAYER_CALL.sub(lambda m: m.group(1), technique)
    return None if arch.found_strict(stripped) is not None else stripped


def _bare(name: str) -> str:
    """A network name as the ladder and the wall spell it, so a name ruled out here is
    the same name the ladder leaves off the line."""
    return vl.model_name({"model": name})


def _loadable(technique: str) -> set[str]:
    return {_bare(name) for name in vl.models_named(technique)}


def _priceable() -> set[str]:
    """The networks timm's table sizes, the only ones the wall can price."""
    from iterate.core import prices

    return set(prices.timm_sizes()[0])


def _is_network(name: str) -> bool:
    return name not in vl.FROM_ZERO and (name in vl.CATALOG or bool(vl.models_named(name)))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]*[a-z0-9]", text.lower()))


def _catalog_listing(exclude: Collection[str] = ()) -> str:
    """The catalog as a 12B can read it: one line a family, its lightest networks by
    timm's published weight count first, and only networks that table sizes, so every
    name offered is one the wall can price."""
    from iterate.core import prices

    sizes, _ = prices.timm_sizes()
    families: dict[str, list[tuple[float, str]]] = {}
    for name in vl.CATALOG:
        rows = sizes.get(name)
        family = _FAMILY.match(name)
        if (
            not rows
            or family is None
            or name in exclude
            or name in vl.FIT_BACKBONES
            or name.startswith(_TOY)
        ):
            continue
        weights = min(row.param_count_m for row in rows)
        families.setdefault(family.group(0), []).append((weights, name))
    return "\n".join(
        ", ".join(name for _, name in sorted(members)[:_CATALOG_PER_FAMILY])
        for _, members in sorted(families.items())
    )


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


__all__ = ["CATALOG_CITATION", "Findings", "Researcher", "Setup", "Suggestion", "credited"]


_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "into", "from", "that", "this", "using", "use", "data"}
)


def _content_words(text: str) -> set[str]:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return {w for w in cleaned.split() if len(w) > 3 and w not in _STOPWORDS}


def credited(findings: Findings | None, brief: str, *, networks: bool = False) -> list[str]:
    """Citations for the suggestions this brief actually took up.

    Deliberately conservative. Stamping every citation from the pass onto every
    later candidate would claim a paper informed work it never touched, and an
    unearned citation is no better than an invented one — the whole point of this
    specialist is that its provenance can be trusted. So a suggestion is credited
    only when the brief and the technique share at least two content words, or the
    technique phrase appears outright. A technique that names a network a library
    loads is credited only to a brief naming that same network, because "fine-tune
    ... with pretrained weights on all layers" shares its words with every such brief.
    That rule is for image runs (``networks``); a table or prompt technique names no
    network, and its credit is read as it always was. Under-attribution is the safe
    failure.
    """
    if findings is None or not brief.strip():
        return []
    brief_words = _content_words(brief)
    brief_models = _loadable(brief) if networks else set()
    lowered = brief.lower()
    out: list[str] = []
    for suggestion in findings.suggestions:
        technique = suggestion.technique.strip()
        if not technique:
            continue
        named = _loadable(technique) if networks else set()
        if named and not named & brief_models:
            continue
        overlap = _content_words(technique) & brief_words
        matched = technique.lower() in lowered or len(overlap) >= 2
        if matched and suggestion.citation not in out:
            out.append(suggestion.citation)
    return out
