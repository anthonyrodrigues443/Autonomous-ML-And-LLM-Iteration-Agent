"""The Supervisor — the strategist that briefs the coding agent across experiments.

It reads the full run history (what was tried, the components used, the scores),
COMPRESSES it into a short summary, and hands the coding agent a brief for the next
experiment: 1-2 lines of "what's been learned so far" plus a specific strategy to
try. It can also decide to stop.

For now the Supervisor is a single LLM that does this via its own tool
(`plan_next`); the summarization work is a natural candidate to graduate into a
dedicated Summarizer agent later — the tool boundary is the future agent boundary.
The coding agent never sees raw history; it only sees the Supervisor's brief, which
is what keeps each experiment's context tight and lifts a weaker coder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.core.scoring import direction
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment, ExperimentResult


class SupervisorError(RuntimeError):
    """The supervisor failed to return a usable plan after retries."""


_PROMPTS = PROMPTS["supervisor"]
_HISTORY_LIMIT = 12  # most recent experiments shown


@dataclass(frozen=True)
class SupervisorDecision:
    """What to do next: stop, or run an experiment described by ``brief``."""

    stop: bool
    title: str  # short label for the leaderboard
    brief: str  # the instruction handed to the coding agent (summary + strategy)


def _build_tool() -> ToolSpec:
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "stop": {"type": "boolean", "description": fields["stop"]},
                "title": {"type": "string", "description": fields["title"]},
                "brief": {"type": "string", "description": fields["brief"]},
            },
            "required": ["stop", "brief"],
        },
    )


PLAN_NEXT = _build_tool()


class Supervisor:
    """Briefs the coding agent for the next experiment from run history."""

    def __init__(
        self,
        client: LLMClient,
        *,
        metric: str,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._metric = metric
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def decide(
        self, *, data_summary: str, baseline: ExperimentResult, history: list[Experiment]
    ) -> SupervisorDecision:
        if baseline.metrics is None:
            raise SupervisorError("baseline has no metrics")
        messages = _build_messages(
            data_summary=data_summary,
            metric=self._metric,
            direction=direction(self._metric),
            score=baseline.metrics.primary_value,
            history=history,
        )
        detail = ""
        for _ in range(self._max_retries + 1):
            response = self._client.chat(
                messages, tools=[PLAN_NEXT], temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next((c for c in response.tool_calls if c.name == PLAN_NEXT.name), None)
            if call is not None:
                return _to_decision(call.arguments)
            detail = "model replied without calling plan_next"
            messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
        raise SupervisorError(f"no plan after {self._max_retries + 1} attempt(s): {detail}")


def _to_decision(args: dict[str, Any]) -> SupervisorDecision:
    stop = bool(args.get("stop"))
    brief = str(args.get("brief") or "").strip()
    title = str(args.get("title") or "").strip() or (brief[:60] or "experiment")
    if not stop and not brief:
        raise SupervisorError("plan_next returned neither stop nor a brief")
    return SupervisorDecision(stop=stop, title=title, brief=brief)


def _build_messages(
    *, data_summary: str, metric: str, direction: str, score: float, history: list[Experiment]
) -> list[Message]:
    system = _PROMPTS["system"].format(metric=metric, direction=direction)
    if history:
        recent = history[-_HISTORY_LIMIT:]
        lines = _format_history(recent, metric)
        # the ledger scans the FULL history: a lever tried before the display
        # window is still tried.
        extras = "".join(
            block + "\n\n"
            for block in (_technique_table(recent, metric), _lever_ledger(history))
            if block
        )
        history_section = _PROMPTS["history_header"] + "\n" + "\n".join(lines) + "\n\n" + extras
    else:
        history_section = "No experiments yet — brief the first one.\n\n"
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        metric=metric,
        score=f"{score:.4f}",
        direction=direction,
        history_section=history_section,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


_FLOAT = re.compile(r"\d+\.\d+")


def _validation_trail(exp: Experiment) -> str:
    """The within-session validation scores an experiment printed, in order — the
    knowledge that otherwise dies with the session (every IMPROVE attempt that lost,
    and how far a FAILED session actually got before it died). Scans stdout lines
    that look like a validation/score print and takes the last float on each (the
    metric value usually trails labels like 'Validation F1 score:')."""
    cells = exp.candidate.changes.get("cells")
    if not isinstance(cells, list):
        return ""
    seen: list[str] = []
    for cell in cells:
        stdout = cell.get("stdout") if isinstance(cell, dict) else None
        for line in (stdout or "").splitlines():
            low = line.lower()
            if "val" not in low and "score" not in low:
                continue
            floats = _FLOAT.findall(line)
            if floats and (not seen or seen[-1] != floats[-1]):
                seen.append(floats[-1])
    if len(seen) < 2:  # one value adds nothing beyond the final score
        return ""
    return f" (val tries: {' -> '.join(seen[-6:])})"


def _format_history(history: list[Experiment], metric: str) -> list[str]:
    lines = []
    for exp in history:
        desc = exp.candidate.description.strip()[:90]
        code = exp.candidate.changes.get("code")
        used = ""
        if isinstance(code, str):
            comps = codegen.components_used(code)
            if comps:
                used = f" [used: {', '.join(comps)}]"
        result = exp.result
        if result is not None and result.succeeded and result.metrics is not None:
            outcome = f"{metric}={result.metrics.primary_value:.4f}"
        elif result is not None and result.error:
            outcome = f"FAILED ({result.error.splitlines()[0][:60]})"
        else:
            outcome = "not run"
        lines.append(f"- {desc}{used} -> {outcome}{_validation_trail(exp)}")
        # The Summarizer's digest is the cross-notebook knowledge: what the data
        # showed and what helped or hurt, so the supervisor can compound winners.
        if exp.digest is not None:
            for label, items in (
                ("data", exp.digest.data_insights),
                ("helped", exp.digest.what_helped),
                ("hurt", exp.digest.what_hurt),
            ):
                if items:
                    lines.append(f"    {label}: " + "; ".join(items[:4]))
            if exp.digest.takeaway:
                lines.append(f"    next-idea: {exp.digest.takeaway}")
    return lines


# Technique LEVER classes, each detected by deterministic code markers. The ledger
# built from these makes coverage explicit ("what is done and what is not") — run
# c7ddda92 left the imbalance lever untouched for 9 of 10 experiments while the
# supervisor orbited model swaps, because nothing surfaced the untried classes.
# Markers are matched case-insensitively against each experiment's full code.
_LEVER_MARKERS: dict[str, tuple[str, ...]] = {
    "categorical-encoding": ("onehotencoder", "targetencoder", "ordinalencoder", "get_dummies"),
    "numeric-transform": ("powertransformer", "quantiletransformer", "log1p", "np.log"),
    "imbalance-or-threshold": ("class_weight", "scale_pos_weight", "smote", "threshold"),
    "interactions-or-ratios": ("polynomialfeatures", "interaction", "ratio", "_per_"),
    "feature-selection": ("selectkbest", "selectfrommodel", "rfe(", "feature_importances"),
    "ensembling": ("votingclassifier", "stackingclassifier", "votingregressor", "stackingregressor"),
    "hyperparameter-search": ("gridsearchcv", "randomizedsearchcv", "halvinggridsearchcv"),
}


def _lever_ledger(history: list[Experiment]) -> str:
    """One explicit done/not-done line across the technique lever classes, scanned
    from every experiment's code (full history, not just the display window). The
    supervisor should pull from the NOT-yet-tried side instead of re-orbiting the
    tried side."""
    if not history:
        return ""
    tried: set[str] = set()
    for exp in history:
        code = exp.candidate.changes.get("code")
        if not isinstance(code, str):
            continue
        low = code.lower()
        for lever, markers in _LEVER_MARKERS.items():
            if lever not in tried and any(m in low for m in markers):
                tried.add(lever)
    tried_s = ", ".join(lv for lv in _LEVER_MARKERS if lv in tried) or "none"
    untried_s = ", ".join(lv for lv in _LEVER_MARKERS if lv not in tried) or "none"
    return f"Levers tried: {tried_s} | Levers NOT yet tried: {untried_s}"


def _technique_table(history: list[Experiment], metric: str) -> str:
    """Best score reached whenever each technique appeared, aggregated across all
    digests, so the pattern 'this technique tends to score well' is explicit rather
    than left for the model to infer from scattered lines."""
    best: dict[str, float] = {}
    seen: dict[str, int] = {}
    for exp in history:
        if exp.digest is None or exp.digest.score is None:
            continue
        score = exp.digest.score
        for tech in exp.digest.techniques:
            seen[tech] = seen.get(tech, 0) + 1
            if tech not in best or score > best[tech]:
                best[tech] = score
    if not best:
        return ""
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    cells = [f"{t} {best[t]:.4f} (x{seen[t]})" for t, _ in ranked]
    return f"Technique scoreboard (best {metric} when each appeared): " + " | ".join(cells)


__all__ = ["PLAN_NEXT", "Supervisor", "SupervisorDecision", "SupervisorError"]
