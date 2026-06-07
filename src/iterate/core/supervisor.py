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
        lines = _format_history(history[-_HISTORY_LIMIT:], metric)
        history_section = _PROMPTS["history_header"] + "\n" + "\n".join(lines) + "\n\n"
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
    return lines


__all__ = ["PLAN_NEXT", "Supervisor", "SupervisorDecision", "SupervisorError"]
