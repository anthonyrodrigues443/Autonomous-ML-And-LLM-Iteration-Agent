"""The Summarizer — distills ONE finished experiment notebook into a compact,
structured `ExperimentDigest`.

This is the specialist the Supervisor's docstring anticipated: the tool boundary
graduating into its own agent. It runs once, right after an experiment completes,
reads that one session (the cells the coder ran and what they printed), and writes
a digest. The digest, not the notebook, is what crosses to the next experiment:
the Supervisor reasons over many small digests instead of holding raw notebooks,
which would bloat context and induce hallucination by mid-run.

A deterministic skeleton (the components actually instantiated, the score, the
within-session validation trail) is filled in by code; the LLM adds the insight
fields (what the data showed, what helped or hurt, what to try next). If the LLM
call fails for any reason the skeleton is returned alone, so a digest failure can
never kill the run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.prompts import PROMPTS
from iterate.schemas.experiment import ExperimentDigest
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment

_PROMPTS = PROMPTS["summarizer"]
_FLOAT = re.compile(r"\d+\.\d+")
_CELL_OUTPUT_TAIL = 600  # per-cell stdout shown to the summarizer
_SESSION_CAP = 16000  # total session text handed to the summarizer (one bounded call)


def _build_tool() -> ToolSpec:
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    str_list = {"type": "array", "items": {"type": "string"}}
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "data_insights": {**str_list, "description": fields["data_insights"]},
                "what_helped": {**str_list, "description": fields["what_helped"]},
                "what_hurt": {**str_list, "description": fields["what_hurt"]},
                "takeaway": {"type": "string", "description": fields["takeaway"]},
            },
            "required": ["takeaway"],
        },
    )


SUMMARIZE = _build_tool()


class Summarizer:
    """Turns one finished `Experiment` into an `ExperimentDigest` (never raises)."""

    def __init__(
        self,
        client: LLMClient,
        *,
        metric: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._metric = metric
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def summarize(self, experiment: Experiment) -> ExperimentDigest:
        skeleton = _skeleton(experiment, self._metric)
        session = _render_session(experiment)
        if not session:
            return skeleton  # nothing to read (e.g. a pre-run crash): skeleton only
        messages = _build_messages(
            metric=self._metric,
            description=experiment.candidate.description,
            score=skeleton.score,
            techniques=skeleton.techniques,
            val_trail=skeleton.val_trail,
            session=session,
        )
        for _ in range(self._max_retries + 1):
            try:
                response = self._client.chat(
                    messages, tools=[SUMMARIZE], temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception:  # an LLM/backend failure must not cost the run a digest
                return skeleton
            call = next((c for c in response.tool_calls if c.name == SUMMARIZE.name), None)
            if call is not None:
                return _merge(skeleton, call.arguments)
            messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
        return skeleton  # the model never called the tool; skeleton is still useful


def _skeleton(experiment: Experiment, metric: str) -> ExperimentDigest:
    """The deterministic part: techniques actually instantiated, the score, and the
    within-session validation trail. Always correct, no LLM."""
    code = experiment.candidate.changes.get("code")
    techniques = codegen.components_used(code) if isinstance(code, str) else []
    result = experiment.result
    score = (
        result.metrics.primary_value
        if result is not None and result.succeeded and result.metrics is not None
        else None
    )
    return ExperimentDigest(
        techniques=techniques, score=score, val_trail=_val_trail(experiment)
    )


def _val_trail(experiment: Experiment) -> str:
    """The validation scores the session printed, in order (raw 'a -> b -> c')."""
    cells = experiment.candidate.changes.get("cells")
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
    return " -> ".join(seen[-8:])


def _render_session(experiment: Experiment) -> str:
    """The agent cells (code + a tail of what they printed + any error), capped so
    the summarizer is one bounded call no matter how long the session ran."""
    cells = experiment.candidate.changes.get("cells")
    if not isinstance(cells, list):
        return ""
    parts: list[str] = []
    for i, cell in enumerate(c for c in cells if isinstance(c, dict) and c.get("source") == "agent"):
        code = (cell.get("code") or "").strip()
        block = [f"# cell {i + 1}", code]
        out = (cell.get("stdout") or "").strip()
        if out:
            block.append("# output: " + _tail(out, _CELL_OUTPUT_TAIL))
        err = (cell.get("error") or "").strip()
        if err:
            block.append("# ERROR: " + err.splitlines()[-1][:160])
        parts.append("\n".join(block))
    return _tail("\n\n".join(parts), _SESSION_CAP)


def _tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "...(truncated)\n" + text[-limit:]


def _merge(skeleton: ExperimentDigest, args: dict[str, Any]) -> ExperimentDigest:
    def _strs(key: str) -> list[str]:
        v = args.get(key)
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    helped, hurt = _strs("what_helped"), _strs("what_hurt")
    takeaway = str(args.get("takeaway") or "").strip()
    if not takeaway:  # the takeaway must never be empty — synthesize one from the evidence
        if helped:
            takeaway = f"Push further on what helped: {helped[0]}."
        elif skeleton.score is not None:
            takeaway = f"Refine the best approach so far (scored {skeleton.score:.4f}) with one new feature."
        else:
            takeaway = "Repair the failure and secure a valid baseline submission."
    return skeleton.model_copy(
        update={
            "data_insights": _strs("data_insights"),
            "what_helped": helped,
            "what_hurt": hurt,
            "takeaway": takeaway,
        }
    )


def _build_messages(
    *,
    metric: str,
    description: str,
    score: float | None,
    techniques: list[str],
    val_trail: str,
    session: str,
) -> list[Message]:
    system = _PROMPTS["system"].format(metric=metric)
    user = _PROMPTS["user_template"].format(
        metric=metric,
        description=description,
        score=("FAILED (no valid predictions)" if score is None else f"{score:.4f}"),
        techniques=", ".join(techniques) or "(none detected)",
        val_trail=val_trail or "(none printed)",
        session=session,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


__all__ = ["SUMMARIZE", "Summarizer"]
