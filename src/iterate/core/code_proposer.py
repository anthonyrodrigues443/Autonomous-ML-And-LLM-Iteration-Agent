"""The CodeProposer — the LLM that WRITES the next experiment as code.

Sibling of `Proposer` and `Reconstructor`: same `LLMClient` protocol, same
tool-calling + retry machinery, different intent. Where the spec `Proposer` picks
an allow-listed estimator and emits ``changes = {"model", "params"}``, the
CodeProposer writes a `train_and_predict` function and emits
``changes = {"code": "<source>"}``. Both produce a plain `Candidate`; the executor
routes on the presence of ``"code"`` (`core.codegen.is_code_candidate`).

There is deliberately NO library allow-list on this path: the agent imports
whatever it wants and we install its imports before running (Day-5 executor wiring;
the deterministic `core.codegen.required_imports` extractor it relies on already
exists). Before a candidate is accepted, its code passes a cheap static check
(`core.codegen.validate_train_and_predict`) so malformed snippets become a targeted
re-prompt rather than a wasted run.

Security boundary is unchanged: this runs the agent's OWN generated code in the
sandbox, never any user-supplied source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iterate.core.codegen import validate_train_and_predict
from iterate.prompts import PROMPTS
from iterate.schemas.experiment import Candidate
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment, ExperimentResult


class CodeProposerError(RuntimeError):
    """The model failed to return a usable code candidate after retries."""


_PROMPTS = PROMPTS["code_proposer"]
_HISTORY_DESC_LIMIT = 120


def _build_tool() -> ToolSpec:
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": fields["code"]},
                "description": {"type": "string", "description": fields["description"]},
                "rationale": {"type": "string", "description": fields["rationale"]},
                "expected_metric_delta": {
                    "type": "number",
                    "description": fields["expected_metric_delta"],
                },
            },
            "required": ["code", "description", "rationale"],
        },
    )


PROPOSE_CODE = _build_tool()


class CodeProposer:
    """Asks the LLM to WRITE the next `Candidate` as a train_and_predict function."""

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,  # larger than the spec proposer — it emits a whole function
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def propose(
        self,
        *,
        data_summary: str,
        baseline: ExperimentResult,
        history: list[Experiment] | None = None,
    ) -> Candidate:
        if baseline.metrics is None:
            raise CodeProposerError("baseline has no metrics to improve on")
        messages = _build_messages(
            data_summary=data_summary,
            metric=baseline.metrics.primary,
            direction=baseline.metrics.direction,
            score=baseline.metrics.primary_value,
            history=history or [],
        )
        detail = ""
        for _ in range(self._max_retries + 1):
            response = self._client.chat(
                messages,
                tools=[PROPOSE_CODE],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next((c for c in response.tool_calls if c.name == PROPOSE_CODE.name), None)
            if call is None:
                detail = "model replied without calling propose_code"
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            code = call.arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                detail = "tool call carried no 'code'"
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            reason = validate_train_and_predict(code)
            if reason is not None:
                detail = reason
                messages.append(
                    Message(role="user", content=_PROMPTS["invalid_code_nudge"].format(reason=reason))
                )
                continue
            return _to_candidate(call.arguments, code)
        raise CodeProposerError(
            f"no usable code candidate after {self._max_retries + 1} attempt(s): {detail}"
        )


def _to_candidate(args: dict[str, Any], code: str) -> Candidate:
    description = str(args.get("description") or "").strip() or "generated train_and_predict"
    rationale = str(args.get("rationale") or "").strip() or "(no rationale given)"
    return Candidate(
        description=description,
        changes={"code": code.strip()},
        rationale=rationale,
        source="proposer",
        expected_improvement=_as_float(args.get("expected_metric_delta")),
    )


def _build_messages(
    *,
    data_summary: str,
    metric: str,
    direction: str,
    score: float,
    history: list[Experiment],
) -> list[Message]:
    system = _PROMPTS["system"].format(metric=metric, direction=direction)
    if history:
        history_section = (
            _PROMPTS["history_header"] + "\n" + "\n".join(_format_history(history, metric)) + "\n\n"
        )
    else:
        history_section = ""
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        metric=metric,
        score=f"{score:.4f}",
        direction=direction,
        history_section=history_section,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _format_history(history: list[Experiment], metric: str) -> list[str]:
    """Summarize past attempts by description + outcome.

    Code candidates carry their whole function source in ``changes``; echoing that
    back would flood the prompt, so we summarize by the one-line description only.
    """
    lines = []
    for exp in history:
        desc = exp.candidate.description.strip()
        if len(desc) > _HISTORY_DESC_LIMIT:
            desc = desc[: _HISTORY_DESC_LIMIT - 3] + "..."
        result = exp.result
        if result is None:
            outcome = "not run"
        elif result.succeeded and result.metrics is not None:
            outcome = f"{metric}={result.metrics.primary_value:.4f}"
        else:
            outcome = f"FAILED ({result.error})"
        lines.append(f"- {desc} -> {outcome}")
    return lines


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["PROPOSE_CODE", "CodeProposer", "CodeProposerError"]
