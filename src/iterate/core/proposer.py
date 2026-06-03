"""The Proposer — the LLM that proposes the next experiment.

Given a data summary, the re-measured baseline, the current model, and what's
already been tried, the Proposer asks the LLM to choose the most appropriate model
+ hyperparameters for the next `Candidate`. It speaks only the provider-agnostic
`LLMClient` protocol and gets a structured proposal via a `propose_candidate` tool
call — with a text-reply retry fallback, because the protocol exposes no
`tool_choice` and a model may answer in prose.

Reconstructing a baseline from a user-supplied source (md/txt/notebook) reads it as
text ONLY and never executes it — user-supplied code is untrusted (malware/RCE
risk), so we rebuild the approach as our own spec and re-measure through our eval.
That reuses this same machinery and lands later in Week 3. (Note: the e2b sandbox at
v0.2 runs the agent's OWN generated code, never the user's.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from iterate.prompts import PROMPTS
from iterate.schemas.experiment import Candidate
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment, ExperimentResult


class ProposerError(RuntimeError):
    """The model failed to return a usable candidate after retries."""


class SupportsPropose(Protocol):
    """What the Orchestrator needs from any proposer (spec `Proposer` or `CodeProposer`).

    ``current_model`` is meaningful only on the spec path (it names the estimator
    in play); the code path accepts it for a uniform call site and ignores it.
    """

    def propose(
        self,
        *,
        data_summary: str,
        baseline: ExperimentResult,
        current_model: str,
        history: list[Experiment] | None = ...,
    ) -> Candidate: ...


_PROMPTS = PROMPTS["proposer"]


def _build_tool() -> ToolSpec:
    """Build the propose_candidate ToolSpec — wording from prompts.yaml, types here."""
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": fields["model"]},
                "params": {
                    "type": "object",
                    "description": fields["params"],
                    "additionalProperties": True,
                },
                "description": {"type": "string", "description": fields["description"]},
                "rationale": {"type": "string", "description": fields["rationale"]},
                "expected_metric_delta": {
                    "type": "number",
                    "description": fields["expected_metric_delta"],
                },
            },
            "required": ["model", "description", "rationale"],
        },
    )


PROPOSE_CANDIDATE = _build_tool()


def summarize_dataset(dataset: TabularDataset) -> str:
    """A compact, LLM-facing brief of a tabular dataset."""
    numeric = dataset.train_features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in dataset.features if c not in numeric]
    lines = [
        f"Rows: {dataset.n_train} train / {dataset.n_test} test (sealed holdout).",
        f"Target column: {dataset.target!r}.",
        f"Features ({len(dataset.features)}): {len(numeric)} numeric, {len(categorical)} categorical.",
    ]
    if numeric:
        lines.append(f"Numeric: {', '.join(numeric[:20])}.")
    if categorical:
        lines.append(f"Categorical: {', '.join(categorical[:20])}.")
    return "\n".join(lines)


class Proposer:
    """Asks the LLM for the next `Candidate` to try."""

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 2,  # 3 total attempts — local models occasionally reply in prose
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
        current_model: str,
        history: list[Experiment] | None = None,
    ) -> Candidate:
        if baseline.metrics is None:
            raise ProposerError("baseline has no metrics to improve on")
        messages = _build_messages(
            data_summary=data_summary,
            metric=baseline.metrics.primary,
            direction=baseline.metrics.direction,
            score=baseline.metrics.primary_value,
            current_model=current_model,
            history=history or [],
        )
        detail = ""
        for _ in range(self._max_retries + 1):
            response = self._client.chat(
                messages,
                tools=[PROPOSE_CANDIDATE],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next((c for c in response.tool_calls if c.name == PROPOSE_CANDIDATE.name), None)
            if call is not None:
                return _to_candidate(call.arguments)
            detail = "model replied without calling propose_candidate"
            messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
        raise ProposerError(f"no candidate after {self._max_retries + 1} attempt(s): {detail}")


def _to_candidate(args: dict[str, Any]) -> Candidate:
    model = args.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ProposerError("proposal is missing a 'model'")
    changes: dict[str, Any] = {"model": model.strip()}
    params = args.get("params")
    if isinstance(params, dict) and params:
        changes["params"] = params
    description = str(args.get("description") or "").strip() or f"try {model.strip()}"
    rationale = str(args.get("rationale") or "").strip() or "(no rationale given)"
    return Candidate(
        description=description,
        changes=changes,
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
    current_model: str,
    history: list[Experiment],
) -> list[Message]:
    system = _PROMPTS["system"].format(
        metric=metric, direction=direction, current_model=current_model
    )
    if history:
        history_section = (
            _PROMPTS["history_header"] + "\n" + "\n".join(_format_history(history, metric)) + "\n\n"
        )
    else:
        history_section = ""
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        current_model=current_model,
        metric=metric,
        score=f"{score:.4f}",
        direction=direction,
        history_section=history_section,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _format_history(history: list[Experiment], metric: str) -> list[str]:
    lines = []
    for exp in history:
        result = exp.result
        if result is None:
            outcome = "not run"
        elif result.succeeded and result.metrics is not None:
            outcome = f"{metric}={result.metrics.primary_value:.4f}"
        else:
            outcome = f"FAILED ({result.error})"
        lines.append(f"- {exp.candidate.changes} -> {outcome}")
    return lines


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "PROPOSE_CANDIDATE",
    "Proposer",
    "ProposerError",
    "SupportsPropose",
    "summarize_dataset",
]
