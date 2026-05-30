"""The Orchestrator — closes the agentic loop.

Drives ``baseline → propose → execute → score → record → decide → repeat`` over
the Week-2 substrate. Holds history in memory and runs its own minimal stop
checks (`max_iterations` + `patience`). The Memory store (Day 4, sqlite) replaces
the in-memory list; the Terminator (Day 3) takes over stop logic via a delegated
protocol — both YAGNI-deferred so this lands clean.

`current_model` follows the best-so-far candidate (so the Proposer's prompt always
reflects "what's currently in use"). On the first iteration it's the model the
baseline used, supplied by the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from iterate.core.proposer import ProposerError
from iterate.schemas.experiment import Experiment

if TYPE_CHECKING:
    from iterate.adapters.compute.local import LocalExecutor
    from iterate.core.proposer import Proposer
    from iterate.schemas.experiment import ExperimentResult
    from iterate.targets.base import BenchmarkTarget


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Outcome of an ``Orchestrator.run()`` — the audit trail of one autonomous run."""

    baseline: ExperimentResult
    history: list[Experiment]
    best: Experiment | None
    stopped_because: str  # "max_iterations" | "patience" | "baseline_failed"


def _now() -> datetime:
    return datetime.now(UTC)


def _improves(
    candidate_result: ExperimentResult,
    best: Experiment | None,
    baseline: ExperimentResult,
    direction: str,
) -> bool:
    """Did ``candidate_result`` improve on the current bar (best so far, else baseline)?"""
    if candidate_result.metrics is None:
        return False
    bar: float | None = None
    if best is not None and best.result is not None and best.result.metrics is not None:
        bar = best.result.metrics.primary_value
    elif baseline.metrics is not None:
        bar = baseline.metrics.primary_value
    if bar is None:
        return True
    new = candidate_result.metrics.primary_value
    return new < bar if direction == "minimize" else new > bar


class Orchestrator:
    """Closes the agentic loop on a single target."""

    def __init__(
        self,
        target: BenchmarkTarget,
        proposer: Proposer,
        executor: LocalExecutor,
        *,
        data_summary: str,
        baseline_model: str,
        max_iterations: int = 10,
        patience: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self._target = target
        self._proposer = proposer
        self._executor = executor
        self._data_summary = data_summary
        self._initial_model = baseline_model
        self._max_iterations = max_iterations
        self._patience = patience

    def run(self) -> RunResult:
        baseline = self._executor.execute(self._target)
        if not baseline.succeeded or baseline.metrics is None:
            log.warning("orchestrator: baseline failed (%s); aborting", baseline.error)
            return RunResult(
                baseline=baseline,
                history=[],
                best=None,
                stopped_because="baseline_failed",
            )

        direction = baseline.metrics.direction
        metric = baseline.metrics.primary
        history: list[Experiment] = []
        best: Experiment | None = None
        current_model = self._initial_model
        non_improving = 0
        stopped_because = "max_iterations"

        for iteration in range(1, self._max_iterations + 1):
            try:
                candidate = self._proposer.propose(
                    data_summary=self._data_summary,
                    baseline=baseline,
                    current_model=current_model,
                    history=history,
                )
            except ProposerError as exc:
                log.warning("orchestrator: iteration %d proposer failed: %s", iteration, exc)
                non_improving += 1
                if non_improving >= self._patience:
                    stopped_because = "patience"
                    break
                continue

            experiment = Experiment(
                candidate=candidate,
                target=self._target.name,
                hypothesis=candidate.description,
                status="running",
                iteration=iteration,
                started_at=_now(),
            )
            result = self._executor.execute(self._target, candidate)
            # Link the result back to *this* experiment (the executor wrote candidate.id).
            result = result.model_copy(update={"experiment_id": experiment.id})
            experiment = experiment.model_copy(
                update={
                    "status": "completed" if result.succeeded else "failed",
                    "result": result,
                    "finished_at": _now(),
                }
            )
            history.append(experiment)

            if result.succeeded and result.metrics is not None:
                log.info(
                    "orchestrator: iteration %d %s -> %s=%.4f",
                    iteration,
                    candidate.changes.get("model"),
                    metric,
                    result.metrics.primary_value,
                )
            else:
                log.info(
                    "orchestrator: iteration %d %s -> failed (%s)",
                    iteration,
                    candidate.changes.get("model"),
                    result.error,
                )

            if result.succeeded and _improves(result, best, baseline, direction):
                best = experiment
                model = candidate.changes.get("model")
                if isinstance(model, str):
                    current_model = model
                non_improving = 0
            else:
                non_improving += 1

            if non_improving >= self._patience:
                stopped_because = "patience"
                break

        return RunResult(
            baseline=baseline,
            history=history,
            best=best,
            stopped_because=stopped_because,
        )


__all__ = ["Orchestrator", "RunResult"]
