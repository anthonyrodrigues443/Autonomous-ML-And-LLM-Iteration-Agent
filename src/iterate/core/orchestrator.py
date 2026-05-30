"""The Orchestrator — closes the agentic loop.

Drives ``baseline → propose → execute → score → record → decide → repeat`` over
the Week-2 substrate. History and proposer failures land in a `Memory` (Day 4,
sqlite or in-memory); stop logic is delegated to a `Terminator` (Day 3). The
Proposer reads cross-run history straight from Memory so previous sessions inform
the next proposal.

`current_model` follows the best-so-far candidate (so the Proposer's prompt always
reflects "what's currently in use"). On the first iteration it's the model the
baseline used, supplied by the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING

from iterate.core.proposer import ProposerError
from iterate.core.terminator import AttemptOutcome, LoopState
from iterate.schemas.experiment import Experiment

if TYPE_CHECKING:
    from iterate.adapters.compute.local import LocalExecutor
    from iterate.core.memory import Memory
    from iterate.core.proposer import Proposer
    from iterate.core.terminator import Terminator
    from iterate.schemas.experiment import Candidate, ExperimentResult
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
        terminator: Terminator,
        memory: Memory,
        *,
        data_summary: str,
        baseline_model: str,
        baseline_candidate: Candidate | None = None,
    ) -> None:
        self._target = target
        self._proposer = proposer
        self._executor = executor
        self._terminator = terminator
        self._memory = memory
        self._data_summary = data_summary
        self._initial_model = baseline_model
        self._baseline_candidate = baseline_candidate

    def run(self) -> RunResult:
        baseline = (
            self._executor.execute(self._target, self._baseline_candidate)
            if self._baseline_candidate is not None
            else self._executor.execute(self._target)
        )
        if not baseline.succeeded or baseline.metrics is None:
            log.warning("orchestrator: baseline failed (%s); aborting", baseline.error)
            return RunResult(
                baseline=baseline,
                history=[],
                best=None,
                stopped_because="baseline_failed",
            )

        run_id = self._memory.start_run(self._target.name, baseline)
        direction = baseline.metrics.direction
        metric = baseline.metrics.primary
        current_run: list[Experiment] = []
        best: Experiment | None = None
        current_model = self._initial_model
        started_at = perf_counter()
        stopped_because = "exhausted"  # safety: a well-formed terminator always fires
        iteration = 0

        while True:
            iteration += 1
            outcome: AttemptOutcome
            last_experiment: Experiment | None = None

            try:
                candidate = self._proposer.propose(
                    data_summary=self._data_summary,
                    baseline=baseline,
                    current_model=current_model,
                    history=self._memory.history(self._target.name),
                )
            except ProposerError as exc:
                log.warning("orchestrator: iteration %d proposer failed: %s", iteration, exc)
                self._memory.record_proposer_failure(run_id, iteration, current_model, str(exc))
                outcome = "proposer_error"
            else:
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
                current_run.append(experiment)
                self._memory.record(run_id, experiment)
                last_experiment = experiment

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
                    outcome = "improved"
                else:
                    outcome = "no_improvement"

            state = LoopState(
                iteration=iteration,
                baseline=baseline,
                best=best,
                last_experiment=last_experiment,
                last_attempt_outcome=outcome,
                elapsed_seconds=perf_counter() - started_at,
            )
            reason = self._terminator.update_and_check(state)
            if reason is not None:
                stopped_because = reason
                break

        self._memory.finish_run(run_id, stopped_because)
        return RunResult(
            baseline=baseline,
            history=current_run,
            best=best,
            stopped_because=stopped_because,
        )


__all__ = ["Orchestrator", "RunResult"]
