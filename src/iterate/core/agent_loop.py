"""The supervised agent loop — the multi-agent (Supervisor + Coder) code path.

Mirrors the spec `Orchestrator`, but each iteration is: the Supervisor reads the
history and briefs the next experiment → the Coding agent runs that brief as a
cell-by-cell session and returns a scored result. Reuses `Memory`, the `Terminator`,
and the spec baseline as the bar to beat. Returns the same `RunResult` so the CLI
treats both paths uniformly.

The Coder is rebuilt per experiment (a fresh kernel = a fresh session); the cells
are stored on the candidate so the notebook deliverable can render the real session.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING

from iterate.adapters.compute.local import run_in_process
from iterate.core.orchestrator import RunResult
from iterate.core.supervisor import SupervisorError
from iterate.core.terminator import AttemptOutcome, LoopState
from iterate.schemas.experiment import Candidate, Experiment

if TYPE_CHECKING:
    from collections.abc import Callable

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.core.coder import CodingAgent
    from iterate.core.memory import Memory
    from iterate.core.supervisor import Supervisor, SupervisorDecision
    from iterate.core.terminator import Terminator
    from iterate.schemas.experiment import ExperimentResult
    from iterate.targets.base import BenchmarkTarget

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _improves(
    result: ExperimentResult, best: Experiment | None, baseline: ExperimentResult, direction: str
) -> bool:
    if result.metrics is None:
        return False
    bar: float | None = None
    if best is not None and best.result is not None and best.result.metrics is not None:
        bar = best.result.metrics.primary_value
    elif baseline.metrics is not None:
        bar = baseline.metrics.primary_value
    if bar is None:
        return True
    new = result.metrics.primary_value
    return new < bar if direction == "minimize" else new > bar


def run_supervised(
    *,
    target: BenchmarkTarget,
    dataset: TabularDataset,
    supervisor: Supervisor,
    make_coder: Callable[[], CodingAgent],
    terminator: Terminator,
    memory: Memory,
    data_summary: str,
) -> RunResult:
    """Run the Supervisor + Coder loop until the terminator (or supervisor) stops."""
    baseline = run_in_process(target)  # spec default = the bar to beat
    if not baseline.succeeded or baseline.metrics is None:
        log.warning("agent loop: baseline failed (%s); aborting", baseline.error)
        return RunResult(baseline=baseline, history=[], best=None, stopped_because="baseline_failed")

    run_id = memory.start_run(target.name, baseline)
    direction = baseline.metrics.direction
    current_run: list[Experiment] = []
    best: Experiment | None = None
    started_at = perf_counter()
    stopped_because = "exhausted"
    iteration = 0

    while True:
        iteration += 1
        outcome: AttemptOutcome
        last_experiment: Experiment | None = None
        try:
            # Memory already holds every recorded experiment (line below records each
            # one) — adding current_run would feed this run's experiments in twice.
            decision = supervisor.decide(
                data_summary=data_summary,
                baseline=baseline,
                history=memory.history(target.name),
            )
        except SupervisorError as exc:
            log.warning("agent loop: iteration %d supervisor failed: %s", iteration, exc)
            memory.record_proposer_failure(run_id, iteration, "supervisor", str(exc))
            outcome = "proposer_error"
        else:
            if decision.stop:
                stopped_because = "supervisor"
                break
            start_code = _winning_code(best)  # carry the best working code forward
            start_score = (
                best.result.metrics.primary_value
                if best is not None and best.result is not None and best.result.metrics is not None
                else None
            )
            try:
                experiment = _run_experiment(
                    make_coder(), dataset, decision, iteration, target.name, start_code, start_score
                )
            except Exception as exc:  # one bad experiment must not kill the run
                # e.g. the LLM backend timing out after retries, or a kernel dying.
                # Record it and let the terminator (patience) decide, like a failed cell.
                log.warning("agent loop: iteration %d coder failed: %s", iteration, exc)
                memory.record_proposer_failure(run_id, iteration, "coder", str(exc))
                outcome = "proposer_error"
            else:
                current_run.append(experiment)
                memory.record(run_id, experiment)
                last_experiment = experiment
                result = experiment.result
                assert result is not None
                if result.succeeded and result.metrics is not None:
                    log.info(
                        "agent loop: iteration %d %r -> %s=%.4f",
                        iteration, decision.title, result.metrics.primary,
                        result.metrics.primary_value,
                    )
                else:
                    log.info("agent loop: iteration %d %r -> failed (%s)", iteration,
                             decision.title, result.error)
                if result.succeeded and _improves(result, best, baseline, direction):
                    best = experiment
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
        reason = terminator.update_and_check(state)
        if reason is not None:
            stopped_because = reason
            break

    memory.finish_run(run_id, stopped_because)
    return RunResult(
        baseline=baseline, history=current_run, best=best,
        stopped_because=stopped_because, run_id=run_id,
    )


def _winning_code(best: Experiment | None) -> str | None:
    """The working code to seed the next experiment from: the SUCCESSFUL agent cells
    of the best session so far, concatenated in order. Sessions now build in stages
    (PREPARE -> MODEL -> SUBMIT), so no single cell is self-contained — the pipeline
    lives across cells, and re-running the successful ones in order reproduces it.
    Errored cells are dropped so a fixed-after-failure step doesn't carry the broken
    attempt forward."""
    if best is None:
        return None
    cells = best.candidate.changes.get("cells")
    if isinstance(cells, list):
        good = [
            str(cell["code"])
            for cell in cells
            if cell.get("source") == "agent" and not cell.get("error") and cell.get("code")
        ]
        if good:
            return "\n\n".join(good)
    code = best.candidate.changes.get("code")
    return str(code) if isinstance(code, str) else None


def _run_experiment(
    coder: CodingAgent,
    dataset: TabularDataset,
    decision: SupervisorDecision,
    iteration: int,
    target_name: str,
    starting_code: str | None,
    starting_score: float | None,
) -> Experiment:
    coding = coder.run(
        dataset=dataset, brief=decision.brief, experiment_id="pending",
        starting_code=starting_code, starting_score=starting_score,
    )
    cells = [
        {"code": c.code, "stdout": c.stdout, "error": c.error, "source": c.source,
         "outputs": c.outputs}
        for c in coding.cells
    ]
    code = "\n\n".join(c.code for c in coding.cells if c.source == "agent") or "# (no code)"
    candidate = Candidate(
        description=decision.title,
        changes={"code": code, "cells": cells},
        rationale=decision.brief,
        source="proposer",
    )
    experiment = Experiment(
        candidate=candidate,
        target=target_name,
        hypothesis=decision.brief,
        status="completed" if coding.result.succeeded else "failed",
        iteration=iteration,
        result=coding.result,
        started_at=_now(),
        finished_at=_now(),
    )
    # link the result back to this experiment's id
    linked = coding.result.model_copy(update={"experiment_id": experiment.id})
    return experiment.model_copy(update={"result": linked})


__all__ = ["run_supervised"]
