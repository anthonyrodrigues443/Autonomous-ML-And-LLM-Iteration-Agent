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
from iterate.core.supervisor import SupervisorError, lever_markers_for_brief
from iterate.core.terminator import AttemptOutcome, LoopState
from iterate.schemas.experiment import Candidate, Experiment

if TYPE_CHECKING:
    from collections.abc import Callable

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.core.coder import CodingAgent
    from iterate.core.memory import Memory
    from iterate.core.summarizer import Summarizer
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
    summarizer: Summarizer | None = None,
    on_experiment: Callable[..., None] | None = None,
) -> RunResult:
    """Run the Supervisor + Coder loop until the terminator (or supervisor) stops.

    ``summarizer`` (optional) distills each finished experiment into a compact
    `ExperimentDigest` attached to the experiment before it is recorded, so the next
    Supervisor reasons over digests instead of raw notebooks (cross-notebook
    knowledge transfer). It never raises; a failed digest is simply absent.

    ``on_experiment`` (optional) is invoked after EVERY completed experiment —
    success or failure — with ``experiment=, baseline=, is_best=, run_id=`` keyword
    arguments. The CLI uses it to save each iteration's notebook the moment it
    finishes, so a crash or Ctrl-C mid-run still leaves every finished iteration's
    deliverable on disk. A failing hook is logged and never kills the run."""
    baseline = run_in_process(target)  # spec default = the bar to beat
    if not baseline.succeeded or baseline.metrics is None:
        log.warning("agent loop: baseline failed (%s); aborting", baseline.error)
        return RunResult(baseline=baseline, history=[], best=None, stopped_because="baseline_failed")

    run_id = memory.start_run(target.name, baseline)
    direction = baseline.metrics.direction
    current_run: list[Experiment] = []
    best: Experiment | None = None
    seen_digests: set[str] = set()  # sha256 of EVERY submission so far, for the no-op gate
    started_at = perf_counter()
    stopped_because = "exhausted"
    iteration = 0

    try:
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
                    # the loop's ACTUAL carried best — the brief's "so far:" slot is
                    # grounded on this so it describes the code the coder receives,
                    # never a cross-run best the coder does not hold.
                    carried_best=best,
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
                    experiment, preds_digest = _run_experiment(
                        make_coder(), dataset, decision, iteration, target.name,
                        start_code, start_score, seen_digests=frozenset(seen_digests),
                    )
                except Exception as exc:  # one bad experiment must not kill the run
                    # e.g. the LLM backend timing out after retries, or a kernel dying.
                    # Record it and let the terminator (patience) decide, like a failed cell.
                    log.warning("agent loop: iteration %d coder failed: %s", iteration, exc)
                    memory.record_proposer_failure(run_id, iteration, "coder", str(exc))
                    outcome = "proposer_error"
                else:
                    if preds_digest and preds_digest in seen_digests:
                        # Byte-identical to an earlier submission: stamp it so the
                        # supervisor's history shows a re-run, not a fresh result.
                        experiment.candidate.changes["duplicate_submission"] = True
                        log.info(
                            "agent loop: iteration %d submission duplicates an earlier experiment",
                            iteration,
                        )
                    elif preds_digest:
                        seen_digests.add(preds_digest)
                    if summarizer is not None:
                        experiment = _digest(summarizer, experiment, iteration)
                    experiment = _sanitize_unmeasured_digest(experiment, decision.brief)
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
                    if on_experiment is not None:
                        try:
                            on_experiment(
                                experiment=experiment, baseline=baseline,
                                is_best=(best is experiment), run_id=run_id,
                            )
                        except Exception:  # a deliverable hook must never kill the run
                            log.warning("agent loop: on_experiment hook failed", exc_info=True)

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

    except KeyboardInterrupt:
        # Ctrl-C: keep what the run already earned. Memory still gets finalized and the
        # best-so-far notebook is already on disk (on_experiment saves per iteration), so
        # an interrupt exits like a short run, not a stack trace.
        log.warning("agent loop: interrupted; finalizing with %d kept experiment(s)", len(current_run))
        stopped_because = "interrupted"

    memory.finish_run(run_id, stopped_because)
    return RunResult(
        baseline=baseline, history=current_run, best=best,
        stopped_because=stopped_because, run_id=run_id,
    )


def _sanitize_unmeasured_digest(experiment: Experiment, brief: str) -> Experiment:
    """Strip digest claims the machine verdict contradicts. Two live failure modes:
    a session that never executed its lever fabricated an 'implied weighting' win;
    a byte-duplicate submission's Findings claimed a settled optimization as the
    session's own win. Nothing RAISED a score in either case — so a duplicate keeps
    no what_helped at all, and an unexecuted lever keeps no claims naming it. The
    what_hurt channel (the valuable measured losses) survives untouched."""
    if experiment.digest is None:
        return experiment
    changes = experiment.candidate.changes
    if changes.get("duplicate_submission"):
        kept: list[str] = []
    elif changes.get("lever_unmeasured"):
        markers = lever_markers_for_brief(brief)
        if not markers:
            return experiment
        kept = [
            item for item in experiment.digest.what_helped
            if not any(m in item.lower() for m in markers)
        ]
    else:
        return experiment
    if len(kept) == len(experiment.digest.what_helped):
        return experiment
    log.info("agent loop: dropped what-helped claims contradicted by the machine verdict")
    return experiment.model_copy(
        update={"digest": experiment.digest.model_copy(update={"what_helped": kept})}
    )


def _digest(summarizer: Summarizer, experiment: Experiment, iteration: int) -> Experiment:
    """Attach the Summarizer's digest to the experiment. Never raises: a digest is
    a nice-to-have for the next Supervisor, not worth failing a recorded run over."""
    try:
        digest = summarizer.summarize(experiment)
    except Exception:  # the Summarizer already guards internally; this is belt-and-braces
        log.warning("agent loop: iteration %d summarizer failed", iteration, exc_info=True)
        return experiment
    return experiment.model_copy(update={"digest": digest})


def _winning_code(best: Experiment | None) -> str | None:
    """The working code to seed the next experiment from: the SUCCESSFUL agent cells
    of the best session so far, concatenated in order. Sessions now build in stages
    (PREPARE -> MODEL -> SUBMIT), so no single cell is self-contained — the pipeline
    lives across cells, and re-running the successful ones in order reproduces it.
    Errored cells are dropped so a fixed-after-failure step doesn't carry the broken
    attempt forward. A "fallback" cell (the harness's floor submit) is kept: when a
    session's submission came from the fallback, that cell IS the pipeline that
    produced the recorded score — it is self-contained and runs last, so appending
    it keeps the carried code reproducing what was actually scored."""
    if best is None:
        return None
    cells = best.candidate.changes.get("cells")
    if isinstance(cells, list):
        good = [
            str(cell["code"])
            for cell in cells
            if cell.get("source") in ("agent", "fallback")
            and not cell.get("error")
            and cell.get("code")
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
    *,
    seen_digests: frozenset[str] = frozenset(),
) -> tuple[Experiment, str | None]:
    """Run one briefed session; returns the experiment and the sha256 of its
    submitted predictions (for later sessions' identical-submission gate)."""
    from iterate.core.coder import lever_executed

    markers = lever_markers_for_brief(decision.brief)
    coding = coder.run(
        dataset=dataset, brief=decision.brief, experiment_id=f"iter-{iteration:02d}",
        starting_code=starting_code, starting_score=starting_score,
        brief_markers=markers,
        seen_digests=seen_digests,
    )
    cells = [
        {"code": c.code, "stdout": c.stdout, "error": c.error, "source": c.source,
         "outputs": c.outputs, "thinking": c.thinking}
        for c in coding.cells
    ]
    # The code fingerprint includes fallback cells: when the submission came from the
    # harness floor, the score-bearing pipeline must be what the lever ledger, the
    # technique scoreboard, and the grounded brief attribute — not the dead-end agent
    # cells alone. Agent cells stay too (errored or not): they are what was TRIED.
    code = (
        "\n\n".join(c.code for c in coding.cells if c.source in ("agent", "fallback"))
        or "# (no code)"
    )
    changes: dict[str, object] = {"code": code, "cells": cells}
    if markers and not lever_executed(coding.cells, markers, starting_code):
        # The commissioned lever never ran successfully — the score is the carried
        # pipeline's, not the lever's, and the supervisor must not credit it.
        changes["lever_unmeasured"] = True
    candidate = Candidate(
        description=decision.title,
        changes=changes,
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
    return experiment.model_copy(update={"result": linked}), coding.predictions_sha256


__all__ = ["run_supervised"]
