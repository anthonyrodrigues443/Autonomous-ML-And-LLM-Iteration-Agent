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

import json
import logging
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from iterate.adapters.compute.local import run_in_process
from iterate.core import vision_levers as vl
from iterate.core.critic import stamp as stamp_verdict
from iterate.core.critic import was_rejected
from iterate.core.orchestrator import RunResult
from iterate.core.researcher import credited
from iterate.core.supervisor import (
    SupervisorError,
    lever_markers_for_brief,
    render_live_cells,
    run_ledger,
)
from iterate.core.terminator import AttemptOutcome, LoopState
from iterate.schemas.experiment import Candidate, Experiment

if TYPE_CHECKING:
    from collections.abc import Callable

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.core.coder import CodingAgent
    from iterate.core.critic import Critic
    from iterate.core.interactive import RunController
    from iterate.core.memory import Memory
    from iterate.core.researcher import Findings, Researcher
    from iterate.core.serving import Wall
    from iterate.core.summarizer import Summarizer
    from iterate.core.supervisor import Supervisor, SupervisorDecision
    from iterate.core.terminator import Terminator
    from iterate.schemas.experiment import ExperimentResult
    from iterate.targets.base import BenchmarkTarget

log = logging.getLogger(__name__)

# What a typed line has to look like to move the wall: "budget $80", "budget 80 a month",
# "serving budget is $120/month"; and "no budget" (or drop / lift / remove the budget) to
# take it down. Read before the Supervisor routes the line, so a budget is never a steer.
_BUDGET_LINE = re.compile(
    r"^\s*(?:serving\s+)?budget\s*(?:is|=|:|to|of)?\s*\$?\s*"
    r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?:usd|dollars)?\s*(?:(?:a|per|/)\s*(?:month|mo))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_NO_BUDGET_LINE = re.compile(
    r"^\s*(?:no|drop|lift|remove|clear)\s+(?:the\s+)?(?:serving\s+)?budget\s*[.!]?\s*$",
    re.IGNORECASE,
)


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
    researcher: Researcher | None = None,
    critic: Critic | None = None,
    max_research_calls: int = 3,
    max_inspect_calls: int = 2,
    on_experiment: Callable[..., None] | None = None,
    controller: RunController | None = None,
    # Which family's arm each iteration takes. "vision" carries the recipe the coder
    # starts from and gates the briefed lever on this run's own helper lines.
    family: str = "tabular",
    # The serving budget as a wall, shared with the Supervisor. After every experiment
    # the finished work is priced from what it did; one over the budget is stamped with
    # its price and can never become the best, whatever it scored.
    wall: Wall | None = None,
) -> RunResult:
    """Run the Supervisor + Coder loop until the terminator (or supervisor) stops.

    The optional collaborators never raise into the loop. ``summarizer`` attaches a
    digest to each experiment before it is recorded. ``researcher`` runs when the
    supervisor asks via ``want_research``, always on iteration 1, and at most
    ``max_research_calls`` times. ``want_inspect`` runs an unscored session that
    records no experiment and spends no patience, capped by ``max_inspect_calls``.
    ``on_experiment`` is called after EVERY completed experiment with
    ``experiment=, baseline=, is_best=, run_id=``. ``controller`` is the interactive
    seam: checkpoints at every iteration boundary, deadline suspended while paused,
    drained guidance and rules folded into ``decide``."""
    baseline = run_in_process(target)  # spec default = the bar to beat
    if not baseline.succeeded or baseline.metrics is None:
        log.warning("agent loop: baseline failed (%s); aborting", baseline.error)
        return RunResult(
            baseline=baseline, history=[], best=None, stopped_because="baseline_failed"
        )

    run_id = memory.start_run(target.name, baseline)
    vision = _vision_refs(baseline, target) if family == "vision" else None
    direction = baseline.metrics.direction
    current_run: list[Experiment] = []
    best: Experiment | None = None
    seen_digests: set[str] = set()  # sha256 of EVERY submission so far, for the no-op gate
    started_at = perf_counter()
    stopped_because = "exhausted"
    iteration = 0
    wants_research = False  # the supervisor's ask, carried to the next iteration
    research_calls = 0
    last_findings: Findings | None = None
    wants_inspect = False  # the same ask, for a free look at the data
    inspect_calls = 0
    last_inspection = ""  # facts persist once printed — re-reading them is free
    last_brief = ""  # the plan the supervisor had when it asked to look first
    if controller is not None:
        # Q&A is scoped to THIS run: current_run is appended in place below, so
        # the closure always sees exactly the experiments the user is watching —
        # never a prior run's iteration 3 answering for this run's iteration 3.
        controller.interpreter = _make_interpreter(
            controller=controller,
            supervisor=supervisor,
            experiments=current_run,
            baseline=baseline,
            data_summary=data_summary,
            wall=wall,
        )

        def _snapshot() -> RunResult:
            # The hard quit renders the summary from whatever has FINISHED at
            # this instant; the loop itself may still be blocked inside a cell.
            return RunResult(
                baseline=baseline,
                history=list(current_run),
                best=best,
                stopped_because="stopped-by-user",
                run_id=run_id,
            )

        controller.snapshot = _snapshot
        if vision is not None:
            # Image runs only: the two layer classes open on what the user typed, and
            # the note has to carry the mark the harness read out of it. The reader sees
            # this run's experiments, so the reply can say an ask opens nothing.
            def _read_ask(text: str) -> tuple[str, str]:
                return vl.ask_note(text, current_run, vl.recipe_of(best))

            controller.note_reader = _read_ask

    try:
        while True:
            iteration += 1
            guidance: list[str] = []
            rules: tuple[str, ...] = ()
            if controller is not None:
                controller.status = "between experiments"
                controller.checkpoint(None)
                if controller.abort_requested:
                    stopped_because = "stopped-by-user"
                    break
                guidance = controller.take_brief_notes()
                rules = controller.rules
            outcome: AttemptOutcome
            last_experiment: Experiment | None = None
            try:
                # Only pass the interactive kwargs when there is something to say, so
                # a controller-less run calls decide exactly as before.
                extra: dict[str, Any] = {}
                if guidance:
                    extra["user_guidance"] = "; ".join(guidance)
                if rules:
                    extra["standing_rules"] = rules
                # Iteration 1 always researches: with no history there is nothing for
                # the supervisor to base a want_research judgement on. After that it
                # is the supervisor's ask, bounded by the run cap.
                if (
                    researcher is not None
                    and (iteration == 1 or wants_research)
                    and (research_calls < max_research_calls)
                ):
                    research_calls += 1
                    findings = researcher.research(
                        profile=data_summary,
                        tried=(
                            vl.tried_components(current_run)
                            if vision is not None
                            else run_ledger(memory.history(target.name)).tried_components
                        ),
                    )
                    wants_research = False
                    if findings:
                        last_findings = findings
                        extra["research"] = findings.render()
                        log.info(
                            "agent loop: researched %d papers -> %d suggestions",
                            findings.papers_seen,
                            len(findings.suggestions),
                        )
                if wants_inspect and inspect_calls < max_inspect_calls:
                    inspect_calls += 1
                    wants_inspect = False
                    observed = make_coder().inspect(
                        dataset=dataset,
                        brief=last_brief,
                        experiment_id=f"inspect-{inspect_calls:02d}",
                    )
                    if observed:
                        last_inspection = observed
                        log.info(
                            "agent loop: inspected the data -> %d facts (free, unscored)",
                            len(observed.splitlines()),
                        )
                if last_inspection:
                    extra["inspection"] = last_inspection
                if vision is not None:
                    # Every image run is named "vision-model", so memory can hold an
                    # earlier run's rows; the levers open and close on this run alone.
                    extra["this_run"] = list(current_run)
                    if last_findings is not None:
                        extra["known_findings"] = last_findings.render()
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
                    **extra,
                )
                # The budget this brief was judged under. A budget typed while the
                # session runs holds from the NEXT brief, as the reply promised.
                briefed_budget = wall.budget if wall is not None else None
                wants_research = decision.want_research
                wants_inspect = decision.want_inspect
                last_brief = decision.brief
            except SupervisorError as exc:
                log.warning("agent loop: iteration %d supervisor failed: %s", iteration, exc)
                memory.record_proposer_failure(run_id, iteration, "supervisor", str(exc))
                outcome = "proposer_error"
                if controller is not None:  # the drained steers must survive the retry
                    for note in guidance:
                        controller.requeue_brief_note(note)
            else:
                if decision.stop:
                    stopped_because = "supervisor"
                    break
                if controller is not None:
                    controller.emit(
                        "brief", iteration=iteration, title=decision.title, brief=decision.brief
                    )
                start_code = (
                    _own_code(best) if vision is not None else _winning_code(best)
                )  # carry the best working code forward
                start_score = (
                    best.result.metrics.primary_value
                    if best is not None
                    and best.result is not None
                    and best.result.metrics is not None
                    else None
                )
                try:
                    experiment, preds_digest = _run_experiment(
                        make_coder(),
                        dataset,
                        decision,
                        iteration,
                        target.name,
                        start_code,
                        start_score,
                        seen_digests=frozenset(seen_digests),
                        findings=last_findings,
                        vision=vision,
                        best=best,
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
                    # Audit trail: which human words shaped this experiment (the
                    # same post-hoc stamp pattern as duplicate_submission above).
                    if guidance:
                        experiment.candidate.changes["user_guidance"] = "; ".join(guidance)
                    if rules:
                        experiment.candidate.changes["user_rules"] = list(rules)
                    _hold_against_the_wall(experiment, wall, iteration, briefed_budget)
                    if summarizer is not None:
                        experiment = _digest(summarizer, experiment, iteration)
                    experiment = _sanitize_unmeasured_digest(experiment, decision.brief)
                    current_run.append(experiment)
                    if critic is not None:
                        bar = (
                            best.result.metrics.primary_value
                            if best is not None
                            and best.result is not None
                            and best.result.metrics is not None
                            else None
                        )
                        stamp_verdict(experiment, critic.review(experiment, previous_best=bar))
                    memory.record(run_id, experiment)
                    last_experiment = experiment
                    result = experiment.result
                    assert result is not None
                    if result.succeeded and result.metrics is not None:
                        log.info(
                            "agent loop: iteration %d %r -> %s=%.4f",
                            iteration,
                            decision.title,
                            result.metrics.primary,
                            result.metrics.primary_value,
                        )
                    else:
                        log.info(
                            "agent loop: iteration %d %r -> failed (%s)",
                            iteration,
                            decision.title,
                            result.error,
                        )
                    if (
                        result.succeeded
                        and _improves(result, best, baseline, direction)
                        and not was_rejected(experiment)
                        and "over_budget" not in experiment.candidate.changes
                    ):
                        best = experiment
                        outcome = "improved"
                    else:
                        outcome = "no_improvement"
                    if controller is not None:
                        controller.emit(
                            "score",
                            iteration=iteration,
                            score=(
                                result.metrics.primary_value if result.metrics is not None else None
                            ),
                            error=result.error,
                            is_best=(best is experiment),
                        )
                    if on_experiment is not None:
                        try:
                            on_experiment(
                                experiment=experiment,
                                baseline=baseline,
                                is_best=(best is experiment),
                                run_id=run_id,
                            )
                        except Exception:  # a deliverable hook must never kill the run
                            log.warning("agent loop: on_experiment hook failed", exc_info=True)

            paused_total = controller.paused_seconds_total if controller is not None else 0.0
            state = LoopState(
                iteration=iteration,
                baseline=baseline,
                best=best,
                last_experiment=last_experiment,
                last_attempt_outcome=outcome,
                # The --until deadline measures WORKING time: paused time (accrued at
                # any checkpoint, including inside a coding session) is subtracted.
                elapsed_seconds=max(0.0, perf_counter() - started_at - paused_total),
            )
            # The user's stop outranks the terminator's label: a stop typed during
            # the session must read "stopped-by-user", not whatever patience or the
            # deadline happens to conclude from the wound-down iteration.
            if controller is not None and controller.abort_requested:
                stopped_because = "stopped-by-user"
                break
            reason = terminator.update_and_check(state)
            if reason is not None:
                stopped_because = reason
                break

    except KeyboardInterrupt:
        # Ctrl-C: keep what the run already earned. Memory still gets finalized and the
        # best-so-far notebook is already on disk (on_experiment saves per iteration), so
        # an interrupt exits like a short run, not a stack trace.
        log.warning(
            "agent loop: interrupted; finalizing with %d kept experiment(s)", len(current_run)
        )
        stopped_because = "interrupted"

    memory.finish_run(run_id, stopped_because)
    return RunResult(
        baseline=baseline,
        history=current_run,
        best=best,
        stopped_because=stopped_because,
        run_id=run_id,
    )


def _make_interpreter(
    *,
    controller: RunController,
    supervisor: Supervisor,
    experiments: list[Experiment],
    baseline: ExperimentResult,
    data_summary: str,
    wall: Wall | None = None,
) -> Callable[[list[str], bool], None]:
    """The plain-English message interpreter the controller calls at boundaries.

    The Supervisor labels each line's intent (it is the role that knows the run);
    THIS closure executes the routing deterministically. ``experiments`` is the
    loop's live current-run list (appended in place), so Q&A always answers about
    THIS run. Duck-typed lookups keep old fakes and the frozen spec path working:
    a supervisor without ``route_message`` just means every line is a steer."""
    route = getattr(supervisor, "route_message", None)
    answer = getattr(supervisor, "answer", None)

    def interpret(batch: list[str], live_session: bool) -> None:
        for text in batch:
            if wall is not None and (moved := _move_the_wall(text, wall)) is not None:
                # Only the user moves the wall, and this is how: the new budget holds
                # from the next brief on, and nothing already recorded is re-judged.
                controller.reply(moved)
                continue
            kind = "steer_now"
            if callable(route):
                try:
                    kind = route(text, live_session=live_session)
                except Exception:  # classification degrades, never crashes
                    log.warning(
                        "agent loop: route_message failed; treating as a steer", exc_info=True
                    )
            if kind == "question":
                if callable(answer):
                    try:
                        live = (
                            render_live_cells(controller.live_cells)
                            if controller.live_cells
                            else None
                        )
                        reply = answer(
                            text,
                            history=experiments,
                            baseline=baseline,
                            data_summary=data_summary,
                            live_session=live,
                        )
                    except Exception as exc:
                        reply = f"(could not answer: {exc})"
                else:
                    reply = "(questions are not supported on this run)"
                controller.reply(reply)
            elif kind == "rule":
                controller.add_rule(text)
                controller.reply("standing rule added — every future brief will carry it")
            elif kind == "steer_later":
                controller.add_brief_note(text)
                controller.reply("noted for the NEXT experiment's brief")
            else:  # steer_now, or any fallback: the least-destructive route
                controller.add_brief_note(text)
                if live_session:
                    controller.add_session_note(text)
                    controller.reply(
                        "delivering to the running session at its next cell "
                        "(also visible when the next experiment is planned)"
                    )
                else:
                    controller.reply("folding into the next experiment's brief")

    return interpret


def _move_the_wall(text: str, wall: Wall) -> str | None:
    """What a typed budget line does to the wall, and the reply; None for any other line."""
    from iterate.schemas.serving import money

    if _NO_BUDGET_LINE.match(text):
        wall.budget = None
        return "the serving budget is lifted; from here on nothing is refused for its price"
    found = _BUDGET_LINE.match(text)
    if found is None:
        return None
    budget = float(found.group(1).replace(",", ""))
    if budget <= 0:
        return 'a budget is dollars a month above zero, e.g. "budget $80"; the wall is unchanged'
    wall.budget = budget
    return (
        f"serving budget is now {money(budget)} a month at {wall.requests_per_hour:,} requests "
        "an hour; it holds from the next brief on, and nothing already recorded is re-judged"
    )


def _hold_against_the_wall(
    experiment: Experiment, wall: Wall | None, iteration: int, budget: float | None
) -> None:
    """Price the finished experiment from what it actually did, against the budget it was
    briefed under, and stamp it when that cannot cover it. The stamp is what keeps it
    from ever becoming the best; a network the table cannot price is left alone."""
    if wall is None:
        return
    priced = wall.stamp(experiment, budget=budget)
    if priced is not None:
        from iterate.schemas.serving import money

        log.info(
            "agent loop: iteration %d costs about $%.0f a month to serve on %s, over the %s "
            "a month serving budget; it cannot be the winner",
            iteration,
            priced.usd_per_month or 0.0,
            priced.host,
            money(budget) if budget is not None else "the wall's",
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
            item
            for item in experiment.digest.what_helped
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


def _vision_refs(baseline: ExperimentResult, target: Any) -> dict[str, Any]:
    """What an image run measures its tries against: the recipe the host's own baseline
    scored, and the image size the session decodes at."""
    raw = baseline.artifacts.get("recipe.json") or "{}"
    try:
        scored = json.loads(raw)
    except ValueError:
        scored = {}
    base = {k: v for k, v in scored.items() if k not in ("epochs_planned", "epochs_run")}
    meta = json.loads(target.meta_json()) if hasattr(target, "meta_json") else {}
    return {"baseline": base, "default_size": meta.get("image_size")}


def _own_code(best: Experiment | None) -> str | None:
    """The best's cells up to its submission when that submission was the agent's own
    model, the only way back to it. A fit() best travels as its recipe instead, so no
    session spends a fit rebuilding what the harness already holds."""
    if best is None or "model" not in vl.recipe_of(best):
        return None
    from iterate.core.critic import vision_submit_code

    cells = best.candidate.changes.get("cells")
    return vision_submit_code(cells) if isinstance(cells, list) else None


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
    findings: Findings | None = None,
    vision: dict[str, Any] | None = None,
    best: Experiment | None = None,
) -> tuple[Experiment, str | None]:
    """Run one briefed session; returns the experiment and the sha256 of its
    submitted predictions (for later sessions' identical-submission gate)."""
    from iterate.core import codegen
    from iterate.core.coder import lever_executed

    markers = () if vision is not None else lever_markers_for_brief(decision.brief)
    extra: dict[str, Any] = {}
    gate: Callable[[list[Any]], bool] | None = None
    started_from: dict[str, Any] | None = None
    if vision is not None:
        # What the run carried in, for the gate and the prompt, and separately the
        # recipe the kernel's fit() may depart from: an own-model win is the carried
        # best for both, but it is not a recipe, so incumbent.json takes its session's
        # last fit instead.
        carried = vl.recipe_of(best) or vision["baseline"]
        incumbent = vl.fit_recipe_of(best) or vision["baseline"]
        started_from = incumbent
        lever = vl.lever_class(decision.brief) or ""

        def gate(cells: list[Any]) -> bool:
            if lever:
                return vl.moved(lever, cells, carried)
            return bool(vl.moved_levers(cells, carried))

        extra = {
            "starting_files": {codegen.INCUMBENT_JSON: json.dumps(incumbent).encode()},
            "starting_recipe": carried,
            "lever_gate": gate,
            "lever_name": lever,
        }
    coding = coder.run(
        dataset=dataset,
        brief=decision.brief,
        experiment_id=f"iter-{iteration:02d}",
        starting_code=starting_code,
        starting_score=starting_score,
        brief_markers=markers,
        seen_digests=seen_digests,
        **extra,
    )
    cells = [
        {
            "code": c.code,
            "stdout": c.stdout,
            "error": c.error,
            "source": c.source,
            "outputs": c.outputs,
            "thinking": c.thinking,
        }
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
    if vision is not None and gate is not None:
        # The recipe the kernel was handed, kept so the delivered notebook can hand the
        # same one to its own Run All: the kernel's folder is gone by then.
        changes["started_from"] = started_from
        agent_cells = [c for c in cells if c["source"] == "agent"]
        carried = vl.recipe_of(best) or vision["baseline"]
        if (recipe := vl.submitted(agent_cells)) is not None:
            changes["recipe"] = recipe
        changes["levers_moved"] = vl.moved_levers(agent_cells, carried)
        if not gate(agent_cells):
            changes["lever_unmeasured"] = True
    elif markers and not lever_executed(coding.cells, markers, starting_code):
        # The commissioned lever never ran successfully — the score is the carried
        # pipeline's, not the lever's, and the supervisor must not credit it.
        changes["lever_unmeasured"] = True
    citations = credited(findings, decision.brief)
    candidate = Candidate(
        description=decision.title,
        changes=changes,
        rationale=decision.brief,
        # "researcher" only when the brief actually took up a suggestion. A
        # research pass that the supervisor read and ignored leaves the candidate
        # a plain proposer candidate, because claiming otherwise would put a
        # citation on work no paper informed.
        source="researcher" if citations else "proposer",
        citations=citations,
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
