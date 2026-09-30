"""Tests for the supervised agent loop (Supervisor + Coder), with fakes."""

from __future__ import annotations

import pytest

from iterate.core.agent_loop import _winning_code, run_supervised
from iterate.core.coder import Cell, CodingResult
from iterate.core.memory import InMemoryMemory
from iterate.core.supervisor import SupervisorDecision
from iterate.core.terminator import MaxIterations
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

pytestmark = pytest.mark.unit


def _result(score: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="x",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )


class _FakeTarget:
    name = "tabular-model"

    def baseline(self) -> ExperimentResult:
        return _result(0.50)  # the bar to beat

    def run(self, candidate: object) -> ExperimentResult:  # pragma: no cover - unused
        raise NotImplementedError


class _FakeSupervisor:
    def __init__(self, decisions: list[SupervisorDecision]) -> None:
        self._decisions = list(decisions)
        self.seen_history_lens: list[int] = []
        self.seen_carried: list[object] = []

    def decide(
        self, *, data_summary: str, baseline: object, history: list,
        carried_best: object = None,
    ) -> SupervisorDecision:
        self.seen_history_lens.append(len(history))
        self.seen_carried.append(carried_best)
        return self._decisions.pop(0)


class _FakeCoder:
    def __init__(self, result: ExperimentResult, predictions_sha256: str | None = None) -> None:
        self._result = result
        self._digest = predictions_sha256
        self.seen_kwargs: list[dict] = []

    def run(
        self, *, dataset: object, brief: str, experiment_id: str,
        starting_code: str | None = None, starting_score: float | None = None,
        brief_markers: object = None, seen_digests: frozenset | None = None,
    ) -> CodingResult:
        self.seen_kwargs.append(
            {"brief_markers": brief_markers, "seen_digests": seen_digests}
        )
        cells = [
            Cell("# preamble", "loaded", "", None, "preamble"),
            Cell("model.fit(); to_csv('predictions.csv')", "ok", "", None, "agent"),
        ]
        return CodingResult(result=self._result, cells=cells, predictions_sha256=self._digest)


def _loop(supervisor: object, coders: list[_FakeCoder], terminator: object,
          on_experiment=None, summarizer=None, memory=None, controller=None, wall=None):
    it = iter(coders)
    return run_supervised(
        target=_FakeTarget(),  # type: ignore[arg-type]
        dataset=object(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        make_coder=lambda: next(it),  # type: ignore[arg-type,return-value]
        terminator=terminator,  # type: ignore[arg-type]
        memory=memory if memory is not None else InMemoryMemory(),
        data_summary="d",
        summarizer=summarizer,
        on_experiment=on_experiment,
        controller=controller,
        wall=wall,
    )


def test_loop_runs_experiments_and_tracks_best() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    result = _loop(sup, [_FakeCoder(_result(0.60)), _FakeCoder(_result(0.55))], MaxIterations(2))
    assert result.stopped_because == "max_iterations"
    assert len(result.history) == 2
    # each experiment reaches the supervisor exactly once (memory only, no double-count)
    assert sup.seen_history_lens == [0, 1]
    # the loop's carried best is handed to the supervisor for brief grounding
    assert sup.seen_carried[0] is None
    assert sup.seen_carried[1] is result.history[0]
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.60  # the better of the two, beats baseline
    # the session cells are stored on the candidate for the notebook
    assert result.best.candidate.changes["cells"][0]["source"] == "preamble"


class _ExplodingCoder:
    def run(self, **kwargs: object) -> CodingResult:
        raise TimeoutError("backend timed out after retries")


def test_a_crashing_coder_fails_the_iteration_not_the_run() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    result = _loop(
        sup, [_ExplodingCoder(), _FakeCoder(_result(0.60))], MaxIterations(2)  # type: ignore[list-item]
    )
    # iteration 1 exploded mid-experiment; the loop survived and iteration 2 scored
    assert result.stopped_because == "max_iterations"
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.60


class _InterruptingSupervisor:
    """Briefs once, then raises Ctrl-C on the next decision — like a user hitting
    Ctrl-C after the first experiment finished."""

    def __init__(self, first: SupervisorDecision) -> None:
        self._first = first
        self._calls = 0

    def decide(
        self, *, data_summary: str, baseline: object, history: list,
        carried_best: object = None,
    ) -> SupervisorDecision:
        self._calls += 1
        if self._calls == 1:
            return self._first
        raise KeyboardInterrupt


def test_ctrl_c_finalizes_the_run_and_keeps_what_it_earned() -> None:
    mem = InMemoryMemory()
    sup = _InterruptingSupervisor(SupervisorDecision(False, "a", "try a"))
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(5), memory=mem)
    # the interrupt exits cleanly, not as a stack trace
    assert result.stopped_because == "interrupted"
    # the one experiment that finished before Ctrl-C is kept, with its best tracked
    assert len(result.history) == 1
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.60
    # memory is finalized (not left dangling) so the run reads as "interrupted" on disk
    assert mem._runs[result.run_id]["stopped_because"] == "interrupted"


def test_ctrl_c_during_the_very_first_decision_still_finalizes() -> None:
    # Ctrl-C before any experiment finished: empty history, run still finalized.
    mem = InMemoryMemory()
    sup = _InterruptingSupervisor(SupervisorDecision(False, "a", "try a"))
    sup._calls = 1  # force the next decide() to interrupt immediately
    result = _loop(sup, [], MaxIterations(5), memory=mem)
    assert result.stopped_because == "interrupted"
    assert result.history == []
    assert result.best is None
    assert mem._runs[result.run_id]["stopped_because"] == "interrupted"


def test_on_experiment_hook_fires_per_finished_experiment() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    seen: list[tuple[bool, str]] = []

    def hook(*, experiment, baseline, is_best, run_id) -> None:  # type: ignore[no-untyped-def]
        assert baseline.metrics is not None  # the bar is handed along
        seen.append((is_best, run_id))

    _loop(sup, [_FakeCoder(_result(0.60)), _FakeCoder(_result(0.55))], MaxIterations(2), hook)
    # called once per experiment, the moment it finished; only the winner is best
    assert [b for b, _ in seen] == [True, False]
    assert len({rid for _, rid in seen}) == 1  # both carry the same run id


def test_a_failing_on_experiment_hook_does_not_kill_the_run() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )

    def hook(**kwargs) -> None:  # type: ignore[no-untyped-def]
        raise OSError("disk full")  # a deliverable write failing mid-run

    result = _loop(sup, [_FakeCoder(_result(0.60)), _FakeCoder(_result(0.55))], MaxIterations(2), hook)
    assert result.stopped_because == "max_iterations"  # the run survived both failures
    assert len(result.history) == 2


def test_summarizer_digest_is_attached_to_recorded_experiments() -> None:
    from iterate.core.memory import InMemoryMemory
    from iterate.schemas.experiment import ExperimentDigest

    class _FakeSummarizer:
        def __init__(self) -> None:
            self.seen = 0

        def summarize(self, experiment) -> ExperimentDigest:  # type: ignore[no-untyped-def]
            self.seen += 1
            return ExperimentDigest(techniques=["OneHotEncoder"], score=0.60,
                                    takeaway="try target encoding")

    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    summ = _FakeSummarizer()
    mem = InMemoryMemory()
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1),
                   summarizer=summ, memory=mem)
    assert summ.seen == 1  # the finished experiment was summarized once
    assert result.best is not None
    assert result.best.digest is not None
    assert result.best.digest.takeaway == "try target encoding"
    # and it persisted into memory, so the NEXT supervisor would read it
    assert mem.history("tabular-model")[0].digest is not None


def test_a_failing_summarizer_does_not_kill_the_run() -> None:
    class _BoomSummarizer:
        def summarize(self, experiment) -> object:  # type: ignore[no-untyped-def]
            raise RuntimeError("summarizer exploded")

    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), summarizer=_BoomSummarizer())
    assert result.best is not None  # run survived; experiment recorded without a digest
    assert result.best.digest is None


def test_supervisor_stop_ends_the_loop_immediately() -> None:
    sup = _FakeSupervisor([SupervisorDecision(True, "", "")])
    result = _loop(sup, [], MaxIterations(5))
    assert result.stopped_because == "supervisor"
    assert result.history == []
    assert result.best is None


def _exp_with_cells(cells: list[Cell]) -> Experiment:
    return Experiment(
        candidate=Candidate(
            description="d",
            changes={"cells": [c.__dict__ for c in cells]},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=_result(0.6),
    )


def test_winning_code_concatenates_successful_staged_cells() -> None:
    # a staged session: prepare -> (errored attempt) -> model -> submit.
    # carry-forward keeps the successful cells in order and drops the errored one.
    best = _exp_with_cells(
        [
            Cell("# preamble", "", "", None, "preamble"),
            Cell("X_tr = prepare(X_train)", "shape (100, 8)", "", None, "agent"),
            Cell("model.fit(broken)", "", "", "NameError: broken", "agent"),
            Cell("print(validation_f1)", "0.61", "", None, "agent"),
            Cell("write_predictions()", "wrote 100", "", None, "agent"),
        ]
    )
    code = _winning_code(best)
    assert code is not None
    assert "X_tr = prepare(X_train)" in code
    assert "write_predictions()" in code
    assert "model.fit(broken)" not in code  # the errored cell is dropped
    assert "# preamble" not in code  # only agent cells carry forward
    # order preserved: prepare before submit
    assert code.index("X_tr = prepare") < code.index("write_predictions")


def test_winning_code_is_none_without_a_best() -> None:
    assert _winning_code(None) is None


def test_loop_hands_every_prior_digest_and_brief_markers_to_the_next_session() -> None:
    # EVERY prior submission's hash + the brief's lever markers reach the next
    # coder, powering the in-session no-op gates — a run wasted 4 iterations on
    # sibling duplicates when only the best's digest was checked.
    sup = _FakeSupervisor([
        SupervisorDecision(False, "a", "try a"),
        SupervisorDecision(False, "b", "next: imbalance-or-threshold: set class_weight."),
        SupervisorDecision(False, "c", "try c"),
    ])
    coder1 = _FakeCoder(_result(0.60), predictions_sha256="digest-one")
    coder2 = _FakeCoder(_result(0.55), predictions_sha256="digest-two")  # NOT the best
    coder3 = _FakeCoder(_result(0.50))
    _loop(sup, [coder1, coder2, coder3], MaxIterations(3))
    # iteration 1: nothing submitted yet; brief names no lever class
    assert coder1.seen_kwargs[0] == {"brief_markers": (), "seen_digests": frozenset()}
    # iteration 2: the first digest is handed over; the named class maps to markers
    kwargs = coder2.seen_kwargs[0]
    assert kwargs["seen_digests"] == frozenset({"digest-one"})
    assert "class_weight" in kwargs["brief_markers"]
    # iteration 3: BOTH prior digests, including the non-best sibling's
    assert coder3.seen_kwargs[0]["seen_digests"] == frozenset({"digest-one", "digest-two"})


def test_a_duplicate_submission_is_stamped_on_the_recorded_experiment() -> None:
    # the loop knows the digest matched an earlier submission; the stamp is what the
    # supervisor's history renders as "duplicate — no new information".
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    coder1 = _FakeCoder(_result(0.60), predictions_sha256="same-bytes")
    coder2 = _FakeCoder(_result(0.60), predictions_sha256="same-bytes")
    result = _loop(sup, [coder1, coder2], MaxIterations(2))
    assert "duplicate_submission" not in result.history[0].candidate.changes
    assert result.history[1].candidate.changes["duplicate_submission"] is True


def test_an_unexecuted_commissioned_lever_is_stamped_on_the_experiment() -> None:
    # run 13 i3/i6: the brief commissioned a lever that never ran successfully; the
    # recorded score is the carried pipeline's and must be labeled as such.
    class _SkippingCoder:
        def run(self, *, dataset, brief, experiment_id, starting_code=None, starting_score=None,
                brief_markers=None, seen_digests=None):
            cells = [
                Cell("# preamble", "loaded", "", None, "preamble"),
                Cell("model = HGB().fit(Xa, ya)  # rebuild only", "ok", "", None, "agent"),
                Cell("to_csv('predictions.csv')", "wrote", "", None, "agent"),
            ]
            return CodingResult(result=_result(0.60), cells=cells)

    sup = _FakeSupervisor([
        SupervisorDecision(False, "w", "next: imbalance-or-threshold: set class_weight balanced.")
    ])
    result = _loop(sup, [_SkippingCoder()], MaxIterations(1))  # type: ignore[list-item]
    assert result.history[0].candidate.changes["lever_unmeasured"] is True


def test_an_executed_lever_is_not_stamped() -> None:
    class _ComplyingCoder:
        def run(self, *, dataset, brief, experiment_id, starting_code=None, starting_score=None,
                brief_markers=None, seen_digests=None):
            cells = [
                Cell("# preamble", "loaded", "", None, "preamble"),
                Cell("model = HGB(class_weight='balanced').fit(Xa, ya)", "ok", "", None, "agent"),
                Cell("to_csv('predictions.csv')", "wrote", "", None, "agent"),
            ]
            return CodingResult(result=_result(0.60), cells=cells)

    sup = _FakeSupervisor([
        SupervisorDecision(False, "w", "next: imbalance-or-threshold: set class_weight balanced.")
    ])
    result = _loop(sup, [_ComplyingCoder()], MaxIterations(1))  # type: ignore[list-item]
    assert "lever_unmeasured" not in result.history[0].candidate.changes


def test_fabricated_helped_claims_about_an_unexecuted_lever_are_dropped() -> None:
    # run 16 i4: class_weight never reached a constructor, yet the digest claimed an
    # "implied weighting" win — which re-queued the dead lever via the knowledge
    # channel. The machine verdict overrides the narrative.
    from iterate.schemas.experiment import ExperimentDigest

    class _FabricatingSummarizer:
        def summarize(self, experiment):  # type: ignore[no-untyped-def]
            return ExperimentDigest(
                techniques=["HistGradientBoosting"], score=0.6254,
                what_helped=["implied class_weight weighting improved from 0.5568",
                             "target encoding of the Contract column: small gain"],
                what_hurt=[], data_insights=[], takeaway="t",
            )

    class _SkippingCoder:
        def run(self, *, dataset, brief, experiment_id, starting_code=None, starting_score=None,
                brief_markers=None, seen_digests=None):
            cells = [
                Cell("# preamble", "loaded", "", None, "preamble"),
                Cell("model = HGB().fit(Xa, ya)  # never applies the lever", "ok", "", None, "agent"),
                Cell("to_csv('predictions.csv')", "wrote", "", None, "agent"),
            ]
            return CodingResult(result=_result(0.6254), cells=cells)

    sup = _FakeSupervisor([
        SupervisorDecision(False, "w", "next: imbalance-or-threshold: set class_weight balanced.")
    ])
    result = _loop(sup, [_SkippingCoder()], MaxIterations(1), summarizer=_FabricatingSummarizer())  # type: ignore[list-item]
    exp = result.history[0]
    assert exp.candidate.changes["lever_unmeasured"] is True
    assert exp.digest is not None
    # the fabricated lever claim is gone; the unrelated claim survives
    assert exp.digest.what_helped == ["target encoding of the Contract column: small gain"]


def test_a_duplicate_submissions_helped_claims_are_stripped() -> None:
    # run 17 i9: a byte-dup's Findings claimed a settled optimization as the
    # session's own win. Nothing raised a score on an identical submission.
    from iterate.schemas.experiment import ExperimentDigest

    class _MisattributingSummarizer:
        def summarize(self, experiment):  # type: ignore[no-untyped-def]
            return ExperimentDigest(
                techniques=["HistGradientBoosting"], score=0.6311,
                what_helped=["grid search: 0.6387 -> 0.6625"],
                what_hurt=["min_samples_leaf tuning: val 0.6534, below best"],
                data_insights=[], takeaway="t",
            )

    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    coder1 = _FakeCoder(_result(0.6311), predictions_sha256="same-bytes")
    coder2 = _FakeCoder(_result(0.6311), predictions_sha256="same-bytes")
    result = _loop(sup, [coder1, coder2], MaxIterations(2), summarizer=_MisattributingSummarizer())
    dup = result.history[1]
    assert dup.candidate.changes["duplicate_submission"] is True
    assert dup.digest is not None
    assert dup.digest.what_helped == []  # the misattributed win is gone
    assert dup.digest.what_hurt == ["min_samples_leaf tuning: val 0.6534, below best"]  # the loss survives


def test_a_floor_banked_experiment_records_the_fallback_code_in_its_fingerprint() -> None:
    # the recorded changes["code"] feeds the lever ledger, the technique scoreboard,
    # and the grounded brief — when the score came from the harness floor, the
    # score-bearing fallback pipeline must be in it, not the dead-end cells alone.
    class _RescuedCoder:
        def run(self, *, dataset, brief, experiment_id, starting_code=None, starting_score=None,
                brief_markers=None, seen_digests=None):
            cells = [
                Cell("# preamble", "loaded", "", None, "preamble"),
                Cell("GridSearchCV(...)  # never submitted", "", "", "timeout", "agent"),
                Cell("hgb_floor_submit()", "banked", "", None, "fallback"),
            ]
            return CodingResult(result=_result(0.58), cells=cells)

    sup = _FakeSupervisor([SupervisorDecision(False, "tuning", "try tuning")])
    result = _loop(sup, [_RescuedCoder()], MaxIterations(1))  # type: ignore[list-item]
    assert result.best is not None
    code = result.best.candidate.changes["code"]
    assert "hgb_floor_submit()" in code  # the pipeline that actually scored
    assert "GridSearchCV" in code  # what the agent tried stays visible too


def test_winning_code_keeps_a_fallback_floor_submit() -> None:
    # a session whose submission came from the harness fallback: the fallback cell IS
    # the pipeline that produced the recorded score, so it carries forward too.
    best = _exp_with_cells(
        [
            Cell("# preamble", "", "", None, "preamble"),
            Cell("X_tr = prepare(X_train)", "shape (100, 8)", "", None, "agent"),
            Cell("model.fit(broken)", "", "", "NameError: broken", "agent"),
            Cell("hgb_floor_submit()", "banked", "", None, "fallback"),
        ]
    )
    code = _winning_code(best)
    assert code is not None
    assert "hgb_floor_submit()" in code
    assert "model.fit(broken)" not in code  # errored agent cell still dropped


# ─── Interactive controller wiring (v0.3) ────────────────────────────────────


class _GuidedSupervisor(_FakeSupervisor):
    """A fake that also records the interactive kwargs the loop passes."""

    def __init__(self, decisions: list[SupervisorDecision]) -> None:
        super().__init__(decisions)
        self.seen_guidance: list[str | None] = []
        self.seen_rules: list[tuple] = []

    def decide(
        self, *, data_summary: str, baseline: object, history: list,
        carried_best: object = None, user_guidance: str | None = None,
        standing_rules: tuple = (),
    ) -> SupervisorDecision:
        self.seen_guidance.append(user_guidance)
        self.seen_rules.append(tuple(standing_rules))
        return super().decide(
            data_summary=data_summary, baseline=baseline, history=history,
            carried_best=carried_best,
        )


def test_a_typed_message_becomes_guidance_and_is_stamped() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    ctrl.submit_line("prefer smaller models")
    sup = _GuidedSupervisor([SupervisorDecision(False, "a", "try a")])
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl)
    # no route_message on the fake supervisor -> the interpreter defaults to a steer
    assert sup.seen_guidance == ["prefer smaller models"]
    assert result.history[0].candidate.changes["user_guidance"] == "prefer smaller models"


def test_stop_before_planning_ends_the_run_and_finalizes_memory() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    ctrl.submit_line("stop")
    mem = InMemoryMemory()
    sup = _FakeSupervisor([])  # must never be consulted
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(3), memory=mem, controller=ctrl)
    assert result.stopped_because == "stopped-by-user"
    assert result.history == []
    assert result.run_id is not None
    assert mem._runs[result.run_id]["stopped_because"] == "stopped-by-user"


def test_paused_time_is_excluded_from_the_deadline_clock() -> None:
    from iterate.core.interactive import RunController

    class _SpyTerminator:
        def __init__(self) -> None:
            self.elapsed: list[float] = []

        def update_and_check(self, state) -> str:
            self.elapsed.append(state.elapsed_seconds)
            return "spied"

    ctrl = RunController()
    ctrl.paused_seconds_total = 3600.0  # pretend an hour of pause already accrued
    spy = _SpyTerminator()
    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], spy, controller=ctrl)
    # real elapsed is milliseconds; the paused credit dominates and the clamp holds
    assert spy.elapsed == [0.0]


class _QASupervisor(_FakeSupervisor):
    """Routes every line as a question and records what context answer() saw."""

    def __init__(self, decisions: list[SupervisorDecision]) -> None:
        super().__init__(decisions)
        self.answered_over: list[int] = []
        self.seen_summaries: list[str] = []
        self.seen_live: list[object] = []

    def route_message(self, text: str, *, live_session: bool) -> str:
        return "question"

    def answer(
        self, question: str, *, history: list, baseline: object,
        data_summary: str = "", live_session: object = None,
    ) -> str:
        self.answered_over.append(len(history))
        self.seen_summaries.append(data_summary)
        self.seen_live.append(live_session)
        return "answered"


def test_questions_are_answered_from_the_current_run_only() -> None:
    from iterate.core.interactive import RunController

    # An earlier run on the same target sits in memory; its experiments must be
    # invisible to Q&A about THIS run.
    mem = InMemoryMemory()
    old_run = mem.start_run("tabular-model", _result(0.50))
    old_exp = Experiment(
        candidate=Candidate(description="old", changes={"code": "x"}, rationale="r"),
        target="tabular-model", hypothesis="h", status="completed", iteration=3,
        result=_result(0.55),
    )
    mem.record(old_run, old_exp)
    mem.finish_run(old_run, "max_iterations")

    ctrl = RunController()
    ctrl.submit_line("why did iteration 3 fail?")
    sup = _QASupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), memory=mem, controller=ctrl)
    assert sup.answered_over == [0]  # the current run had no experiments yet


def test_stop_during_an_iteration_wins_the_stop_reason() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()

    def hook(**kwargs) -> None:  # type: ignore[no-untyped-def]
        ctrl.submit_line("stop")  # the user stops while the iteration finishes

    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), hook, controller=ctrl)
    # MaxIterations(1) would also fire here; the user's stop outranks its label
    assert result.stopped_because == "stopped-by-user"


def test_two_messages_in_one_window_both_reach_the_brief() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    first = "use a much smaller learning rate for the gradient boosting model " * 3
    ctrl.submit_line(first)
    ctrl.submit_line("and never touch the test split")
    sup = _GuidedSupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl)
    (guidance,) = sup.seen_guidance
    assert guidance is not None
    assert "never touch the test split" in guidance


def test_brief_and_score_events_are_emitted_for_the_ui() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    events: list[tuple[str, dict]] = []
    ctrl.on_event = lambda kind, payload: events.append((kind, payload))
    sup = _FakeSupervisor([SupervisorDecision(False, "class weights", "next: balance")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl)
    kinds = [k for k, _ in events]
    assert kinds == ["brief", "score"]
    brief = dict(events[0][1])
    assert brief["iteration"] == 1
    assert brief["title"] == "class weights"
    score = dict(events[1][1])
    assert score["score"] == 0.60
    assert score["is_best"] is True


def test_questions_get_the_dataset_profile_even_with_no_experiments() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    ctrl.submit_line("how many numeric and categorical columns?")
    sup = _QASupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl)
    assert sup.answered_over == [0]  # no experiments yet, and that is fine now:
    assert sup.seen_summaries == ["d"]  # the profile still grounds the answer


def test_snapshot_reflects_finished_work_for_the_hard_quit() -> None:
    from iterate.core.interactive import RunController

    ctrl = RunController()
    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl)
    assert callable(ctrl.snapshot)
    snap = ctrl.snapshot()
    assert snap.stopped_because == "stopped-by-user"
    assert len(snap.history) == 1
    assert snap.best is not None
    assert snap.best.result.metrics.primary_value == 0.60


# ─── the free inspect step (carry-in 5) ───────────────────────────────────────


class _InspectingSupervisor(_FakeSupervisor):
    """Records what reached each decide(), and asks to look before deciding."""

    def __init__(self, decisions: list[SupervisorDecision]) -> None:
        super().__init__(decisions)
        self.seen_inspection: list[str] = []

    def decide(  # type: ignore[override]
        self, *, data_summary: str, baseline: object, history: list,
        carried_best: object = None, inspection: str = "", **kwargs: object,
    ) -> SupervisorDecision:
        self.seen_inspection.append(inspection)
        return super().decide(
            data_summary=data_summary, baseline=baseline, history=history,
            carried_best=carried_best,
        )


class _InspectingCoder(_FakeCoder):
    """One instance serves every session in these tests, so the counts are totals.

    An inspection takes its own `make_coder()` — the coder owns a kernel, and the
    inspect session starts and closes one — so a per-iteration fake would run dry
    exactly when the step is exercised.
    """

    def __init__(self, result: ExperimentResult, findings: str = "- rows 1000") -> None:
        super().__init__(result)
        self._findings = findings
        self.inspections = 0

    def inspect(self, *, dataset: object, brief: str = "", experiment_id: str = "i") -> str:
        self.inspections += 1
        return self._findings


def _loop_shared(supervisor: object, coder: object, terminator: object):
    from iterate.core.agent_loop import run_supervised

    return run_supervised(
        target=_FakeTarget(),  # type: ignore[arg-type]
        dataset=object(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        make_coder=lambda: coder,  # type: ignore[arg-type,return-value]
        terminator=terminator,  # type: ignore[arg-type]
        memory=InMemoryMemory(),
        data_summary="d",
    )


def test_an_inspection_reaches_the_next_decision() -> None:
    sup = _InspectingSupervisor(
        [
            SupervisorDecision(False, "a", "try a", want_inspect=True),
            SupervisorDecision(False, "b", "try b"),
        ]
    )
    coder = _InspectingCoder(_result(0.60))

    _loop_shared(sup, coder, MaxIterations(2))

    assert sup.seen_inspection[0] == ""  # nothing asked for yet
    assert "rows 1000" in sup.seen_inspection[1]
    assert coder.inspections == 1


def test_an_inspection_costs_neither_an_iteration_nor_patience() -> None:
    """The whole point of the step: exploration stops costing a scored iteration.
    Two scored experiments run under MaxIterations(2) with an inspection between
    them, and the run still ends on max_iterations rather than one short."""
    from iterate.core.terminator import Composite, Patience

    sup = _InspectingSupervisor(
        [
            SupervisorDecision(False, "a", "try a", want_inspect=True),
            SupervisorDecision(False, "b", "try b"),
        ]
    )
    coder = _InspectingCoder(_result(0.60))

    result = _loop_shared(sup, coder, Composite(MaxIterations(2), Patience(2)))

    assert result.stopped_because == "max_iterations"
    assert len(result.history) == 2  # the inspection is not one of them
    assert coder.inspections == 1


def test_the_inspect_budget_is_capped() -> None:
    """A supervisor that keeps asking cannot spend the run on looking."""
    sup = _InspectingSupervisor(
        [SupervisorDecision(False, f"d{i}", "b", want_inspect=True) for i in range(4)]
    )
    coder = _InspectingCoder(_result(0.60))

    _loop_shared(sup, coder, MaxIterations(4))

    assert coder.inspections == 2  # max_inspect_calls


def test_an_empty_inspection_is_not_carried_into_the_prompt() -> None:
    """A step that found nothing must not add an empty block to a planning prompt
    the June revert showed is sensitive to density."""
    sup = _InspectingSupervisor(
        [
            SupervisorDecision(False, "a", "try a", want_inspect=True),
            SupervisorDecision(False, "b", "try b"),
        ]
    )
    coder = _InspectingCoder(_result(0.60), findings="")

    _loop_shared(sup, coder, MaxIterations(2))

    assert sup.seen_inspection == ["", ""]
    assert coder.inspections == 1  # it ran; it just found nothing worth carrying


def test_a_table_experiment_records_no_recipe_it_started_from() -> None:
    """`started_from` is the image family's: the other two hand their session no recipe,
    and their recorded experiments must not grow a key for one."""
    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1))
    assert "started_from" not in result.history[0].candidate.changes


# ─── the serving budget is a wall (Sprint 5 Day 3) ──────────────────────────


def _linear_wall(budget: float | None):
    """A wall that reads every experiment as a linear pipeline on the shipped table."""
    from iterate.core import serving

    def linear(_experiment):
        return serving.facts_from_code(None, "LogisticRegression()", n_features=3)

    prices = serving.load_prices()
    price = serving.profile(linear(None), 1000, prices).chosen.usd_per_month
    wall = serving.Wall(requests_per_hour=1000, prices=prices, budget=budget, facts_of=linear)
    return wall, price


def test_an_experiment_over_the_serving_budget_is_stamped_and_never_the_best() -> None:
    wall, price = _linear_wall(None)
    wall.budget = price - 1
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    result = _loop(
        sup, [_FakeCoder(_result(0.60)), _FakeCoder(_result(0.70))], MaxIterations(2), wall=wall
    )
    assert result.best is None  # both beat the baseline, neither can be served
    assert [e.candidate.changes["over_budget"] for e in result.history] == [round(price, 2)] * 2
    # Named as the run names them, at the budget they were judged under.
    assert [(r.what, r.usd_per_month, r.budget, r.kind) for r in wall.refused] == [
        ("a", price, price - 1, "trained"),
        ("b", price, price - 1, "trained"),
    ]


def test_an_experiment_within_the_budget_is_not_stamped_and_can_be_the_best() -> None:
    wall, price = _linear_wall(None)
    wall.budget = price + 1
    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    result = _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall)
    assert result.best is not None
    assert "over_budget" not in result.best.candidate.changes
    assert wall.refused == []


def test_a_typed_budget_moves_the_wall_and_is_never_a_steer() -> None:
    from iterate.core.interactive import RunController

    said: list[str] = []
    ctrl = RunController(reply=said.append)
    wall, _ = _linear_wall(None)
    ctrl.submit_line("budget $1,200 a month")
    sup = _GuidedSupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl, wall=wall)
    assert wall.budget == 1200.0
    assert sup.seen_guidance == [None]
    assert (
        "serving budget is now $1,200 a month at 1,000 requests an hour; it holds from the "
        "next brief on, and nothing already recorded is re-judged"
    ) in said


def test_no_budget_lifts_the_wall_and_a_zero_leaves_it_where_it_was() -> None:
    from iterate.core.interactive import RunController

    said: list[str] = []
    ctrl = RunController(reply=said.append)
    wall, _ = _linear_wall(50.0)
    ctrl.submit_line("budget 0")
    ctrl.submit_line("no budget")
    sup = _GuidedSupervisor([SupervisorDecision(False, "a", "try a")])
    _loop(sup, [_FakeCoder(_result(0.60))], MaxIterations(1), controller=ctrl, wall=wall)
    assert wall.budget is None
    assert sup.seen_guidance == [None]
    assert (
        'a budget is dollars a month above zero, e.g. "budget $80"; the wall is unchanged'
    ) in said
    assert "the serving budget is lifted; from here on nothing is refused for its price" in said


@pytest.mark.parametrize(
    ("line", "budget"),
    [
        ("budget $80", 80.0),
        ("Budget: 80", 80.0),
        ("serving budget is $120/month", 120.0),
        ("budget 1,500 dollars a month.", 1500.0),
        ("budget to 42.5", 42.5),
    ],
)
def test_the_shapes_a_typed_budget_takes(line: str, budget: float) -> None:
    from iterate.core.agent_loop import _move_the_wall

    wall, _ = _linear_wall(None)
    assert _move_the_wall(line, wall) is not None
    assert wall.budget == budget


@pytest.mark.parametrize(
    "line",
    [
        "the budget for epochs is 6",
        "prefer smaller models",
        "budget",
        "no budget for that",
        "budget 1,,5",
    ],
)
def test_a_line_that_is_not_a_budget_leaves_the_wall_and_goes_to_the_supervisor(line: str) -> None:
    from iterate.core.agent_loop import _move_the_wall

    wall, _ = _linear_wall(50.0)
    assert _move_the_wall(line, wall) is None
    assert wall.budget == 50.0


# ─── the Researcher and the Pricer go back and forth (v0.7 Day 4) ────────────


_OVER = SupervisorDecision(
    stop=True, title="over the serving budget", brief="", stopped_because="over_budget"
)
_OWN = SupervisorDecision(False, "efficientnet_b0", "next: own-model: write torch code for efficientnet_b0")


class _WalledSupervisor(_FakeSupervisor):
    """Plays the real Supervisor's side of the wall: its over-budget stop leaves the
    network the split refused on the wall, and every decide records the extras it saw."""

    def __init__(
        self, decisions: list[SupervisorDecision], wall: object, network: str = "convnext_tiny"
    ) -> None:
        super().__init__(decisions)
        self._wall = wall
        self._network = network
        self.seen_extra: list[dict[str, object]] = []

    def decide(  # type: ignore[override]
        self, *, data_summary: str, baseline: object, history: list,
        carried_best: object = None, **extra: object,
    ) -> SupervisorDecision:
        self.seen_extra.append(dict(extra))
        decision = super().decide(
            data_summary=data_summary, baseline=baseline, history=history,
            carried_best=carried_best,
        )
        if decision.stopped_because == "over_budget":
            self._wall.record_refusal(  # type: ignore[attr-defined]
                f"keep the recipe and swap the backbone to {self._network}",
                37.0,
                "off the line",
                network=self._network,
            )
        return decision


class _RoundTripResearcher:
    """The iteration's own pass finds nothing; each budget round answers from a script,
    an empty string being a round that names nothing."""

    def __init__(self, rounds: list[str]) -> None:
        self._rounds = list(rounds)
        self.calls: list[dict[str, object]] = []

    def research(
        self, *, profile: str, tried: list[str] | tuple[str, ...] = (),
        ruled_out: list[str] | tuple[str, ...] = (), round: int = 1, **kwargs: object,
    ) -> object:
        from iterate.core.researcher import Findings, Suggestion

        self.calls.append(
            {"profile": profile, "tried": list(tried), "ruled_out": list(ruled_out), "round": round}
        )
        if round == 1 or not self._rounds:
            return Findings()
        technique = self._rounds.pop(0)
        if not technique:
            return Findings(papers_seen=8)
        return Findings(
            suggestions=[Suggestion(technique, "small and strong on satellite tiles", "doi:10.1/x")],
            papers_seen=8,
        )


def _walled_loop(
    supervisor: object, coders: list[_FakeCoder], terminator: object, *, wall: object,
    researcher: object = None, controller: object = None, memory: object = None,
    max_budget_rounds: int = 2, max_research_calls: int = 3,
):
    it = iter(coders)
    return run_supervised(
        target=_FakeTarget(),  # type: ignore[arg-type]
        dataset=object(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        make_coder=lambda: next(it),  # type: ignore[arg-type,return-value]
        terminator=terminator,  # type: ignore[arg-type]
        memory=memory if memory is not None else InMemoryMemory(),  # type: ignore[arg-type]
        data_summary="d",
        researcher=researcher,  # type: ignore[arg-type]
        controller=controller,  # type: ignore[arg-type]
        wall=wall,  # type: ignore[arg-type]
        max_budget_rounds=max_budget_rounds,
        max_research_calls=max_research_calls,
    )


def _answering(*lines: str):
    """A controller whose user types ``lines`` the moment the run asks for a budget."""
    from iterate.core.interactive import RunController

    said: list[str] = []
    ctrl = RunController()

    def reply(text: str) -> None:
        said.append(text)
        if text.startswith("over the serving budget"):
            for line in lines:
                ctrl.submit_line(line)

    ctrl.bind_reply(reply)
    return ctrl, said


def _asks(said: list[str]) -> list[str]:
    return [s for s in said if s.startswith("over the serving budget")]


def test_a_round_that_names_a_new_model_re_decides_the_same_iteration() -> None:
    from iterate.core.terminator import Composite, Patience

    wall, _ = _linear_wall(20.0)
    mem = InMemoryMemory()
    sup = _WalledSupervisor([_OVER, _OWN], wall)
    researcher = _RoundTripResearcher(["fine-tune efficientnet_b0 with pretrained weights"])
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], Composite(MaxIterations(1), Patience(1)),
        wall=wall, researcher=researcher, memory=mem,
    )
    assert result.stopped_because == "max_iterations"
    (experiment,) = result.history
    assert experiment.iteration == 1
    assert sup.seen_history_lens == [0, 0]  # decided twice inside iteration 1
    assert experiment.candidate.changes["research_rounds"] == 1
    assert [c["round"] for c in researcher.calls] == [1, 2]
    assert researcher.calls[1]["ruled_out"] == ["convnext_tiny"]
    assert "$" not in repr(researcher.calls)
    assert "efficientnet_b0" in str(sup.seen_extra[1]["research"])
    assert sup.seen_extra[1]["known_findings"] == sup.seen_extra[1]["research"]
    assert mem.proposer_failures("tabular-model") == []


def test_a_scripted_run_stops_over_budget_when_the_researcher_runs_dry() -> None:
    wall, _ = _linear_wall(20.0)
    mem = InMemoryMemory()
    sup = _WalledSupervisor([_OVER], wall)
    researcher = _RoundTripResearcher(["", ""])
    result = _walled_loop(
        sup, [], MaxIterations(5), wall=wall, researcher=researcher, memory=mem
    )
    assert result.stopped_because == "over_budget"
    assert result.history == []
    assert [c["round"] for c in researcher.calls] == [1, 2, 3]
    assert sup.seen_history_lens == [0]  # a dry round never re-decides
    assert mem._runs[result.run_id]["stopped_because"] == "over_budget"
    assert mem.proposer_failures("tabular-model") == []


def test_a_round_naming_only_a_refused_network_is_dry() -> None:
    wall, _ = _linear_wall(20.0)
    sup = _WalledSupervisor([_OVER], wall, network="efficientnet_b0")
    researcher = _RoundTripResearcher(["efficientnet_b0", "efficientnet_b0 again"])
    result = _walled_loop(sup, [], MaxIterations(5), wall=wall, researcher=researcher)
    assert result.stopped_because == "over_budget"
    assert [c["ruled_out"] for c in researcher.calls[1:]] == [["efficientnet_b0"]] * 2
    assert sup.seen_history_lens == [0]


def test_the_budget_rounds_are_capped_apart_from_the_research_calls() -> None:
    wall, _ = _linear_wall(20.0)
    researcher = _RoundTripResearcher(["", "", ""])
    _walled_loop(
        _WalledSupervisor([_OVER], wall), [], MaxIterations(5), wall=wall,
        researcher=researcher, max_research_calls=1,
    )
    assert [c["round"] for c in researcher.calls] == [1, 2, 3]  # the one pass, then two rounds

    researcher = _RoundTripResearcher(["", "", ""])
    _walled_loop(
        _WalledSupervisor([_OVER], wall), [], MaxIterations(5), wall=wall,
        researcher=researcher, max_budget_rounds=1,
    )
    assert [c["round"] for c in researcher.calls] == [1, 2]


def test_the_models_own_stop_never_starts_a_round_trip() -> None:
    wall, _ = _linear_wall(20.0)
    researcher = _RoundTripResearcher(["efficientnet_b0"])
    result = _walled_loop(
        _WalledSupervisor([SupervisorDecision(True, "", "")], wall), [], MaxIterations(5),
        wall=wall, researcher=researcher,
    )
    assert result.stopped_because == "supervisor"
    assert [c["round"] for c in researcher.calls] == [1]


@pytest.mark.parametrize(
    ("line", "budget"),
    [("budget $80", 80.0), ("no budget", None), ("budget $20", 20.0)],
)
def test_a_dry_interactive_run_waits_for_the_user_to_move_the_wall(
    line: str, budget: float | None
) -> None:
    wall, _ = _linear_wall(20.0)
    ctrl, said = _answering(line)
    mem = InMemoryMemory()
    sup = _WalledSupervisor([_OVER, SupervisorDecision(False, "a", "try a")], wall)
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall, controller=ctrl,
        memory=mem,
    )
    assert len(_asks(said)) == 1
    assert wall.budget == budget
    assert wall.moves == 1  # the same number still moves it, and ends the wait
    assert result.stopped_because == "max_iterations"
    assert [e.iteration for e in result.history] == [1]
    assert sup.seen_history_lens == [0, 0]
    assert "research_rounds" not in result.history[0].candidate.changes
    assert mem.proposer_failures("tabular-model") == []


def test_stop_typed_at_the_budget_ask_ends_the_run_as_the_users() -> None:
    wall, _ = _linear_wall(20.0)
    ctrl, said = _answering("stop")
    mem = InMemoryMemory()
    result = _walled_loop(
        _WalledSupervisor([_OVER], wall), [], MaxIterations(3), wall=wall, controller=ctrl,
        memory=mem,
    )
    assert len(_asks(said)) == 1
    assert result.stopped_because == "stopped-by-user"
    assert result.history == []
    assert wall.budget == 20.0
    assert mem._runs[result.run_id]["stopped_because"] == "stopped-by-user"


def test_the_wait_for_a_budget_never_burns_the_deadline() -> None:
    import threading

    from iterate.core.interactive import RunController
    from iterate.core.terminator import Composite, Deadline

    wall, _ = _linear_wall(20.0)
    ctrl = RunController()

    def reply(text: str) -> None:
        if text.startswith("over the serving budget"):
            threading.Timer(0.5, lambda: ctrl.submit_line("budget $80")).start()

    ctrl.bind_reply(reply)
    sup = _WalledSupervisor([_OVER, SupervisorDecision(False, "a", "try a")], wall)
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], Composite(Deadline(0.3), MaxIterations(1)),
        wall=wall, controller=ctrl,
    )
    assert ctrl.paused_seconds_total >= 0.4
    assert result.stopped_because == "max_iterations"  # "deadline" had the wait counted


def test_what_the_user_types_during_the_wait_reaches_the_brief_it_ends_in() -> None:
    wall, _ = _linear_wall(20.0)
    ctrl, _ = _answering("prefer a small network", "budget $80")
    sup = _WalledSupervisor([_OVER, SupervisorDecision(False, "a", "try a")], wall)
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall, controller=ctrl
    )
    assert "user_guidance" not in sup.seen_extra[0]
    assert sup.seen_extra[1]["user_guidance"] == "prefer a small network"
    assert result.history[0].candidate.changes["user_guidance"] == "prefer a small network"


def test_the_budget_typed_at_the_ask_is_the_one_the_next_experiment_is_held_to() -> None:
    wall, price = _linear_wall(20.0)
    ctrl, _ = _answering("budget $5")  # under the linear pipeline's price
    sup = _WalledSupervisor([_OVER, SupervisorDecision(False, "a", "try a")], wall)
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall, controller=ctrl
    )
    assert result.history[0].candidate.changes["over_budget"] == round(price, 2)
    assert result.best is None


def test_dry_rounds_come_before_the_wait_and_start_again_once_the_wall_moves() -> None:
    wall, _ = _linear_wall(20.0)
    ctrl, said = _answering("budget $25")
    sup = _WalledSupervisor([_OVER, _OVER, SupervisorDecision(False, "a", "try a")], wall)
    researcher = _RoundTripResearcher(["", "", "", "fine-tune efficientnet_b0"])
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall, controller=ctrl,
        researcher=researcher,
    )
    (ask,) = _asks(said)
    assert "The Researcher was asked twice more, and nothing it named fits." in ask
    # two dry rounds under $20; then, under $25, a dry round and one that names a model
    assert [c["round"] for c in researcher.calls] == [1, 2, 3, 2, 3]
    assert result.history[0].candidate.changes["research_rounds"] == 2
    assert sup.seen_history_lens == [0, 0, 0]


def test_the_ask_lists_what_this_budget_refused_cheapest_first_in_one_plain_paragraph() -> None:
    from iterate.core import serving
    from iterate.core.agent_loop import _over_budget_ask

    wall = serving.Wall(requests_per_hour=400_000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("keep the recipe and swap the backbone to convnext_tiny", 37.0, "off the line")
    wall.record_refusal("keep the best and train at 128 px", 24.53, "off the line")
    wall.record_refusal("an older refusal", 12.0, "trained", budget=10.0)
    ask = _over_budget_ask(wall, 2)
    assert ask == (
        "over the serving budget ($20 a month at 400,000 requests an hour): keep the best "
        "and train at 128 px ($25); keep the recipe and swap the backbone to convnext_tiny "
        "($37). The Researcher was asked twice more, and nothing it named fits. Type "
        '"budget $N" to raise it, "no budget" to lift it, or "stop" to end the run; the '
        "clocks are suspended while this waits."
    )
    assert "\n" not in ask
    assert "[" not in ask


def test_the_ask_caps_its_list_and_says_when_nothing_was_looked_for() -> None:
    from iterate.core import serving
    from iterate.core.agent_loop import _over_budget_ask

    wall = serving.Wall(requests_per_hour=1_000, prices=serving.load_prices(), budget=36.8)
    for i in range(7):
        wall.record_refusal(f"entry {i}", 40.0 + i, "off the line")
    ask = _over_budget_ask(wall, 0)
    assert ask.startswith("over the serving budget ($36.80 a month at 1,000 requests an hour)")
    assert "entry 4 ($44); and 2 more." in ask
    assert "entry 5" not in ask
    assert "No other models were looked for." in ask


def test_every_typed_budget_moves_the_wall_even_to_the_number_it_had() -> None:
    from iterate.core.agent_loop import _move_the_wall

    wall, _ = _linear_wall(20.0)
    _move_the_wall("budget $20", wall)
    assert (wall.budget, wall.moves) == (20.0, 1)
    _move_the_wall("no budget", wall)
    assert (wall.budget, wall.moves) == (None, 2)
    _move_the_wall("budget 0", wall)  # refused: the wall is where it was
    _move_the_wall("prefer smaller models", wall)
    assert (wall.budget, wall.moves) == (None, 2)


def test_a_price_the_pricer_could_not_finish_never_crashes_the_ask() -> None:
    from iterate.core import serving
    from iterate.core.agent_loop import _over_budget_ask

    wall = serving.Wall(requests_per_hour=1_000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("write torch code for vit_base_patch16_224", float("nan"), "off the line")
    wall.record_refusal("keep the best and train at 128 px", 24.53, "off the line")
    ask = _over_budget_ask(wall, 1)
    assert (
        ": keep the best and train at 128 px ($25); write torch code for vit_base_patch16_224 "
        "(not priced). The Researcher was asked once more, and nothing it named fits."
    ) in ask



# ─── what the integration check added (v0.7 Day 4) ───────────────────────────


def test_a_round_naming_only_a_model_already_tried_is_dry() -> None:
    wall, _ = _linear_wall(20.0)
    sup = _WalledSupervisor([_OWN, _OVER], wall)
    researcher = _RoundTripResearcher(["", "fine-tune efficientnet_b0", "efficientnet_b0 again"])
    result = _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(5), wall=wall, researcher=researcher
    )
    assert result.stopped_because == "over_budget"
    assert len(result.history) == 1
    assert sup.seen_history_lens == [0, 1]  # iteration 2 never re-decided


def test_the_supervisor_reads_this_rounds_findings_and_the_ladder_reads_them_all() -> None:
    from iterate.core.researcher import Findings, Suggestion

    class _Earlier(_RoundTripResearcher):
        def research(self, **kw: object) -> object:  # type: ignore[override]
            if kw.get("round", 1) == 1:
                self.calls.append(dict(kw))
                return Findings(
                    suggestions=[Suggestion("fine-tune convnext_large", "big", "doi:10.1/b")],
                    papers_seen=4,
                )
            return super().research(**kw)  # type: ignore[arg-type]

    wall, _ = _linear_wall(20.0)
    sup = _WalledSupervisor([_OVER, _OWN], wall)
    researcher = _Earlier(["fine-tune efficientnet_b0 with pretrained weights"])
    _walled_loop(
        sup, [_FakeCoder(_result(0.60))], MaxIterations(1), wall=wall, researcher=researcher
    )
    redecided = sup.seen_extra[1]
    assert "efficientnet_b0" in str(redecided["research"])
    assert "convnext_large" not in str(redecided["research"])
    assert "convnext_large" in str(redecided["known_findings"])
    assert "efficientnet_b0" in str(redecided["known_findings"])


def test_the_ask_rounds_to_the_dollar_and_lists_an_unpriced_refusal_last() -> None:
    from iterate.core.agent_loop import _over_budget_ask

    wall, _ = _linear_wall(20.0)
    wall.record_refusal("own model: google/vit-base-patch16-224", None, "off the line")
    wall.record_refusal("swap the backbone to convnext_tiny", 36.79, "off the line")
    wall.record_refusal("train at 128 px", 24.4, "off the line")
    ask = _over_budget_ask(wall, 2)
    assert (
        "train at 128 px ($24); swap the backbone to convnext_tiny ($37); "
        "own model: google/vit-base-patch16-224 (not priced)"
    ) in ask


def test_a_failed_re_decide_keeps_the_findings_the_round_found() -> None:
    from iterate.core.agent_loop import _decide_within_the_wall
    from iterate.core.supervisor import SupervisorError

    wall, _ = _linear_wall(20.0)
    calls = {"n": 0}

    def decide(**extra: object) -> SupervisorDecision:
        calls["n"] += 1
        if calls["n"] == 1:
            wall.record_refusal("swap to convnext_tiny", 37.0, "off the line", network="convnext_tiny")
            return _OVER
        raise SupervisorError("model replied without calling plan_next")

    researcher = _RoundTripResearcher(["fine-tune efficientnet_b0 with pretrained weights"])
    with pytest.raises(SupervisorError) as raised:
        _decide_within_the_wall(
            decide, {}, [], wall=wall, researcher=researcher, controller=None,  # type: ignore[arg-type]
            profile="d", this_run=[], findings=None, max_rounds=2, iteration=1,
        )
    assert raised.value.findings is not None
    assert "efficientnet_b0" in raised.value.findings.render()


def test_a_run_whose_input_has_closed_stops_instead_of_waiting_for_ever() -> None:
    from iterate.core.agent_loop import _decide_within_the_wall
    from iterate.core.interactive import RunController

    wall, _ = _linear_wall(20.0)
    ctrl = RunController()
    ctrl.close_input()
    decision, _, _ = _decide_within_the_wall(
        lambda **extra: _OVER, {}, [], wall=wall, researcher=None, controller=ctrl,
        profile="d", this_run=[], findings=None, max_rounds=2, iteration=1,
    )
    assert decision.stopped_because == "over_budget"
    assert ctrl.wait_until(lambda: False, ask="?") == 0.0


def test_a_stop_typed_during_the_rounds_ends_them_before_the_next_ask() -> None:
    from iterate.core.agent_loop import _decide_within_the_wall
    from iterate.core.interactive import RunController

    wall, _ = _linear_wall(20.0)
    ctrl = RunController()
    ctrl.request_graceful_stop()
    researcher = _RoundTripResearcher(["fine-tune efficientnet_b0"])
    decision, _, _ = _decide_within_the_wall(
        lambda **extra: _OVER, {}, [], wall=wall, researcher=researcher, controller=ctrl,  # type: ignore[arg-type]
        profile="d", this_run=[], findings=None, max_rounds=2, iteration=1,
    )
    assert decision.stopped_because == "stopped-by-user"
    assert researcher.calls == []


def test_a_pass_that_lost_records_is_compared_on_the_records_both_have() -> None:
    """The ruler is the target's paired score when it gives one; the stored scores
    stand when it gives none."""
    from iterate.core.agent_loop import _improves

    def scored(value: float) -> ExperimentResult:
        return ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": value}, primary="f1", direction="maximize"),
        )

    baseline = scored(0.8)
    assert not _improves(scored(0.7), None, baseline, "maximize")
    assert _improves(scored(0.7), None, baseline, "maximize", lambda a, b: (0.9, 0.8))
    assert not _improves(scored(0.9), None, baseline, "maximize", lambda a, b: (0.7, 0.8))
    assert _improves(scored(0.9), None, baseline, "maximize", lambda a, b: None)
    nan = float("nan")
    assert not _improves(scored(0.9), None, baseline, "maximize", lambda a, b: (nan, nan))


def test_the_loop_weighs_a_pass_by_the_targets_paired_score() -> None:
    """A prompt target that lost records scores the two passes on the records both
    have; the loop takes that ruler over the stored scores."""

    class _PairingTarget(_FakeTarget):
        def paired_scores(
            self, result: ExperimentResult, bar: ExperimentResult
        ) -> tuple[float, float]:
            return (0.9, 0.5)

    sup = _FakeSupervisor([SupervisorDecision(False, "a", "try a")])
    coders = iter([_FakeCoder(_result(0.40))])
    result = run_supervised(
        target=_PairingTarget(),  # type: ignore[arg-type]
        dataset=object(),  # type: ignore[arg-type]
        supervisor=sup,  # type: ignore[arg-type]
        make_coder=lambda: next(coders),  # type: ignore[arg-type,return-value]
        terminator=MaxIterations(1),  # type: ignore[arg-type]
        memory=InMemoryMemory(),
        data_summary="d",
    )
    # Stored, 0.40 loses to the baseline's 0.50; on the common records it wins.
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.40
