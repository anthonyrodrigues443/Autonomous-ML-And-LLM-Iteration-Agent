"""Tests for the supervised agent loop (Supervisor + Coder), with fakes."""

from __future__ import annotations

from iterate.core.agent_loop import _winning_code, run_supervised
from iterate.core.coder import Cell, CodingResult
from iterate.core.memory import InMemoryMemory
from iterate.core.supervisor import SupervisorDecision
from iterate.core.terminator import MaxIterations
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics


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
          on_experiment=None, summarizer=None, memory=None):
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
