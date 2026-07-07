"""Tests for the Supervisor — briefs the coder from run history (fake LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core.supervisor import Supervisor, SupervisorError, lever_markers_for_brief
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, Message, ToolCall

if TYPE_CHECKING:
    from iterate.schemas.llm import ToolSpec


class _FakeLLM:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _plan(stop: bool, title: str, brief: str) -> ChatResponse:
    return ChatResponse(
        model="fake-model",
        tool_calls=[ToolCall(id="p", name="plan_next", arguments={"stop": stop, "title": title, "brief": brief})],
    )


def _text() -> ChatResponse:
    return ChatResponse(model="fake-model", content="let me think")


def _baseline() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.57}, primary="f1", direction="maximize", n_samples=100),
    )


def test_decide_returns_a_brief() -> None:
    fake = _FakeLLM([_plan(False, "target-encode", "so far: only one-hot tried. Try target encoding.")])
    d = Supervisor(fake, metric="f1").decide(data_summary="120 rows", baseline=_baseline(), history=[])
    assert not d.stop
    assert d.title == "target-encode"
    assert "target encoding" in d.brief


def test_decide_can_stop() -> None:
    fake = _FakeLLM([_plan(True, "", "")])
    d = Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[])
    assert d.stop


def test_no_tool_call_retries_then_raises() -> None:
    fake = _FakeLLM([_text(), _text()])
    with pytest.raises(SupervisorError, match="no plan"):
        Supervisor(fake, metric="f1", max_retries=1).decide(
            data_summary="d", baseline=_baseline(), history=[]
        )
    assert len(fake.calls) == 2


class _ErroringThenPlanLLM:
    """First chat() raises a backend reject (like groq's tool_use_failed 400); the
    retry returns a valid plan."""

    def __init__(self, plan: ChatResponse) -> None:
        self._plan = plan
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("400 tool_use_failed: /stop expected boolean, got string")
        return self._plan


def test_backend_reject_is_retried_not_fatal() -> None:
    # a malformed-tool-call 400 must not crash the run: nudge, retry, recover.
    fake = _ErroringThenPlanLLM(_plan(False, "ok", "so far: x. next: y"))
    d = Supervisor(fake, metric="f1", max_retries=1).decide(
        data_summary="d", baseline=_baseline(), history=[]
    )
    assert fake.calls == 2  # first errored, second succeeded
    assert d.brief == "so far: x. next: y"


def test_persistent_backend_error_degrades_to_supervisor_error() -> None:
    class _AlwaysErrors:
        @property
        def model(self) -> str:
            return "fake-model"

        def chat(self, messages, *, tools=None, temperature=None, max_tokens=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("400 tool_use_failed")

    with pytest.raises(SupervisorError, match="backend error"):
        Supervisor(_AlwaysErrors(), metric="f1", max_retries=1).decide(
            data_summary="d", baseline=_baseline(), history=[]
        )


def test_stop_as_the_string_false_does_not_stop_the_run() -> None:
    # bool("false") is True — a weak model emitting the STRING "false" must not be read
    # as a request to stop. The brief is what matters; coerce by meaning.
    fake = _FakeLLM([
        ChatResponse(
            model="fake-model",
            tool_calls=[ToolCall(id="p", name="plan_next",
                                 arguments={"stop": "false", "title": "go", "brief": "next: try x"})],
        )
    ])
    d = Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[])
    assert d.stop is False
    assert d.brief == "next: try x"


def _scored_experiment(desc: str, score: float, code: str) -> Experiment:
    return Experiment(
        candidate=Candidate(description=desc, changes={"code": code}, rationale="r"),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id=desc,
            metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize"),
        ),
    )


def test_brief_so_far_is_composed_from_the_carried_best_not_the_llm() -> None:
    # a live run's supervisor claimed "PowerTransformer + threshold scored 0.6437"
    # when the real best was 0.6132 — the so-far slot must come from the record.
    prior = _scored_experiment(
        "imbalance weighting",
        0.6132,
        "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder()",
    )
    fake = _FakeLLM([
        _plan(False, "tune", "so far: PowerTransformer scored 0.6437. next: hyperparameter-search: tune max_iter.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.brief.startswith("so far: best f1=0.6132")
    assert "0.6437" not in d.brief  # the hallucinated score is discarded
    assert "OneHotEncoder" in d.brief  # the real carried components are stated
    assert d.brief.endswith("next: hyperparameter-search: tune max_iter.")


def test_brief_without_a_next_marker_is_wrapped_not_lost() -> None:
    prior = _scored_experiment("baseline", 0.57, "x = 1")
    fake = _FakeLLM([_plan(False, "encode", "target-encode the PaymentMethod column")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert "next: target-encode the PaymentMethod column" in d.brief
    assert d.brief.startswith("so far:")


def test_no_marker_with_a_hallucinated_so_far_is_stripped() -> None:
    # the model disobeys twice: no "next:" marker AND a hallucinated so-far claim.
    # The claim must not ride into the grounded brief inside the wrapped move.
    prior = _scored_experiment("baseline", 0.57, "x = 1")
    fake = _FakeLLM([_plan(False, "tune", "so far: PowerTransformer scored 0.99. tune max_iter")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert "0.99" not in d.brief
    assert d.brief.endswith("next: tune max_iter")
    assert d.brief.count("so far:") == 1  # only the harness-composed slot survives


def test_next_marker_matches_regardless_of_casing_and_spacing() -> None:
    prior = _scored_experiment("baseline", 0.57, "x = 1")
    for brief in ("SO FAR: junk 0.99. NEXT: tune max_iter.", "Next : tune max_iter."):
        fake = _FakeLLM([_plan(False, "t", brief)])
        d = Supervisor(fake, metric="f1").decide(
            data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
        )
        assert "0.99" not in d.brief
        assert "tune max_iter." in d.brief


def test_first_experiment_brief_passes_through_ungrounded() -> None:
    # experiment 1 has no history to ground against; the baseline brief is verbatim.
    fake = _FakeLLM([_plan(False, "baseline", "one-hot encode, median-impute, fit HGB.")])
    d = Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[])
    assert d.brief == "one-hot encode, median-impute, fit HGB."


def test_so_far_states_the_banked_config_and_searched_grid() -> None:
    # 4 of 5 residual duplicates came from re-briefing levers already embodied in
    # the best: the brief must state the incumbent's exact config and searched grid.
    code = (
        "param_grid = {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}\n"
        "search = GridSearchCV(HistGradientBoostingClassifier(random_state=42), param_grid)\n"
        "model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_iter=100)\n"
        "model.fit(Xa, ya)\n"
        "preds = (model.predict_proba(Xb)[:, 1] >= 0.43).astype(int)\n"
    )
    prior = _scored_experiment("grid winner", 0.626, code)
    fake = _FakeLLM([_plan(False, "next", "next: ensembling: stack the top models.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert "final model: HistGradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_iter=100)" in d.brief
    assert "decision threshold 0.43 already applied" in d.brief
    assert "already grid-searched: learning_rate/max_depth" in d.brief


def test_so_far_names_the_carried_decision_threshold_last_match_wins() -> None:
    # a stale initializer early in the code loses to the threshold actually applied
    # in the submit cell (last by construction).
    code = (
        "threshold = 0.5\n"
        "model = HistGradientBoostingClassifier().fit(Xa, ya)\n"
        "preds = (model.predict_proba(Xb)[:, 1] >= 0.35).astype(int)\n"
    )
    prior = _scored_experiment("threshold tuning", 0.61, code)
    fake = _FakeLLM([_plan(False, "next", "next: feature-selection: drop noisy one-hots.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert "decision threshold 0.35" in d.brief
    assert "threshold 0.5" not in d.brief  # the stale initializer lost


def test_carried_threshold_resolves_the_applied_variable_not_a_sweep_literal() -> None:
    # run 16: briefs carried "0.4 already applied" scraped from sweep code while the
    # banked threshold was 0.2802 — the corrupted fact poisoned the whole run.
    from iterate.core.supervisor import _carried_threshold

    code = (
        "for t in [0.2, 0.4, 0.6]:\n"
        "    scores.append(f1_score(yb, (proba_b >= t).astype(int)))\n"
        "best_threshold = 0.2802\n"
        "preds = (model.predict_proba(Xh)[:, 1] >= best_threshold).astype(int)\n"
    )
    assert _carried_threshold(code) == "0.2802"


def test_carried_threshold_is_none_when_unresolvable() -> None:
    # a sweep-derived variable cannot be resolved statically: no fact > wrong fact.
    from iterate.core.supervisor import _carried_threshold

    code = (
        "best_threshold = thresholds[np.argmax(scores)]\n"
        "preds = (model.predict_proba(Xh)[:, 1] >= best_threshold).astype(int)\n"
    )
    assert _carried_threshold(code) is None


def test_a_persistent_guard_violation_falls_back_to_an_untried_lever() -> None:
    # run 16 i7: the measured-lost guard fired, the model re-emitted the same lever
    # on the retry, and the known re-commission was accepted. Now the harness
    # composes a deterministic move from an untried class instead.
    best = _scored_experiment("baseline", 0.6254, "model = HGB().fit(Xa, ya)")
    lost = Experiment(
        candidate=Candidate(description="imbalance", changes={"code": "y"}, rationale="r"),
        target="t",
        hypothesis="so far: x. next: imbalance-or-threshold: set class_weight balanced.",
        status="completed",
        result=ExperimentResult(
            experiment_id="l",
            metrics=Metrics(values={"f1": 0.5965}, primary="f1", direction="maximize"),
        ),
    )
    same = "next: imbalance-or-threshold: apply class_weight balanced weighting."
    fake = _FakeLLM([_plan(False, "again", same), _plan(False, "again", same)])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[best, lost], carried_best=best
    )
    assert d.title.startswith("untried lever:")
    assert "class_weight" not in d.brief.split("next:", 1)[1].split("Known dead ends")[0]
    assert d.brief.startswith("so far:")  # the fallback move is grounded like any other


def test_so_far_without_a_carried_best_states_the_bar_only() -> None:
    # failures so far (or a cross-run history the coder does not hold): the so-far
    # slot must not claim a best the coder was never given.
    failed = Experiment(
        candidate=Candidate(description="broken", changes={"code": "x"}, rationale="r"),
        target="t",
        hypothesis="h",
        status="failed",
        result=ExperimentResult(experiment_id="f", error="boom"),
    )
    fake = _FakeLLM([_plan(False, "retry", "next: repair the NameError.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[failed]
    )
    assert d.brief.startswith("so far: baseline f1=0.5700 is the bar to beat; no carried best yet.")
    assert d.brief.endswith("next: repair the NameError.")


def test_history_marks_a_floor_banked_score_so_the_lever_is_not_credited() -> None:
    banked = Experiment(
        candidate=Candidate(
            description="gridsearch tuning",
            changes={
                "code": "GridSearchCV(HistGradientBoostingClassifier(), grid)",
                "cells": [
                    {"code": "GridSearchCV(...)", "source": "agent", "error": "timeout"},
                    {"code": "floor_submit()", "source": "fallback", "error": None},
                ],
            },
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.61}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "next: try x")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[banked])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "[floor: harness fallback submission" in sent


def test_history_shows_components_to_the_supervisor() -> None:
    prior = Experiment(
        candidate=Candidate(
            description="logreg one-hot",
            changes={"code": "from sklearn.preprocessing import OneHotEncoder\nOneHotEncoder()"},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.55}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "try interactions")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "used: OneHotEncoder" in sent
    assert "f1=0.5500" in sent


def test_history_renders_digests_and_a_technique_scoreboard() -> None:
    from iterate.schemas.experiment import ExperimentDigest

    def _exp(desc: str, score: float, techs: list[str], digest: ExperimentDigest) -> Experiment:
        return Experiment(
            candidate=Candidate(description=desc, changes={"code": "x=1"}, rationale="r"),
            target="t", hypothesis="h", status="completed",
            result=ExperimentResult(
                experiment_id=desc,
                metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize"),
            ),
            digest=digest,
        )

    hist = [
        _exp("one-hot baseline", 0.55, ["OneHotEncoder"], ExperimentDigest(
            techniques=["OneHotEncoder", "HistGradientBoosting"], score=0.55,
            data_insights=["27% positive class"], what_helped=[], what_hurt=[],
            takeaway="Try target encoding on PaymentMethod.")),
        _exp("target encoding", 0.61, ["TargetEncoder"], ExperimentDigest(
            techniques=["TargetEncoder", "HistGradientBoosting"], score=0.61,
            what_helped=["target encoding: 0.55 -> 0.61"], what_hurt=[],
            data_insights=[], takeaway="Add an interaction feature.")),
    ]
    fake = _FakeLLM([_plan(False, "next", "build on target encoding")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=hist)
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    # the digests' insight reaches the supervisor
    assert "target encoding: 0.55 -> 0.61" in sent
    assert "next-idea: Add an interaction feature." in sent
    assert "27% positive class" in sent
    # the technique scoreboard surfaces the best score per technique
    assert "Technique scoreboard" in sent
    assert "TargetEncoder 0.6100" in sent


def test_lever_ledger_lists_tried_and_untried_classes() -> None:
    # one experiment used one-hot + class_weight + an interaction column; the ledger
    # must mark those classes tried and surface the rest as NOT yet tried.
    prior = Experiment(
        candidate=Candidate(
            description="baseline plus imbalance",
            changes={
                "code": (
                    "from sklearn.preprocessing import OneHotEncoder\n"
                    "model = RandomForestClassifier(class_weight='balanced')\n"
                    "X['tenure_per_charge'] = X['tenure'] / X['MonthlyCharges']\n"
                )
            },
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.6}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "pull an untried lever")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "Levers tried:" in sent
    tried_line = next(ln for ln in sent.splitlines() if ln.startswith("Levers tried:"))
    tried, untried = tried_line.split("| Levers NOT yet tried:")
    assert "categorical-encoding" in tried
    assert "imbalance-or-threshold" in tried
    assert "interactions-or-ratios" in tried
    assert "numeric-transform" in untried  # never touched -> explicitly surfaced
    assert "ensembling" in untried


def test_baseline_rebrief_with_history_is_rejected_once_and_retried() -> None:
    # rung 1's trigger is "no history"; with experiments on record a baseline
    # rebuild duplicates a scored experiment (wasted an iteration in every live run).
    prior = _scored_experiment("first", 0.58, "x = 1")
    fake = _FakeLLM([
        _plan(False, "Baseline Model",
              "next: Baseline: one-hot encode categoricals, median-impute, fit HGB."),
        _plan(False, "tune", "next: hyperparameter-search: tune the learning rate."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "tune"  # the corrected second plan won
    assert len(fake.calls) == 2
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "BASELINE rung must NOT fire" in sent


def test_baseline_rebrief_on_the_last_attempt_falls_back_to_an_untried_lever() -> None:
    # with no retry budget left, a violating brief is not accepted anymore — the
    # harness composes a deterministic move from an untried class instead.
    prior = _scored_experiment("first", 0.58, "x = 1")
    fake = _FakeLLM([
        _plan(False, "Baseline Model", "next: Baseline: one-hot, median-impute, HGB.")
    ])
    d = Supervisor(fake, metric="f1", max_retries=0).decide(
        data_summary="d", baseline=_baseline(), history=[prior]
    )
    assert d.title.startswith("untried lever:")
    assert d.brief.startswith("so far:")


def test_a_lever_brief_that_mentions_the_baseline_is_not_rejected() -> None:
    prior = _scored_experiment("first", 0.58, "x = 1")
    fake = _FakeLLM([
        _plan(False, "weights",
              "next: imbalance-or-threshold: set class_weight balanced to beat the baseline.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior]
    )
    assert d.title == "weights"
    assert len(fake.calls) == 1  # accepted on the first attempt


def test_history_marks_a_duplicate_submission_so_the_lever_is_not_recredited() -> None:
    dup = Experiment(
        candidate=Candidate(
            description="threshold re-tune",
            changes={"code": "x = 1", "duplicate_submission": True},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.6173}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "next: try x")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[dup])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "[duplicate submission of an earlier experiment" in sent


def test_dead_ends_from_digests_reach_the_brief() -> None:
    # failure knowledge must reach the CODER: without it, sessions re-probed the
    # same rejected feature in 8 of 10 notebooks of a live run.
    from iterate.schemas.experiment import ExperimentDigest

    prior = _scored_experiment("imbalance", 0.61, "x = 1")
    with_digest = prior.model_copy(update={
        "digest": ExperimentDigest(
            techniques=["HistGradientBoosting"], score=0.61,
            what_hurt=["tenure*MonthlyCharges interaction: 0.62 -> 0.53",
                       "PowerTransformer on numerics: no change"],
            what_helped=[], data_insights=[], takeaway="t",
        ),
    })
    fake = _FakeLLM([_plan(False, "next", "next: feature-selection: drop noisy one-hots.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[with_digest], carried_best=with_digest
    )
    assert "Known dead ends this run (do NOT retry):" in d.brief
    assert "tenure*MonthlyCharges" in d.brief
    assert "PowerTransformer" in d.brief


def test_dead_ends_are_capped_and_deduped() -> None:
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    def _with(hurts: list[str]) -> Experiment:
        e = _scored_experiment("e", 0.6, "x = 1")
        return e.model_copy(update={"digest": ExperimentDigest(
            techniques=[], score=0.6, what_hurt=hurts, what_helped=[],
            data_insights=[], takeaway="t")})

    hist = [_with(["alpha encoding trick: hurt"]),
            _with(["alpha encoding variant: hurt too", "bravo scaling probe: hurt"]),
            _with(["charlie binning probe: hurt", "delta stacking probe: hurt"])]
    line = _dead_ends(hist)
    assert line.count(";") == 2  # capped at 3 items
    # the RECURRING idea (alpha encoding, 2 occurrences via shared tokens) leads
    # and shows its latest phrasing; one-off ideas fill the remaining slots
    assert "alpha encoding variant" in line
    assert line.lower().count("alpha") == 1  # grouped, not listed twice


def test_recurring_dead_end_outlives_a_flood_of_one_offs() -> None:
    # live run: the pet dead idea was recorded early, then evicted by newer one-off
    # failures under a most-recent policy — and the re-probing resumed immediately.
    # Recurrence ranking must keep it on the list.
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    def _with(hurts: list[str]) -> Experiment:
        e = _scored_experiment("e", 0.6, "x = 1")
        return e.model_copy(update={"digest": ExperimentDigest(
            techniques=[], score=0.6, what_hurt=hurts, what_helped=[],
            data_insights=[], takeaway="t")})

    hist = [
        _with(["tenure_monthly interaction: 0.628 -> 0.623"]),
        _with(["gridsearch max_iter bump: lowered the val"]),
        _with(["polynomial expansion on tenure_monthly interaction: no gain"]),
        _with(["highcharge manual flag: 0.638 -> 0.603"]),
        _with(["usageintensity engineered column: NameErrors"]),
        _with(["robustscaler pass: no change"]),
    ]
    line = _dead_ends(hist)
    assert "tenure_monthly interaction" in line  # 2 occurrences -> sticks despite the flood


def test_no_digests_means_no_dead_ends_line() -> None:
    prior = _scored_experiment("first", 0.58, "x = 1")
    fake = _FakeLLM([_plan(False, "next", "next: try x")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert "dead ends" not in d.brief


def _dup_experiment(brief_class: str) -> Experiment:
    return Experiment(
        candidate=Candidate(
            description="dup",
            changes={"code": "x = 1", "duplicate_submission": True},
            rationale="r",
        ),
        target="t",
        hypothesis=f"so far: x. next: {brief_class}: push it further.",
        status="completed",
        result=ExperimentResult(
            experiment_id="d",
            metrics=Metrics(values={"f1": 0.6}, primary="f1", direction="maximize"),
        ),
    )


def test_rebriefing_the_class_that_just_duplicated_is_rejected_once() -> None:
    # observed in two runs: the family that just no-opped got briefed again and
    # no-opped again. One corrective retry toward a different class.
    prev = _dup_experiment("hyperparameter-search")
    fake = _FakeLLM([
        _plan(False, "again", "next: hyperparameter-search: tune depth once more."),
        _plan(False, "encode", "next: categorical-encoding: target-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prev], carried_best=prev
    )
    assert d.title == "encode"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "DUPLICATE" in sent


def test_same_class_rebrief_is_fine_when_the_previous_was_not_a_duplicate() -> None:
    prev = _dup_experiment("hyperparameter-search")
    prev.candidate.changes.pop("duplicate_submission")  # a real, novel result
    fake = _FakeLLM([_plan(False, "again", "next: hyperparameter-search: tune depth once more.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prev], carried_best=prev
    )
    assert d.title == "again"
    assert len(fake.calls) == 1  # accepted first time — refinement is legitimate


def test_different_class_after_a_duplicate_is_accepted() -> None:
    prev = _dup_experiment("hyperparameter-search")
    fake = _FakeLLM([_plan(False, "sel", "next: feature-selection: drop weak one-hots.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prev], carried_best=prev
    )
    assert d.title == "sel"
    assert len(fake.calls) == 1


_BANKED_CODE = (
    "X_fe['Tenure_Monthly'] = X_fe['tenure'] * X_fe['MonthlyCharges']\n"
    "param_grid = {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}\n"
    "model = HistGradientBoostingClassifier(class_weight='balanced', learning_rate=0.1, max_depth=3)\n"
    "model.fit(Xa, ya)\n"
)


def test_briefing_class_weight_already_in_the_banked_model_is_rejected() -> None:
    # run 11 i3: the so-far line SAID class_weight was set; the move briefed it anyway.
    prior = _scored_experiment("best", 0.6264, _BANKED_CODE)
    fake = _FakeLLM([
        _plan(False, "again", "next: imbalance-or-threshold: set class_weight balanced on the model."),
        _plan(False, "encode", "next: categorical-encoding: target-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "encode"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "already set in the carried best" in sent


def test_rebriefing_the_already_searched_grid_is_rejected_but_a_new_dimension_passes() -> None:
    prior = _scored_experiment("best", 0.6264, _BANKED_CODE)
    # same grid re-briefed -> rejected, corrected plan wins
    fake = _FakeLLM([
        _plan(False, "again", "next: hyperparameter-search: tune learning_rate and max_depth."),
        _plan(False, "l2", "next: hyperparameter-search: search l2_regularization, a new dimension."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "l2"
    # a NEW dimension named up front -> accepted first time
    fake2 = _FakeLLM([_plan(False, "l2", "next: hyperparameter-search: search l2_regularization values.")])
    d2 = Supervisor(fake2, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d2.title == "l2"
    assert len(fake2.calls) == 1


def test_rung_labeled_grid_rebriefs_are_rejected_without_the_class_name() -> None:
    # run 12 i8/i9: the moves opened "refine-best:" (a ladder rung, not a lever
    # class) and sailed past a class-gated check while re-commissioning the
    # already-searched grid — one vaguely, one naming the exact searched params.
    prior = _scored_experiment("best", 0.6333, _BANKED_CODE)
    for move in (
        "next: refine-best: perform a deeper search on the GridSearchCV winner.",
        "next: refine-best: refine the search space for learning_rate and max_depth.",
    ):
        fake = _FakeLLM([
            _plan(False, "again", move),
            _plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod."),
        ])
        d = Supervisor(fake, metric="f1").decide(
            data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
        )
        assert d.title == "enc", move
        sent = "\n".join(m.content or "" for m in fake.calls[1])
        assert "already searched" in sent


def test_a_threshold_refine_move_is_not_mistaken_for_a_grid_re_search() -> None:
    prior = _scored_experiment("best", 0.6333, _BANKED_CODE)
    fake = _FakeLLM([
        _plan(False, "thr", "next: refine-best: tune the decision threshold to finer values around 0.4.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "thr"
    assert len(fake.calls) == 1  # accepted first time


def test_carried_threshold_resolves_the_split_proba_idiom() -> None:
    # run 20 i2's actual carried code: predict_proba assigned to a variable, the
    # comparison on a LATER line — the same-line-only check returned None, dropped
    # the fact from the brief, and disarmed the re-tune guard.
    from iterate.core.supervisor import _carried_threshold

    code = (
        "probs_h = model.predict_proba(Xh_final)[:, 1]\n"
        "preds_h = (probs_h > 0.4).astype(int)\n"
        "pd.Series(preds_h).to_csv('predictions.csv', index=False, header=False)\n"
    )
    assert _carried_threshold(code) == "0.4"


def test_submit_components_exclude_null_probes_left_in_the_code() -> None:
    # run 20: PowerTransformer was probed, measured a no-op, excluded from the
    # submission — but the whole-code fingerprint still credited it and the next
    # brief said "keep ... with PowerTransformer".
    from iterate.core.supervisor import _submit_components

    code = (
        "num_imputer = SimpleImputer(strategy='median')\n"
        "Xa_num = pd.DataFrame(num_imputer.fit_transform(Xa_raw[num_cols]))\n"
        "pt = PowerTransformer()\n"
        "Xa_pt = pd.DataFrame(pt.fit_transform(Xa_num))  # probe: measured a no-op\n"
        "enc = OneHotEncoder(handle_unknown='ignore')\n"
        "Xa_cat = pd.DataFrame(enc.fit_transform(Xa_raw[cat_cols]))\n"
        "Xa = pd.concat([Xa_num, Xa_cat], axis=1)\n"
        "model = HistGradientBoostingClassifier(random_state=42)\n"
        "model.fit(Xa, ya)\n"
        "Xh_num = pd.DataFrame(num_imputer.transform(X_holdout[num_cols]))\n"
        "Xh_cat = pd.DataFrame(enc.transform(X_holdout[cat_cols]))\n"
        "Xh = pd.concat([Xh_num, Xh_cat], axis=1)\n"
        "pd.Series(model.predict(Xh)).to_csv('predictions.csv', index=False, header=False)\n"
    )
    components = _submit_components(code)
    assert "PowerTransformer" not in components
    assert "SimpleImputer" in components
    assert "OneHotEncoder" in components
    assert "HistGradientBoostingClassifier" in components


def test_carried_config_resolves_the_submit_path_estimator_not_the_last_constructor() -> None:
    # run 19 i3: the banked model was class_weight='balanced' but the LAST
    # constructor in the code was a plain probe — the corrupted so-far fact let the
    # brief re-commission the already-applied lever.
    from iterate.core.supervisor import _carried_config

    code = (
        "model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)\n"
        "model.fit(Xa, ya)\n"
        "probe = HistGradientBoostingClassifier(random_state=42)  # a later losing probe\n"
        "probe.fit(Xa, ya)\n"
        "preds = model.predict_proba(Xh)[:, 1]\n"
        "pd.Series((preds >= 0.28).astype(int)).to_csv('predictions.csv', index=False, header=False)\n"
    )
    config = _carried_config(code)
    assert config is not None
    assert "class_weight='balanced'" in config


def test_rebriefing_class_weight_is_rejected_even_when_a_probe_constructor_comes_last() -> None:
    # end-to-end: with the submit-path resolution, the class_weight guard fires.
    code = (
        "model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)\n"
        "model.fit(Xa, ya)\n"
        "probe = HistGradientBoostingClassifier(random_state=42)\n"
        "preds = model.predict_proba(Xh)[:, 1]\n"
    )
    prior = _scored_experiment("imbalance", 0.6140, code)
    fake = _FakeLLM([
        _plan(False, "again", "next: imbalance-or-threshold: train with class_weight balanced."),
        _plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "enc"


def test_settled_dead_end_reasons_are_never_truncated() -> None:
    # run 19 i6's brief shipped "settled: its;" — the suffix must survive whole.
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    best = _scored_experiment("best", 0.6254, "x = 1")
    long_claim = (
        "HistGradientBoostingClassifier with class_weight balanced and a long tail "
        "of extra explanatory detail that overflows the per-item cap easily"
    )
    loser = _scored_experiment("lost", 0.6140, "x = 1").model_copy(update={
        "digest": ExperimentDigest(
            techniques=[], score=0.6140, what_helped=[long_claim],
            what_hurt=[], data_insights=[], takeaway="t"),
    })
    line = _dead_ends([best, loser])
    assert "settled: its holdout was 0.6140, not the best" in line  # intact suffix


def test_retuning_an_already_applied_threshold_is_rejected() -> None:
    # run 17 i2/i4: the banked best already applied a tuned threshold (0.2790); two
    # iterations burned on re-tuning it, each reproducing the incumbent exactly.
    code = (
        "best_threshold = 0.279\n"
        "preds = (model.predict_proba(Xh)[:, 1] >= best_threshold).astype(int)\n"
    )
    prior = _scored_experiment("baseline", 0.6247, code)
    fake = _FakeLLM([
        _plan(False, "again", "next: imbalance-or-threshold: refine the threshold sweep to finer steps."),
        _plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "enc"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "already applied in the carried best" in sent


def test_keeping_the_threshold_as_context_does_not_trip_the_guard() -> None:
    # the like-for-like rule makes briefs SAY "keep the threshold" — context
    # mentions must pass (the run-14 over-blocking lesson).
    code = (
        "best_threshold = 0.279\n"
        "preds = (model.predict_proba(Xh)[:, 1] >= best_threshold).astype(int)\n"
    )
    prior = _scored_experiment("baseline", 0.6247, code)
    fake = _FakeLLM([
        _plan(False, "grid", "next: hyperparameter-search: search l2_regularization, keeping the 0.279 threshold.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "grid"
    assert len(fake.calls) == 1


def test_technique_scoreboard_skips_stamped_experiments() -> None:
    # run 17 i9: a duplicate's digest credited its techniques with the incumbent's
    # score — the scoreboard must not absorb stamped experiments.
    from iterate.schemas.experiment import ExperimentDigest

    real = _scored_experiment("winner", 0.6311, "x = 1").model_copy(update={
        "digest": ExperimentDigest(techniques=["GridSearchCV"], score=0.6311,
                                   what_helped=[], what_hurt=[], data_insights=[], takeaway="t"),
    })
    dup = _scored_experiment("dup", 0.6311, "x = 1")
    dup.candidate.changes["duplicate_submission"] = True
    dup = dup.model_copy(update={
        "digest": ExperimentDigest(techniques=["RandomForest"], score=0.6311,
                                   what_helped=[], what_hurt=[], data_insights=[], takeaway="t"),
    })
    fake = _FakeLLM([_plan(False, "next", "next: try x")])
    Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[real, dup], carried_best=real
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "GridSearchCV 0.6311" in sent
    assert "RandomForest" not in sent.split("Technique scoreboard")[1].splitlines()[0]


def test_rebriefing_an_engineered_feature_the_best_already_builds_is_rejected() -> None:
    # run 11 i6: the config compression dropped the feature set, so the supervisor
    # re-commissioned the incumbent's own Tenure_Monthly win.
    prior = _scored_experiment("best", 0.6264, _BANKED_CODE)
    fake = _FakeLLM([
        _plan(False, "again", "next: interactions-or-ratios: add the Tenure_Monthly interaction feature."),
        _plan(False, "sel", "next: feature-selection: drop weak one-hot columns."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "sel"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "'Tenure_Monthly' already exists" in sent


def test_novel_moves_pass_the_banked_guard_untouched() -> None:
    prior = _scored_experiment("best", 0.6264, _BANKED_CODE)
    fake = _FakeLLM([_plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "enc"
    assert len(fake.calls) == 1


def test_a_measured_lost_technique_is_not_silently_recommissioned() -> None:
    # run 13 i3: class_weight was measured fairly in i2 (0.6002, lost to 0.6251)
    # and re-briefed verbatim — the loss was invisible to the so-far facts.
    best = _scored_experiment("baseline", 0.6251, "model = HGB().fit(Xa, ya)")
    lost = Experiment(
        candidate=Candidate(description="imbalance", changes={"code": "x"}, rationale="r"),
        target="t",
        hypothesis="so far: x. next: imbalance-or-threshold: set class_weight balanced.",
        status="completed",
        result=ExperimentResult(
            experiment_id="l",
            metrics=Metrics(values={"f1": 0.6002}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([
        _plan(False, "again", "next: imbalance-or-threshold: use class_weight balanced weighting."),
        _plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[best, lost], carried_best=best
    )
    assert d.title == "enc"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "already measured this run" in sent
    assert "0.6002" in sent


def test_class_names_and_so_far_facts_do_not_trip_the_lost_guard() -> None:
    # run 14 over-fire: 'imbalance-or-threshold' (the class name the prompt requires)
    # and the so-far's "decision threshold 0.4 already applied" both contain the
    # 'threshold' marker — only TECHNIQUE mentions in the move part may count.
    best = _scored_experiment("baseline", 0.6251, "model = HGB().fit(Xa, ya)")
    lost = Experiment(
        candidate=Candidate(description="enc", changes={"code": "x"}, rationale="r"),
        target="t",
        # a grounded hypothesis whose SO-FAR mentions threshold, but whose move is
        # target encoding — this experiment never measured a threshold technique
        hypothesis=(
            "so far: best f1=0.6251 via 'b' (decision threshold 0.4 already applied). "
            "next: categorical-encoding: target-encode PaymentMethod."
        ),
        status="completed",
        result=ExperimentResult(
            experiment_id="l",
            metrics=Metrics(values={"f1": 0.60}, primary="f1", direction="maximize"),
        ),
    )
    # the new move names the CLASS (containing 'threshold') but its technique is
    # class_weight — no prior class_weight loss exists, so no rejection
    fake = _FakeLLM([
        _plan(False, "w", "next: imbalance-or-threshold: set class_weight balanced on the model.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[best, lost], carried_best=best
    )
    assert d.title == "w"
    assert len(fake.calls) == 1  # accepted first time


def test_refining_the_winning_technique_is_not_blocked_by_the_lost_guard() -> None:
    # only strictly-losing priors count: re-refining the winner stays legitimate.
    best = Experiment(
        candidate=Candidate(description="thr", changes={"code": "x"}, rationale="r"),
        target="t",
        hypothesis="so far: x. next: imbalance-or-threshold: tune the threshold.",
        status="completed",
        result=ExperimentResult(
            experiment_id="b",
            metrics=Metrics(values={"f1": 0.62}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "finer", "next: imbalance-or-threshold: tune the threshold finer.")])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[best], carried_best=best
    )
    assert d.title == "finer"
    assert len(fake.calls) == 1


def test_a_move_claiming_a_phantom_best_score_is_rejected() -> None:
    # run 13 i6: the move cited "previous best was 0.6552" — a validation-split
    # number that never existed on the holdout.
    prior = _scored_experiment("best", 0.6251, "model = HGB().fit(Xa, ya)")
    fake = _FakeLLM([
        _plan(False, "bad", "next: ensembling: the previous best was 0.6552, stack models to beat it."),
        _plan(False, "ok", "next: ensembling: stack the top two models for variance reduction."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "ok"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "neither the banked best nor the baseline" in sent


def test_a_move_fusing_two_lever_classes_is_rejected() -> None:
    prior = _scored_experiment("best", 0.6251, "model = HGB().fit(Xa, ya)")
    fake = _FakeLLM([
        _plan(False, "fused", "next: hyperparameter-search: imbalance-or-threshold: tune both."),
        _plan(False, "one", "next: feature-selection: drop weak one-hots."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "one"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "two lever classes" in sent


def test_a_move_citing_the_real_best_or_baseline_passes_the_lint() -> None:
    prior = _scored_experiment("best", 0.6251, "model = HGB().fit(Xa, ya)")
    fake = _FakeLLM([
        _plan(False, "ok", "next: ensembling: the best scored 0.6251, stack models to beat it.")
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "ok"
    assert len(fake.calls) == 1


def test_false_api_claims_are_kept_out_of_the_dead_ends() -> None:
    # run 13: "HGB lacks class_weight" was banked while the prior session's executed
    # code contained a successful class_weight fit.
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    exp = _scored_experiment("e", 0.6, "x = 1")
    exp.candidate.changes["cells"] = [
        {"code": "model = HGB(class_weight='balanced').fit(Xa, ya)", "source": "agent", "error": None}
    ]
    poisoned = exp.model_copy(update={"digest": ExperimentDigest(
        techniques=[], score=0.6,
        what_hurt=["HistGradientBoostingClassifier lacks a class_weight parameter"],
        what_helped=[], data_insights=[], takeaway="t")})
    assert _dead_ends([poisoned]) == ""  # the hallucinated claim never enters


def test_a_settled_unbriefed_model_swap_is_not_recommissioned() -> None:
    # run 15 i8: iter 5 drifted into a GradientBoosting swap (unbriefed — its
    # hypothesis said feature-selection), submitted it, stamped holdout 0.6180 <
    # best 0.6254. The re-commission must be caught via the SUBMITTED estimator.
    best = _scored_experiment(
        "baseline", 0.6254,
        "model = HistGradientBoostingClassifier(random_state=42).fit(Xa, ya)",
    )
    settled = Experiment(
        candidate=Candidate(
            description="feature selection",
            changes={"code": (
                "sel = SelectFromModel(HistGradientBoostingClassifier())\n"
                "gb = GradientBoostingClassifier(random_state=42)\n"
                "gb.fit(Xa_sel, ya)\n"
            )},
            rationale="r",
        ),
        target="t",
        hypothesis="so far: x. next: feature-selection: drop weak features.",
        status="completed",
        result=ExperimentResult(
            experiment_id="s",
            metrics=Metrics(values={"f1": 0.6180}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([
        _plan(False, "swap", "next: model-swap: switch to GradientBoostingClassifier for stability."),
        _plan(False, "enc", "next: categorical-encoding: frequency-encode PaymentMethod."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[best, settled], carried_best=best
    )
    assert d.title == "enc"
    sent = "\n".join(m.content or "" for m in fake.calls[1])
    assert "'gradientboosting' was already measured this run" in sent
    assert "0.6180" in sent


def test_the_default_hgb_baseline_does_not_read_as_a_model_swap() -> None:
    # "gradientboosting" is a substring of "histgradientboosting": the baseline
    # must not mark model-swap tried, and a baseline brief must not trip guards.
    prior = _scored_experiment(
        "baseline", 0.6, "model = HistGradientBoostingClassifier().fit(Xa, ya)"
    )
    fake = _FakeLLM([_plan(False, "next", "next: feature-selection: drop weak one-hots.")])
    Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    tried_line = next(ln for ln in sent.splitlines() if ln.startswith("Levers tried:"))
    _, untried = tried_line.split("| Levers NOT yet tried:")
    assert "model-swap" in untried


def test_a_real_swap_marks_the_model_swap_lever_tried() -> None:
    prior = _scored_experiment(
        "rf swap", 0.6, "model = RandomForestClassifier(random_state=42).fit(Xa, ya)"
    )
    fake = _FakeLLM([_plan(False, "next", "next: feature-selection: drop weak one-hots.")])
    Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    tried_line = next(ln for ln in sent.splitlines() if ln.startswith("Levers tried:"))
    tried, _ = tried_line.split("| Levers NOT yet tried:")
    assert "model-swap" in tried


def test_settled_val_wins_that_lost_on_holdout_enter_the_dead_ends() -> None:
    # run 15: iter 5's digest recorded the GB swap as HELPED (val 0.6593) though
    # its holdout stamped 0.6180 — a val-vs-holdout mirage that invited the i8
    # re-commission. It must enter the do-NOT-retry channel with the real stamp.
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    best = _scored_experiment("baseline", 0.6254, "x = 1")
    loser = _scored_experiment("drifted", 0.6180, "x = 1").model_copy(update={
        "digest": ExperimentDigest(
            techniques=["GradientBoosting"], score=0.6180,
            what_helped=["Switch to GradientBoostingClassifier: 0.6480 -> 0.6593"],
            what_hurt=[], data_insights=[], takeaway="t"),
    })
    line = _dead_ends([best, loser])
    assert "GradientBoostingClassifier" in line
    assert "settled: its holdout was 0.6180" in line


def test_the_best_experiments_helped_items_are_not_marked_settled() -> None:
    from iterate.core.supervisor import _dead_ends
    from iterate.schemas.experiment import ExperimentDigest

    winner = _scored_experiment("best", 0.6254, "x = 1").model_copy(update={
        "digest": ExperimentDigest(
            techniques=[], score=0.6254,
            what_helped=["threshold tuning: 0.55 -> 0.6254"],
            what_hurt=[], data_insights=[], takeaway="t"),
    })
    assert _dead_ends([winner]) == ""  # the winner's helped items stay out


def test_top_performing_val_claims_trip_the_move_lint() -> None:
    # run 15 i10: "the top-performing model (0.6593)" — a same-fold val number.
    prior = _scored_experiment("best", 0.6254, "model = HGB().fit(Xa, ya)")
    fake = _FakeLLM([
        _plan(False, "bad", "next: model-swap: use the top-performing model (0.6593) as primary."),
        _plan(False, "ok", "next: feature-selection: drop weak one-hots."),
    ])
    d = Supervisor(fake, metric="f1").decide(
        data_summary="d", baseline=_baseline(), history=[prior], carried_best=prior
    )
    assert d.title == "ok"


def test_lever_markers_for_brief_maps_the_named_class() -> None:
    markers = lever_markers_for_brief(
        "so far: best f1=0.60. next: imbalance-or-threshold: set class_weight balanced."
    )
    assert "class_weight" in markers
    # no known class named -> gate stays off
    assert lever_markers_for_brief("next: just try something different") == ()


def test_history_shows_within_session_validation_trail() -> None:
    # a session that printed several validation scores as it iterated — the supervisor
    # must see the trail (incl. the attempts that lost), not just the final score.
    prior = Experiment(
        candidate=Candidate(
            description="iterated session",
            changes={
                "code": "from sklearn.linear_model import LogisticRegression\nLogisticRegression()",
                "cells": [
                    {"source": "agent", "stdout": "Validation F1 score: 0.5800", "error": None},
                    {"source": "agent", "stdout": "validation f1: 0.6100", "error": None},
                    {"source": "agent", "stdout": "Validation F1 score: 0.5900", "error": None},
                ],
            },
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.61}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "refine")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "val tries: 0.5800 -> 0.6100 -> 0.5900" in sent
