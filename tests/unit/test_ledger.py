"""Tests for the tried/untried ledger.

Two dimensions with different failure modes: lever classes are string-matched and
can false-positive, components are AST-extracted and cannot. The tests pin the
distinction, because the whole point of adding the component dimension is that it
answers questions marker matching gets wrong.
"""

from __future__ import annotations

from typing import Any

from iterate.core import ledger
from iterate.core.supervisor import _fallback_move, run_ledger
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

_MARKERS = {
    "encoding": ("targetencoder", "onehotencoder"),
    "class-weight": ("class_weight",),
    "model-swap": ("gradientboosting",),
}
_ORDER = ("encoding", "class-weight", "model-swap")


def _exp(code: str | None) -> Experiment:
    return Experiment(
        candidate=Candidate(
            description="d", changes={"code": code, "cells": []}, rationale="r"
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(
                values={"f1": 0.6}, primary="f1", direction="maximize", n_samples=10
            ),
        ),
    )


def _build(history: list[Experiment], **kw: Any) -> ledger.Ledger:
    return ledger.build(history, lever_markers=_MARKERS, lever_order=_ORDER, **kw)


def test_an_empty_run_has_tried_nothing() -> None:
    led = _build([])
    assert led.tried_levers == frozenset()
    assert led.untried_levers == _ORDER
    assert led.first_untried() == "encoding"
    assert led.experiments == 0


def test_levers_are_marked_tried_from_submitted_code() -> None:
    led = _build([_exp("enc = OneHotEncoder()"), _exp("m = X(class_weight='balanced')")])
    assert led.tried_levers == {"encoding", "class-weight"}
    assert led.untried_levers == ("model-swap",)
    assert led.first_untried() == "model-swap"


def test_untried_order_follows_the_canonical_order_not_discovery() -> None:
    led = _build([_exp("m = X(class_weight='balanced')")])
    assert led.untried_levers == ("encoding", "model-swap")


def test_every_lever_tried_leaves_nothing_to_fall_back_to() -> None:
    led = _build(
        [_exp("OneHotEncoder(); X(class_weight='balanced'); GradientBoostingClassifier()")]
    )
    assert led.untried_levers == ()
    assert led.first_untried() is None


def test_the_normalizer_prevents_a_marker_false_positive() -> None:
    """`histgradientboosting` contains `gradientboosting`; without the caller's
    neutralising pass the default estimator would mark model-swap as tried in every
    session, since each one rebuilds the carried pipeline."""
    code = "m = HistGradientBoostingClassifier()"
    assert "model-swap" in _build([_exp(code)]).tried_levers
    neutralised = _build([_exp(code)], normalize=lambda t: t.replace("histgradientboosting", "hgb"))
    assert "model-swap" not in neutralised.tried_levers


def test_components_are_exact_where_markers_are_fuzzy() -> None:
    """The reason the component dimension exists: it can distinguish two estimators
    whose names contain one another, which substring matching cannot."""
    led = _build([_exp("m = HistGradientBoostingClassifier()")])
    assert led.component_tried("HistGradientBoostingClassifier")
    assert not led.component_tried("GradientBoostingClassifier")


def test_component_matching_is_case_insensitive() -> None:
    led = _build([_exp("enc = OneHotEncoder()")])
    assert led.component_tried("onehotencoder")


def test_components_accumulate_in_first_seen_order_without_duplicates() -> None:
    led = _build([_exp("OneHotEncoder(); SimpleImputer()"), _exp("SimpleImputer()")])
    assert led.tried_components == ("OneHotEncoder", "SimpleImputer")


def test_spec_candidates_without_code_are_skipped_not_counted() -> None:
    led = _build([_exp(None), _exp("enc = OneHotEncoder()")])
    assert led.experiments == 1
    assert led.tried_levers == {"encoding"}


# ─── the supervisor still behaves identically through the ledger ─────────────


def test_fallback_move_is_unchanged_by_the_refactor() -> None:
    """`_fallback_move` now reads the ledger instead of recomputing the tried set
    inline. Same contract: a novel-by-construction move, or None when exhausted."""
    assert _fallback_move([]) is not None
    title, move = _fallback_move([])  # type: ignore[misc]
    assert title.startswith("untried lever: ")
    assert move.startswith("next: ")


def test_run_ledger_uses_the_supervisors_own_marker_vocabulary() -> None:
    led = run_ledger([_exp("enc = OneHotEncoder()")])
    assert led.experiments == 1
    assert led.tried_components == ("OneHotEncoder",)
    assert led.first_untried() is not None


def test_lever_ledger_line_keeps_display_order_not_fallback_order() -> None:
    """`_LEVER_MARKERS` (display) and `_CANONICAL_MOVES` (fallback priority) hold the
    same keys in deliberately different orders. Sharing the tried SET is safe;
    sharing the sequence would silently reorder text in the supervisor's prompt."""
    from iterate.core.supervisor import _CANONICAL_MOVES, _LEVER_MARKERS, _lever_ledger

    assert set(_LEVER_MARKERS) == set(_CANONICAL_MOVES)
    assert list(_LEVER_MARKERS) != list(_CANONICAL_MOVES)

    line = _lever_ledger([_exp("enc = OneHotEncoder()")])
    untried = line.split("NOT yet tried: ")[1].split(", ")
    expected = [lv for lv in _LEVER_MARKERS if lv != "categorical-encoding"]
    assert untried == expected


def test_the_fallback_menu_drops_levers_that_cannot_move_the_metric() -> None:
    """Measured across five datasets: threshold tuning moves f1 by ~0.02 and moves
    average_precision / roc_auc by EXACTLY 0.0000. On a threshold-free metric that
    lever is not weak, it is a no-op, so the harness stops offering it rather than
    trusting a 12B to reason it out."""
    from iterate.core.supervisor import _CANONICAL_MOVES, canonical_moves

    assert canonical_moves("f1") == _CANONICAL_MOVES
    ranking = canonical_moves("average_precision")
    assert "imbalance-or-threshold" not in ranking
    assert "model-swap" in ranking  # levers that DO move ranking survive
    assert len(ranking) == len(_CANONICAL_MOVES) - 1


def test_the_fallback_move_respects_the_metric() -> None:
    from iterate.core.supervisor import _fallback_move

    for _ in range(8):  # walk the whole menu on a threshold-free metric
        got = _fallback_move([], "roc_auc")
        if got is None:
            break
        assert "imbalance-or-threshold" not in got[0]
    assert _fallback_move([], "f1") is not None


def test_a_threshold_lever_brief_is_rejected_on_a_ranking_metric() -> None:
    """The prompt note alone did NOT change gemma4:12b's behaviour — it reached the
    prompt and the model briefed imbalance weighting anyway, on iteration 2, in
    three runs out of three. Guards beat prompt nudges on weak models, so this is a
    guard; the note stays as the explanation the corrective retry needs."""
    from iterate.core.supervisor import dead_lever_reason

    brief = "next: imbalance-or-threshold: train with class_weight='balanced'."
    assert dead_lever_reason(brief, "average_precision") is not None
    assert dead_lever_reason(brief, "roc_auc") is not None
    # the same brief is perfectly good when the metric IS threshold-dependent
    assert dead_lever_reason(brief, "f1") is None
    # and a ranking-moving lever is never rejected
    assert dead_lever_reason("next: model-swap: try XGBoost.", "average_precision") is None
