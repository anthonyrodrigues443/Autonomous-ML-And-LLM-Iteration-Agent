"""The single-ruler contract: what `core.scoring` computes and when it refuses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iterate.core.scoring import (
    AVERAGES,
    CLASSIFICATION_METRICS,
    LABEL_METRICS,
    PANEL,
    PROBA_METRICS,
    REGISTRY,
    REGRESSION_METRICS,
    direction,
    requires_proba,
    resolve_average,
    score,
    task_for_metric,
)

BINARY_TRUE = [0, 1, 0, 1, 1, 0]
BINARY_PRED = [0, 1, 0, 1, 0, 0]
BINARY_PROBA = [0.1, 0.9, 0.2, 0.8, 0.4, 0.3]

MULTI_TRUE = [0, 1, 2, 1, 0, 2]
MULTI_PRED = [0, 1, 2, 1, 0, 1]
MULTI_PROBA = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.2, 0.7, 0.1],
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
]


def test_proba_metrics_are_classification_metrics() -> None:
    assert PROBA_METRICS <= CLASSIFICATION_METRICS
    assert LABEL_METRICS <= CLASSIFICATION_METRICS
    for metric in PROBA_METRICS:
        assert task_for_metric(metric) == "classification"


def test_log_loss_and_brier_minimize_roc_auc_maximizes() -> None:
    assert direction("log_loss") == "minimize"
    assert direction("brier") == "minimize"
    assert direction("roc_auc") == "maximize"


def test_requires_proba_only_for_the_probability_panel() -> None:
    assert all(requires_proba(m) for m in PROBA_METRICS)
    assert not any(requires_proba(m) for m in LABEL_METRICS | REGRESSION_METRICS)


def test_label_panel_unchanged_when_no_probabilities_given() -> None:
    values = score("classification", BINARY_TRUE, BINARY_PRED)
    assert set(values) == LABEL_METRICS


def test_binary_probabilities_add_the_full_panel() -> None:
    values = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    assert set(values) == {n for n, s in PANEL.items() if s.task == "classification"}
    # Label metrics are still computed, so history stays comparable across
    # iterations that did and didn't emit probabilities.
    labels_only = score("classification", BINARY_TRUE, BINARY_PRED)
    for name, value in labels_only.items():
        assert values[name] == value


def test_binary_accepts_a_two_column_probability_matrix() -> None:
    matrix = [[1 - p, p] for p in BINARY_PROBA]
    from_matrix = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=matrix)
    from_column = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    assert from_matrix == pytest.approx(from_column)


def test_multiclass_panel_omits_brier() -> None:
    values = score("classification", MULTI_TRUE, MULTI_PRED, y_proba=MULTI_PROBA)
    assert "roc_auc" in values
    assert "log_loss" in values
    # Multiclass Brier landed after the scikit-learn>=1.5 floor this package ships.
    assert "brier" not in values


def test_confidently_wrong_probabilities_stay_finite() -> None:
    """`Metrics` rejects non-finite values, so an unclipped inf here would sink
    an otherwise-valid experiment. sklearn clips log-loss; assert it stays true."""
    values = score("classification", [0, 1], [0, 1], y_proba=[1.0, 0.0])
    assert math.isfinite(values["log_loss"])
    assert values["log_loss"] > 0


def test_regression_ignores_probabilities() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    assert score("regression", y_true, y_pred, y_proba=[0.5] * 4) == score(
        "regression", y_true, y_pred
    )


def test_regression_panel_is_the_four_known_metrics() -> None:
    values = score("regression", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert set(values) == {n for n, s in PANEL.items() if s.task == "regression"}
    assert values["rmse"] == pytest.approx(math.sqrt(values["mse"]))


@pytest.mark.parametrize(
    "proba",
    [
        [[0.1, 0.2, 0.7]] * 6,  # three columns for a binary target
        np.zeros((6, 3, 2)).tolist(),  # 3-D
    ],
)
def test_malformed_binary_probabilities_raise(proba: object) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        score("classification", BINARY_TRUE, BINARY_PRED, y_proba=proba)


def test_multiclass_probabilities_need_one_column_per_class() -> None:
    with pytest.raises(ValueError, match="one column per class"):
        score("classification", MULTI_TRUE, MULTI_PRED, y_proba=[[0.5, 0.5]] * 6)
    with pytest.raises(ValueError, match="one column per class"):
        score("classification", MULTI_TRUE, MULTI_PRED, y_proba=[0.5] * 6)


def test_binary_string_labels_score_without_probabilities() -> None:
    """Regression: the label panel used to inherit sklearn's pos_label=1 default,
    so any binary string target (the common Yes/No churn encoding) raised."""
    values = score("classification", ["no", "yes", "no", "yes"], ["no", "yes", "no", "no"])
    assert set(values) == LABEL_METRICS
    assert values["recall"] == pytest.approx(0.5)


def test_string_labels_score_against_the_greater_class_as_positive() -> None:
    y_true = ["no", "yes", "no", "yes"]
    y_pred = ["no", "yes", "no", "no"]
    values = score("classification", y_true, y_pred, y_proba=[0.1, 0.9, 0.2, 0.4])
    assert values["roc_auc"] == pytest.approx(1.0)
    assert math.isfinite(values["brier"])


def test_unknown_metric_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="roc_auc"):
        task_for_metric("nonsense")
    with pytest.raises(ValueError, match="roc_auc"):
        direction("nonsense")


# ─── The registry is the single source of truth ──────────────────────────────


def test_every_exported_set_is_derived_from_the_registry() -> None:
    """The bug this guards: the panel, the direction table and the CLI's metric
    names used to be independent lists, and they drifted."""
    assert set(REGISTRY) == CLASSIFICATION_METRICS | REGRESSION_METRICS
    assert not CLASSIFICATION_METRICS & REGRESSION_METRICS
    for name, spec in REGISTRY.items():
        assert task_for_metric(name) == spec.task
        assert direction(name) == spec.direction
        assert requires_proba(name) is spec.needs_proba


def test_every_panel_metric_is_scorable() -> None:
    """A panel row without a working compute would otherwise only surface live."""
    binary = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    regression = score("regression", [1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(binary) | set(regression) == set(PANEL)


def test_every_exported_set_is_derived_from_the_right_table() -> None:
    """Vocabulary sets answer "may this be selected"; LABEL_METRICS answers "what
    does the panel always contain". Conflating them silently changes what a run
    computes, so pin which table each reads."""
    assert set(REGISTRY) == CLASSIFICATION_METRICS | REGRESSION_METRICS
    assert set(PANEL) >= LABEL_METRICS
    assert len(REGISTRY) > len(PANEL)  # the vocabulary is genuinely wider


def test_pr_auc_is_registered_and_maximizes() -> None:
    assert requires_proba("average_precision")
    assert direction("average_precision") == "maximize"
    assert task_for_metric("average_precision") == "classification"


def test_pr_auc_scores_on_both_binary_and_multiclass() -> None:
    """PR-AUC is the metric that matters on an imbalanced target, so it has to
    survive the multiclass path too — where sklearn needs a binarized y_true."""
    binary = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    multi = score("classification", MULTI_TRUE, MULTI_PRED, y_proba=MULTI_PROBA)
    assert 0.0 <= binary["average_precision"] <= 1.0
    assert 0.0 <= multi["average_precision"] <= 1.0


# ─── Configurable averaging ──────────────────────────────────────────────────


def test_average_defaults_preserve_pre_v04_behaviour() -> None:
    assert resolve_average(None, binary=True) == "binary"
    assert resolve_average(None, binary=False) == "macro"


@pytest.mark.parametrize("average", AVERAGES)
def test_every_advertised_average_resolves(average: str) -> None:
    assert resolve_average(average, binary=average == "binary") == average


def test_micro_averaging_changes_the_label_metrics() -> None:
    macro = score("classification", MULTI_TRUE, MULTI_PRED, average="macro")
    micro = score("classification", MULTI_TRUE, MULTI_PRED, average="micro")
    assert macro["f1"] != micro["f1"]


def test_binary_averaging_on_a_multiclass_target_is_refused() -> None:
    with pytest.raises(ValueError, match="two-class"):
        score("classification", MULTI_TRUE, MULTI_PRED, average="binary")


def test_unknown_average_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="weighted"):
        score("classification", BINARY_TRUE, BINARY_PRED, average="nonsense")


def test_averaging_does_not_touch_the_probability_panel() -> None:
    macro = score("classification", MULTI_TRUE, MULTI_PRED, y_proba=MULTI_PROBA, average="macro")
    micro = score("classification", MULTI_TRUE, MULTI_PRED, y_proba=MULTI_PROBA, average="micro")
    for name in PROBA_METRICS & set(macro):
        assert macro[name] == micro[name]


# ─── the derived vocabulary ──────────────────────────────────────────────────


def test_the_vocabulary_is_derived_from_sklearn_not_hand_written() -> None:
    """CANARY. The derivation reads private scorer attributes (_sign, _score_func,
    _kwargs, _response_method). They have been stable for years, but if a sklearn
    upgrade moves them this must fail HERE, in CI, rather than silently shrinking
    a user's metric vocabulary back to the 12 curated ones."""
    from sklearn.metrics import get_scorer_names

    assert len(REGISTRY) > 40, "derivation produced almost nothing — sklearn API moved?"
    for name in ("matthews_corrcoef", "balanced_accuracy", "f1_weighted", "jaccard"):
        assert name in get_scorer_names()
        assert name in REGISTRY


def test_direction_for_derived_metrics_comes_from_sklearns_sign() -> None:
    """The point of deriving: an agent may propose any of these and still cannot
    invert the loop, because it never supplies the direction."""
    assert direction("matthews_corrcoef") == "maximize"
    assert direction("balanced_accuracy") == "maximize"
    # Registered without sklearn's "neg_" prefix: we report the raw quantity and
    # say minimize, rather than printing a positive number under a negative name.
    assert direction("log_loss") == "minimize"
    assert direction("root_mean_squared_error") == "minimize"
    assert direction("mean_absolute_error") == "minimize"
    assert not [n for n in REGISTRY if n.startswith("neg_")]


def test_clustering_scorers_are_excluded_from_the_vocabulary() -> None:
    """sklearn also registers clustering scores. They compare two label
    assignments rather than a prediction against a target, so offering them would
    invite the agent to select something meaningless here."""
    for name in ("rand_score", "v_measure_score", "adjusted_mutual_info_score"):
        assert name not in REGISTRY


def test_a_selected_metric_outside_the_panel_is_computed_on_top() -> None:
    values = score(
        "classification", BINARY_TRUE, BINARY_PRED, include=("matthews_corrcoef",)
    )
    assert "matthews_corrcoef" in values
    assert set(values) >= LABEL_METRICS  # the panel is still there


def test_include_is_ignored_for_the_wrong_task_or_missing_probabilities() -> None:
    regression = score("regression", [1.0, 2.0], [1.1, 1.9], include=("matthews_corrcoef",))
    assert "matthews_corrcoef" not in regression
    no_proba = score("classification", BINARY_TRUE, BINARY_PRED, include=("log_loss",))
    assert "log_loss" not in no_proba


def test_derived_binary_metrics_honour_the_datasets_own_positive_label() -> None:
    """sklearn bakes pos_label=1 into its binary scorers — the exact default that
    made string targets unscorable. The derived compute re-points it."""
    values = score(
        "classification",
        ["no", "yes", "no", "yes"],
        ["no", "yes", "no", "no"],
        include=("jaccard",),
    )
    assert "jaccard" in values


def test_a_derived_probability_metric_scores_from_probabilities() -> None:
    values = score(
        "classification",
        BINARY_TRUE,
        BINARY_PRED,
        y_proba=BINARY_PROBA,
        include=("log_loss",),
    )
    assert values["log_loss"] > 0  # the raw loss, with direction saying minimize


# ─── threshold-free metrics (v0.4 certification finding) ─────────────────────


def test_threshold_free_is_exactly_the_probability_panel() -> None:
    """A metric scored from probabilities has no decision threshold by
    construction, so the predicate needs no new data — only a name."""
    from iterate.core.scoring import requires_proba, threshold_free

    for name in REGISTRY:
        assert threshold_free(name) is requires_proba(name)
    assert threshold_free("average_precision")
    assert threshold_free("roc_auc")
    assert not threshold_free("f1")
    assert not threshold_free("accuracy")


def test_guidance_is_given_only_where_it_changes_the_playbook() -> None:
    """Both v0.4 certification runs picked average_precision and then spent
    iteration 2 on class weighting — the right lever for f1, worth nothing on a
    ranking metric. An f1 run must not carry the note; every line competes for a
    weak model's attention."""
    from iterate.core.scoring import metric_guidance

    ranking = metric_guidance("average_precision")
    assert "RANKED PROBABILITIES" in ranking
    assert "threshold" in ranking.lower()
    # Threshold metrics get the importable-name hint but NOT the ranking advice,
    # which would be wrong for them and is another line competing for attention.
    assert "RANKED PROBABILITIES" not in metric_guidance("f1")
    assert "RANKED PROBABILITIES" not in metric_guidance("rmse")


def test_every_metric_names_an_importable_sklearn_function() -> None:
    """The v0.4 runs burned 16 cells on `from sklearn.metrics import
    average_precision`, which does not exist — the metric NAME is a scorer name,
    the importable symbol is `average_precision_score`. Every name we hand an agent
    must resolve to something it can actually import."""
    import sklearn.metrics as skm

    from iterate.core.scoring import sklearn_function

    missing = []
    for name in REGISTRY:
        func = sklearn_function(name)
        if not func or not hasattr(skm, func):
            missing.append(name)
    assert not missing, f"no importable sklearn function for: {sorted(missing)}"


def test_the_importable_name_reaches_the_guidance() -> None:
    from iterate.core.scoring import metric_guidance

    assert "average_precision_score" in metric_guidance("average_precision")
    assert "f1_score" in metric_guidance("f1")  # threshold metrics get it too
