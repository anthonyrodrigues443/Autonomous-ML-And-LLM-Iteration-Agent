"""Tests for resolving a run's metric and starting model.

The property that matters: the agent can only ever UPGRADE the deterministic
default. Every failure path — an unknown name, the wrong task, a dead network, a
model that will not answer — lands on exactly what v0.3 would have run. Removing an
input must not add a way for a run to fail to start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.core.researcher import Setup
from iterate.core.setup import default_setup, resolve, target_task, validate

if TYPE_CHECKING:
    from pathlib import Path


def _dataset(values: list[object], tmp_path: Path, name: str = "y"):  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {"a": range(len(values)), "b": [i % 7 for i in range(len(values))], name: values}
    )
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return load_csv(path, target=name)


def _binary(tmp_path: Path):  # type: ignore[no-untyped-def]
    return _dataset([0, 1] * 30, tmp_path)


def _regression(tmp_path: Path):  # type: ignore[no-untyped-def]
    return _dataset([i * 1.7 for i in range(60)], tmp_path)


# ─── the deterministic floor ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("values", "task", "metric"),
    [
        ([0, 1] * 30, "classification", "f1"),
        (["yes", "no"] * 30, "classification", "f1"),
        ([0, 1, 2] * 20, "classification", "f1_macro"),
        ([i * 1.7 for i in range(60)], "regression", "rmse"),
    ],
)
def test_the_default_matches_what_v03_would_have_run(
    values: list[object], task: str, metric: str, tmp_path: Path
) -> None:
    ds = _dataset(values, tmp_path)
    assert target_task(ds) == task
    assert default_setup(ds).metric == metric


def test_task_agrees_with_the_split_the_loader_actually_made(tmp_path: Path) -> None:
    """One definition of "is this classification", shared with the loader. Two would
    eventually disagree, and a metric chosen for a task the data was never split for
    is a run that cannot be scored. A float target with few distinct values is the
    case where a second heuristic would most plausibly have diverged."""
    from iterate.adapters.data.tabular import looks_like_classification

    for values in ([0, 1] * 30, ["a", "b"] * 30, [0.0, 1.0] * 30, [i * 1.7 for i in range(60)]):
        ds = _dataset(values, tmp_path)
        expected = "classification" if looks_like_classification(ds.train_target) else "regression"
        assert target_task(ds) == expected


# ─── explicit always wins ────────────────────────────────────────────────────


def test_an_explicit_metric_is_used_as_given(tmp_path: Path) -> None:
    got = resolve(_binary(tmp_path), explicit="roc_auc")
    assert got.metric == "roc_auc"
    assert not got.chosen_by_agent
    assert got.render() == ""  # nothing to announce; the user chose it


def test_an_explicit_metric_beats_a_proposal(tmp_path: Path) -> None:
    got = resolve(
        _binary(tmp_path), explicit="f1", proposed=Setup(metric="average_precision", why="w")
    )
    assert got.metric == "f1"


# ─── the agent path ──────────────────────────────────────────────────────────


def test_a_valid_proposal_is_taken_and_announced(tmp_path: Path) -> None:
    got = resolve(
        _binary(tmp_path),
        proposed=Setup(
            metric="average_precision",
            starting_model="HistGradientBoostingClassifier",
            why="target is 73/27 imbalanced",
        ),
    )
    assert got.metric == "average_precision"
    assert got.chosen_by_agent
    assert "average_precision" in got.render()
    assert "73/27" in got.render()


def test_an_unknown_metric_falls_back(tmp_path: Path) -> None:
    got = resolve(_binary(tmp_path), proposed=Setup(metric="best_metric_ever"))
    assert got.metric == "f1"
    assert not got.chosen_by_agent
    assert "not a known metric" in got.why


def test_a_regression_metric_on_a_label_falls_back(tmp_path: Path) -> None:
    """The failure that would otherwise surface as a crash at scoring time."""
    got = resolve(_binary(tmp_path), proposed=Setup(metric="rmse"))
    assert got.metric == "f1"
    assert "regression metric" in got.why


def test_a_classification_metric_on_a_continuous_target_falls_back(tmp_path: Path) -> None:
    got = resolve(_regression(tmp_path), proposed=Setup(metric="f1"))
    assert got.metric == "rmse"
    assert not got.chosen_by_agent


def test_no_proposal_falls_back_silently(tmp_path: Path) -> None:
    got = resolve(_binary(tmp_path), proposed=None)
    assert got.metric == "f1"
    assert got.why == ""


def test_an_empty_metric_falls_back(tmp_path: Path) -> None:
    assert resolve(_binary(tmp_path), proposed=Setup(metric="   ")).metric == "f1"


# ─── the starting model is a weaker claim than the metric ────────────────────


def test_a_junk_starting_model_falls_back_without_losing_the_metric(tmp_path: Path) -> None:
    """The agent rewrites the model every iteration anyway, so a bad one costs one
    experiment — it must not also cost the metric choice."""
    got = resolve(
        _binary(tmp_path),
        proposed=Setup(metric="average_precision", starting_model="!!! not a class !!!"),
    )
    assert got.metric == "average_precision"
    assert got.starting_model.endswith("HistGradientBoostingClassifier")


def test_a_plausible_starting_model_is_kept(tmp_path: Path) -> None:
    got = resolve(
        _binary(tmp_path), proposed=Setup(metric="f1", starting_model="xgboost.XGBClassifier")
    )
    assert got.starting_model == "xgboost.XGBClassifier"


# ─── validation is the guard the whole dial rests on ─────────────────────────


def test_validate_names_the_reason(tmp_path: Path) -> None:
    ds = _binary(tmp_path)
    assert validate("f1", ds) is None
    assert validate("average_precision", ds) is None
    assert "not a known metric" in (validate("nope", ds) or "")
    assert "regression metric" in (validate("rmse", ds) or "")


def test_every_registry_metric_of_the_right_task_validates(tmp_path: Path) -> None:
    """The vocabulary and the validator must agree, or the agent gets told its
    legal choices are illegal."""
    from iterate.core.scoring import CLASSIFICATION_METRICS

    ds = _binary(tmp_path)
    for name in CLASSIFICATION_METRICS:
        assert validate(name, ds) is None, name
