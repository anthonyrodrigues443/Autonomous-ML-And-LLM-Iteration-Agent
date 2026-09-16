"""Tests for the vision target with no torch: a fake runner stands in for the
backbone, so CI covers the recipe, the budget, the ruler, the seal and the code job
for classes and for numbers without a download or a GPU."""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from evals.config import REPO_ROOT
from iterate.adapters.compute.base import SupportsCodeGen
from iterate.adapters.compute.runner import LocalCodeRunner, RunResult
from iterate.adapters.data.images import Prepared, prepare_images
from iterate.adapters.data.tabular import load_csv, load_split
from iterate.schemas.experiment import Candidate
from iterate.targets import dl
from iterate.targets.base import BenchmarkTarget
from iterate.targets.dl import DLModelTarget, FitJob, FitReport, Recipe, RecipeError

pytestmark = pytest.mark.unit


def _write(root: Path, classes: int, per_class: int, *, offset: int = 0) -> list[dict[str, Any]]:
    """One colour per class, one shade per image, so every file's bytes differ."""
    rows = []
    for k in range(classes):
        for i in range(per_class):
            path = root / "images" / f"{k:02d}_{offset + i:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            colour = (20 * k % 256, (37 * k + 60) % 256, (7 * (offset + i) + 90) % 256)
            Image.new("RGB", (16, 16), colour).save(path)
            rows.append({"image": f"images/{path.name}", "label": k})
    return rows


def _write_labels(
    root: Path, labels: list[int], per_class: int, *, offset: int = 0
) -> list[dict[str, Any]]:
    rows = []
    for k in labels:
        for i in range(per_class):
            path = root / "images" / f"{k:02d}_{offset + i:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            colour = (20 * k % 256, (37 * k + 60) % 256, (7 * (offset + i) + 90) % 256)
            Image.new("RGB", (16, 16), colour).save(path)
            rows.append({"image": f"images/{path.name}", "label": k})
    return rows


def _prepared(tmp_path: Path, *, classes: int = 3, per_class: int = 10) -> Prepared:
    root = tmp_path / "data"
    pd.DataFrame(_write(root, classes, per_class)).to_csv(root / "data.csv", index=False)
    loaded = load_csv(root / "data.csv", target="label", task="classification")
    return prepare_images(loaded, root / "data.csv", into=tmp_path / "cache")


def _numbers(tmp_path: Path, rows: int = 30) -> Prepared:
    """A fractional label that rises with the red and green channels."""
    root = tmp_path / "data"
    records = []
    for i in range(rows):
        path = root / "images" / f"{i:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (8 * i, (5 * i + 40) % 256, 90)).save(path)
        records.append({"image": f"images/{path.name}", "label": 1.5 * i + 0.25})
    pd.DataFrame(records).to_csv(root / "data.csv", index=False)
    loaded = load_csv(root / "data.csv", target="label", task="regression")
    return prepare_images(loaded, root / "data.csv", into=tmp_path / "cache")


def _ramp(n: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


class FakeRunner:
    """Channel means stand in for a backbone; a fine-tune is a nearest-centroid fit on
    the same features, so every class colour is told apart. A number comes back as a
    fixed ramp of standardised outputs, so its score can be worked out by hand."""

    device = "cpu"

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.jobs: list[FitJob] = []
        self._fail = fail

    def embed(self, pixels: np.ndarray, *, backbone: str) -> np.ndarray:
        return pixels.reshape(len(pixels), 3, -1).mean(axis=2).astype(np.float32)

    def fit(self, job: FitJob) -> FitReport:
        self.jobs.append(job)
        if self._fail is not None:
            raise self._fail
        if job.task == "regression":
            return FitReport(_ramp(len(job.holdout)), job.recipe.epochs, job.recipe.epochs)
        train = self.embed(job.train, backbone="")
        held = self.embed(job.holdout, backbone="")
        centres = np.stack([train[job.labels == c].mean(axis=0) for c in range(job.outputs)])
        logits = -((held[:, None, :] - centres[None]) ** 2).sum(-1)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        return FitReport(p / p.sum(axis=1, keepdims=True), job.recipe.epochs, job.recipe.epochs)


def _target(prepared: Prepared, *, metric: str = "accuracy", runner: Any = None) -> DLModelTarget:
    return DLModelTarget(
        prepared.dataset,
        column=prepared.column.column,
        metric=metric,
        runner=runner or FakeRunner(),
        image_size=prepared.image_size,
        profile=prepared.profile,
    )


def _cand(changes: dict[str, Any]) -> Candidate:
    return Candidate(description="d", changes=changes, rationale="r")


# ─── the contract ────────────────────────────────────────────────────────────


def test_the_target_meets_both_contracts(tmp_path: Path) -> None:
    target = _target(_prepared(tmp_path))
    assert isinstance(target, BenchmarkTarget)
    assert isinstance(target, SupportsCodeGen)


def test_the_baseline_scores_the_holdout_and_records_its_recipe(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    result = _target(prepared).baseline()
    assert result.error is None
    assert result.metrics is not None
    assert result.metrics.primary_value == 1.0
    assert result.metrics.n_samples == prepared.dataset.n_test
    recorded = json.loads(result.artifacts[dl.RECIPE_JSON])
    assert (recorded["backbone"], recorded["unfreeze"], recorded["epochs_run"]) == (
        "simple_cnn",
        "all",
        20,
    )


def test_the_baseline_recipe_is_the_small_cnn_from_zero_with_basic_prep() -> None:
    expected = Recipe(
        backbone="simple_cnn",
        image_size=None,
        unfreeze="all",
        epochs=20,
        batch_size=64,
        lr=1e-3,
        optimizer="adamw",
        schedule="cosine",
        augment="none",
        label_smoothing=0.0,
        head_init="random",
        seed=42,
    )
    assert expected == dl.BASELINE
    assert dl.BASELINE_SIZE == 64


@pytest.mark.parametrize(
    ("numbers", "metric", "outputs"), [(False, "accuracy", 3), (True, "rmse", 1)]
)
def test_the_baseline_is_one_fixed_fit_with_no_deadline(
    tmp_path: Path, numbers: bool, metric: str, outputs: int
) -> None:
    prepared = _numbers(tmp_path) if numbers else _prepared(tmp_path)
    runner = FakeRunner()
    assert _target(prepared, metric=metric, runner=runner).baseline().error is None
    (job,) = runner.jobs
    assert job.recipe == replace(dl.BASELINE, image_size=prepared.image_size)
    assert (job.fixed, job.deadline) == (True, math.inf)
    assert (job.task, job.outputs) == (prepared.dataset.task, outputs)


def test_the_baseline_size_is_capped_at_64_and_a_smaller_one_is_never_upscaled(
    tmp_path: Path,
) -> None:
    dataset = _prepared(tmp_path).dataset
    for size, expected in ((160, 64), (64, 64), (48, 48), (32, 32)):
        runner = FakeRunner()
        DLModelTarget(
            dataset, column="image", metric="accuracy", runner=runner, image_size=size
        ).baseline()
        assert runner.jobs[0].recipe.image_size == expected


def test_an_agent_try_has_a_deadline_and_is_not_fixed(tmp_path: Path) -> None:
    runner = FakeRunner()
    changes = {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 30}
    assert _target(_prepared(tmp_path), runner=runner).run(_cand(changes)).error is None
    (job,) = runner.jobs
    assert not job.fixed
    assert math.isfinite(job.deadline)


def test_a_fine_tune_runs_through_the_runner_and_records_its_epochs(tmp_path: Path) -> None:
    runner = FakeRunner()
    result = _target(_prepared(tmp_path), runner=runner).run(
        _cand({"unfreeze": "all", "epochs": 2})
    )
    assert result.error is None
    assert result.metrics is not None
    assert result.metrics.primary_value == 1.0
    assert json.loads(result.artifacts[dl.RECIPE_JSON])["epochs_planned"] == 2
    assert len(runner.jobs) == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"unfreeze": "all", "epochs": "3"}, "epochs must be int, got '3'"),
        ({"unfreeze": "all", "epochs": 3, "lr": True}, "lr must be float, got True"),
        ({"unfreeze": "all", "epochs": 3, "seed": -1}, "seed=-1 is outside"),
        ({"unfreeze": "all", "epochs": 3, "seed": 2**64}, "is outside 0 to"),
        ({"backbone": "efficientnet_b0"}, "backbone must be one of"),
        ({"unfreeze": "all", "epochs": 0}, "the probe takes 0 epochs"),
        ({"unfreeze": "all", "epochs": 3, "image_size": 16}, "image_size=16 is outside"),
        ({"code": "def train_and_predict(): ..."}, "unknown recipe keys ['code']"),
        ({"unfreeze": "all", "epochs": 3, "batch_size": 0}, "batch_size=0 is outside"),
        ({"unfreeze": "all", "epochs": 3, "batch_size": 257}, "batch_size=257 is outside"),
        ({"unfreeze": "all", "epochs": 3, "lr": 0}, "lr=0 is outside"),
        ({"unfreeze": "all", "epochs": 3, "lr": 2.0}, "lr=2.0 is outside"),
        ({"unfreeze": "all", "epochs": 13}, "epochs=13 is outside 0 to 12 for a pretrained"),
        ({"unfreeze": "head", "epochs": 13}, "epochs=13 is outside 0 to 12 for a pretrained"),
        (
            {"unfreeze": "all", "epochs": 31},
            "epochs=31 is outside 0 to 12 for a pretrained backbone; lower epochs, or use simple_cnn",
        ),
        (
            {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 31},
            "epochs=31 is outside 0 to 30",
        ),
        ({"backbone": "simple_cnn"}, "simple_cnn trains from zero: set unfreeze to all"),
        (
            {"backbone": "simple_cnn", "unfreeze": "head", "epochs": 3},
            "simple_cnn trains from zero: set unfreeze to all",
        ),
        (
            {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 0},
            "simple_cnn trains from zero: set unfreeze to all with 1 to 30 epochs",
        ),
        (
            {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 3, "head_init": "probe"},
            "simple_cnn has no probe to start its head from; set head_init to random",
        ),
        (
            {"unfreeze": "all", "epochs": 3, "label_smoothing": 0.5},
            "label_smoothing=0.5 is outside",
        ),
        ({"unfreeze": "none", "epochs": 3}, "the probe takes 0 epochs"),
    ],
)
def test_a_recipe_is_refused_by_name_as_a_failed_result(
    tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    runner = FakeRunner()
    result = _target(_prepared(tmp_path), runner=runner).run(_cand(changes))
    assert result.metrics is None
    assert result.error is not None
    assert result.error.startswith("recipe refused: ")
    assert reason in result.error
    assert runner.jobs == []


def test_a_path_that_is_not_a_byte_named_copy_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    pd.DataFrame(_write(root, 3, 6)).to_csv(root / "data.csv", index=False)
    raw = load_csv(root / "data.csv", target="label", task="classification")
    with pytest.raises(ValueError, match="must hold the byte-named copies"):
        DLModelTarget(raw, column="image", metric="accuracy", runner=FakeRunner())


def test_a_metric_for_the_other_kind_of_label_is_refused_both_ways(tmp_path: Path) -> None:
    classes = _prepared(tmp_path / "classes")
    with pytest.raises(
        ValueError, match="rmse is a regression metric and the labels read as classification"
    ):
        _target(classes, metric="rmse")
    numbers = _numbers(tmp_path / "numbers")
    with pytest.raises(
        ValueError, match="accuracy is a classification metric and the labels read as regression"
    ):
        _target(numbers, metric="accuracy")
    relabelled = replace(classes.dataset, task="regression")
    with pytest.raises(ValueError, match="accuracy is a classification metric"):
        DLModelTarget(relabelled, column="image", metric="accuracy", runner=FakeRunner())


# ─── numbers ─────────────────────────────────────────────────────────────────


def test_a_number_is_fitted_standardised_and_scored_in_its_own_units(tmp_path: Path) -> None:
    prepared = _numbers(tmp_path)
    runner = FakeRunner()
    changes = {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 2}
    result = _target(prepared, metric="rmse", runner=runner).run(_cand(changes))
    assert result.error is None
    assert result.metrics is not None
    (job,) = runner.jobs
    train = prepared.dataset.train_target.to_numpy(dtype=float)
    truth = prepared.dataset.test_target.to_numpy(dtype=float)
    assert job.labels.dtype == np.float32
    np.testing.assert_allclose(job.labels, (train - train.mean()) / train.std(), atol=1e-6)
    predicted = _ramp(len(truth)) * train.std() + train.mean()
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(predicted, truth, strict=True)) / len(truth))
    assert result.metrics.primary == "rmse"
    assert result.metrics.direction == "minimize"
    assert result.metrics.primary_value == pytest.approx(rmse)
    assert {"rmse", "r2", "pearson", "spearman"} <= set(result.metrics.values)
    assert result.metrics.values["spearman"] == pytest.approx(
        pd.Series(predicted).corr(pd.Series(truth), method="spearman")
    )
    recorded = json.loads(result.artifacts[dl.RECIPE_JSON])
    assert (recorded["epochs_planned"], recorded["epochs_run"]) == (2, 2)


@pytest.mark.parametrize(("bad_train", "bad_held"), [(3, 1), (0, 2)])
def test_labels_that_are_not_numbers_are_refused_with_both_counts(
    tmp_path: Path, bad_train: int, bad_held: int
) -> None:
    dataset = _numbers(tmp_path).dataset
    train = dataset.train_target.astype(object)
    train.iloc[:bad_train] = "n/a"
    held = dataset.test_target.astype(object)
    held.iloc[:bad_held] = None
    broken = replace(dataset, train_target=train, test_target=held)
    with pytest.raises(
        ValueError, match=f"{bad_train} training and {bad_held} holdout labels are not numbers"
    ):
        DLModelTarget(broken, column="image", metric="rmse", runner=FakeRunner())


def test_text_in_the_training_labels_reaches_the_counted_refusal_through_prepare(
    tmp_path: Path,
) -> None:
    _numbers(tmp_path)
    root = tmp_path / "data"
    frame = pd.read_csv(root / "data.csv").astype({"label": object})
    frame.loc[[0, 1], "label"] = "abc"
    frame.iloc[:24].to_csv(root / "train.csv", index=False)
    frame.iloc[24:].to_csv(root / "holdout.csv", index=False)
    split = load_split(root / "train.csv", root / "holdout.csv", target="label", task="regression")
    prepared = prepare_images(split, root / "train.csv", into=tmp_path / "split-cache")
    with pytest.raises(ValueError, match="2 training and 0 holdout labels are not numbers"):
        _target(prepared, metric="rmse")


@pytest.mark.parametrize("step", [1e306, 1e160])
def test_labels_too_large_to_standardise_are_refused_before_any_fit(
    tmp_path: Path, step: float
) -> None:
    dataset = _numbers(tmp_path).dataset
    huge = replace(dataset, train_target=pd.Series(np.arange(len(dataset.train_target)) * step))
    runner = FakeRunner()
    with pytest.raises(ValueError, match="too large to standardise"):
        DLModelTarget(huge, column="image", metric="rmse", runner=runner)
    assert runner.jobs == []


def test_a_label_that_is_the_same_number_everywhere_is_refused(tmp_path: Path) -> None:
    dataset = _numbers(tmp_path).dataset
    same = replace(dataset, train_target=dataset.train_target * 0 + 7.5)
    with pytest.raises(ValueError, match="every training label is the same number"):
        DLModelTarget(same, column="image", metric="rmse", runner=FakeRunner())


def test_label_smoothing_is_refused_for_a_number_and_kept_for_classes(tmp_path: Path) -> None:
    changes = {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 2, "label_smoothing": 0.1}
    runner = FakeRunner()
    refused = _target(_numbers(tmp_path / "numbers"), metric="rmse", runner=runner).run(
        _cand(changes)
    )
    assert refused.error == (
        "recipe refused: label_smoothing is for classes; set it to 0 to predict a number"
    )
    assert runner.jobs == []
    assert _target(_prepared(tmp_path / "classes")).run(_cand(changes)).error is None


class NaNRunner(FakeRunner):
    def fit(self, job: FitJob) -> FitReport:
        report = super().fit(job)
        outputs = report.outputs.copy()
        outputs[:1] = np.nan
        outputs[1:2] = np.inf
        return FitReport(outputs, report.epochs_planned, report.epochs_run)


def test_numbers_that_are_not_finite_are_a_failed_result_that_counts_them(tmp_path: Path) -> None:
    prepared = _numbers(tmp_path)
    result = _target(prepared, metric="mae", runner=NaNRunner()).baseline()
    assert result.metrics is None
    assert result.error == f"mae needs finite predictions; 2 of {prepared.dataset.n_test} are not"


def test_the_probe_of_a_number_is_a_ridge_fit_whose_head_predicts_the_same(
    tmp_path: Path,
) -> None:
    prepared = _numbers(tmp_path)
    target = _target(prepared, metric="rmse")
    result = target.run(_cand({"unfreeze": "none"}))
    assert result.error is None
    assert result.metrics is not None
    _scaler, model, out = target._probes[("resnet18", prepared.image_size)]
    assert isinstance(model, Ridge)
    assert model.alpha == 1.0
    train_pixels, held_pixels = target._decoded(prepared.image_size)
    features = FakeRunner().embed(train_pixels, backbone="resnet18")
    held = FakeRunner().embed(held_pixels, backbone="resnet18")
    standard = StandardScaler().fit(features)
    expected = Ridge(alpha=1.0).fit(standard.transform(features), target._labels)
    np.testing.assert_allclose(out, expected.predict(standard.transform(held)), rtol=1e-5)
    train = prepared.dataset.train_target.to_numpy(dtype=float)
    truth = prepared.dataset.test_target.to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((out * train.std() + train.mean() - truth) ** 2)))
    assert result.metrics.primary_value == pytest.approx(rmse)
    weight, bias = target._probe_head("resnet18", prepared.image_size)
    assert (weight.shape, bias.shape) == ((1, 3), (1,))
    np.testing.assert_allclose((held @ weight.T + bias)[:, 0], out, rtol=1e-4, atol=1e-4)


def test_simple_cnn_can_neither_embed_nor_take_a_copied_head() -> None:
    head = (np.zeros((3, 128), np.float32), np.zeros(3, np.float32))
    for outputs, given in ((None, None), (3, head)):
        with pytest.raises(RecipeError, match="simple_cnn has no pretrained features"):
            dl._build(None, "simple_cnn", outputs, given)


def test_the_meta_names_the_task_the_spread_the_baseline_and_every_backbone(
    tmp_path: Path,
) -> None:
    classes = json.loads(_target(_prepared(tmp_path / "classes")).meta_json())
    prepared = _numbers(tmp_path / "numbers")
    numbers = json.loads(_target(prepared, metric="rmse").meta_json())
    assert (classes["task"], classes["task_kind"]) == ("classification", "classification")
    assert (classes["classes"], classes["target_spread"]) == ([0, 1, 2], None)
    assert (numbers["task"], numbers["task_kind"]) == ("regression", "regression")
    assert numbers["classes"] is None
    assert prepared.profile.target_spread is not None
    assert numbers["target_spread"] == pytest.approx(list(prepared.profile.target_spread))
    train = prepared.dataset.train_target
    assert numbers["target_spread"] == pytest.approx(
        [train.mean(), train.std(), train.min(), train.max()]
    )
    for meta in (classes, numbers):
        assert meta["baseline"] == asdict(replace(dl.BASELINE, image_size=32))
        assert meta["backbones"] == ["resnet18", "resnet50", "convnext_tiny", "simple_cnn"]


# ─── one ruler ───────────────────────────────────────────────────────────────


def test_integer_classes_score_the_same_through_run_and_the_code_job(tmp_path: Path) -> None:
    """Twelve integer classes: sorted as text, 10 and 11 would sit between 1 and 2,
    and the same probabilities scored 1.0 one way and 0.60 the other."""
    prepared = _prepared(tmp_path, classes=12, per_class=8)
    target = _target(prepared, metric="roc_auc")
    by_run = target.run(_cand({"unfreeze": "none"}))
    assert by_run.metrics is not None
    classes = json.loads(target.meta_json())["classes"]
    assert classes == list(range(12))
    proba = target._probes[("resnet18", prepared.image_size)][2]
    predictions = "\n".join(str(classes[i]) for i in proba.argmax(1)).encode()
    probabilities = "\n".join(",".join(f"{v:.10f}" for v in row) for row in proba).encode()
    outputs = {"predictions.csv": predictions, "probabilities.csv": probabilities}
    by_code = target.score_code_job(
        RunResult(stdout="", stderr="", exit_code=0, outputs=outputs), "code"
    )
    assert by_code.metrics is not None
    for name in ("roc_auc", "accuracy", "log_loss"):
        assert by_code.metrics.values[name] == pytest.approx(by_run.metrics.values[name], abs=1e-6)


# ─── the seal ────────────────────────────────────────────────────────────────


def _user_split_with_a_holdout_only_class(tmp_path: Path) -> Prepared:
    root = tmp_path / "data"
    pd.DataFrame(_write(root, 3, 8)).to_csv(root / "train.csv", index=False)
    pd.DataFrame(_write(root, 4, 3, offset=100)).to_csv(root / "holdout.csv", index=False)
    loaded = load_split(
        root / "train.csv", root / "holdout.csv", target="label", task="classification"
    )
    return prepare_images(loaded, root / "train.csv", into=tmp_path / "cache")


def test_the_runner_sees_training_labels_and_holdout_pixels_only(tmp_path: Path) -> None:
    prepared = _user_split_with_a_holdout_only_class(tmp_path)
    runner = FakeRunner()
    target = _target(prepared, runner=runner)
    assert json.loads(target.meta_json())["classes"] == [0, 1, 2]
    assert json.loads(target.meta_json())["family"] == "vision"
    result = target.run(_cand({"unfreeze": "all", "epochs": 1}))
    assert result.error is None
    (job,) = runner.jobs
    assert len(job.labels) == prepared.dataset.n_train
    assert set(job.labels.tolist()) == {0, 1, 2}
    assert len(job.holdout) == prepared.dataset.n_test
    assert job.outputs == 3


def test_a_probability_metric_with_a_holdout_only_class_fails_with_the_reason(
    tmp_path: Path,
) -> None:
    prepared = _user_split_with_a_holdout_only_class(tmp_path)
    result = _target(prepared, metric="roc_auc").baseline()
    assert result.metrics is None
    assert result.error == "roc_auc needs one probability column per holdout class"


# ─── out of memory ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("MPS backend out of memory (MPS allocated: 1.75 GiB, max allowed: 1.78 GiB)", "oom"),
        ("CUDA out of memory. Tried to allocate 2.00 GiB", "oom"),
        (
            "[enforce fail at alloc_cpu.cpp:135] DefaultCPUAllocator: can't allocate memory: "
            "you tried to allocate 1125899906842624 bytes. Error code 12 (Cannot allocate memory)",
            "oom",
        ),
        ("Invalid buffer size: 372.53 GiB", "buffer"),
        ("mat1 and mat2 shapes cannot be multiplied", None),
    ],
)
def test_out_of_memory_is_read_from_the_text(text: str, kind: str | None) -> None:
    assert dl.oom_kind(RuntimeError(text)) == kind
    assert dl.oom_kind(ValueError(text)) is None


def test_an_out_of_memory_error_is_raised_without_its_chain() -> None:
    def boom() -> None:
        raise RuntimeError("MPS backend out of memory (MPS allocated: 1 GiB)")

    with pytest.raises(dl.DeviceOutOfMemoryError) as caught:
        dl._guarded("cpu", boom, "batch_size=64, image_size=224")
    assert "halve one of them" in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_a_buffer_too_large_is_not_retryable_and_other_errors_propagate() -> None:
    def buffer() -> None:
        raise RuntimeError("Invalid buffer size: 372.53 GiB")

    def other() -> None:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

    with pytest.raises(dl.DeviceOutOfMemoryError, match="not retryable"):
        dl._guarded("cpu", buffer, "image_size=384")
    with pytest.raises(RuntimeError, match="shapes"):
        dl._guarded("cpu", other, "x")


def test_a_device_failure_in_a_fit_is_a_failed_result(tmp_path: Path) -> None:
    failure = dl.DeviceOutOfMemoryError("out of memory on mps at batch_size=64: halve one of them")
    result = _target(_prepared(tmp_path), runner=FakeRunner(fail=failure)).run(
        _cand({"unfreeze": "all", "epochs": 1})
    )
    assert result.metrics is None
    assert result.error == "out of memory on mps at batch_size=64: halve one of them"


# ─── the budget ──────────────────────────────────────────────────────────────


def test_epoch_one_alone_over_the_budget_is_refused_with_advice() -> None:
    with pytest.raises(RecipeError, match="halve image_size or use resnet18"):
        dl.plan_epochs(3, 600.0, 540.0)
    with pytest.raises(RecipeError, match=r"are left; halve image_size$"):
        dl.plan_epochs(3, 600.0, 540.0, backbone="simple_cnn")


def test_the_plan_is_the_whole_epochs_that_fit() -> None:
    assert dl.plan_epochs(5, 120.0, 500.0) == 4
    assert dl.plan_epochs(3, 21.0, 500.0) == 3


def test_an_epoch_the_deadline_cuts_is_not_counted_and_the_log_says_so() -> None:
    logs: list[str] = []
    summaries = iter(["loss=1.0000 train_acc=0.5000", None])
    assert dl.run_epochs(lambda _epoch: next(summaries), 3, logs.append) == 1
    assert logs[0].startswith("epoch 1/3 loss=1.0000")
    assert logs[1] == "stopped in epoch 2/3: the fit budget ran out"


def test_pixels_over_a_quarter_of_ram_are_refused_with_their_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "_ram_bytes", lambda: 100_000)

    def never(paths: list[str], size: int) -> np.ndarray:
        raise AssertionError("decoded before the RAM check")

    monkeypatch.setattr(dl, "decode", never)
    result = _target(_prepared(tmp_path)).baseline()
    assert result.error is not None
    assert "GiB decoded, over a quarter of this machine's RAM; lower image_size" in result.error


def test_only_one_decoded_size_is_kept(tmp_path: Path) -> None:
    target = _target(_prepared(tmp_path))
    target.baseline()
    target.run(_cand({"image_size": 64}))
    assert list(target._pixels) == [64]


@pytest.mark.parametrize("classes", [2, 4])
def test_the_probe_head_predicts_what_the_probe_predicts(tmp_path: Path, classes: int) -> None:
    prepared = _prepared(tmp_path, classes=classes, per_class=10)
    target = _target(prepared)
    target.baseline()
    weight, bias = target._probe_head("resnet18", prepared.image_size)
    _, held = target._decoded(prepared.image_size)
    logits = FakeRunner().embed(held, backbone="resnet18") @ weight.T + bias
    probe = target._probes[("resnet18", prepared.image_size)][2]
    assert (logits.argmax(1) == probe.argmax(1)).all()


# ─── the code job ────────────────────────────────────────────────────────────

_MAJORITY = (
    "def train_and_predict(X_train, y_train, X_holdout):\n"
    "    top = y_train.mode()[0]\n"
    "    return [top] * len(X_holdout)\n"
)


def test_the_code_job_hands_over_paths_without_labels_and_the_vision_meta(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    job = _target(prepared).build_code_job(_cand({"code": _MAJORITY}))
    holdout = pd.read_csv(io.BytesIO(job.inputs["holdout.csv"]))
    assert list(holdout.columns) == ["image"]
    assert len(holdout) == prepared.dataset.n_test
    meta = json.loads(job.inputs["meta.json"])
    assert meta["family"] == "vision"
    assert meta["classes"] == [0, 1, 2]
    assert job.outputs == ["predictions.csv", "probabilities.csv"]


def test_a_majority_answer_scores_through_the_local_runner(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    target = _target(prepared)
    job = target.build_code_job(_cand({"code": _MAJORITY}))
    run = LocalCodeRunner().run(job.script, inputs=job.inputs, outputs=job.outputs, timeout=60)
    assert run.succeeded, run.stderr
    result = target.score_code_job(run, "majority")
    assert result.metrics is not None
    assert result.metrics.n_samples == prepared.dataset.n_test
    assert 0.0 < result.metrics.primary_value < 1.0


def test_a_failed_script_is_a_failed_result(tmp_path: Path) -> None:
    result = _target(_prepared(tmp_path)).score_code_job(
        RunResult(stdout="", stderr="Traceback: boom", exit_code=1, outputs={}), "x"
    )
    assert result.metrics is None
    assert result.error is not None
    assert result.error.startswith("code script failed")


# ─── imports ─────────────────────────────────────────────────────────────────


def test_the_target_and_the_sweep_import_without_torch() -> None:
    code = (
        "import sys\n"
        "import iterate.targets.dl, evals.vision_ceilings\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "False"


# ─── what the review forced ──────────────────────────────────────────────────


def test_an_int_for_a_float_field_is_accepted(tmp_path: Path) -> None:
    changes = {"unfreeze": "all", "epochs": 1, "lr": 1, "label_smoothing": 0}
    assert _target(_prepared(tmp_path)).run(_cand(changes)).error is None


def test_either_half_of_the_copy_guard_refuses_on_its_own(tmp_path: Path) -> None:
    dataset = _prepared(tmp_path).dataset
    renamed = dataset.test_features.copy()
    first = Path(str(renamed["image"].iloc[0]))
    renamed.iloc[0, renamed.columns.get_loc("image")] = str(first.with_name("cat_001.png"))
    for broken in (replace(dataset, test_features=renamed), replace(dataset, data_hash="0" * 16)):
        with pytest.raises(ValueError, match="must hold the byte-named copies"):
            DLModelTarget(broken, column="image", metric="accuracy", runner=FakeRunner())


def test_one_training_class_or_fractional_labels_are_refused(tmp_path: Path) -> None:
    dataset = _prepared(tmp_path).dataset
    one = replace(dataset, train_target=dataset.train_target * 0)
    with pytest.raises(ValueError, match="at least two training classes"):
        DLModelTarget(one, column="image", metric="accuracy", runner=FakeRunner())
    halves = replace(dataset, train_target=dataset.train_target.astype(float) + 0.5)
    with pytest.raises(ValueError, match="fractional labels"):
        DLModelTarget(halves, column="image", metric="accuracy", runner=FakeRunner())


def test_a_holdout_with_as_many_classes_but_another_set_cannot_take_probabilities(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    pd.DataFrame(_write_labels(root, [0, 1, 2], 8)).to_csv(root / "train.csv", index=False)
    pd.DataFrame(_write_labels(root, [0, 1, 3], 3, offset=100)).to_csv(
        root / "holdout.csv", index=False
    )
    loaded = load_split(
        root / "train.csv", root / "holdout.csv", target="label", task="classification"
    )
    prepared = prepare_images(loaded, root / "train.csv", into=tmp_path / "cache")
    failed = _target(prepared, metric="roc_auc").baseline()
    assert failed.error == "roc_auc needs one probability column per holdout class"
    assert _target(prepared).baseline().error is None


def test_probabilities_that_are_not_finite_fail_a_probability_metric_and_not_accuracy(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    fine_tune = _cand({"unfreeze": "all", "epochs": 1})
    failed = _target(prepared, metric="roc_auc", runner=NaNRunner()).run(fine_tune)
    assert failed.error == "roc_auc needs finite probabilities"
    scored = _target(prepared, runner=NaNRunner()).run(fine_tune)
    assert scored.error is None
    assert scored.metrics is not None


def test_the_budget_counts_from_the_top_of_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    clock = {"now": 1000.0}
    monkeypatch.setattr(dl, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    real_decode = dl.decode

    def slow_decode(paths: list[str], size: int) -> np.ndarray:
        clock["now"] += 100.0
        return real_decode(paths, size)

    monkeypatch.setattr(dl, "decode", slow_decode)
    runner = FakeRunner()
    target = DLModelTarget(
        prepared.dataset,
        column="image",
        metric="accuracy",
        runner=runner,
        image_size=prepared.image_size,
        budget_seconds=540.0,
    )
    target.run(_cand({"unfreeze": "all", "epochs": 1}))
    (job,) = runner.jobs
    assert job.deadline == 1000.0 + 540.0
    assert not job.fixed


def test_release_clears_the_kept_traceback_and_empties_the_device_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emptied: list[str] = []
    stub = SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: emptied.append("mps")))
    monkeypatch.setattr(dl, "_torch", lambda: stub)
    for name in ("last_value", "last_traceback"):
        monkeypatch.setattr(sys, name, "kept", raising=False)

    def boom() -> None:
        raise RuntimeError("MPS backend out of memory")

    with pytest.raises(dl.DeviceOutOfMemoryError):
        dl._guarded("mps", boom, "batch_size=64")
    assert sys.last_value is None
    assert sys.last_traceback is None
    assert emptied == ["mps"]


def test_a_16_bit_and_a_float_image_decode_like_their_8_bit_version(tmp_path: Path) -> None:
    ramp = np.tile(np.linspace(0.0, 1.0, 64), (64, 1))
    Image.fromarray((ramp * 255).astype(np.uint8)).save(tmp_path / "eight.png")
    Image.fromarray((ramp * 65535).astype(np.uint16)).save(tmp_path / "sixteen.png")
    Image.fromarray(ramp.astype(np.float32)).save(tmp_path / "float.tiff")
    names = ("eight.png", "sixteen.png", "float.tiff")
    eight, sixteen, floating = dl.decode([str(tmp_path / n) for n in names], 32)
    assert abs(float(sixteen.mean()) - float(eight.mean())) < 2.0
    assert abs(float(floating.mean()) - float(eight.mean())) < 2.0


def test_an_image_over_the_pixel_limit_decodes_blank_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new("RGB", (64, 64), (200, 10, 10)).save(tmp_path / "big.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    (pixels,) = dl.decode([str(tmp_path / "big.png")], 32)
    assert int(pixels.max()) == 0
