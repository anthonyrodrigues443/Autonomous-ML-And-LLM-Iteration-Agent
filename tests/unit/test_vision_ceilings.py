"""The vision sweep with a fake runner: the ladder, the fixed baseline as row 0, a
failing recipe recorded, a trimmed recipe visible, the record it stores, and what that
record does to an older sweep's."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evals import vision_ceilings
from evals.corpus import Dataset
from evals.store import Ceiling
from iterate.targets.dl import BASELINE, DeviceOutOfMemoryError, FitJob, FitReport, Recipe

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit


class Fake:
    """A probe on noisy features, so it misses some; a fine-tune on clean features that
    gets one holdout row wrong unless the backbone is convnext_tiny, so the rows differ
    and the ceiling is a single row. A number is a least-squares fit on clean features,
    shifted less for convnext_tiny than for the rest."""

    device = "cpu"

    def __init__(self) -> None:
        self.jobs: list[FitJob] = []

    def embed(self, pixels: np.ndarray, *, backbone: str) -> np.ndarray:
        means = pixels.reshape(len(pixels), 3, -1).mean(axis=2).astype(np.float32)
        if backbone == "clean":
            return means
        noise = np.random.default_rng(len(pixels)).normal(0.0, 200.0, means.shape)
        return (means + noise).astype(np.float32)

    def fit(self, job: FitJob) -> FitReport:
        self.jobs.append(job)
        if job.recipe.backbone == "resnet50":
            raise DeviceOutOfMemoryError("out of memory on cpu at batch_size=64: halve one of them")
        train = self.embed(job.train, backbone="clean")
        held = self.embed(job.holdout, backbone="clean")
        ran = 2 if job.recipe.epochs == 5 else job.recipe.epochs
        if job.task == "regression":
            coef = np.linalg.lstsq(np.c_[train, np.ones(len(train))], job.labels, rcond=None)[0]
            shift = 0.1 if job.recipe.backbone == "convnext_tiny" else 0.5
            return FitReport(np.c_[held, np.ones(len(held))] @ coef + shift, job.recipe.epochs, ran)
        centres = np.stack([train[job.labels == c].mean(axis=0) for c in range(job.outputs)])
        logits = -((held[:, None, :] - centres[None]) ** 2).sum(-1)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        if job.recipe.backbone != "convnext_tiny":
            p[:1] = np.roll(p[:1], 1, axis=1)
        return FitReport(p, job.recipe.epochs, ran)


def _dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "tiny"
    rows = []
    for k in range(3):
        for i in range(10):
            path = root / "images" / f"{k}_{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (70 * k, 40 + 60 * k, 90 + 7 * i)).save(path)
            rows.append({"image": f"images/{path.name}", "label": f"c{k}"})
    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    return Dataset(
        name="tiny", path=root / "data.csv", target="label", metric="accuracy", family="vision"
    )


def _numbers(tmp_path: Path) -> Dataset:
    """A number that rises with the red channel, scored by rmse."""
    root = tmp_path / "ramp"
    rows = []
    for i in range(30):
        path = root / "images" / f"{i:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (8 * i, 60, 90)).save(path)
        rows.append({"image": f"images/{path.name}", "value": 1.5 * i + 0.25})
    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    return Dataset(
        name="ramp", path=root / "data.csv", target="value", metric="rmse", family="vision"
    )


def _v1_ladder(base: int) -> list[Recipe]:
    """The twelve rungs as vision_recipe_sweep_v1 ran them."""
    bench = Recipe(unfreeze="all", epochs=3, image_size=base)
    large = min(224, base * 2)
    return [
        Recipe(image_size=base),
        Recipe(backbone="convnext_tiny", image_size=base),
        bench,
        replace(bench, epochs=5),
        replace(bench, unfreeze="head", lr=1e-2),
        replace(bench, head_init="probe", optimizer="sgd", lr=3e-3),
        replace(bench, label_smoothing=0.1),
        replace(bench, augment="flip_crop"),
        replace(bench, backbone="resnet50"),
        replace(bench, backbone="convnext_tiny"),
        Recipe(image_size=large),
        replace(bench, image_size=large),
    ]


def test_the_method_is_version_two() -> None:
    assert vision_ceilings.METHOD == "vision_recipe_sweep_v2"


def test_the_ladder_is_the_baseline_then_the_twelve_v1_rungs_grouped_by_size() -> None:
    ladder = vision_ceilings.recipes_for(160)
    assert ladder == [replace(BASELINE, image_size=64), *_v1_ladder(160)]
    assert [r.image_size for r in ladder] == [64] + [160] * 10 + [224] * 2
    assert [vision_ceilings.larger_size(s) for s in (64, 96, 160)] == [128, 192, 224]


@pytest.mark.parametrize(("base", "size"), [(160, 64), (64, 64), (48, 48), (32, 32)])
def test_the_baseline_rung_is_capped_at_64_and_never_upscaled(base: int, size: int) -> None:
    assert vision_ceilings.recipes_for(base)[0] == replace(BASELINE, image_size=size)


def test_a_number_ladder_leaves_out_the_label_smoothing_rung() -> None:
    ladder = vision_ceilings.recipes_for(160, "regression")
    kept = [r for r in _v1_ladder(160) if r.label_smoothing == 0]
    assert ladder == [replace(BASELINE, image_size=64), *kept]
    assert len(ladder) == 12


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_rung_is_a_recipe_the_agent_could_submit(task: str) -> None:
    for recipe in vision_ceilings.recipes_for(64, task):
        assert Recipe.from_changes(asdict(recipe), task=task) == recipe


def test_a_label_names_what_differs_from_the_bench() -> None:
    bench = Recipe(unfreeze="all", epochs=3, image_size=64)
    assert vision_ceilings.label(Recipe(image_size=64)) == "probe resnet18 64px"
    assert vision_ceilings.label(bench) == "fine-tune resnet18 64px 3ep"
    lp_ft = replace(bench, head_init="probe", optimizer="sgd", lr=3e-3)
    assert (
        vision_ceilings.label(lp_ft)
        == "fine-tune resnet18 64px 3ep optimizer=sgd lr=0.003 head_init=probe"
    )


def test_a_model_from_zero_is_labelled_against_the_baseline() -> None:
    baseline = replace(BASELINE, image_size=64)
    assert vision_ceilings.label(baseline) == "from zero simple_cnn 64px 20ep"
    assert (
        vision_ceilings.label(replace(baseline, lr=1e-2, augment="flip"))
        == "from zero simple_cnn 64px 20ep lr=0.01 augment=flip"
    )


def test_the_sweep_records_every_rung_and_measures_row_0_as_the_fixed_baseline(
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    fake = Fake()
    ceiling, rows = vision_ceilings.sweep(
        _dataset(tmp_path),
        runner=fake,
        into=tmp_path / "cache",
        on_progress=lambda r: seen.append(r.label),
    )
    assert len(rows) == 13
    assert seen == [r.label for r in rows]
    assert rows[0].label == "from zero simple_cnn 32px 20ep"
    first, *others = fake.jobs
    assert (first.recipe, first.fixed, first.deadline) == (
        vision_ceilings.recipes_for(32)[0],
        True,
        math.inf,
    )
    assert others
    assert all(not job.fixed and math.isfinite(job.deadline) for job in others)
    scored = [r.score for r in rows if r.score is not None]
    assert ceiling.ceiling == max(scored) == 1.0
    assert ceiling.baseline == rows[0].score
    assert rows[0].score is not None
    assert rows[0].score < ceiling.ceiling
    assert (rows[0].epochs_planned, rows[0].epochs_run) == (20, 20)
    assert ceiling.method.endswith("best: fine-tune convnext_tiny 32px 3ep)")
    failed = next(r for r in rows if "resnet50" in r.label)
    assert failed.score is None
    assert failed.error.startswith("out of memory on cpu")
    trimmed = next(r for r in rows if " 5ep" in r.label)
    assert (trimmed.epochs_planned, trimmed.epochs_run) == (5, 2)
    assert ceiling.method.startswith(f"{vision_ceilings.METHOD} (12 of 13 recipes on cpu, best: ")
    detail = json.loads(ceiling.detail)
    assert detail["device"] == "cpu"
    assert (detail["standard_error"], detail["standard_error_note"]) == (0.0, "")
    assert len(detail["recipes"]) == 13
    assert all("pearson" not in row and "spearman" not in row for row in detail["recipes"])
    assert detail["image_size"] == 32


def test_a_number_sweep_records_both_correlations_and_no_error_bar(tmp_path: Path) -> None:
    ceiling, rows = vision_ceilings.sweep(
        _numbers(tmp_path), runner=Fake(), into=tmp_path / "cache"
    )
    assert len(rows) == 12
    assert not any("label_smoothing" in r.label for r in rows)
    assert ceiling.direction == "minimize"
    assert rows[0].label == "from zero simple_cnn 32px 20ep"
    assert rows[0].score is not None
    assert ceiling.baseline == rows[0].score
    assert ceiling.method.endswith("best: fine-tune convnext_tiny 32px 3ep)")
    detail = json.loads(ceiling.detail)
    for row, result in zip(detail["recipes"], rows, strict=True):
        assert (row["pearson"], row["spearman"]) == (result.pearson, result.spearman)
        if result.score is None:
            assert (result.pearson, result.spearman) == (None, None)
        else:
            assert isinstance(result.pearson, float)
            assert isinstance(result.spearman, float)
    assert detail["standard_error"] is None
    assert detail["standard_error_note"] == vision_ceilings.NO_ERROR_BAR


def test_a_dataset_with_a_holdout_file_is_swept_on_that_split(tmp_path: Path) -> None:
    root = tmp_path / "split"
    sides: dict[str, list[dict[str, str]]] = {"train": [], "holdout": []}
    for side, count in (("train", 10), ("holdout", 3)):
        for k in range(3):
            for i in range(count):
                path = root / "images" / f"{side}_{k}_{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                blue = 90 + 7 * i + (100 if side == "holdout" else 0)
                Image.new("RGB", (16, 16), (70 * k, 40 + 60 * k, blue)).save(path)
                sides[side].append({"image": f"images/{path.name}", "label": f"c{k}"})
    for side, rows in sides.items():
        pd.DataFrame(rows).to_csv(root / f"{side}.csv", index=False)
    dataset = Dataset(
        name="split",
        path=root / "train.csv",
        target="label",
        metric="accuracy",
        family="vision",
        holdout=root / "holdout.csv",
    )
    fake = Fake()
    ceiling, _ = vision_ceilings.sweep(dataset, runner=fake, into=tmp_path / "cache")
    assert fake.jobs
    assert all((len(job.train), len(job.holdout)) == (30, 9) for job in fake.jobs)
    assert json.loads(ceiling.detail)["holdout"] == 9


@pytest.mark.parametrize(
    ("metric", "value", "n", "expected"),
    [
        ("accuracy", 0.9, 100, math.sqrt(0.9 * 0.1 / 100)),
        ("rmse", 4.0, 50, None),
        ("mae", 4.0, 50, None),
        ("f1_macro", 0.9, 100, None),
        ("accuracy", 0.9, 0, None),
    ],
)
def test_only_accuracy_gets_an_error_bar_and_a_number_label_says_why_not(
    metric: str, value: float, n: int, expected: float | None
) -> None:
    error_bar, note = vision_ceilings.standard_error(metric, value, n)
    assert error_bar == (pytest.approx(expected) if expected is not None else None)
    assert note == (vision_ceilings.NO_ERROR_BAR if metric in ("rmse", "mae") and n else "")


def _stored(method: str, ceiling: float, direction: str = "maximize") -> Ceiling:
    return Ceiling(
        dataset="tiny",
        dataset_hash="abc",
        metric="accuracy",
        ceiling=ceiling,
        direction=direction,
        baseline=0.89,
        method=method,
        measured_at="2026-09-15T00:00:00+00:00",
        detail='{"old": true}',
    )


def _new(ceiling: float, direction: str = "maximize") -> Ceiling:
    return Ceiling(
        dataset="tiny",
        dataset_hash="abc",
        metric="accuracy",
        ceiling=ceiling,
        direction=direction,
        baseline=0.55,
        method="vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: x)",
        measured_at="2026-09-16T00:00:00+00:00",
        detail='{"new": true}',
    )


V1 = "vision_recipe_sweep_v1 (12 of 12 recipes on mps, best: y)"


def test_nothing_stored_writes_the_new_ceiling_by_the_usual_rule() -> None:
    assert vision_ceilings.carry_stored(None, _new(0.95)) == (_new(0.95), False)


def test_the_same_version_stored_is_left_to_keep_the_better() -> None:
    stored = _stored("vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: z)", 0.99)
    assert vision_ceilings.carry_stored(stored, _new(0.95)) == (_new(0.95), False)


@pytest.mark.parametrize(
    ("direction", "old", "new"), [("maximize", 0.97, 0.95), ("minimize", 3.0, 3.5)]
)
def test_an_older_vision_sweep_that_was_better_is_replaced_with_its_ceiling_carried(
    direction: str, old: float, new: float
) -> None:
    written, replace_it = vision_ceilings.carry_stored(
        _stored(V1, old, direction), _new(new, direction)
    )
    assert replace_it is True
    assert written.ceiling == old
    assert written.method == (
        "vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: x), "
        "ceiling carried from vision_recipe_sweep_v1"
    )
    assert written.method.split()[0] == vision_ceilings.METHOD
    assert (written.baseline, written.detail) == (0.55, '{"new": true}')
    assert written.measured_at == "2026-09-16T00:00:00+00:00"


def test_a_ceiling_carried_twice_still_names_the_sweep_that_measured_it() -> None:
    v2, _ = vision_ceilings.carry_stored(_stored(V1, 0.97), _new(0.95))
    v3 = replace(_new(0.93), method="vision_recipe_sweep_v3 (13 of 13 recipes on mps, best: w)")

    written, replace_it = vision_ceilings.carry_stored(v2, v3)

    assert (replace_it, written.ceiling) == (True, 0.97)
    assert written.method == (
        "vision_recipe_sweep_v3 (13 of 13 recipes on mps, best: w), "
        "ceiling carried from vision_recipe_sweep_v1"
    )


def test_an_older_vision_sweep_that_was_worse_is_replaced_by_the_new_ceiling() -> None:
    assert vision_ceilings.carry_stored(_stored(V1, 0.90), _new(0.95)) == (_new(0.95), True)


@pytest.mark.parametrize(
    "method",
    [
        "brute_force_sweep_v1 (9 models)",
        "feature_treatment_sweep_v2 (best: frequency-encoding, 8 treatments)",
    ],
)
def test_a_tabular_sweep_stored_is_never_replaced(method: str) -> None:
    assert vision_ceilings.carry_stored(_stored(method, 0.99), _new(0.95)) == (_new(0.95), False)


def _ceiling() -> Ceiling:
    return Ceiling(
        dataset="tiny",
        dataset_hash="abc",
        metric="accuracy",
        ceiling=0.96,
        direction="maximize",
        baseline=0.89,
        method="vision_recipe_sweep_v1 (12 of 12 recipes on mps, best: x)",
        measured_at="2026-09-16T00:00:00+00:00",
        detail="{}",
    )


class _Child:
    """Stands in for the child process: the lines it would print, then its exit code."""

    argv: ClassVar[list[str]] = []
    killed: ClassVar[list[bool]] = []

    def __init__(self, argv: list[str], lines: list[str], code: int, *, boom: bool = False) -> None:
        type(self).argv = argv
        self.stdout = self._read(lines, boom)
        self._code = code

    @staticmethod
    def _read(lines: list[str], boom: bool) -> Iterator[str]:
        for line in lines:
            yield f"{line}\n"
        if boom:
            raise RuntimeError("a line that cannot be read")

    def __enter__(self) -> _Child:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def kill(self) -> None:
        type(self).killed.append(True)

    def wait(self) -> int:
        return self._code


def test_the_sweep_runs_in_a_child_and_its_ceiling_comes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = ["tiny (accuracy) sweeping vision recipes:", "    probe resnet18 32px  1.0000  1s"]
    result_line = "ceiling-json: " + json.dumps(asdict(_ceiling()))
    monkeypatch.setattr(
        vision_ceilings.subprocess,
        "Popen",
        lambda argv, **_: _Child(argv, [*lines, result_line], 0),
    )
    shown: list[str] = []
    ceiling = vision_ceilings.sweep_in_child(_dataset(tmp_path), on_line=shown.append)
    assert ceiling == _ceiling()
    assert shown == lines
    assert _Child.argv[-3:] == ["-m", "evals.vision_ceilings", "tiny"]


def test_a_child_that_dies_is_an_error_not_a_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vision_ceilings.subprocess, "Popen", lambda argv, **_: _Child(argv, ["boom"], -11)
    )
    with pytest.raises(RuntimeError, match="the vision sweep's child exited with -11"):
        vision_ceilings.sweep_in_child(_dataset(tmp_path))


def test_the_child_entry_point_wants_exactly_one_dataset() -> None:
    assert vision_ceilings.main([]) == 2


@pytest.mark.parametrize(("lines", "code"), [(["no result line"], 0), (["ceiling-json: {}"], -11)])
def test_a_child_needs_both_a_result_line_and_a_clean_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lines: list[str], code: int
) -> None:
    monkeypatch.setattr(
        vision_ceilings.subprocess, "Popen", lambda argv, **_: _Child(argv, lines, code)
    )
    with pytest.raises(RuntimeError, match="the vision sweep's child exited with"):
        vision_ceilings.sweep_in_child(_dataset(tmp_path))


def test_a_child_whose_output_cannot_be_read_is_killed_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Child.killed = []
    monkeypatch.setattr(
        vision_ceilings.subprocess,
        "Popen",
        lambda argv, **_: _Child(argv, ["one line"], 0, boom=True),
    )
    with pytest.raises(RuntimeError, match="a line that cannot be read"):
        vision_ceilings.sweep_in_child(_dataset(tmp_path), on_line=lambda _line: None)
    assert _Child.killed == [True]
