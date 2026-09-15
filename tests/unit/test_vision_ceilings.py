"""The vision sweep with a fake runner: the ladder, the probe as the baseline, a
failing recipe recorded, a trimmed recipe visible, and the record it stores."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evals import vision_ceilings
from evals.corpus import Dataset
from evals.store import Ceiling
from iterate.targets.dl import DeviceOutOfMemoryError, FitJob, FitReport, Recipe

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class Fake:
    device = "cpu"

    def embed(self, pixels: np.ndarray, *, backbone: str) -> np.ndarray:
        return pixels.reshape(len(pixels), 3, -1).mean(axis=2).astype(np.float32)

    def fit(self, job: FitJob) -> FitReport:
        if job.recipe.backbone == "resnet50":
            raise DeviceOutOfMemoryError("out of memory on cpu at batch_size=64: halve one of them")
        train, held = self.embed(job.train, backbone=""), self.embed(job.holdout, backbone="")
        centres = np.stack([train[job.labels == c].mean(axis=0) for c in range(job.n_classes)])
        logits = -((held[:, None, :] - centres[None]) ** 2).sum(-1)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        ran = 2 if job.recipe.epochs == 5 else job.recipe.epochs
        return FitReport(p / p.sum(axis=1, keepdims=True), job.recipe.epochs, ran)


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


def test_the_ladder_is_twelve_recipes_grouped_by_size() -> None:
    ladder = vision_ceilings.recipes_for(160)
    assert len(ladder) == 12
    assert ladder[0] == Recipe(image_size=160)
    assert [r.image_size for r in ladder] == [160] * 10 + [224] * 2
    assert [vision_ceilings.larger_size(s) for s in (64, 96, 160)] == [128, 192, 224]


def test_every_rung_is_a_recipe_the_agent_could_submit() -> None:
    for recipe in vision_ceilings.recipes_for(64):
        assert Recipe.from_changes(asdict(recipe)) == recipe


def test_a_label_names_what_differs_from_the_bench() -> None:
    bench = Recipe(unfreeze="all", epochs=3, image_size=64)
    assert vision_ceilings.label(Recipe(image_size=64)) == "probe resnet18 64px"
    assert vision_ceilings.label(bench) == "fine-tune resnet18 64px 3ep"
    lp_ft = replace(bench, head_init="probe", optimizer="sgd", lr=3e-3)
    assert (
        vision_ceilings.label(lp_ft)
        == "fine-tune resnet18 64px 3ep optimizer=sgd lr=0.003 head_init=probe"
    )


def test_the_sweep_records_every_rung_and_keeps_the_probe_as_the_baseline(tmp_path: Path) -> None:
    seen: list[str] = []
    ceiling, rows = vision_ceilings.sweep(
        _dataset(tmp_path),
        runner=Fake(),
        into=tmp_path / "cache",
        on_progress=lambda r: seen.append(r.label),
    )
    assert len(rows) == 12
    assert seen == [r.label for r in rows]
    assert rows[0].label == "probe resnet18 32px"
    assert ceiling.baseline == rows[0].score == 1.0
    assert ceiling.ceiling == 1.0
    failed = next(r for r in rows if "resnet50" in r.label)
    assert failed.score is None
    assert failed.error.startswith("out of memory on cpu")
    trimmed = next(r for r in rows if " 5ep" in r.label)
    assert (trimmed.epochs_planned, trimmed.epochs_run) == (5, 2)
    assert ceiling.method.startswith(f"{vision_ceilings.METHOD} (11 of 12 recipes on cpu, best: ")
    detail = json.loads(ceiling.detail)
    assert detail["device"] == "cpu"
    assert detail["standard_error"] == 0.0
    assert len(detail["recipes"]) == 12
    assert detail["image_size"] == 32


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

    def __init__(self, argv: list[str], lines: list[str], code: int) -> None:
        type(self).argv = argv
        self.stdout = iter(f"{line}\n" for line in lines)
        self._code = code

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
