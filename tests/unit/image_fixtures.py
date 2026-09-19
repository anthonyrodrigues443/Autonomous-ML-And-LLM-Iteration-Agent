"""Tiny PNG fixtures for the folder tests. The class is folded into the colour, so a
class tree is clean by construction: no two classes share an image's bytes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

CLASSES = ("cat", "dog", "emu")


def png(path: Path, seed: int = 0, *, klass: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), ((seed * 9 + klass * 71) % 255, (60 + klass * 40) % 255, 120)).save(
        path
    )


def class_tree(root: Path, *, per_class: int = 4, seed: int = 0) -> Path:
    for k, c in enumerate(CLASSES):
        for i in range(per_class):
            png(root / c / f"{c}_{seed + i}.png", seed + i, klass=k)
    return root


def stub_image_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """The real `iterate run` up to the loop, and no further: torch reads as installed,
    the setup pass finds nothing, image copies land under tmp_path, and the loop entry
    records what reached it. Nothing trains and no model is called."""
    from iterate.adapters.compute import deps
    from iterate.core import agent_loop
    from iterate.core.orchestrator import RunResult
    from iterate.core.researcher import Findings, Researcher
    from iterate.schemas.experiment import ExperimentResult

    calls: list[dict[str, Any]] = []

    def loop(**kw: Any) -> RunResult:
        calls.append(kw)
        return RunResult(
            baseline=ExperimentResult(experiment_id="baseline", error="stubbed loop"),
            history=[],
            best=None,
            stopped_because="baseline_failed",
        )

    real = deps.installed()
    monkeypatch.setattr(agent_loop, "run_supervised", loop)
    monkeypatch.setattr(deps, "installed", lambda: {**real, "torch": "2.9", "torchvision": "0.24"})
    monkeypatch.setattr(Researcher, "research", lambda self, **_: Findings())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return calls


class FakeVisionRunner:
    """A backbone stand-in with no torch: channel means are the features, a fine-tune is
    a nearest-centroid fit on them, and a number comes back as a ramp of standardised
    outputs. One epoch short of the plan, so a FIT line's planned and run differ."""

    device = "cpu"

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.jobs: list[Any] = []
        self._fail = fail

    def embed(self, pixels: Any, *, backbone: str) -> Any:
        import numpy as np

        return pixels.reshape(len(pixels), 3, -1).mean(axis=2).astype(np.float32)

    def fit(self, job: Any) -> Any:
        import numpy as np

        from iterate.targets.dl import FitReport

        self.jobs.append(job)
        if self._fail is not None:
            raise self._fail
        ran = max(1, job.recipe.epochs - 1)
        if job.task == "regression":
            return FitReport(np.linspace(-1.0, 1.0, len(job.holdout)), job.recipe.epochs, ran)
        train = self.embed(job.train, backbone="")
        held = self.embed(job.holdout, backbone="")
        centres = np.stack([train[job.labels == c].mean(axis=0) for c in range(job.outputs)])
        logits = -((held[:, None, :] - centres[None]) ** 2).sum(-1)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        return FitReport(p / p.sum(axis=1, keepdims=True), job.recipe.epochs, ran)


def vision_session(
    tmp_path: Path,
    *,
    task: str = "classification",
    metric: str = "accuracy",
    classes: int = 3,
    per_class: int = 5,
    holdout: int = 6,
    size: int = 32,
    budget: float = 60.0,
    carried: dict[str, Any] | None = None,
    runner: Any = None,
    average: str | None = None,
) -> Any:
    """A kernel session on real PNGs, built the way the preamble builds one but without
    `start()`, which imports torch."""
    import json
    from dataclasses import asdict, replace

    import pandas as pd

    from iterate.core import codegen
    from iterate.core.vision_session import Session
    from iterate.targets import dl

    def images(root: Path, labels: list[Any], tag: str, colours: list[int]) -> pd.DataFrame:
        rows = []
        for i, (label, colour) in enumerate(zip(labels, colours, strict=True)):
            path = root / f"{tag}_{i:03d}.png"
            png(path, seed=i, klass=colour)
            rows.append({"image": str(path), "label": label})
        return pd.DataFrame(rows)

    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    if task == "classification":
        # The colour follows the class, so a fake backbone can tell them apart.
        labels: list[Any] = [f"c{k}" for k in range(classes) for _ in range(per_class)]
        held: list[Any] = [f"c{i % classes}" for i in range(holdout)]
        names = sorted({str(v) for v in labels})
        train_colours = [names.index(str(v)) for v in labels]
        held_colours = [names.index(str(v)) for v in held]
    else:
        labels = [float(i) for i in range(classes * per_class)]
        held = [float(i) for i in range(holdout)]
        names = []
        train_colours = [int(v) for v in labels]
        held_colours = [int(v) for v in held]
    train = images(tmp_path / "train", labels, "t", train_colours)
    test = images(tmp_path / "held", held, "h", held_colours)
    train.to_csv(work / codegen.TRAIN_CSV, index=False)
    test.drop(columns=["label"]).to_csv(work / codegen.HOLDOUT_CSV, index=False)
    meta = {
        "target": "label",
        "task": task,
        "metric": metric,
        "average": average,
        "image_column": "image",
        "classes": names or None,
        "image_size": size,
        "baseline": asdict(replace(dl.BASELINE, image_size=min(dl.BASELINE_SIZE, size))),
        "budget_seconds": budget,
        "seed": 42,
    }
    (work / codegen.META_JSON).write_text(json.dumps(meta))
    if carried is not None:
        (work / codegen.INCUMBENT_JSON).write_text(json.dumps(carried))
    return Session(meta, train, test, workdir=work, runner=runner or FakeVisionRunner())


def session_names(session: Any) -> dict[str, Any]:
    """The names `start()` binds, for a cell run outside a kernel."""
    train_px, holdout_px = session.pixels(session.size)
    return {
        "SESSION": session,
        "TASK": session.task,
        "CLASSES": session.classes or None,
        "DEVICE": session.device,
        "IMAGE_SIZE": session.size,
        "train_px": train_px,
        "holdout_px": holdout_px,
        "labels": session.labels,
        "FIT_IDX": session.fit_idx,
        "VAL_IDX": session.val_idx,
        "fit": session.fit,
        "submit": session.submit,
        "evaluate": session.evaluate,
        "submit_probabilities": session.submit_probabilities,
        "submit_numbers": session.submit_numbers,
        "seconds_left": session.seconds_left,
        "pixels": session.pixels,
        "batches": session.batches,
        "as_input": session.as_input,
        "as_labels": session.as_labels,
        "predict": session.predict,
    }
