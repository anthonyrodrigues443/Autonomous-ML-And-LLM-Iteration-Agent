"""Checks for the real torch runner, each run in a child process by test_dl_torch.

torch and lightgbm each ship their own OpenMP runtime, and on macOS a fit in one
after the other in the same process crashes or hangs, so torch never loads inside
the pytest process. Each check prints one JSON line.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from iterate.adapters.data.images import prepare_images
from iterate.adapters.data.tabular import load_csv
from iterate.targets import dl
from iterate.targets.dl import DLModelTarget, FitJob, Recipe, RecipeError, TorchRunner

_REAL_BUILD = dl._build


def _tiny_build(torch_: Any, backbone: str, outputs: int | None, head: Any) -> Any:
    """A pinned tiny net whose head is named like resnet's, so freezing by name works."""

    class Tiny(torch_.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.body = torch_.nn.Sequential(
                torch_.nn.Conv2d(3, 4, 3, padding=1),
                torch_.nn.BatchNorm2d(4),
                torch_.nn.ReLU(),
                torch_.nn.AdaptiveAvgPool2d(1),
                torch_.nn.Flatten(),
            )
            self.fc = torch_.nn.Identity() if outputs is None else torch_.nn.Linear(4, outputs)

        def forward(self, x: Any) -> Any:
            return self.fc(self.body(x))

    torch_.manual_seed(0)
    model = Tiny()
    if head is not None:
        with torch_.no_grad():
            model.fc.weight.copy_(torch_.from_numpy(head[0]))
            model.fc.bias.copy_(torch_.from_numpy(head[1]))
    return model


def _data(per_class: int, classes: int = 3, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(per_class)
    xs, ys = [], []
    for c in range(classes):
        base = np.zeros((3, size, size), np.int64)
        base[c % 3] = 200
        for _ in range(per_class):
            xs.append(np.clip(base + rng.integers(0, 30, base.shape), 0, 255).astype(np.uint8))
            ys.append(c)
    return np.stack(xs), np.array(ys)


def _job(epochs: int, deadline: float, log: list[str]) -> FitJob:
    train, labels = _data(8)
    holdout, _ = _data(3)
    fit = Recipe(unfreeze="all", epochs=epochs, batch_size=8, image_size=16)
    return FitJob(train, holdout, labels, fit, 3, None, deadline, log.append)


def fit_prints_epochs() -> dict[str, Any]:
    log: list[str] = []
    report = TorchRunner("cpu").fit(_job(2, time.perf_counter() + 120, log))
    return {
        "shape": list(report.outputs.shape),
        "sums": bool(np.allclose(report.outputs.sum(axis=1), 1.0)),
        "epochs": [report.epochs_planned, report.epochs_run],
        "heads": [line.split()[:2] for line in log],
    }


def trimmed_plan() -> dict[str, Any]:
    dl._torch()
    # A frozen clock: under load the real one moves enough to trim a planned epoch.
    dl.time = SimpleNamespace(perf_counter=lambda: 100.0)  # type: ignore[assignment]
    dl.time_steps = lambda step, k=10: 1.0  # type: ignore[assignment]
    seen: dict[str, float] = {}
    real_schedule, real_plan = dl._schedule, dl.plan_epochs

    def spy_schedule(torch_: Any, opt: Any, recipe: Recipe, steps: int) -> Any:
        seen["steps"] = steps
        return real_schedule(torch_, opt, recipe, steps)

    def spy_plan(wanted: int, epoch_seconds: float, seconds_left: float, **kwargs: Any) -> int:
        seen["epoch_seconds"], seen["left"] = epoch_seconds, seconds_left
        return real_plan(wanted, epoch_seconds, seconds_left, **kwargs)

    dl._schedule = spy_schedule  # type: ignore[assignment]
    dl.plan_epochs = spy_plan  # type: ignore[assignment]
    report = TorchRunner("cpu").fit(_job(5, 100.0 + 2.2 + 7.5, []))
    return {
        "epochs": [report.epochs_planned, report.epochs_run],
        "steps": seen["steps"],
        "epoch_seconds": round(seen["epoch_seconds"], 3),
        "left": round(seen["left"], 2),
    }


def cut_epoch() -> dict[str, Any]:
    ticks = iter(range(10_000))
    dl.time = SimpleNamespace(perf_counter=lambda: float(next(ticks)))  # type: ignore[assignment]
    dl.time_steps = lambda step, k=10: 0.5  # type: ignore[assignment]
    log: list[str] = []
    report = TorchRunner("cpu").fit(_job(5, 20.1, log))
    return {
        "epochs": [report.epochs_planned, report.epochs_run],
        "last": log[-1],
        "shape": list(report.outputs.shape),
    }


def probe_after_fine_tune() -> dict[str, Any]:
    runner = TorchRunner("cpu")
    holdout, _ = _data(3)
    before = runner.embed(holdout, backbone="resnet18")
    runner.fit(_job(1, time.perf_counter() + 120, []))
    return {"same": bool(np.array_equal(before, runner.embed(holdout, backbone="resnet18")))}


def head_copy() -> dict[str, Any]:
    torch = dl._torch()
    runner = TorchRunner("cpu")
    holdout, _ = _data(3)
    features = runner.embed(holdout, backbone="resnet18")
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(3, 4)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    model = _tiny_build(torch, "resnet18", 3, (weight, bias))
    predicted = dl._predict(torch, model, holdout, torch.device("cpu"), "classification")
    return {"same": bool((predicted.argmax(1) == (features @ weight.T + bias).argmax(1)).all())}


def mps_starts() -> dict[str, Any]:
    torch = dl._torch()
    if not torch.backends.mps.is_available():
        return {"skip": "MPS only"}
    torch.zeros(1, device="mps")
    return {
        "started": True,
        "high": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        "low": os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
    }


def mps_cap() -> dict[str, Any]:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.05"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.04"
    torch = dl._torch()
    if not torch.backends.mps.is_available():
        return {"skip": "MPS only"}
    try:
        x = torch.ones(2**30, device="mps")
        (x + 1).sum().item()
        return {"kind": None}
    except RuntimeError as exc:
        return {"kind": dl.oom_kind(exc)}


def timing_moves_nothing() -> dict[str, Any]:
    torch = dl._torch()
    model = _tiny_build(torch, "resnet18", 3, None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    train, labels = _data(8)
    xb = dl._to_device(torch, train[:8], torch.device("cpu"))
    yb = torch.from_numpy(labels[:8])

    def step() -> tuple[Any, Any, Any]:
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        return loss, logits, yb

    seconds = dl._time_train_step(model, opt, step, lambda out, y: int((out.argmax(1) == y).sum()))
    after = model.state_dict()
    return {
        "same": all(bool(torch.equal(before[k], after[k])) for k in before),
        "lr": opt.param_groups[0]["lr"],
        "state": len(opt.state),
        "timed": seconds > 0,
    }


def head_only_keeps_batch_norm() -> dict[str, Any]:
    torch = dl._torch()
    built: list[Any] = []

    def keep(torch_: Any, backbone: str, n_classes: int | None, head: Any) -> Any:
        model = _tiny_build(torch_, backbone, n_classes, head)
        built.append(model)
        return model

    dl._build = keep  # type: ignore[assignment]
    out: dict[str, bool] = {}
    for unfreeze in ("head", "all"):
        built.clear()
        train, labels = _data(8)
        holdout, _ = _data(3)
        fit = Recipe(unfreeze=unfreeze, epochs=2, batch_size=8, image_size=16)
        job = FitJob(train, holdout, labels, fit, 3, None, time.perf_counter() + 120, print)
        TorchRunner("cpu").fit(job)
        fresh = _tiny_build(torch, "resnet18", 3, None)
        out[unfreeze] = bool(torch.equal(built[0].body[1].running_mean, fresh.body[1].running_mean))
    return out


def simple_cnn_parameters() -> dict[str, Any]:
    torch = dl._torch()
    counts = {}
    for outputs in (1, 10, 102):
        model = _REAL_BUILD(torch, "simple_cnn", outputs, None)
        counts[str(outputs)] = sum(p.numel() for p in model.parameters())
    model = _REAL_BUILD(torch, "simple_cnn", 10, None).eval()
    with torch.no_grad():
        shapes = [list(model(torch.zeros(2, 3, s, s)).shape) for s in (32, 64)]
    return {
        "counts": counts,
        "children": [name for name, _ in model.named_children()],
        "shapes": shapes,
    }


def _brightness(n: int, seed: int, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Noisy grey images whose label is their mean pixel value."""
    rng = np.random.default_rng(seed)
    level = rng.integers(20, 236, n)
    noise = rng.integers(-20, 21, (n, 3, size, size))
    pixels = np.clip(level[:, None, None, None] + noise, 0, 255).astype(np.uint8)
    return pixels, pixels.reshape(n, -1).mean(axis=1)


def regression_fit() -> dict[str, Any]:
    dl._build = _REAL_BUILD
    train, numbers = _brightness(160, 0)
    holdout, truth = _brightness(40, 1)
    centre, scale = numbers.mean(), numbers.std()
    labels = ((numbers - centre) / scale).astype(np.float32)
    fit = replace(dl.BASELINE, epochs=15, batch_size=16, image_size=16)
    log: list[str] = []
    job = FitJob(
        train, holdout, labels, fit, 1, None, math.inf, log.append, task="regression", fixed=True
    )
    report = TorchRunner("cpu").fit(job)
    predicted = report.outputs * scale + centre
    r2 = 1 - ((predicted - truth) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()
    return {
        "shape": list(report.outputs.shape),
        "epochs": [report.epochs_planned, report.epochs_run],
        "r2": float(r2),
        "r2_lines": sum(" train_r2=" in line for line in log),
        "acc_lines": sum("train_acc=" in line for line in log),
        "loss_and_r2": [
            [float(m[1]), float(m[2])]
            for line in log
            if (m := re.search(r"loss=(\S+) train_r2=(\S+)", line))
        ],
    }


def ridge_head_copy() -> dict[str, Any]:
    import torchvision.models as tvm

    torch = dl._torch()
    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    cached = checkpoints / Path(tvm.ResNet18_Weights.IMAGENET1K_V1.url).name
    if not cached.is_file():
        return {"skip": f"resnet18 weights are not cached at {cached}; this check never downloads"}
    dl._build = _REAL_BUILD
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "data"
        (folder / "images").mkdir(parents=True)
        pixels, numbers = _brightness(40, 2, size=32)
        rows = []
        for i, image in enumerate(pixels):
            Image.fromarray(image.transpose(1, 2, 0)).save(folder / "images" / f"{i:03d}.png")
            rows.append({"image": f"images/{i:03d}.png", "label": float(numbers[i])})
        pd.DataFrame(rows).to_csv(folder / "data.csv", index=False)
        loaded = load_csv(folder / "data.csv", target="label", task="regression")
        prepared = prepare_images(loaded, folder / "data.csv", into=Path(tmp) / "cache")
        target = DLModelTarget(
            prepared.dataset,
            column=prepared.column.column,
            metric="rmse",
            runner=TorchRunner("cpu"),
            image_size=32,
        )
        probe = target._probe("resnet18", 32)[2]
        head = target._probe_head("resnet18", 32)
        _, held = target._decoded(32)
        model = dl._build(torch, "resnet18", 1, head)
        copied = dl._predict(torch, model, held, torch.device("cpu"), "regression")
    return {
        "shape": [list(copied.shape), list(probe.shape)],
        "gap": float(np.abs(copied - probe).max()),
    }


def fixed_runs_every_epoch() -> dict[str, Any]:
    dl.time_steps = lambda step, k=10: 1000.0  # type: ignore[assignment]
    calls: list[int] = []
    real_plan = dl.plan_epochs

    def spy_plan(*args: Any, **kwargs: Any) -> int:
        calls.append(1)
        return real_plan(*args, **kwargs)

    dl.plan_epochs = spy_plan  # type: ignore[assignment]
    log: list[str] = []
    job = replace(_job(3, math.inf, log), fixed=True)
    report = TorchRunner("cpu").fit(job)
    lines, fixed_calls = len(log), len(calls)
    try:
        scratch = replace(job.recipe, backbone="simple_cnn")
        TorchRunner("cpu").fit(
            replace(job, recipe=scratch, fixed=False, deadline=time.perf_counter() + 120)
        )
        refused = ""
    except RecipeError as exc:
        refused = str(exc)
    return {
        "epochs": [report.epochs_planned, report.epochs_run],
        "lines": lines,
        "plan_calls": [fixed_calls, len(calls)],
        "refused": refused,
    }


CHECKS = {
    f.__name__: f
    for f in (
        timing_moves_nothing,
        head_only_keeps_batch_norm,
        fit_prints_epochs,
        trimmed_plan,
        cut_epoch,
        probe_after_fine_tune,
        head_copy,
        mps_starts,
        mps_cap,
        simple_cnn_parameters,
        regression_fit,
        ridge_head_copy,
        fixed_runs_every_epoch,
    )
}


if __name__ == "__main__":
    dl._build = _tiny_build  # type: ignore[assignment]
    print(json.dumps(CHECKS[sys.argv[1]]()))
