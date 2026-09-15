"""Checks for the real torch runner, each run in a child process by test_dl_torch.

torch and lightgbm each ship their own OpenMP runtime, and on macOS a fit in one
after the other in the same process crashes or hangs, so torch never loads inside
the pytest process. Each check prints one JSON line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from iterate.targets import dl
from iterate.targets.dl import FitJob, Recipe, TorchRunner


def _tiny_build(torch_: Any, backbone: str, n_classes: int | None, head: Any) -> Any:
    """A pinned tiny net whose head is named like resnet's, so freezing by name works."""

    class Tiny(torch_.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.body = torch_.nn.Sequential(
                torch_.nn.Conv2d(3, 4, 3, padding=1),
                torch_.nn.AdaptiveAvgPool2d(1),
                torch_.nn.Flatten(),
            )
            self.fc = torch_.nn.Identity() if n_classes is None else torch_.nn.Linear(4, n_classes)

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
        "shape": list(report.probabilities.shape),
        "sums": bool(np.allclose(report.probabilities.sum(axis=1), 1.0)),
        "epochs": [report.epochs_planned, report.epochs_run],
        "heads": [line.split()[:2] for line in log],
    }


def trimmed_plan() -> dict[str, Any]:
    dl.time_steps = lambda step, k=10: 1.0  # type: ignore[assignment]
    seen: dict[str, int] = {}
    real = dl._schedule

    def spy(torch_: Any, opt: Any, recipe: Recipe, steps: int) -> Any:
        seen["steps"] = steps
        return real(torch_, opt, recipe, steps)

    dl._schedule = spy  # type: ignore[assignment]
    report = TorchRunner("cpu").fit(_job(5, time.perf_counter() + 2 + 7.5, []))
    return {"epochs": [report.epochs_planned, report.epochs_run], "steps": seen["steps"]}


def cut_epoch() -> dict[str, Any]:
    ticks = iter(range(10_000))
    dl.time = SimpleNamespace(perf_counter=lambda: float(next(ticks)))  # type: ignore[assignment]
    dl.time_steps = lambda step, k=10: 0.5  # type: ignore[assignment]
    log: list[str] = []
    report = TorchRunner("cpu").fit(_job(5, 20.0, log))
    return {
        "epochs": [report.epochs_planned, report.epochs_run],
        "last": log[-1],
        "shape": list(report.probabilities.shape),
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
    predicted = dl._predict(torch, model, holdout, torch.device("cpu"))
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


CHECKS = {
    f.__name__: f
    for f in (
        fit_prints_epochs,
        trimmed_plan,
        cut_epoch,
        probe_after_fine_tune,
        head_copy,
        mps_starts,
        mps_cap,
    )
}


if __name__ == "__main__":
    dl._build = _tiny_build  # type: ignore[assignment]
    print(json.dumps(CHECKS[sys.argv[1]]()))
