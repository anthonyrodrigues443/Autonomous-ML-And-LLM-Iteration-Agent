"""Checks for the real torch runner, each run in a child process by test_dl_torch.

torch and lightgbm each ship their own OpenMP runtime, and on macOS a fit in one
after the other in the same process crashes or hangs, so torch never loads inside
the pytest process. Each check prints one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from iterate.adapters.data.images import prepare_images
from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen
from iterate.targets import dl, net
from iterate.targets import layers as arch
from iterate.targets.dl import DLModelTarget, FitJob, Network, Recipe, RecipeError, TorchRunner
from tests.unit.image_fixtures import vision_session

_REAL_BUILD = dl._build
# A stack with one of each layer, small enough to build in a moment.
STACK: list[Any] = [("conv", 32), ("pool",), ("conv", 64), ("pool",), ("dropout", 0.3)]
HEAD: list[Any] = [("linear", 64), ("dropout", 0.5)]

# Every shape of recipe `fit()` accepts. A saved network of each one has to open again,
# and test_net fails when a backbone, an unfreeze value or a recipe field is not here.
ROUND_TRIP_SHAPES: list[dict[str, Any]] = [
    {"backbone": backbone, "unfreeze": unfreeze, "epochs": epochs, "task": "classification"}
    for backbone in dl.BACKBONES
    for unfreeze, epochs in (("none", 0), ("head", 2), ("all", 2))
] + [
    {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 2, "task": "classification"},
    {"backbone": "simple_cnn", "unfreeze": "all", "epochs": 2, "task": "regression"},
    {"backbone": "resnet18", "unfreeze": "none", "epochs": 0, "task": "regression"},
    {"backbone": "resnet18", "unfreeze": "all", "epochs": 2, "task": "regression"},
    {
        "backbone": "layers_net",
        "layers": STACK,
        "unfreeze": "all",
        "epochs": 2,
        "task": "classification",
    },
    {
        "backbone": "resnet18",
        "head": HEAD,
        "unfreeze": "head",
        "epochs": 2,
        "task": "classification",
    },
    {
        "backbone": "resnet50",
        "head": HEAD,
        "drop_stages": 1,
        "unfreeze": "head",
        "epochs": 2,
        "task": "regression",
    },
    {"backbone": "resnet18", "unfreeze": "last_block", "epochs": 2, "task": "classification"},
]


def _tiny_build(
    torch_: Any, backbone: str, outputs: int | None, head: Any, *, pretrained: bool = True
) -> Any:
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


def _reopened(torch: Any, path: Path, build: Any = None) -> tuple[dict[str, Any], Any]:
    """The steps `iterate.vision.load` takes, so a file that passes here opens there."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = net.checked_meta(saved, path)
    extra = {} if build is None else {"build": build}
    model = net.model_for(
        torch, meta["recipe"], int(meta["outputs"]), None, pretrained=False, **extra
    )
    model.load_state_dict(saved["state_dict"])
    return meta, model


def staging_moves_no_output() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "1.pt"
        job = _job(2, time.perf_counter() + 240, [])
        plain = TorchRunner("cpu").fit(job)
        saving = TorchRunner("cpu").fit(replace(job, save_to=staged))
        return {
            "same": bool(np.array_equal(plain.outputs, saving.outputs)),
            "staged": staged.is_file(),
        }


def _session_round_trip(tmp: Path, build: Any, **changes: Any) -> dict[str, Any]:
    torch = dl._torch()
    task = changes.pop("task", "classification")
    metric = "rmse" if task == "regression" else "accuracy"
    session = vision_session(tmp, task=task, metric=metric, per_class=6, runner=TorchRunner("cpu"))
    session.submit(session.fit(**changes))
    work = session.workdir
    path = work / codegen.NETWORK_PT
    recorded = json.loads((work / codegen.RECIPE_JSON).read_text())
    meta, model = _reopened(torch, path, build)
    pixels = net.pixels_of(session.holdout_paths, int(meta["image_size"]))
    out = net._predict(torch, model, pixels, torch.device("cpu"), task)
    written = (work / codegen.PREDICTIONS_CSV).read_text().splitlines()
    if task == "regression":
        numbers = out * float(meta["label_spread"]) + float(meta["label_centre"])
        gap = float(np.abs(numbers - np.array([float(v) for v in written])).max())
        same = True
    else:
        probs = np.loadtxt(work / codegen.PROBABILITIES_CSV, delimiter=",")
        gap = float(np.abs(out - probs).max())
        same = [str(meta["classes"][i]) for i in out.argmax(1)] == written
    return {
        "gap": gap,
        "same": same,
        "digest": recorded["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": [meta["recipe"]["backbone"], meta["recipe"]["unfreeze"], meta["task"]],
        "leftovers": sorted(p.name for p in work.glob("*.part")),
    }


def saved_fit_round_trip() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return _session_round_trip(Path(tmp), _tiny_build, backbone="resnet18", epochs=2)


def saved_probe_round_trip() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return _session_round_trip(Path(tmp), _tiny_build, backbone="resnet18", unfreeze="none")


def saved_regression_round_trip() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return _session_round_trip(
            Path(tmp), _tiny_build, backbone="resnet18", epochs=2, task="regression"
        )


def saved_simple_cnn_round_trip() -> dict[str, Any]:
    dl._build = _REAL_BUILD
    with tempfile.TemporaryDirectory() as tmp:
        return _session_round_trip(Path(tmp), None, epochs=1)


def every_recipe_shape_round_trips() -> dict[str, Any]:
    """The real builders with no weights, so nothing downloads: each shape is staged or
    folded the way a session does it, saved, and opened the way `load` opens it."""
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp) / "hub"
        hub.mkdir()
        os.environ["TORCH_HOME"] = str(hub)
        torch = dl._torch()

        def unweighted(torch_: Any, backbone: str, outputs: int | None, head: Any) -> Any:
            torch_.manual_seed(0)
            return _REAL_BUILD(torch_, backbone, outputs, head, pretrained=False)

        dl._build = unweighted  # type: ignore[assignment]
        pixels, _ = _data(1, size=32)
        cpu, gaps = torch.device("cpu"), {}
        for shape in ROUND_TRIP_SHAPES:
            task = shape["task"]
            recipe = Recipe(**{k: v for k, v in shape.items() if k != "task"}, image_size=32)
            outputs = 1 if task == "regression" else 3
            head, staged = None, None
            if recipe.unfreeze == "none":
                body = unweighted(torch, recipe.backbone, None, None).eval()
                width = body(torch.zeros(1, 3, 32, 32)).shape[1]
                rng = np.random.default_rng(1)
                head = (
                    rng.normal(size=(outputs, width)).astype(np.float32),
                    rng.normal(size=outputs).astype(np.float32),
                )
            model = net.model_for(torch, asdict(recipe), outputs, head, build=unweighted)
            if head is None:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(0.01)
                staged = Path(tmp) / "staged.pt"
                dl._stage(torch, model, staged, print)
            expected = net._predict(torch, model, pixels, cpu, task)
            classes = [] if task == "regression" else ["a", "b", "c"]
            network = Network(recipe, task, classes, outputs, weights=staged, head=head)
            path = Path(tmp) / "net.pt"
            TorchRunner("cpu").save(network, path)
            _, opened = _reopened(torch, path)
            again = net._predict(torch, opened, pixels, cpu, task)
            gaps[f"{dl.recipe_name(recipe)}/{recipe.unfreeze}/{task}"] = float(
                np.abs(again - expected).max()
            )
        return {"gaps": gaps, "downloaded": sorted(p.name for p in hub.rglob("*"))}


def spec_matches_simple_cnn() -> dict[str, Any]:
    """The stack simple_cnn is: the same modules, the same weights, the same seed. The
    baseline is not routed through the builder, so this is what says they agree."""
    torch = dl._torch()
    torch.manual_seed(11)
    stock = net._simple_cnn(torch, 7)
    torch.manual_seed(11)
    built = arch.build_scratch(torch, arch.SIMPLE_CNN, 7)
    a, b = stock.state_dict(), built.state_dict()
    return {
        "text": arch.text(arch.SIMPLE_CNN),
        "keys": list(a) == list(b),
        "same": all(bool(torch.equal(a[k], b[k])) for k in a),
        "modules": [type(m).__name__ for m in built.modules()]
        == [type(m).__name__ for m in stock.modules()],
    }


def layers_shapes_and_counts() -> dict[str, Any]:
    """Real parameters against the pure-Python counter, and the output shape at two
    image sizes."""
    torch = dl._torch()
    out: dict[str, Any] = {}
    specs = {
        "simple_cnn": arch.SIMPLE_CNN,
        "stack": STACK,
        "strided": [("conv", 16, 5, 2), ("pool", "avg"), ("conv", 32), ("linear", 128)],
    }
    for name, raw in specs.items():
        spec = arch.parse(raw)
        assert spec is not None
        for outputs in (1, 10):
            model = arch.build_scratch(torch, spec, outputs)
            real = sum(p.numel() for p in model.parameters())
            out[f"{name}/{outputs}"] = [real, arch.count_weights(spec, outputs)]
    stack = arch.parse(STACK)
    assert stack is not None
    model = arch.build_scratch(torch, stack, 10).eval()
    with torch.no_grad():
        out["shapes"] = [list(model(torch.zeros(2, 3, s, s)).shape) for s in (32, 64)]
    return out


def layers_fit() -> dict[str, Any]:
    """A whole network from a stack, trained on the CPU. Four stride-2 convs take 16px
    down to a 1px map, and the 25-row set leaves a last batch of one, which batch norm
    in train mode cannot take at that size: without the skip this fit raises."""
    dl._build = _REAL_BUILD
    train, labels = _data(9)
    holdout, _ = _data(3)
    recipe = Recipe(
        backbone="layers_net",
        layers=[("conv", 16, 3, 2)] * 4,
        unfreeze="all",
        epochs=2,
        batch_size=8,
        image_size=16,
    )
    log: list[str] = []
    job = FitJob(
        train[:25], holdout, labels[:25], recipe, 3, None, time.perf_counter() + 240, log.append
    )
    report = TorchRunner("cpu").fit(job)
    return {
        "rows": len(train[:25]),
        "shape": list(report.outputs.shape),
        "sums": bool(np.allclose(report.outputs.sum(axis=1), 1.0)),
        "epochs": [report.epochs_planned, report.epochs_run],
        "lines": len(log),
    }


def drop_stage_widths() -> dict[str, Any]:
    """What the head sees after 0, 1 and 2 stages come off, on every backbone, with no
    weights downloaded."""
    torch = dl._torch()
    dl._build = _REAL_BUILD
    out: dict[str, Any] = {}
    for backbone in net.BACKBONES:
        for dropped in (0, 1, 2):
            recipe = Recipe(
                backbone=backbone,
                unfreeze="head",
                epochs=2,
                drop_stages=dropped,
                head=HEAD,
                image_size=32,
            )
            torch.manual_seed(0)
            model = net.model_for(torch, asdict(recipe), 5, None, pretrained=False).eval()
            head = model.get_submodule(net.BACKBONES[backbone][1])
            with torch.no_grad():
                shape = list(model(torch.zeros(2, 3, 64, 64)).shape)
            out[f"{backbone}/{dropped}"] = {
                "width": int(head[0].in_features),
                "table": net.STAGES[backbone][len(net.STAGES[backbone]) - dropped - 1][1],
                "shape": shape,
            }
    return out


def last_block_trains() -> dict[str, Any]:
    """Only the last stage that is left, plus the head; and every frozen batch-norm layer
    keeps its statistics."""
    torch = dl._torch()
    dl._build = _REAL_BUILD
    out: dict[str, Any] = {}
    for backbone in net.BACKBONES:
        for dropped in (0, 1):
            recipe = Recipe(
                backbone=backbone,
                unfreeze="last_block",
                epochs=2,
                drop_stages=dropped,
                image_size=32,
            )
            torch.manual_seed(0)
            model = net.model_for(torch, asdict(recipe), 5, None, pretrained=False)
            dl._optimiser(torch, model, recipe)
            dl._train_mode(torch, model, recipe)
            norms = [
                m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            ]
            out[f"{backbone}/{dropped}"] = {
                "prefixes": list(dl._trainable(recipe) or ()),
                "groups": sorted(
                    {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
                ),
                "training_norms": [sum(m.training for m in norms), len(norms)],
            }
    return out


def dropout_head_keeps_the_probe() -> dict[str, Any]:
    """A head that adds no linear layer still ends in the probe's own weights."""
    torch = dl._torch()
    dl._build = _REAL_BUILD
    rng = np.random.default_rng(5)
    head = (rng.normal(size=(3, 512)).astype(np.float32), rng.normal(size=3).astype(np.float32))
    recipe = Recipe(
        backbone="resnet18",
        unfreeze="head",
        epochs=2,
        head=[("dropout", 0.5)],
        head_init="probe",
        image_size=32,
    )
    torch.manual_seed(0)
    model = net.model_for(torch, asdict(recipe), 3, head, pretrained=False)
    final = model.fc[-1]
    return {
        "modules": [type(m).__name__ for m in model.fc],
        "weight": bool(np.array_equal(final.weight.detach().numpy(), head[0])),
        "bias": bool(np.array_equal(final.bias.detach().numpy(), head[1])),
    }


def golden_state_dict_keys() -> dict[str, Any]:
    """The names a saved file carries. A builder that moves one cannot open yesterday's
    file, so this check failing means SAVED_FORMAT has to go up with it."""
    torch = dl._torch()
    dl._build = _REAL_BUILD
    zero = Recipe(
        backbone="layers_net",
        layers=[*STACK, ("linear", 128)],
        unfreeze="all",
        epochs=2,
        image_size=32,
    )
    topped = Recipe(
        backbone="resnet18", unfreeze="head", epochs=2, drop_stages=1, head=HEAD, image_size=32
    )
    torch.manual_seed(0)
    keys = list(net.model_for(torch, asdict(zero), 4, None, pretrained=False).state_dict())
    torch.manual_seed(0)
    topped_keys = list(net.model_for(torch, asdict(topped), 4, None, pretrained=False).state_dict())
    return {
        "format": net.SAVED_FORMAT,
        "layers": keys,
        "head": [k for k in topped_keys if k.startswith("fc.")],
        "dropped": [k for k in topped_keys if k.startswith("layer4.")],
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
        staging_moves_no_output,
        saved_fit_round_trip,
        saved_probe_round_trip,
        saved_regression_round_trip,
        saved_simple_cnn_round_trip,
        every_recipe_shape_round_trips,
        spec_matches_simple_cnn,
        layers_shapes_and_counts,
        layers_fit,
        drop_stage_widths,
        last_block_trains,
        dropout_head_keeps_the_probe,
        golden_state_dict_keys,
    )
}


if __name__ == "__main__":
    dl._build = _tiny_build  # type: ignore[assignment]
    print(json.dumps(CHECKS[sys.argv[1]]()))
