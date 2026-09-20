"""`DLModelTarget`: images to classes or to a number, on the same loop.

The dataset is the tabular one, a path column and a label column, with every path a
byte-named copy `images.prepare_images` wrote, so the split, the sealed holdout and
the scorer are the ones the other two targets use. `baseline()` is a small CNN trained
from zero for a fixed number of epochs with no time cap, so it means the same thing on
every machine. `run()` takes one typed `Recipe` and fits it through the runner, the
same path the eval sweep calls with no LLM. A number is learned standardised on the
training labels and mapped back before it is scored. torch loads on first use, so this
module imports where it is absent and CI tests the target with a fake runner.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from iterate.adapters.compute.base import CodeJob
from iterate.core import codegen
from iterate.core.scoring import direction, requires_proba, score, task_for_metric
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from iterate.adapters.compute.runner import RunResult
    from iterate.adapters.data.images import ImageProfile
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.experiment import Candidate

DEFAULT_SIZE = 160
BASELINE_SIZE = 64
PRETRAINED_EPOCHS = 12
FIT_BUDGET_SECONDS = 540.0
RECIPE_JSON = "recipe.json"
EMBED_BATCH = 256
TIMED_STEPS = 10
PIXEL_RAM_SHARE = 0.25
# A step in train mode, with its scheduler step, costs a little more than the timed one.
PLAN_MARGIN = 1.1
SHORT_SCHEDULE = 10
# 0.7 of the recommended working set: every recipe measured fits under it, and at 1.0 an
# overshoot paged 10 GB on a 24 GB Mac before it raised. Low above high fails init.
MPS_HIGH_RATIO = "0.7"
MPS_LOW_RATIO = "0.56"
_OUTPUT_TAIL_CHARS = 2000
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_T = TypeVar("_T")
_COPY_NAME = re.compile(r"(missing-)?[0-9a-f]{16}")

# Pinned enums: torchvision's DEFAULT may change between releases and move every score.
BACKBONES: dict[str, tuple[str, str]] = {
    "resnet18": ("ResNet18_Weights.IMAGENET1K_V1", "fc"),
    "resnet50": ("ResNet50_Weights.IMAGENET1K_V2", "fc"),
    "convnext_tiny": ("ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "classifier.2"),
}
# Trained from zero, so no probe and no head-only fit: name to head module.
SCRATCH: dict[str, str] = {"simple_cnn": "head"}
_CHOICES: dict[str, tuple[str, ...]] = {
    "backbone": (*BACKBONES, *SCRATCH),
    "unfreeze": ("none", "head", "all"),
    "optimizer": ("adamw", "sgd"),
    "schedule": ("onecycle", "cosine", "constant"),
    "augment": ("none", "flip", "flip_crop"),
    "head_init": ("random", "probe"),
}
_RANGES: dict[str, tuple[float, float]] = {
    "image_size": (32, 384),
    "epochs": (0, 30),
    "batch_size": (8, 256),
    "lr": (1e-5, 1.0),
    "label_smoothing": (0.0, 0.3),
    "seed": (0, 2**32 - 1),
}


class RecipeError(ValueError):
    """A recipe the runner will not execute, with a reason the agent can act on."""


@dataclass(frozen=True)
class Recipe:
    """All defaults is the probe; a fine-tune with nothing else set is the Day 1 bench."""

    backbone: str = "resnet18"
    image_size: int | None = None
    unfreeze: str = "none"
    epochs: int = 0
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adamw"
    schedule: str = "onecycle"
    augment: str = "flip"
    label_smoothing: float = 0.0
    head_init: str = "random"
    seed: int = 42

    @classmethod
    def from_changes(cls, changes: dict[str, Any], *, task: str = "classification") -> Recipe:
        kinds = {f.name: str(f.type) for f in fields(cls)}
        unknown = sorted(set(changes) - set(kinds) - {"note"})
        if unknown:
            raise RecipeError(f"unknown recipe keys {unknown}; the keys are {sorted(kinds)}")
        for name, value in changes.items():
            if name in kinds and not _is_kind(value, kinds[name]):
                raise RecipeError(f"{name} must be {kinds[name]}, got {value!r}")
        recipe = cls(**{k: v for k, v in changes.items() if k in kinds})
        recipe.validate(task)
        return recipe

    def validate(self, task: str = "classification") -> None:
        for name, allowed in _CHOICES.items():
            if (value := getattr(self, name)) not in allowed:
                raise RecipeError(f"{name} must be one of {list(allowed)}, got {value!r}")
        for name, (low, high) in _RANGES.items():
            pretrained = name == "epochs" and self.backbone not in SCRATCH
            top = PRETRAINED_EPOCHS if pretrained else high
            if (value := getattr(self, name)) is not None and not low <= value <= top:
                too_long = pretrained and value > top
                hint = " for a pretrained backbone; lower epochs, or use simple_cnn"
                raise RecipeError(
                    f"{name}={value!r} is outside {low} to {top}{hint if too_long else ''}"
                )
        if self.backbone in SCRATCH:
            if self.unfreeze != "all" or self.epochs == 0:
                raise RecipeError(
                    f"{self.backbone} trains from zero: set unfreeze to all with 1 to 30 epochs; "
                    "a probe or a head-only fit needs a pretrained backbone"
                )
            if self.head_init != "random":
                raise RecipeError(
                    f"{self.backbone} has no probe to start its head from; set head_init to random"
                )
        elif (self.unfreeze == "none") != (self.epochs == 0):
            raise RecipeError("the probe takes 0 epochs; unfreeze head or all takes 1 to 12")
        if task == "regression" and self.label_smoothing > 0:
            raise RecipeError("label_smoothing is for classes; set it to 0 to predict a number")


def _is_kind(value: Any, kind: str) -> bool:
    if isinstance(value, bool):
        return kind == "bool"
    if kind == "float":
        return isinstance(value, int | float)
    if kind == "int | None":
        return value is None or isinstance(value, int)
    return type(value).__name__ == kind


BASELINE = Recipe(
    backbone="simple_cnn", unfreeze="all", epochs=20, schedule="cosine", augment="none"
)


@dataclass(frozen=True)
class FitJob:
    """One fit. `labels` are TRAINING labels, class indices or standardised numbers; the
    holdout goes in as pixels only. `outputs` is the class count, or 1 for a number. A
    `fixed` job runs every epoch its recipe asks for."""

    train: np.ndarray
    holdout: np.ndarray
    labels: np.ndarray
    recipe: Recipe
    outputs: int
    head: tuple[np.ndarray, np.ndarray] | None
    deadline: float
    log: Callable[[str], None]
    task: str = "classification"
    fixed: bool = False


@dataclass(frozen=True)
class FitReport:
    """`outputs` is (n, classes) probabilities, or (n,) standardised numbers."""

    outputs: np.ndarray
    epochs_planned: int
    epochs_run: int


class DeviceOutOfMemoryError(RuntimeError):
    """The device could not hold the recipe; the text says whether a smaller one can."""


class Runner(Protocol):
    @property
    def device(self) -> str: ...

    def embed(self, pixels: np.ndarray, *, backbone: str) -> np.ndarray: ...

    def fit(self, job: FitJob) -> FitReport: ...


def oom_kind(exc: BaseException) -> str | None:
    """On MPS and on the CPU allocator both failures are plain RuntimeErrors;
    torch.OutOfMemoryError is CUDA's."""
    if not isinstance(exc, RuntimeError):
        return None
    text = str(exc).lower()
    if "out of memory" in text or "allocate memory" in text:
        return "oom"
    return "buffer" if "invalid buffer size" in text else None


def release(device: str) -> None:
    # A traceback the interpreter keeps holds the whole graph: 1.15 GB measured on MPS.
    for name in ("last_exc", "last_value", "last_traceback"):
        if hasattr(sys, name):
            setattr(sys, name, None)
    gc.collect()
    if device in ("cuda", "mps"):
        getattr(_torch(), device).empty_cache()


def _guarded(device: str, work: Callable[[], _T], where: str) -> _T:
    # Raised after the except block: raised inside it, the new error would chain the
    # old one, whose traceback keeps every tensor of the failed step alive.
    failure = ""
    try:
        return work()
    except RuntimeError as exc:
        if (kind := oom_kind(exc)) is None:
            raise
        failure = f"{_oom_text(kind, device, where)} ({str(exc).splitlines()[0][:160]})"
    finally:
        release(device)
    raise DeviceOutOfMemoryError(failure)


def time_steps(step: Callable[[], object], k: int = TIMED_STEPS) -> float:
    """Seconds per step, from k steps after one untimed warm-up."""
    step()
    tick = time.perf_counter()
    for _ in range(k):
        step()
    return (time.perf_counter() - tick) / k


def plan_epochs(
    wanted: int, epoch_seconds: float, seconds_left: float, *, backbone: str = "resnet18"
) -> int:
    """The whole epochs that fit, so the schedule is built over what will run."""
    fits = int(seconds_left // epoch_seconds) if epoch_seconds > 0 else wanted
    if fits < 1:
        advice = "halve image_size" if backbone in SCRATCH else "halve image_size or use resnet18"
        raise RecipeError(
            f"one epoch needs about {epoch_seconds:.0f}s and {max(seconds_left, 0):.0f}s of "
            f"the fit budget are left; {advice}"
        )
    return min(wanted, fits)


def run_epochs(
    one_epoch: Callable[[int], str | None], planned: int, log: Callable[[str], None]
) -> int:
    """`one_epoch` returns None when the deadline cut it short; that epoch is not counted."""
    for epoch in range(1, planned + 1):
        tick = time.perf_counter()
        if (summary := one_epoch(epoch)) is None:
            log(f"stopped in epoch {epoch}/{planned}: the fit budget ran out")
            return epoch - 1
        log(f"epoch {epoch}/{planned} {summary} {time.perf_counter() - tick:.0f}s")
    return planned


def _time_train_step(
    model: Any,
    opt: Any,
    step: Callable[[], tuple[Any, Any, Any]],
    tally: Callable[[Any, Any], float],
) -> float:
    """Seconds per full training step, optimiser and tally included. Timed in eval mode
    at a zero learning rate, so it moves neither the weights nor the batch-norm
    statistics, and the optimiser's state is cleared after."""
    rates = [group["lr"] for group in opt.param_groups]
    for group in opt.param_groups:
        group["lr"] = 0.0
    model.eval()

    def once() -> None:
        opt.zero_grad(set_to_none=True)
        loss, out, labels = step()
        opt.step()
        loss.item()
        tally(out, labels)

    try:
        return time_steps(once)
    finally:
        for group, rate in zip(opt.param_groups, rates, strict=True):
            group["lr"] = rate
        opt.state.clear()
        opt.zero_grad(set_to_none=True)


def _train_mode(torch: Any, model: Any, recipe: Recipe) -> None:
    """Train mode; a head-only fit keeps the frozen backbone's batch-norm statistics,
    so the backbone stays the one the probe head was fitted on."""
    model.train()
    if recipe.unfreeze == "head":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


class TorchRunner:
    """torch loads on first use, so this module imports where it is absent."""

    def __init__(self, device: str | None = None) -> None:
        self._device = device

    @property
    def device(self) -> str:
        if self._device is None:
            torch = _torch()
            cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
            self._device = "cuda" if cuda else "mps" if mps else "cpu"
        return self._device

    def fit(self, job: FitJob) -> FitReport:
        where = f"batch_size={job.recipe.batch_size}, image_size={job.recipe.image_size}"
        return _guarded(self.device, lambda: self._fit(job), where)

    def _fit(self, job: FitJob) -> FitReport:
        torch, recipe, n = _torch(), job.recipe, len(job.train)
        regression = job.task == "regression"
        torch.manual_seed(recipe.seed)
        rng = np.random.default_rng(recipe.seed)
        dev = torch.device(self.device)
        cuda = dev.type == "cuda"
        model = _build(torch, recipe.backbone, job.outputs, job.head).to(dev)
        if cuda:
            torch.backends.cudnn.benchmark = True
            model = model.to(memory_format=torch.channels_last)
        opt = _optimiser(torch, model, recipe)
        if regression:
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)

        def step(idx: np.ndarray) -> tuple[Any, Any, Any]:
            xb = _to_device(torch, _augment(job.train[idx], recipe.augment, rng), dev)
            yb = torch.from_numpy(job.labels[idx]).to(dev)
            if regression:
                yb = yb.float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                out = model(xb)
                if regression:
                    out = out.squeeze(1)
                loss = loss_fn(out, yb)
            loss.backward()
            return loss, out, yb

        def tally(out: Any, yb: Any) -> float:
            if regression:
                return float(((out.detach().float() - yb) ** 2).sum())
            return int((out.argmax(1) == yb).sum())

        first = np.arange(min(n, recipe.batch_size))
        step_seconds = _time_train_step(model, opt, lambda: step(first), tally) * PLAN_MARGIN
        per_epoch = math.ceil(n / recipe.batch_size)
        stop_at = job.deadline - step_seconds * math.ceil(len(job.holdout) / recipe.batch_size)
        left = stop_at - time.perf_counter()
        planned = (
            recipe.epochs
            if job.fixed
            else plan_epochs(
                recipe.epochs, step_seconds * per_epoch, left, backbone=recipe.backbone
            )
        )
        sched = _schedule(torch, opt, recipe, planned * per_epoch)

        def one_epoch(_: int) -> str | None:
            _train_mode(torch, model, recipe)
            order, total, tallied = rng.permutation(n), 0.0, 0.0
            for start in range(0, n, recipe.batch_size):
                if time.perf_counter() > stop_at:
                    return None
                idx = order[start : start + recipe.batch_size]
                opt.zero_grad(set_to_none=True)
                loss, out, yb = step(idx)
                opt.step()
                if sched is not None:
                    sched.step()
                total += loss.item() * len(idx)
                tallied += tally(out, yb)
            if regression:
                return f"loss={total / n:.4f} train_r2={1 - tallied / n:.4f}"
            return f"loss={total / n:.4f} train_acc={tallied / n:.4f}"

        ran = run_epochs(one_epoch, planned, job.log)
        return FitReport(_predict(torch, model, job.holdout, dev, job.task), planned, ran)

    def embed(self, pixels: np.ndarray, *, backbone: str) -> np.ndarray:
        where = f"image_size={pixels.shape[-1]} with {backbone}"
        return _guarded(self.device, lambda: self._embed(pixels, backbone), where)

    def _embed(self, pixels: np.ndarray, backbone: str) -> np.ndarray:
        torch = _torch()
        dev = torch.device(self.device)
        model = _build(torch, backbone, None, None).to(dev).eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(pixels), EMBED_BATCH):
                batch = _to_device(torch, pixels[start : start + EMBED_BATCH], dev)
                parts.append(model(batch).cpu())
        return np.asarray(torch.cat(parts).numpy(), dtype=np.float32)


def _oom_text(kind: str, device: str, where: str) -> str:
    if kind == "oom":
        return f"out of memory on {device} at {where}: halve one of them"
    return f"one request is larger than {device} can ever hold at {where}; not retryable"


def _torch() -> Any:
    # Read at torch's static init, so they have to be set before the first import.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # The default high ratio, 1.7, pages past RAM instead of raising.
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = MPS_HIGH_RATIO
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = MPS_LOW_RATIO
    import torch

    return torch


def _simple_cnn(torch: Any, outputs: int) -> Any:
    nn = torch.nn
    layers: list[Any] = []
    for cin, cout in ((3, 32), (32, 64), (64, 128)):
        conv = nn.Conv2d(cin, cout, 3, padding=1)
        layers += [conv, nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
    body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return nn.Sequential(OrderedDict(body=body, head=nn.Linear(128, outputs)))


def _build(
    torch: Any, backbone: str, outputs: int | None, head: tuple[np.ndarray, np.ndarray] | None
) -> Any:
    if backbone in SCRATCH:
        if outputs is None or head is not None:
            raise RecipeError(
                f"{backbone} has no pretrained features to embed or probe head to copy"
            )
        return _simple_cnn(torch, outputs)
    import torchvision.models as tvm

    weights, head_name = BACKBONES[backbone]
    enum, member = weights.split(".")
    model = getattr(tvm, backbone)(weights=getattr(getattr(tvm, enum), member), progress=False)
    features = model.get_submodule(head_name).in_features
    new = torch.nn.Identity() if outputs is None else torch.nn.Linear(features, outputs)
    if head is not None:
        with torch.no_grad():
            new.weight.copy_(torch.from_numpy(head[0]))
            new.bias.copy_(torch.from_numpy(head[1]))
    model.set_submodule(head_name, new)
    return model


def _head_name(backbone: str) -> str:
    return SCRATCH[backbone] if backbone in SCRATCH else BACKBONES[backbone][1]


def _optimiser(torch: Any, model: Any, recipe: Recipe) -> Any:
    head_name = _head_name(recipe.backbone)
    for name, param in model.named_parameters():
        param.requires_grad_(recipe.unfreeze == "all" or name.startswith(head_name))
    params = [p for p in model.parameters() if p.requires_grad]
    if recipe.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=recipe.lr, weight_decay=1e-4)
    return torch.optim.SGD(params, lr=recipe.lr, momentum=0.9, weight_decay=1e-4)


def _schedule(torch: Any, opt: Any, recipe: Recipe, steps: int) -> Any:
    # A one-cycle over a step or two never leaves its warm-up; a short run keeps the rate.
    if steps < SHORT_SCHEDULE:
        return None
    if recipe.schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.lr, total_steps=steps)
    if recipe.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    return None


def _ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _augment(batch: np.ndarray, augment: str, rng: np.random.Generator) -> np.ndarray:
    out = batch.copy()
    if augment == "none":
        return out
    flip = rng.random(len(out)) < 0.5
    out[flip] = out[flip, :, :, ::-1]
    if augment == "flip_crop":
        size = out.shape[-1]
        pad = max(1, size // 8)
        padded = np.pad(out, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
        for i, (dy, dx) in enumerate(rng.integers(0, 2 * pad + 1, size=(len(out), 2))):
            out[i] = padded[i, :, dy : dy + size, dx : dx + size]
    return out


def _to_device(torch: Any, batch: np.ndarray, dev: Any) -> Any:
    x = torch.from_numpy(np.ascontiguousarray(batch)).to(dev).float().div_(255.0)
    return (x - torch.from_numpy(_MEAN).to(dev)) / torch.from_numpy(_STD).to(dev)


def _predict(torch: Any, model: Any, holdout: np.ndarray, dev: Any, task: str) -> np.ndarray:
    regression = task == "regression"
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(holdout), EMBED_BATCH):
            out = model(_to_device(torch, holdout[start : start + EMBED_BATCH], dev)).float()
            parts.append((out.squeeze(1) if regression else torch.softmax(out, dim=1)).cpu())
    values = np.asarray(torch.cat(parts).numpy(), dtype=np.float64)
    if regression:
        return values
    return np.asarray(values / values.sum(axis=1, keepdims=True), dtype=np.float64)


_WIDE_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})


def _as_rgb(im: Any) -> Any:
    """8-bit RGB. A 16-bit or float image is scaled to 0..255 first: `convert` clips it,
    so a 16-bit image reads as white and a 0..1 float one as black."""
    from PIL import Image

    if im.mode not in _WIDE_MODES:
        return im.convert("RGB")
    values = np.asarray(im, dtype=np.float64)
    if im.mode.startswith("I;16"):
        scale = 65535.0
    else:
        top = float(values.max()) if values.size else 0.0
        scale = next(s for s in (1.0, 255.0, 65535.0, max(top, 1.0)) if top <= s)
    grey = np.clip(values / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(grey).convert("RGB")


def decode(paths: Sequence[str], size: int) -> np.ndarray:
    from PIL import Image, UnidentifiedImageError

    out = np.zeros((len(paths), 3, size, size), dtype=np.uint8)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                rgb = _as_rgb(im)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            continue
        scale = size / min(rgb.size)
        w, h = max(size, round(rgb.width * scale)), max(size, round(rgb.height * scale))
        rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
        left, top = (w - size) // 2, (h - size) // 2
        out[i] = np.asarray(rgb.crop((left, top, left + size, top + size))).transpose(2, 0, 1)
    return out


def _numbers(labels: pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(labels, errors="coerce").astype("float64").to_numpy())


class DLModelTarget:
    """Images to classes or to a number, scored on the sealed holdout. A number is fitted
    standardised by the training labels' mean and population std, and mapped back."""

    def __init__(
        self,
        dataset: TabularDataset,
        *,
        column: str,
        metric: str,
        average: str | None = None,
        name: str = "vision-model",
        image_size: int = DEFAULT_SIZE,
        profile: ImageProfile | None = None,
        runner: Runner | None = None,
        budget_seconds: float = FIT_BUDGET_SECONDS,
    ) -> None:
        self._task = dataset.task
        if (kind := task_for_metric(metric)) != dataset.task:
            raise ValueError(
                f"{metric} is a {kind} metric and the labels read as {dataset.task}; "
                f"pick a {dataset.task} metric"
            )
        self.name = name
        self._dataset, self._column, self._metric, self._average = dataset, column, metric, average
        self._image_size, self._profile, self._budget = image_size, profile, budget_seconds
        self._runner: Runner = runner if runner is not None else TorchRunner()
        paths = [
            Path(str(p)) for f in (dataset.train_features, dataset.test_features) for p in f[column]
        ]
        folders = {p.parent.name for p in paths}
        if folders != {dataset.data_hash} or not all(_COPY_NAME.fullmatch(p.name) for p in paths):
            raise ValueError(f"{column!r} must hold the byte-named copies images.prepare writes")
        self._classes: list[Any] = []
        self._centre, self._scale = 0.0, 1.0
        self._spread: list[float] | None = None
        if self._task == "regression":
            train = _numbers(dataset.train_target)
            bad_train = int((~np.isfinite(train)).sum())
            bad_held = int((~np.isfinite(_numbers(dataset.test_target))).sum())
            if bad_train or bad_held:
                raise ValueError(
                    f"{bad_train} training and {bad_held} holdout labels are not numbers; give "
                    "every row a number, or pick a classification metric"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                self._centre, self._scale = float(train.mean()), float(train.std())
            if not (math.isfinite(self._centre) and math.isfinite(self._scale)):
                raise ValueError("the training labels are too large to standardise; rescale them")
            if self._scale == 0.0:
                raise ValueError("every training label is the same number; there is nothing to fit")
            self._labels = ((train - self._centre) / self._scale).astype(np.float32)
            stats = (train.mean(), train.std(ddof=1), train.min(), train.max())
            self._spread = [float(v) for v in stats]
        else:
            # TRAINING labels only, in the column's own type: the order the code-path scorer uses.
            self._classes = np.unique(dataset.train_target.to_numpy()).tolist()
            if len(self._classes) < 2:
                raise ValueError("the vision target needs at least two training classes")
            if any(isinstance(c, float) and not c.is_integer() for c in self._classes):
                raise ValueError(
                    "fractional labels are a number to predict; pick a regression metric"
                )
            index = {c: i for i, c in enumerate(self._classes)}
            self._labels = np.array([index[c] for c in dataset.train_target.tolist()])
        self._outputs = 1 if self._task == "regression" else len(self._classes)
        self._pixels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._probes: dict[tuple[str, int], tuple[Any, Any, np.ndarray]] = {}

    @property
    def device(self) -> str:
        return self._runner.device

    def baseline(self) -> ExperimentResult:
        return self._evaluate(self._baseline_recipe(), experiment_id="baseline", fixed=True)

    def _baseline_recipe(self) -> Recipe:
        return replace(BASELINE, image_size=min(BASELINE_SIZE, self._image_size))

    def run(self, candidate: Candidate) -> ExperimentResult:
        try:
            recipe = Recipe.from_changes(candidate.changes, task=self._task)
        except RecipeError as exc:
            return ExperimentResult(experiment_id=candidate.id, error=f"recipe refused: {exc}")
        return self._evaluate(recipe, experiment_id=candidate.id)

    def _evaluate(
        self, recipe: Recipe, *, experiment_id: str, fixed: bool = False
    ) -> ExperimentResult:
        deadline = math.inf if fixed else time.perf_counter() + self._budget
        size = recipe.image_size or self._image_size
        recipe, m = replace(recipe, image_size=size), self._metric
        log: list[str] = []
        try:
            if recipe.unfreeze == "none":
                report = FitReport(self._probe(recipe.backbone, size)[2], 0, 0)
            else:
                lp_ft = recipe.head_init == "probe"
                head = self._probe_head(recipe.backbone, size) if lp_ft else None
                train, held = self._decoded(size)
                job = FitJob(
                    train,
                    held,
                    self._labels,
                    recipe,
                    self._outputs,
                    head,
                    deadline,
                    log.append,
                    task=self._task,
                    fixed=fixed,
                )
                report = self._runner.fit(job)
        except (RecipeError, DeviceOutOfMemoryError) as exc:
            error = f"recipe refused: {exc}" if isinstance(exc, RecipeError) else str(exc)
            return ExperimentResult(experiment_id=experiment_id, error=error, logs=_tail(log))
        if self._task == "regression":
            values = self._score_numbers(report.outputs)
        else:
            values = self._score_classes(report.outputs)
        if isinstance(values, str):
            return ExperimentResult(experiment_id=experiment_id, error=values, logs=_tail(log))
        metrics = Metrics(
            values=values, primary=m, direction=direction(m), n_samples=self._dataset.n_test
        )
        ran = {"epochs_planned": report.epochs_planned, "epochs_run": report.epochs_run}
        artifacts = {RECIPE_JSON: json.dumps({**asdict(recipe), **ran})}
        return ExperimentResult(
            experiment_id=experiment_id, metrics=metrics, logs=_tail(log), artifacts=artifacts
        )

    def _score_classes(self, outputs: np.ndarray) -> dict[str, float] | str:
        m = self._metric
        y_true = self._dataset.test_target.to_numpy()
        finite = bool(np.isfinite(outputs).all())
        same_classes = np.unique(y_true).tolist() == self._classes
        proba = outputs if finite and same_classes else None
        if proba is None and requires_proba(m):
            need = "one probability column per holdout class" if finite else "finite probabilities"
            return f"{m} needs {need}"
        predicted = [self._classes[i] for i in outputs.argmax(1)]
        return score(
            "classification", y_true, predicted, y_proba=proba, average=self._average, include=(m,)
        )

    def _score_numbers(self, outputs: np.ndarray) -> dict[str, float] | str:
        m = self._metric
        predicted = np.asarray(outputs, dtype=np.float64) * self._scale + self._centre
        if (bad := int((~np.isfinite(predicted)).sum())) > 0:
            return f"{m} needs finite predictions; {bad} of {len(predicted)} are not"
        include = tuple(dict.fromkeys((m, "pearson", "spearman")))
        return score("regression", _numbers(self._dataset.test_target), predicted, include=include)

    def build_code_job(self, candidate: Candidate) -> CodeJob:
        code = str(candidate.changes["code"])
        inputs = codegen.build_inputs(self._dataset)
        inputs[codegen.META_JSON] = self.meta_json()
        return CodeJob(
            script=codegen.assemble_script(code),
            inputs=inputs,
            outputs=[codegen.PREDICTIONS_CSV, codegen.PROBABILITIES_CSV],
            packages=codegen.required_imports(code),
        )

    def _probe(self, backbone: str, size: int) -> tuple[Any, Any, np.ndarray]:
        if (backbone, size) not in self._probes:
            train, holdout = self._decoded(size)
            features = self._runner.embed(train, backbone=backbone)
            scaler = StandardScaler().fit(features)
            regression = self._task == "regression"
            model: Any = Ridge(alpha=1.0) if regression else LogisticRegression(max_iter=3000)
            model.fit(scaler.transform(features), self._labels)
            held = scaler.transform(self._runner.embed(holdout, backbone=backbone))
            out = model.predict(held) if regression else model.predict_proba(held)
            self._probes[(backbone, size)] = (scaler, model, out)
        return self._probes[(backbone, size)]

    def _probe_head(self, backbone: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        scaler, model, _ = self._probe(backbone, size)
        weight = np.atleast_2d(model.coef_) / scaler.scale_
        bias = np.atleast_1d(model.intercept_) - weight @ scaler.mean_
        if self._task == "classification" and len(weight) == 1:
            weight, bias = np.vstack([np.zeros_like(weight), weight]), np.array([0.0, bias[0]])
        return weight.astype(np.float32), bias.astype(np.float32)

    def _decoded(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        if size not in self._pixels:
            n = self._dataset.n_train + self._dataset.n_test
            if (need := n * 3 * size * size) > PIXEL_RAM_SHARE * _ram_bytes():
                raise RecipeError(
                    f"{n} images at {size} px take {need / 2**30:.1f} GiB decoded, over a "
                    "quarter of this machine's RAM; lower image_size"
                )
            self._pixels.clear()
            self._pixels[size] = (
                decode(list(self._dataset.train_features[self._column]), size),
                decode(list(self._dataset.test_features[self._column]), size),
            )
        return self._pixels[size]

    def score_code_job(self, run_result: RunResult, experiment_id: str) -> ExperimentResult:
        stdout_tail = _tail([run_result.stdout])
        if not run_result.succeeded:
            reason = "timed out" if run_result.timed_out else "script failed"
            detail = _tail([run_result.stderr]) or "(no stderr)"
            return ExperimentResult(
                experiment_id=experiment_id, error=f"code {reason}:\n{detail}", logs=stdout_tail
            )
        result = codegen.score_predictions(
            self._dataset,
            run_result.outputs.get(codegen.PREDICTIONS_CSV),
            probabilities_csv=run_result.outputs.get(codegen.PROBABILITIES_CSV),
            metric=self._metric,
            experiment_id=experiment_id,
            average=self._average,
        )
        return result.model_copy(update={"logs": stdout_tail})

    def meta_json(self) -> bytes:
        payload = {
            "target": self._dataset.target,
            "task": self._task,
            "task_kind": self._task,
            "features": list(self._dataset.features),
            "image_column": self._column,
            "classes": self._classes if self._task == "classification" else None,
            "target_spread": self._spread,
            "metric": self._metric,
            "average": self._average,
            "family": "vision",
            "image_size": self._image_size,
            "baseline": asdict(self._baseline_recipe()),
            "backbones": [*BACKBONES, *SCRATCH],
            "budget_seconds": self._budget,
            "seed": self._dataset.seed,
        }
        return json.dumps(payload).encode()

    def data_summary(self) -> str:
        return self._profile.render() if self._profile is not None else ""


def _tail(lines: list[str], limit: int = _OUTPUT_TAIL_CHARS) -> str | None:
    text = "\n".join(lines).strip()
    if not text:
        return None
    return text if len(text) <= limit else "...(truncated)\n" + text[-limit:]


__all__ = [
    "BACKBONES",
    "BASELINE",
    "BASELINE_SIZE",
    "DEFAULT_SIZE",
    "FIT_BUDGET_SECONDS",
    "RECIPE_JSON",
    "SCRATCH",
    "DLModelTarget",
    "DeviceOutOfMemoryError",
    "FitJob",
    "FitReport",
    "Recipe",
    "RecipeError",
    "Runner",
    "TorchRunner",
    "oom_kind",
    "plan_epochs",
    "run_epochs",
    "time_steps",
]
