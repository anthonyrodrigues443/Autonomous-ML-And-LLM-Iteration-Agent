"""The network an image run trains, saves and opens again, and the pixels it reads.

Light on purpose: numpy, PIL, torch and the pure-Python layer grammar are the only
imports, so `iterate.vision` opens a saved network inside a user's app without pandas,
scikit-learn or the agent. torch loads on first use. `dl` re-imports every name that
moved here, so `dl._build` is still the name a test replaces.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

from iterate.targets import layers as arch
from iterate.targets.layers import RecipeError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

SAVED_FORMAT = 1
SAVED_KEY = "iterate_model"
MIN_TORCH = "2.6"
SIZE_RANGE = (32, 384)
EMBED_BATCH = 256
MAX_ASPECT = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_RESIZE = "short side to image_size, bilinear, then the centre square"

# Pinned enums: torchvision's DEFAULT may change between releases and move every score.
BACKBONES: dict[str, tuple[str, str]] = {
    "resnet18": ("ResNet18_Weights.IMAGENET1K_V1", "fc"),
    "resnet50": ("ResNet50_Weights.IMAGENET1K_V2", "fc"),
    "convnext_tiny": ("ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "classifier.2"),
}
# Trained from zero, so no probe and no head-only fit: name to head module.
SCRATCH: dict[str, str] = {"simple_cnn": "head", "layers_net": "head"}
# Each backbone's stages from the stem up, with the feature width each one ends on.
# `drop_stages=N` turns the last N into Identity. Read from torchvision 0.26.
STAGES: dict[str, tuple[tuple[tuple[str, ...], int], ...]] = {
    "resnet18": ((("layer1",), 64), (("layer2",), 128), (("layer3",), 256), (("layer4",), 512)),
    "resnet50": ((("layer1",), 256), (("layer2",), 512), (("layer3",), 1024), (("layer4",), 2048)),
    "convnext_tiny": (
        (("features.1",), 96),
        (("features.2", "features.3"), 192),
        (("features.4", "features.5"), 384),
        (("features.6", "features.7"), 768),
    ),
}
MAX_DROP_STAGES = 2


def torch_at_least(wanted: str = MIN_TORCH) -> Any:
    """torch, or a refusal in plain words: before 2.6 a file opened with
    `weights_only=True` could still run code."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "opening a saved network needs torch and torchvision: pip install 'iterate-ai[vision]'"
        ) from exc
    found = str(torch.__version__)
    if _release(found) < _release(wanted):
        raise RuntimeError(
            f"opening a saved network needs torch {wanted} or newer, and this is torch {found}: "
            "an older torch can run code hidden in a model file"
        )
    return torch


def _release(version: str) -> tuple[int, int]:
    found = re.match(r"(\d+)\.(\d+)", version)
    return (int(found[1]), int(found[2])) if found else (0, 0)


def pick_device(torch: Any, device: str | None = None) -> str:
    if device is not None:
        return device
    cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
    return "cuda" if cuda else "mps" if mps else "cpu"


def _simple_cnn(torch: Any, outputs: int) -> Any:
    nn = torch.nn
    layers: list[Any] = []
    for cin, cout in ((3, 32), (32, 64), (64, 128)):
        conv = nn.Conv2d(cin, cout, 3, padding=1)
        layers += [conv, nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
    body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return nn.Sequential(OrderedDict(body=body, head=nn.Linear(128, outputs)))


def _build(
    torch: Any,
    backbone: str,
    outputs: int | None,
    head: tuple[np.ndarray, np.ndarray] | None,
    *,
    pretrained: bool = True,
) -> Any:
    if backbone in SCRATCH:
        if outputs is None or head is not None:
            raise RecipeError(
                f"{backbone} has no pretrained features to embed or probe head to copy"
            )
        if backbone != "simple_cnn":
            raise RecipeError(f"{backbone} is the network layers=[...] builds; pass layers= too")
        return _simple_cnn(torch, outputs)
    import torchvision.models as tvm

    weights, head_name = BACKBONES[backbone]
    enum, member = weights.split(".")
    chosen = getattr(getattr(tvm, enum), member) if pretrained else None
    model = getattr(tvm, backbone)(weights=chosen, progress=False)
    features = model.get_submodule(head_name).in_features
    new = torch.nn.Identity() if outputs is None else torch.nn.Linear(features, outputs)
    if head is not None:
        with torch.no_grad():
            new.weight.copy_(torch.from_numpy(head[0]))
            new.bias.copy_(torch.from_numpy(head[1]))
    model.set_submodule(head_name, new)
    return model


def model_for(
    torch: Any,
    recipe: Mapping[str, Any],
    outputs: int,
    probe_head: tuple[np.ndarray, np.ndarray] | None,
    *,
    pretrained: bool = True,
    build: Callable[..., Any] = _build,
) -> Any:
    """The one way a network is made: a fit, a save and a load all come through here
    with the whole recipe, so a saved file always rebuilds.

    `layers` is a network of its own. `head` and `drop_stages` reshape a pretrained one.
    With neither, this is the stock build it has always been, module for module and
    random draw for random draw."""
    # Only passed when off: the tests stand a four-argument fake in for _build.
    extra: dict[str, Any] = {} if pretrained else {"pretrained": False}
    layers = arch.parse(recipe.get("layers"), "layers")
    if layers is not None:
        return arch.build_scratch(torch, layers, outputs)
    head = arch.parse(recipe.get("head"), "head")
    dropped = int(recipe.get("drop_stages") or 0)
    backbone = str(recipe["backbone"])
    if not dropped and head is None:
        return build(torch, backbone, outputs, probe_head, **extra)
    # Built whole and topped again: the stock head is what says how wide the features are.
    model = build(torch, backbone, outputs, None, **extra)
    head_name = BACKBONES[backbone][1]
    stages = STAGES[backbone]
    keep = len(stages) - dropped
    width = stages[keep - 1][1]
    _same_shape_as_the_table(model, backbone, head_name, None if dropped else width)
    for names, _ in stages[keep:]:
        for name in names:
            model.set_submodule(name, torch.nn.Identity())
    if dropped and backbone == "convnext_tiny":
        norm = model.get_submodule("classifier.0")
        model.set_submodule("classifier.0", type(norm)(width, eps=norm.eps))
    new = arch.build_head(torch, head or (), width, outputs)
    if probe_head is not None:
        final = new[-1] if head else new
        with torch.no_grad():
            final.weight.copy_(torch.from_numpy(probe_head[0]))
            final.bias.copy_(torch.from_numpy(probe_head[1]))
    model.set_submodule(head_name, new)
    return model


def _same_shape_as_the_table(model: Any, backbone: str, head_name: str, width: int | None) -> None:
    """STAGES names modules and widths as strings and numbers, so a torchvision that
    renamed or resized one would be cut in the wrong place and fail far from here.
    `width` is the stock head's input width to check against, or None when stages were
    dropped and the head no longer sees it."""
    missing = [name for names, _ in STAGES[backbone] for name in names if not _there(model, name)]
    stock = getattr(_there(model, head_name), "in_features", None)
    if not missing and (width is None or stock == width):
        return
    raise RecipeError(
        f"torchvision {_installed('torchvision')} builds {backbone} in a shape this iterate "
        "does not know, so head= and drop_stages= cannot change it; fit without them, or "
        "install torchvision 0.24 to 0.26"
    )


def _there(model: Any, name: str) -> Any:
    try:
        return model.get_submodule(name)
    except AttributeError:
        return None


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


def _fitted(rgb: Any, size: int) -> np.ndarray:
    """The short side to `size`, then the centre square, as CHW bytes. A fit and a saved
    network's `predict` both resize here, so they see the same pixels."""
    from PIL import Image

    scale = size / min(rgb.size)
    w, h = max(size, round(rgb.width * scale)), max(size, round(rgb.height * scale))
    rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(rgb.crop((left, top, left + size, top + size))).transpose(2, 0, 1)


def pixels_of(images: Sequence[Any], size: int, *, first: int = 0) -> np.ndarray:
    """`decode` for a caller that must not predict on a blank: a path, a PIL image or an
    HWC uint8 array, and one that cannot be read raises. `first` is where `images` starts
    in the caller's own list, for naming one in a refusal."""
    from PIL import Image

    out = np.zeros((len(images), 3, size, size), dtype=np.uint8)
    for i, item in enumerate(images):
        if isinstance(item, np.ndarray):
            im = _not_a_sliver(Image.fromarray(item), f"image {first + i}")
            out[i] = _fitted(_as_rgb(im), size)
        elif isinstance(item, Image.Image):
            out[i] = _fitted(_as_rgb(_not_a_sliver(item, f"image {first + i}")), size)
        else:
            with Image.open(item) as im:
                out[i] = _fitted(_as_rgb(_not_a_sliver(im, str(item))), size)
    return out


def _not_a_sliver(im: Any, name: str) -> Any:
    # The short side is resized to image_size first, so the long side of a sliver
    # becomes an array far larger than the file that held it.
    if max(im.size) > MAX_ASPECT * min(im.size):
        raise ValueError(
            f"{name} is {im.width} x {im.height}: one side is more than {MAX_ASPECT} times "
            "the other, which is not a picture a network can read"
        )
    return im


def saved_meta(
    torch: Any,
    recipe: Mapping[str, Any],
    *,
    task: str,
    classes: Sequence[Any],
    outputs: int,
    centre: float = 0.0,
    spread: float = 1.0,
) -> dict[str, Any]:
    """What a saved network carries beside its weights. Plain Python only: the file must
    open with `weights_only=True`."""
    import platform

    from iterate import __version__

    return {
        "format": SAVED_FORMAT,
        "task": task,
        "classes": [c.item() if hasattr(c, "item") else c for c in classes] or None,
        "outputs": int(outputs),
        "recipe": {k: v.item() if hasattr(v, "item") else v for k, v in recipe.items()},
        "image_size": int(recipe["image_size"]),
        "resize": _RESIZE,
        "mean": [float(v) for v in _MEAN.reshape(3)],
        "std": [float(v) for v in _STD.reshape(3)],
        "label_centre": float(centre),
        "label_spread": float(spread),
        "versions": {
            "iterate": __version__,
            "torch": str(torch.__version__),
            "torchvision": _installed("torchvision"),
            "python": platform.python_version(),
        },
    }


def _installed(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def checked_meta(saved: Any, path: str | Path) -> dict[str, Any]:
    """The metadata of a file `load` opened, with every field checked before anything
    is built from it: the file is only as trusted as whoever wrote it."""
    meta = saved.get(SAVED_KEY) if isinstance(saved, dict) else None
    if not isinstance(meta, dict) or not isinstance(meta.get("format"), int):
        raise ValueError(f"{path} is not a network an iterate image run saved")
    if meta["format"] != SAVED_FORMAT:
        raise ValueError(
            f"{path} is saved-network format {meta['format']} and this iterate reads format "
            f"{SAVED_FORMAT}: open it with the iterate-ai version that wrote it"
        )

    def refuse(what: str) -> ValueError:
        return ValueError(f"{path} cannot be opened: {what}")

    recipe, task, outputs = meta.get("recipe"), meta.get("task"), meta.get("outputs")
    if not isinstance(recipe, dict) or recipe.get("backbone") not in (*BACKBONES, *SCRATCH):
        known = ", ".join((*BACKBONES, *SCRATCH))
        raise refuse(f"its backbone is not one of {known}")
    if task not in ("classification", "regression"):
        raise refuse(f"its task is {task!r}, not classification or regression")
    size, (low, high) = meta.get("image_size"), SIZE_RANGE
    if not _is_int(size) or not low <= size <= high:
        raise refuse(f"its image_size is {size!r}, outside {low} to {high}")
    if not _is_int(outputs) or outputs < 1:
        raise refuse(f"its output count is {outputs!r}")
    classes = meta.get("classes")
    if task == "classification":
        if not isinstance(classes, list) or len(classes) < 2 or len(classes) != outputs:
            raise refuse(f"it names {_count(classes)} classes for {outputs} outputs")
    elif outputs != 1 or not all(_is_finite(meta.get(k)) for k in ("label_centre", "label_spread")):
        raise refuse("a number model needs one output and a finite label centre and spread")
    for name, ours in (("mean", _MEAN), ("std", _STD)):
        theirs = meta.get(name)
        if (
            not isinstance(theirs, list)
            or len(theirs) != 3
            or not all(_is_finite(v) for v in theirs)
            or not np.allclose(theirs, ours.reshape(3), atol=1e-6)
        ):
            raise refuse(f"its pixel {name} is not the one this iterate normalises with")
    state = saved.get("state_dict")
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(k, str) and hasattr(v, "dtype") for k, v in state.items())
    ):
        raise refuse("its weights are not a table of tensors")
    return meta


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _count(classes: Any) -> int | str:
    return len(classes) if isinstance(classes, list) else "no"


__all__ = [
    "BACKBONES",
    "MAX_DROP_STAGES",
    "MIN_TORCH",
    "SAVED_FORMAT",
    "SAVED_KEY",
    "SCRATCH",
    "SIZE_RANGE",
    "STAGES",
    "RecipeError",
    "checked_meta",
    "model_for",
    "pick_device",
    "pixels_of",
    "saved_meta",
    "torch_at_least",
]
