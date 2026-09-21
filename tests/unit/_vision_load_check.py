"""`iterate.vision.load` under real torch, each check run in a child by test_vision_load.

This module stands in for a user's app, so it imports what an app would: numpy, PIL and
`iterate.vision`, and `iterate.targets.net` to write the files it opens. It must never
import pandas, scikit-learn or the agent, or the check that `load` brings none of them
proves nothing. Each check prints one JSON line.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iterate.targets import net
from iterate.vision import load

HEAVY = ("pandas", "sklearn", "lightgbm", "scipy", "iterate.targets.dl", "iterate.core", "openai")
CLASSES = ["cat", "dog", "emu"]
SIZE = 32


def _recipe(backbone: str) -> dict[str, Any]:
    return {"backbone": backbone, "unfreeze": "all", "epochs": 1, "image_size": SIZE}


def _teach(torch: Any, model: Any, paths: list[Path]) -> None:
    """An untrained net answers one class for every image, which would prove nothing
    about which name goes with which output: image i is taught class i % 3."""
    x = net._to_device(torch, net.pixels_of(paths, SIZE), torch.device("cpu"))
    y = torch.arange(len(paths)) % len(CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(40):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def _saved(
    torch: Any,
    path: Path,
    backbone: str = "simple_cnn",
    task: str = "classification",
    teach: list[Path] | None = None,
) -> Any:
    torch.manual_seed(0)
    outputs = 1 if task == "regression" else len(CLASSES)
    model = net.model_for(torch, _recipe(backbone), outputs, None, pretrained=False)
    if teach is not None:
        _teach(torch, model, teach)
    model.eval()
    meta = net.saved_meta(
        torch,
        _recipe(backbone),
        task=task,
        classes=[] if task == "regression" else CLASSES,
        outputs=outputs,
        centre=50.0,
        spread=10.0,
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({net.SAVED_KEY: meta, "state_dict": state}, path)
    return model


def _images(root: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    shapes = [(40, 60, 3), (64, 32, 3), (20, 20, 3), (48, 48, 3)] * 3
    for i, shape in enumerate(shapes):
        pixels = rng.integers(0, 60, shape) + np.array([200 * (i % 3 == c) for c in range(3)])
        paths.append(root / f"{i}.png")
        Image.fromarray(pixels.astype(np.uint8)).save(paths[-1])
    return paths


def load_is_light() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch = net.torch_at_least()
        paths = _images(root)
        kept = _saved(torch, root / "best_model.pt", teach=paths)
        model = load(root / "best_model.pt", device="cpu")
        names = model.predict(paths)
        expected = net._predict(
            torch, kept, net.pixels_of(paths, SIZE), torch.device("cpu"), "classification"
        )
        opened = [Image.open(p) for p in paths]
        return {
            "heavy": [m for m in HEAVY if m in sys.modules],
            "names": names,
            "expected": [CLASSES[i] for i in expected.argmax(1)],
            "gap": float(np.abs(model.predict_proba(paths) - expected).max()),
            "from_text_paths": model.predict([str(p) for p in paths]) == names,
            "from_pil": model.predict(opened) == names,
            "from_arrays": model.predict([np.asarray(im) for im in opened]) == names,
            "empty": model.predict([]),
            "classes": model.classes,
            "repr": repr(model),
        }


def load_downloads_nothing() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hub = root / "hub"
        hub.mkdir()
        os.environ["TORCH_HOME"] = str(hub)
        torch = net.torch_at_least()
        _saved(torch, root / "best_model.pt", backbone="resnet18")
        dialled: list[Any] = []

        def refuse(self: Any, address: Any) -> None:
            dialled.append(address)
            raise OSError("no network in this check")

        socket.socket.connect = refuse  # type: ignore[method-assign]
        model = load(root / "best_model.pt", device="cpu")
        return {
            "predicted": len(model.predict(_images(root))),
            "dialled": [str(a) for a in dialled],
            "downloaded": sorted(p.name for p in hub.rglob("*")),
        }


def regression_units() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch = net.torch_at_least()
        kept = _saved(torch, root / "best_model.pt", task="regression")
        paths = _images(root)
        model = load(root / "best_model.pt", device="cpu")
        raw = net._predict(
            torch, kept, net.pixels_of(paths, SIZE), torch.device("cpu"), "regression"
        )
        numbers = model.predict(paths)
        try:
            model.predict_proba(paths)
            refused = ""
        except ValueError as exc:
            refused = str(exc)
        return {
            "gap": float(np.abs(np.array(numbers) - (raw * 10.0 + 50.0)).max()),
            "plain_floats": all(type(v) is float for v in numbers),
            "refused": refused,
        }


class _Hostile:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return os.mkdir, (str(self.marker),)


def foreign_files() -> dict[str, Any]:
    """What `load` says to each file that is not ours. A hostile pickle must leave no
    mark, and no message may tell the reader to switch `weights_only` off."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch = net.torch_at_least()
        marker = root / "code-ran"
        (root / "text.pt").write_text("not a model")
        (root / "empty.pt").write_bytes(b"")
        torch.save(torch.nn.Linear(2, 2), root / "whole_model.pt")
        torch.save({"weight": torch.zeros(2)}, root / "bare_state_dict.pt")
        torch.save({net.SAVED_KEY: _Hostile(marker)}, root / "hostile.pt")
        _saved(torch, root / "good.pt")
        good = torch.load(root / "good.pt", weights_only=True)
        misfit = {**good, net.SAVED_KEY: {**good[net.SAVED_KEY], "recipe": _recipe("resnet18")}}
        torch.save(misfit, root / "misfit.pt")
        said: dict[str, list[str]] = {}
        for name in (
            "text",
            "empty",
            "whole_model",
            "bare_state_dict",
            "hostile",
            "misfit",
            "gone",
        ):
            try:
                load(root / f"{name}.pt", device="cpu")
                said[name] = ["opened", ""]
            except Exception as exc:
                chain = f"{exc} {exc.__cause__ or ''} {exc.__context__ if not exc.__suppress_context__ else ''}"
                said[name] = [type(exc).__name__, chain.replace(str(root), "<tmp>")]
        return {"said": said, "code_ran": marker.exists()}


def old_torch() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch = net.torch_at_least()
        _saved(torch, root / "best_model.pt")
        opened: list[Any] = []
        real_load = torch.load
        torch.load = lambda *a, **kw: opened.append(a) or real_load(*a, **kw)
        torch.__version__ = "2.5.1"
        try:
            load(root / "best_model.pt")
            said = "opened"
        except RuntimeError as exc:
            said = str(exc)
        return {"said": said, "file_was_opened": bool(opened)}


CHECKS = {
    f.__name__: f
    for f in (load_is_light, load_downloads_nothing, regression_units, foreign_files, old_torch)
}


if __name__ == "__main__":
    print(json.dumps(CHECKS[sys.argv[1]]()))
