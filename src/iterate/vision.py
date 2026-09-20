"""Open the network an image run saved, and predict with it.

    from iterate.vision import load

    model = load(".iterate/runs/<run_id>/best_model.pt")
    model.predict(["a.jpg", "b.jpg"])

Imports only `iterate.targets.net`, so an app that loads a model gets numpy, PIL and
torch and none of the agent. Nothing is downloaded: the weights are in the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


class SavedModel:
    """`images` is a list of paths, PIL images or HWC uint8 arrays. Each is resized and
    normalised the way the run's own fits were."""

    def __init__(self, meta: dict[str, Any], network: Any, device: str) -> None:
        self.meta = meta
        self.device = device
        self._network = network

    @property
    def task(self) -> str:
        return str(self.meta["task"])

    @property
    def classes(self) -> list[Any]:
        return list(self.meta["classes"] or [])

    def predict(self, images: Sequence[Any]) -> list[Any]:
        """One class name per image, or one number per image in the label's own units."""
        out = self._outputs(images)
        if self.task == "regression":
            centre, spread = float(self.meta["label_centre"]), float(self.meta["label_spread"])
            return [float(v) * spread + centre for v in out]
        classes = self.classes
        return [classes[int(i)] for i in out.argmax(axis=1)]

    def predict_proba(self, images: Sequence[Any]) -> np.ndarray:
        """One row per image, one column per entry of `classes`."""
        if self.task == "regression":
            raise ValueError("this model predicts a number: call predict(images)")
        return self._outputs(images)

    def _outputs(self, images: Sequence[Any]) -> np.ndarray:
        import numpy as np

        from iterate.targets import net

        if isinstance(images, str | Path):
            raise TypeError(f"predict takes a list of images: predict([{str(images)!r}])")
        items = list(images)
        if not items:
            shape = (0,) if self.task == "regression" else (0, int(self.meta["outputs"]))
            return np.zeros(shape, dtype=np.float64)
        torch = net.torch_at_least()
        pixels = net.pixels_of(items, int(self.meta["image_size"]))
        return net._predict(torch, self._network, pixels, torch.device(self.device), self.task)

    def __repr__(self) -> str:
        what = "a number" if self.task == "regression" else f"{len(self.classes)} classes"
        backbone, size = self.meta["recipe"]["backbone"], self.meta["image_size"]
        return f"SavedModel({backbone}, {what}, {size} px, on {self.device})"


def load(path: str | Path, *, device: str | None = None) -> SavedModel:
    """The network in `path`, ready to predict, on `device` or the best one here."""
    from iterate.targets import net

    torch = net.torch_at_least()
    try:
        saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    except OSError:
        raise
    except Exception:
        # torch's own message tells the reader to pass weights_only=False, which runs
        # whatever code the file holds.
        raise ValueError(
            f"{path} is not a network an iterate image run saved: torch will not open it "
            "as plain weights"
        ) from None
    meta = net.checked_meta(saved, path)
    network = net.model_for(torch, meta["recipe"], int(meta["outputs"]), None, pretrained=False)
    try:
        network.load_state_dict(saved["state_dict"])
    except RuntimeError:
        raise ValueError(
            f"{path} cannot be opened: its weights do not fit the "
            f"{meta['recipe']['backbone']} it names"
        ) from None
    chosen = net.pick_device(torch, device)
    return SavedModel(meta, network.to(chosen).eval(), chosen)


__all__ = ["SavedModel", "load"]
