"""The image session on a REAL kernel, run in a child process by test_vision_kernel.

torch and lightgbm each ship their own OpenMP runtime, and on macOS a fit in one after
the other in the same process crashes or hangs, so torch never loads inside the pytest
process. This module is the child: it boots a confined LocalKernel over about 40 small
PNGs and drives the session the way the coder does. It prints one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from iterate.adapters.compute import confine
from iterate.adapters.compute.kernel import LocalKernel
from iterate.core import codegen
from iterate.targets import dl

CLASSES = ("cat", "dog")
PER_CLASS = 16
HOLDOUT = 8
SIZE = 32


def _png(path: Path, klass: int, i: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colour = (40 + 150 * klass, 60 + 7 * i % 120, 90)
    Image.new("RGB", (40, 40), colour).save(path)


def _inputs(images: Path, work: Path) -> None:
    train, holdout = [], []
    for k, name in enumerate(CLASSES):
        for i in range(PER_CLASS):
            path = images / f"{name}_{i:02d}.png"
            _png(path, k, i)
            train.append({"image": str(path), "label": name})
    for i in range(HOLDOUT):
        path = images / f"held_{i:02d}.png"
        _png(path, i % len(CLASSES), i)
        holdout.append({"image": str(path)})
    pd.DataFrame(train).to_csv(work / codegen.TRAIN_CSV, index=False)
    pd.DataFrame(holdout).to_csv(work / codegen.HOLDOUT_CSV, index=False)
    meta = {
        "target": "label",
        "task": "classification",
        "task_kind": "classification",
        "features": ["image"],
        "image_column": "image",
        "classes": list(CLASSES),
        "metric": "accuracy",
        "average": None,
        "family": "vision",
        "image_size": SIZE,
        "baseline": asdict(replace(dl.BASELINE, image_size=SIZE, epochs=1)),
        "budget_seconds": 120.0,
        "seed": 42,
    }
    (work / codegen.META_JSON).write_text(json.dumps(meta))


def _tagged(stdout: str, tag: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len(tag) + 1 :])
        for line in stdout.splitlines()
        if line.startswith(tag + " ")
    ]


_OWN_MODEL = """\
import torch
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, len(CLASSES))
    def forward(self, x):
        return self.head(x.mean(dim=(2, 3)))
m = Tiny().to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
m.train()
for rows in batches(FIT_IDX, 16):
    loss = torch.nn.functional.cross_entropy(m(as_input(train_px[rows])), as_labels(rows))
    opt.zero_grad(); loss.backward(); opt.step()
    if seconds_left() < 1:
        break
evaluate(predict(m, train_px[VAL_IDX]), model='tiny_net', image_size=IMAGE_SIZE, epochs=1)
submit_probabilities(predict(m, holdout_px), model='tiny_net')
"""

# A model whose pretrained_cfg fixes its input size: the worked example reads that and
# decodes at the model's size, not the session's.
_FIXED_SIZE = """\
import torch
class Fixed(torch.nn.Module):
    pretrained_cfg = {'input_size': (3, 64, 64), 'fixed_input_size': True}
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, len(CLASSES))
    def forward(self, x):
        assert x.shape[-1] == 64, x.shape
        return self.head(x.mean(dim=(2, 3)))
model = Fixed().to(DEVICE)
cfg = model.pretrained_cfg
size = cfg['input_size'][-1] if cfg.get('fixed_input_size') else IMAGE_SIZE
train_x, hold_x = pixels(size)
evaluate(predict(model, train_x[VAL_IDX]), model='fixed_vit', image_size=size, epochs=0)
print('DECODED', size)


class Answer:
    def __init__(self, logits):
        self.logits = logits


class HuggingFace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, len(CLASSES))
    def forward(self, x):
        return Answer(self.head(x.mean(dim=(2, 3))))


answered = predict(HuggingFace().to(DEVICE), holdout_px)
print('LOGITS', list(answered.shape), round(float(answered.sum()), 3))
"""


def session() -> dict[str, Any]:
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vision-check-") as tmp:
        root = Path(tmp)
        images, work = root / "images", root / "work"
        images.mkdir(parents=True)
        work.mkdir(parents=True)
        _inputs(images, work)
        kernel = LocalKernel(
            confinement=confine.Confinement(reads=(images,), weights=root / "weights")
        )
        out["confined"] = kernel.confined
        kernel.start(
            {
                name: (work / name).read_bytes()
                for name in (
                    codegen.TRAIN_CSV,
                    codegen.HOLDOUT_CSV,
                    codegen.META_JSON,
                )
            }
        )
        try:
            preamble = codegen.vision_session_preamble()
            first = kernel.run_cell(preamble, timeout=300)
            out["preamble_error"] = first.error
            out["loaded"] = "loaded:" in first.stdout
            out["example_printed"] = "HOW TO WORK" in first.stdout

            prefix = codegen.VISION_CELL_PREFIX
            fit = kernel.run_cell(
                prefix + "f = fit(epochs=1, augment='flip')\nsubmit(f)\n", timeout=600
            )
            out["fit_error"] = fit.error
            out["fit"] = _tagged(fit.stdout, "FIT")
            out["submitted"] = _tagged(fit.stdout, "SUBMITTED")
            out["predictions"] = (kernel.read_output(codegen.PREDICTIONS_CSV) or b"").decode()
            out["probabilities_rows"] = len(
                (kernel.read_output(codegen.PROBABILITIES_CSV) or b"").split()
            )
            out["recipe"] = json.loads(kernel.read_output(codegen.RECIPE_JSON) or b"{}")
            network = kernel.read_output(codegen.NETWORK_PT) or b""
            out["network_digest"] = hashlib.sha256(network).hexdigest()
            out["staged"] = kernel.read_output(".fits/1.pt") is not None

            own = kernel.run_cell(prefix + _OWN_MODEL, timeout=600)
            out["own_error"] = own.error
            out["network_after_own"] = kernel.read_output(codegen.NETWORK_PT) is not None
            own_recipe = json.loads(kernel.read_output(codegen.RECIPE_JSON) or b"{}")
            out["own_recipe_names_a_network"] = "model_sha256" in own_recipe
            out["model_lines"] = _tagged(own.stdout, "MODEL")
            out["own_submitted"] = _tagged(own.stdout, "SUBMITTED")

            fixed = kernel.run_cell(prefix + _FIXED_SIZE, timeout=600)
            out["fixed_error"] = fixed.error
            out["fixed_decoded"] = "DECODED 64" in fixed.stdout
            out["logits"] = next(
                (
                    line.split("LOGITS ", 1)[1]
                    for line in fixed.stdout.splitlines()
                    if line.startswith("LOGITS ")
                ),
                None,
            )

            kernel.restart()
            again = kernel.run_cell(preamble, timeout=300)
            out["restart_error"] = again.error
            out["recipe_after_restart"] = next(
                (
                    json.loads(line.split("recipe now: ", 1)[1])
                    for line in again.stdout.splitlines()
                    if line.startswith("recipe now: ")
                ),
                None,
            )

            kernel.run_cell("%reset -f", timeout=120)
            kernel.run_cell(preamble, timeout=300)
            floor = kernel.run_cell(prefix + codegen.vision_fallback_baseline(), timeout=300)
            out["floor_error"] = floor.error
            out["floor_predictions"] = (kernel.read_output(codegen.PREDICTIONS_CSV) or b"").decode()

            outside = kernel.run_cell(f"import os; os.listdir({str(root)!r})", timeout=120)
            blocked = kernel.blocked(outside.error)
            out["outside_blocked"] = None if blocked is None else blocked.program
        finally:
            kernel.close()
    return out


class _FakeLLM:
    """Scripts the cells a coder would write: one fit, then finish."""

    def __init__(self, cells: list[str]) -> None:
        self._cells = list(cells)

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(self, messages: Any, **kw: Any) -> Any:
        from iterate.schemas.llm import ChatResponse, ToolCall

        if self._cells:
            call = ToolCall(id="c", name="run_cell", arguments={"code": self._cells.pop(0)})
        else:
            call = ToolCall(id="f", name="finish", arguments={})
        return ChatResponse(model="fake-model", tool_calls=[call])


class _Remembering(LocalKernel):
    """Keeps what the host read, since the kernel's folder is gone once `run` returns."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.read: dict[str, bytes | None] = {}

    def read_output(self, name: str) -> bytes | None:
        self.read[name] = super().read_output(name)
        return self.read[name]


def _opened_again(kept: Path, images: list[str], read: dict[str, bytes | None]) -> dict[str, Any]:
    """The delivered file through the public loader, against what the session submitted."""
    import numpy as np

    from iterate.vision import load

    model = load(kept)
    written = (read[codegen.PREDICTIONS_CSV] or b"").decode().splitlines()
    rows = (read[codegen.PROBABILITIES_CSV] or b"").decode().splitlines()
    probs = np.array([[float(v) for v in row.split(",")] for row in rows])
    return {
        "loaded_on": model.device,
        "predicted": [str(name) for name in model.predict(images)],
        "written": written,
        "probability_gap": float(np.abs(model.predict_proba(images) - probs).max()),
    }


def experiment() -> dict[str, Any]:
    """One image experiment the way `iterate run` runs it: the real CodingAgent, the
    vision family the CLI passes, a confined kernel, and the host scoring what the
    session submitted. Then the network it left, opened the way a user's app opens it."""
    from iterate.adapters.data.tabular import load_split
    from iterate.core.coder import CodingAgent
    from iterate.deliver import saved_model

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vision-coder-") as tmp:
        root = Path(tmp)
        images, work = root / "images", root / "work"
        images.mkdir(parents=True)
        work.mkdir(parents=True)
        _inputs(images, work)
        held = pd.read_csv(work / codegen.HOLDOUT_CSV)
        held["label"] = [CLASSES[i % len(CLASSES)] for i in range(len(held))]
        held.to_csv(work / "sealed.csv", index=False)
        dataset = load_split(
            work / codegen.TRAIN_CSV, work / "sealed.csv", "label", task="classification"
        )
        meta = json.loads((work / codegen.META_JSON).read_text())
        kernel = _Remembering(
            confinement=confine.Confinement(reads=(images,), weights=root / "weights")
        )
        slot = root / "slot" / saved_model.BEST_MODEL
        slot.parent.mkdir()
        slot.write_bytes(b"an earlier session's network")
        agent = CodingAgent(
            _FakeLLM(["f = fit(epochs=1)\nsubmit(f)\n"]),
            kernel,
            metric="accuracy",
            max_cells=4,
            install=False,
            preamble=codegen.vision_session_preamble(),
            extra_inputs={codegen.META_JSON: json.dumps(meta).encode()},
            floor_cell=codegen.vision_fallback_baseline(),
            family="vision",
            cell_prefix=codegen.VISION_CELL_PREFIX,
            floor_carries_code=False,
            data_summary="Images: 32 train / 8 holdout.",
            cell_timeout=750.0,
            deadline_seconds=2700.0,
            wall_ceiling_seconds=5400.0,
            keep_model=slot,
        )
        coded = agent.run(
            dataset=dataset,
            brief="next: backbone: fit the plain CNN",
            experiment_id="iter-01",
            starting_files={codegen.INCUMBENT_JSON: json.dumps(meta["baseline"]).encode()},
        )
        read = dict(kernel.read)
        out["error"] = coded.result.error
        out["score"] = None if coded.result.metrics is None else coded.result.metrics.primary_value
        out["artifacts"] = sorted(coded.result.artifacts)
        recipe = coded.result.artifacts.get(codegen.RECIPE_JSON)
        out["recipe"] = json.loads(recipe) if recipe else None
        out["sources"] = [c.source for c in coded.cells]
        out["stdout_has_fit"] = any("FIT {" in (c.stdout or "") for c in coded.cells)
        out["kernel_folder_gone"] = kernel.read_output(codegen.PREDICTIONS_CSV) is None
        out["network_kept"] = slot.is_file()
        if slot.is_file() and out["recipe"]:
            digest = hashlib.sha256(slot.read_bytes()).hexdigest()
            out["network_digest_matches"] = digest == out["recipe"].get("model_sha256")
            kept = root / "runs" / "r1" / saved_model.BEST_MODEL
            saved_model.settle(slot, kept, is_best=True)
            # The rows the kernel predicted, in its order: load_split shuffles the holdout.
            asked = [str(p) for p in dataset.test_features["image"]]
            out.update(_opened_again(kept, asked, read))
    return out


def layers_cell() -> dict[str, Any]:
    """`fit(layers=[...])` typed into a confined cell: it builds, it trains, it submits,
    and the line names the stack in the text a cell can paste back."""
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vision-layers-") as tmp:
        root = Path(tmp)
        images, work = root / "images", root / "work"
        images.mkdir(parents=True)
        work.mkdir(parents=True)
        _inputs(images, work)
        kernel = LocalKernel(
            confinement=confine.Confinement(reads=(images,), weights=root / "weights")
        )
        kernel.start(
            {
                name: (work / name).read_bytes()
                for name in (codegen.TRAIN_CSV, codegen.HOLDOUT_CSV, codegen.META_JSON)
            }
        )
        try:
            out["preamble_error"] = kernel.run_cell(
                codegen.vision_session_preamble(), timeout=300
            ).error
            ran = kernel.run_cell(
                codegen.VISION_CELL_PREFIX
                + "f = fit(layers=[('conv', 16), ('pool',), ('conv', 32), ('pool',)], epochs=1)\n"
                "submit(f)\n",
                timeout=600,
            )
            out["error"] = ran.error
            out["fit"] = _tagged(ran.stdout, "FIT")
            out["submitted"] = _tagged(ran.stdout, "SUBMITTED")
            out["said"] = [line for line in ran.stdout.splitlines() if line.startswith("val ")]
            out["predictions"] = len((kernel.read_output(codegen.PREDICTIONS_CSV) or b"").split())
            out["recipe"] = json.loads(kernel.read_output(codegen.RECIPE_JSON) or b"{}")
            out["network"] = bool(kernel.read_output(codegen.NETWORK_PT))
        finally:
            kernel.close()
    return out


CHECKS = {f.__name__: f for f in (session, experiment, layers_cell)}


if __name__ == "__main__":
    print(json.dumps(CHECKS[sys.argv[1] if len(sys.argv) > 1 else "session"]()))
    sys.exit(0)
