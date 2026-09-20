"""`iterate.vision`: the loader a user's app calls.

The first half needs no torch: a stand-in module answers for it, so what `load` asks of
torch, what it refuses and what `predict` returns are all checked on CI. The second half
runs real torch in a child process (torch never loads inside the pytest process) and is
skipped where torch is absent.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from evals.config import REPO_ROOT
from iterate import vision
from iterate.targets import net

pytestmark = pytest.mark.unit

_RECIPE = {"backbone": "resnet18", "unfreeze": "all", "epochs": 2, "image_size": 64}


class _Network:
    def __init__(self, misfit: bool = False) -> None:
        self.loaded: Any = None
        self.moves: list[str] = []
        self._misfit = misfit

    def load_state_dict(self, state: Any) -> None:
        if self._misfit:
            raise RuntimeError("Error(s) in loading state_dict for ResNet: size mismatch")
        self.loaded = state

    def to(self, device: str) -> _Network:
        self.moves.append(device)
        return self

    def eval(self) -> _Network:
        self.moves.append("eval")
        return self


def _file(task: str = "classification", classes: list[Any] | None = None) -> dict[str, Any]:
    names = [] if task == "regression" else (classes or ["cat", "dog", "emu"])
    meta = net.saved_meta(
        SimpleNamespace(__version__="2.9.0"),
        _RECIPE,
        task=task,
        classes=names,
        outputs=len(names) or 1,
        centre=50.0,
        spread=10.0,
    )
    return {net.SAVED_KEY: meta, "state_dict": {"fc.weight": np.zeros(2, np.float32)}}


@pytest.fixture
def torch_(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake = ModuleType("torch")
    fake.__version__ = "2.9.0"  # type: ignore[attr-defined]
    fake.asked = []  # type: ignore[attr-defined]
    fake.file = _file()  # type: ignore[attr-defined]

    def load(path: Path, **kw: Any) -> Any:
        fake.asked.append((path, kw))  # type: ignore[attr-defined]
        if isinstance(fake.file, Exception):  # type: ignore[attr-defined]
            raise fake.file  # type: ignore[attr-defined]
        return fake.file  # type: ignore[attr-defined]

    fake.load = load  # type: ignore[attr-defined]
    fake.device = lambda name: name  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def model_for(torch: Any, recipe: Any, outputs: int, head: Any, **kw: Any) -> _Network:
        calls.append({"recipe": recipe, "outputs": outputs, "head": head, **kw})
        calls[-1]["network"] = _Network(misfit=bool(recipe.get("misfit")))
        return calls[-1]["network"]  # type: ignore[no-any-return]

    monkeypatch.setattr(net, "model_for", model_for)
    return calls


def test_a_module_called_torch_never_stays_behind() -> None:
    assert "torch" not in sys.modules


def test_load_opens_plain_weights_on_the_cpu_and_builds_with_no_download(
    torch_: Any, built: list[dict[str, Any]]
) -> None:
    model = vision.load("runs/r1/best_model.pt", device="cpu")
    assert torch_.asked == [
        (Path("runs/r1/best_model.pt"), {"map_location": "cpu", "weights_only": True})
    ]
    (call,) = built
    assert call["pretrained"] is False
    assert (call["recipe"], call["outputs"], call["head"]) == (_RECIPE, 3, None)
    assert call["network"].loaded is torch_.file["state_dict"]
    assert call["network"].moves == ["cpu", "eval"]
    assert (model.device, model.task, model.classes) == (
        "cpu",
        "classification",
        ["cat", "dog", "emu"],
    )
    assert repr(model) == "SavedModel(resnet18, 3 classes, 64 px, on cpu)"


@pytest.mark.parametrize("version", ["2.5.1", "1.13.1+cpu"])
def test_an_old_torch_is_refused_before_the_file_is_opened(
    torch_: Any, built: list[dict[str, Any]], version: str
) -> None:
    torch_.__version__ = version
    with pytest.raises(RuntimeError, match=r"needs torch 2\.6 or newer, and this is torch"):
        vision.load("best_model.pt")
    assert torch_.asked == []
    assert built == []


def test_no_torch_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pip install 'iterate-ai\[vision\]'"):
        vision.load("best_model.pt")


@pytest.mark.parametrize(
    "raised", [KeyError(101), EOFError(), RuntimeError("PytorchStreamReader failed")]
)
def test_a_file_torch_will_not_open_is_refused_without_torchs_advice(
    torch_: Any, built: list[dict[str, Any]], raised: Exception
) -> None:
    """torch's own message says to pass weights_only=False; ours must not carry it."""
    torch_.file = raised
    with pytest.raises(ValueError, match="is not a network an iterate image run saved") as caught:
        vision.load("whole_model.pt")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert built == []


def test_a_missing_file_is_still_a_missing_file(torch_: Any) -> None:
    torch_.file = FileNotFoundError(2, "No such file or directory")
    with pytest.raises(FileNotFoundError):
        vision.load("gone.pt")


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"format": 2}, "saved-network format 2"),
        ({"recipe": {**_RECIPE, "backbone": "hf_hub:someone/net"}}, "its backbone is not one of"),
        ({"image_size": 100_000}, "its image_size is 100000"),
        ({"classes": ["cat"]}, "it names 1 classes for 3 outputs"),
    ],
)
def test_metadata_is_checked_before_anything_is_built(
    torch_: Any, built: list[dict[str, Any]], change: dict[str, Any], said: str
) -> None:
    torch_.file = {**torch_.file, net.SAVED_KEY: {**torch_.file[net.SAVED_KEY], **change}}
    with pytest.raises(ValueError, match=said):
        vision.load("best_model.pt")
    assert built == []


def test_weights_that_do_not_fit_are_refused_in_plain_words(
    torch_: Any, built: list[dict[str, Any]]
) -> None:
    meta = torch_.file[net.SAVED_KEY]
    torch_.file = {**torch_.file, net.SAVED_KEY: {**meta, "recipe": {**_RECIPE, "misfit": True}}}
    with pytest.raises(ValueError, match="its weights do not fit the resnet18 it names") as caught:
        vision.load("best_model.pt")
    assert "size mismatch" not in str(caught.value)


def _model(
    monkeypatch: pytest.MonkeyPatch, torch_: Any, outputs: np.ndarray, **file: Any
) -> tuple[vision.SavedModel, list[Any]]:
    seen: list[Any] = []

    def pixels_of(images: Any, size: int, *, first: int = 0) -> np.ndarray:
        seen.append((list(images), size))
        return np.full((len(images), 3, size, size), first, np.uint16)

    def predict(torch: Any, network: Any, pixels: np.ndarray, dev: Any, task: str) -> np.ndarray:
        first = int(pixels[0, 0, 0, 0])
        return outputs[first : first + len(pixels)]

    monkeypatch.setattr(net, "pixels_of", pixels_of)
    monkeypatch.setattr(net, "_predict", predict)
    return vision.SavedModel(_file(**file)[net.SAVED_KEY], _Network(), "cpu"), seen


def test_predict_answers_in_class_names_of_the_labels_own_type(
    monkeypatch: pytest.MonkeyPatch, torch_: Any
) -> None:
    probs = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    model, seen = _model(monkeypatch, torch_, probs, classes=[3, 7, 11])
    assert model.predict(["a.jpg", Path("b.jpg")]) == [11, 3]
    assert seen == [(["a.jpg", Path("b.jpg")], 64)]
    assert np.array_equal(model.predict_proba(["a.jpg", "b.jpg"]), probs)


def test_a_number_model_answers_in_the_labels_own_units(
    monkeypatch: pytest.MonkeyPatch, torch_: Any
) -> None:
    model, _ = _model(monkeypatch, torch_, np.array([-1.0, 0.0, 2.5]), task="regression")
    assert model.predict(["a.jpg", "b.jpg", "c.jpg"]) == [40.0, 50.0, 75.0]
    assert model.classes == []
    assert repr(model) == "SavedModel(resnet18, a number, 64 px, on cpu)"
    with pytest.raises(ValueError, match="predicts a number"):
        model.predict_proba(["a.jpg"])


def test_one_path_in_place_of_a_list_is_refused_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, torch_: Any
) -> None:
    model, seen = _model(monkeypatch, torch_, np.zeros((1, 3)))
    with pytest.raises(TypeError, match=r"predict\(\['a\.jpg'\]\)"):
        model.predict("a.jpg")
    assert seen == []


@pytest.mark.parametrize(
    ("one", "fix"),
    [
        (b"a.jpg", r"predict\(\[b'a\.jpg'\]\)"),
        (np.zeros((48, 60, 3), np.uint8), r"predict\(\[array\]\)"),
        (np.zeros((48, 60), np.uint8), r"predict\(\[array\]\)"),
        (Image.new("RGB", (60, 48)), r"predict\(\[image\]\)"),
    ],
)
def test_one_image_in_place_of_a_list_is_refused_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, torch_: Any, one: Any, fix: str
) -> None:
    """`list()` of one HWC array is its pixel rows, and each would be predicted on."""
    model, seen = _model(monkeypatch, torch_, np.zeros((1, 3)))
    with pytest.raises(TypeError, match=fix):
        model.predict(one)
    assert seen == []


def test_a_batch_array_is_one_image_per_row(monkeypatch: pytest.MonkeyPatch, torch_: Any) -> None:
    model, seen = _model(monkeypatch, torch_, np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]))
    assert model.predict(np.zeros((2, 48, 60, 3), np.uint8)) == ["emu", "cat"]
    assert [len(images) for images, _ in seen] == [2]


def test_a_long_list_is_decoded_one_batch_at_a_time(
    monkeypatch: pytest.MonkeyPatch, torch_: Any
) -> None:
    count = 2 * net.EMBED_BATCH + 3
    probs = np.eye(3)[np.arange(count) % 3]
    model, seen = _model(monkeypatch, torch_, probs)
    assert np.array_equal(model.predict_proba([f"{i}.jpg" for i in range(count)]), probs)
    assert [len(images) for images, _ in seen] == [net.EMBED_BATCH, net.EMBED_BATCH, 3]
    assert seen[2][0] == [f"{i}.jpg" for i in range(2 * net.EMBED_BATCH, count)]


def test_no_images_is_no_predictions(monkeypatch: pytest.MonkeyPatch, torch_: Any) -> None:
    model, seen = _model(monkeypatch, torch_, np.zeros((1, 3)))
    assert model.predict([]) == []
    assert model.predict_proba([]).shape == (0, 3)
    assert seen == []


def test_importing_the_loader_brings_no_torch_and_none_of_the_agent() -> None:
    script = (
        "import sys; import iterate.vision; "
        "print([m for m in ('torch', 'pandas', 'sklearn', 'lightgbm', 'iterate.targets.dl', "
        "'iterate.core') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "[]"


# ─── real torch, in a child process ──────────────────────────────────────────

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


def _check(name: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-m", "tests.unit._vision_load_check", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


@needs_torch
def test_load_leaves_pandas_sklearn_and_lightgbm_out_of_the_users_process() -> None:
    result = _check("load_is_light")
    assert result["heavy"] == []
    assert result["names"] == result["expected"]
    assert set(result["names"]) == {"cat", "dog", "emu"}
    assert result["gap"] == 0.0
    assert [result[k] for k in ("from_text_paths", "from_pil", "from_arrays")] == [True] * 3
    assert result["empty"] == []


@needs_torch
def test_load_downloads_nothing_and_dials_nothing() -> None:
    result = _check("load_downloads_nothing")
    assert result == {"predicted": 12, "dialled": [], "downloaded": []}


@needs_torch
def test_a_saved_number_model_predicts_in_the_labels_units() -> None:
    result = _check("regression_units")
    assert result["gap"] < 1e-9
    assert result["plain_floats"]
    assert "predicts a number" in result["refused"]


@needs_torch
def test_a_foreign_file_is_refused_in_plain_words_and_runs_no_code() -> None:
    result = _check("foreign_files")
    assert result["code_ran"] is False
    said = result["said"]
    for name in ("text", "empty", "whole_model", "bare_state_dict", "hostile"):
        kind, words = said[name]
        assert kind == "ValueError", name
        assert f"<tmp>/{name}.pt is not a network an iterate image run saved" in words, name
        assert "weights_only" not in words, name
    assert said["misfit"][0] == "ValueError"
    assert "its weights do not fit the resnet18 it names" in said["misfit"][1]
    assert said["gone"][0] == "FileNotFoundError"


@needs_torch
def test_a_real_torch_that_reads_as_too_old_is_refused_before_the_file_is_opened() -> None:
    result = _check("old_torch")
    assert "needs torch 2.6 or newer, and this is torch 2.5.1" in result["said"]
    assert result["file_was_opened"] is False
