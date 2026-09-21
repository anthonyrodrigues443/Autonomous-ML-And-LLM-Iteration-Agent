"""The light network module with no torch: the pixels a saved network reads, the
metadata it carries and every check a file passes before anything is built from it.
The round trips through real torch live in test_dl_torch."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, fields
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from evals.config import REPO_ROOT
from iterate.targets import dl, net
from iterate.targets.dl import FitJob, Recipe

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


# ─── the pixels ──────────────────────────────────────────────────────────────


def _decode_as_released(paths: list[str], size: int) -> np.ndarray:
    """The decode lines as they stood before the resize was split out, kept here so the
    bytes are compared with no stored hash: Pillow's resize differs between builds."""
    out = np.zeros((len(paths), 3, size, size), dtype=np.uint8)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            rgb = net._as_rgb(im)
        scale = size / min(rgb.size)
        w, h = max(size, round(rgb.width * scale)), max(size, round(rgb.height * scale))
        rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
        left, top = (w - size) // 2, (h - size) // 2
        out[i] = np.asarray(rgb.crop((left, top, left + size, top + size))).transpose(2, 0, 1)
    return out


def _fixtures(folder: Path) -> list[str]:
    rng = np.random.default_rng(7)

    def noise(height: int, width: int) -> np.ndarray:
        return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    ramp = np.tile(np.linspace(0.0, 1.0, 61), (47, 1))
    Image.fromarray(noise(37, 211)).save(folder / "wide.png")
    Image.fromarray(noise(203, 41)).save(folder / "tall.png")
    Image.fromarray(noise(9, 13)).save(folder / "tiny.png")
    Image.fromarray((ramp * 65535).astype(np.uint16)).save(folder / "sixteen.png")
    Image.fromarray(ramp.astype(np.float32)).save(folder / "float.tiff")
    names = ("wide.png", "tall.png", "tiny.png", "sixteen.png", "float.tiff")
    return [str(folder / name) for name in names]


@pytest.mark.parametrize("size", [32, 50, 160])
def test_the_split_out_resize_gives_the_bytes_decode_always_gave(tmp_path: Path, size: int) -> None:
    paths = _fixtures(tmp_path)
    expected = _decode_as_released(paths, size)
    assert expected.any(axis=(1, 2, 3)).all()
    assert np.array_equal(dl.decode(paths, size), expected)
    assert np.array_equal(net.pixels_of(paths, size), expected)


def test_a_path_a_pil_image_and_an_array_give_the_same_pixels(tmp_path: Path) -> None:
    (path, *_rest) = _fixtures(tmp_path)
    with Image.open(path) as im:
        loaded = im.copy()
    from_path, from_image, from_array = net.pixels_of([path, loaded, np.asarray(loaded)], 32)
    assert np.array_equal(from_path, from_image)
    assert np.array_equal(from_path, from_array)


def test_an_unreadable_image_raises_where_decode_leaves_a_blank(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert int(dl.decode([str(broken)], 32).max()) == 0
    with pytest.raises(UnidentifiedImageError):
        net.pixels_of([str(broken)], 32)
    with pytest.raises(FileNotFoundError):
        net.pixels_of([str(tmp_path / "absent.png")], 32)


def test_a_sliver_is_refused_before_it_is_resized(tmp_path: Path) -> None:
    """The short side goes to image_size first, so a 2 x 5000 image asks for gigabytes."""
    fine = np.zeros((100, 100 * net.MAX_ASPECT, 3), np.uint8)
    assert net.pixels_of([fine], 32).shape == (1, 3, 32, 32)
    sliver = np.zeros((2, 5000, 3), np.uint8)
    with pytest.raises(ValueError, match=r"image 1 is 5000 x 2"):
        net.pixels_of([fine, sliver], 32)
    with pytest.raises(ValueError, match=r"image 257 is 5000 x 2"):
        net.pixels_of([fine, Image.fromarray(sliver)], 32, first=256)
    path = tmp_path / "sliver.png"
    Image.fromarray(sliver).save(path)
    with pytest.raises(ValueError, match=r"sliver\.png is 5000 x 2"):
        net.pixels_of([path], 32)


# ─── the one builder ─────────────────────────────────────────────────────────


def test_the_builder_is_only_told_about_pretrained_weights_when_they_are_off() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def build(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "the network"

    recipe = asdict(Recipe(backbone="resnet50", unfreeze="all", epochs=3))
    assert net.model_for("torch", recipe, 5, None, build=build) == "the network"
    net.model_for("torch", recipe, 5, None, pretrained=False, build=build)
    assert calls == [
        (("torch", "resnet50", 5, None), {}),
        (("torch", "resnet50", 5, None), {"pretrained": False}),
    ]


def test_the_names_that_moved_are_still_the_ones_dl_holds() -> None:
    for name in ("_build", "_predict", "_to_device", "_as_rgb", "RecipeError", "BACKBONES"):
        assert getattr(dl, name) is getattr(net, name)


@pytest.mark.parametrize("backbone", sorted(net.BACKBONES))
def test_a_network_built_to_be_loaded_asks_for_no_weights(
    backbone: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, Any] = {}

    def load(**kwargs: Any) -> Any:
        asked.update(kwargs)
        return SimpleNamespace(
            get_submodule=lambda name: SimpleNamespace(in_features=4),
            set_submodule=lambda name, new: None,
        )

    models = ModuleType("torchvision.models")
    setattr(models, backbone, load)
    package = ModuleType("torchvision")
    package.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", package)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)

    torch = SimpleNamespace(nn=SimpleNamespace(Identity=object))
    net._build(torch, backbone, None, None, pretrained=False)

    assert asked == {"weights": None, "progress": False}


def test_importing_the_module_brings_in_none_of_the_agent() -> None:
    code = (
        "import json, sys\n"
        "import iterate.targets.net\n"
        "heavy = ('pandas', 'sklearn', 'lightgbm', 'torch', 'iterate.targets.dl', 'iterate.core')\n"
        "print(json.dumps([name for name in heavy if name in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_every_recipe_shape_fit_accepts_is_in_the_round_trip_list() -> None:
    """A new backbone, a new unfreeze value or a new recipe field can change the
    network a recipe builds. It joins ROUND_TRIP_SHAPES first, and only then this list."""
    from tests.unit._torch_checks import ROUND_TRIP_SHAPES

    assert {shape["backbone"] for shape in ROUND_TRIP_SHAPES} == set(dl._CHOICES["backbone"])
    assert {shape["unfreeze"] for shape in ROUND_TRIP_SHAPES} == set(dl._CHOICES["unfreeze"])
    assert {shape["task"] for shape in ROUND_TRIP_SHAPES} == {"classification", "regression"}
    for shape in ROUND_TRIP_SHAPES:
        changes = {k: v for k, v in shape.items() if k != "task"}
        Recipe.from_changes(changes, task=shape["task"])
    assert {f.name for f in fields(Recipe)} == {
        "backbone",
        "image_size",
        "unfreeze",
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "schedule",
        "augment",
        "label_smoothing",
        "head_init",
        "seed",
        "layers",
        "head",
        "drop_stages",
    }
    assert {bool(shape.get("layers")) for shape in ROUND_TRIP_SHAPES} == {False, True}
    assert {bool(shape.get("head")) for shape in ROUND_TRIP_SHAPES} == {False, True}
    assert {int(shape.get("drop_stages", 0)) for shape in ROUND_TRIP_SHAPES} == {0, 1}


# ─── layers, heads and dropped stages, with no torch ─────────────────────────


class _Fake:
    """A stand-in module: what it is, the numbers it was built with, and its parts."""

    def __init__(self, kind: Any, kids: dict[str, Any] | None = None, **fields: Any) -> None:
        self.kind, self.kids = kind, kids or {}
        self.__dict__.update(fields)

    def __getitem__(self, i: int) -> Any:
        return self.order[i]

    def get_submodule(self, name: str) -> Any:
        node: Any = self
        for step in name.split("."):
            if not isinstance(node, _Fake) or step not in node.kids:
                raise AttributeError(name)
            node = node.kids[step]
        return node

    def set_submodule(self, name: str, new: Any) -> None:
        stem, _, last = name.rpartition(".")
        node = self.get_submodule(stem) if stem else self
        node.kids[last] = new


def _sequential(*parts: Any) -> _Fake:
    if len(parts) == 1 and isinstance(parts[0], dict):
        return _Fake("sequential", kids=dict(parts[0]), order=list(parts[0].values()))
    return _Fake("sequential", order=list(parts))


_FAKE_NN = SimpleNamespace(
    Identity=lambda: _Fake("identity"),
    Linear=lambda a, b: _Fake("linear", in_features=a, out_features=b),
    ReLU=lambda: _Fake("relu"),
    Dropout=lambda p: _Fake("dropout", p=p),
    Dropout2d=lambda p: _Fake("dropout2d", p=p),
    Conv2d=lambda a, b, k, stride=1, padding=0: _Fake("conv", channels=(a, b), kernel=k),
    BatchNorm2d=lambda c: _Fake("batchnorm", channels=c),
    MaxPool2d=lambda k: _Fake("maxpool"),
    AvgPool2d=lambda k: _Fake("avgpool"),
    AdaptiveAvgPool2d=lambda k: _Fake("gap"),
    Flatten=lambda: _Fake("flatten"),
    Sequential=_sequential,
)
_FAKE_TORCH = SimpleNamespace(nn=_FAKE_NN)


def _put(root: _Fake, name: str, node: Any) -> None:
    stem, _, last = name.rpartition(".")
    parent = root
    for step in stem.split(".") if stem else []:
        parent = parent.kids.setdefault(step, _Fake(step))
    parent.kids[last] = node


def _stock(backbone: str, *, features: int | None = None, without: str = "") -> _Fake:
    """The module tree STAGES names, as a torchvision of this shape would hand it over."""
    root = _Fake(backbone)
    for names, width in net.STAGES[backbone]:
        for name in names:
            if name != without:
                _put(root, name, _Fake(f"stage:{name}", width=width))
    if backbone == "convnext_tiny":
        _put(root, "classifier.0", _Fake("layernorm2d", eps=1e-6))
    wide = net.STAGES[backbone][-1][1] if features is None else features
    _put(root, net.BACKBONES[backbone][1], _Fake("linear", in_features=wide, out_features=1000))
    return root


def _torchvision(monkeypatch: pytest.MonkeyPatch, tree: _Fake) -> list[dict[str, Any]]:
    """A torchvision that records every call and hands back one prepared tree."""
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> _Fake:
        calls.append(kwargs)
        return tree

    models = ModuleType("torchvision.models")
    setattr(models, str(tree.kind), factory)
    package = ModuleType("torchvision")
    package.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", package)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    return calls


def _built(recipe: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tree: _Fake) -> Any:
    _torchvision(monkeypatch, tree)
    return net.model_for(_FAKE_TORCH, recipe, 5, None, pretrained=False)


def test_a_head_alone_replaces_the_final_layer_and_leaves_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _stock("resnet18")
    model = _built(
        {"backbone": "resnet18", "head": [("linear", 64), ("dropout", 0.5)]}, monkeypatch, tree
    )
    fc = model.get_submodule("fc")
    assert [part.kind for part in fc.order] == ["linear", "relu", "dropout", "linear"]
    assert (fc[0].in_features, fc[0].out_features) == (512, 64)
    assert (fc[-1].in_features, fc[-1].out_features) == (64, 5)
    assert [model.get_submodule(f"layer{i}").kind for i in (1, 2, 3, 4)] == [
        f"stage:layer{i}" for i in (1, 2, 3, 4)
    ]


@pytest.mark.parametrize(
    ("dropped", "width", "gone"), [(1, 1024, ["layer4"]), (2, 512, ["layer3", "layer4"])]
)
def test_dropping_stages_turns_them_into_identity_and_narrows_the_head(
    monkeypatch: pytest.MonkeyPatch, dropped: int, width: int, gone: list[str]
) -> None:
    tree = _stock("resnet50")
    model = _built({"backbone": "resnet50", "drop_stages": dropped}, monkeypatch, tree)
    assert [
        name
        for name in ("layer1", "layer2", "layer3", "layer4")
        if model.get_submodule(name).kind == "identity"
    ] == gone
    fc = model.get_submodule("fc")
    assert (fc.kind, fc.in_features, fc.out_features) == ("linear", width, 5)


def test_a_dropped_convnext_gets_a_norm_of_the_width_that_is_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its head sits behind a norm sized to the stage below it, so the norm is rebuilt."""
    tree = _stock("convnext_tiny")
    model = _built({"backbone": "convnext_tiny", "drop_stages": 1}, monkeypatch, tree)
    norm = model.get_submodule("classifier.0")
    assert (norm.kind, norm.eps) == (384, 1e-6)
    assert model.get_submodule("features.6").kind == "identity"
    assert model.get_submodule("features.7").kind == "identity"
    assert model.get_submodule("features.5").kind == "stage:features.5"


def test_a_torchvision_that_moved_the_stages_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = _stock("resnet18", without="layer4")
    with pytest.raises(net.RecipeError, match=r"builds resnet18 in a shape this iterate"):
        _built({"backbone": "resnet18", "drop_stages": 1}, monkeypatch, missing)
    resized = _stock("resnet18", features=99)
    with pytest.raises(net.RecipeError, match=r"install torchvision 0\.24 or newer"):
        _built({"backbone": "resnet18", "head": [("linear", 64)]}, monkeypatch, resized)


def test_a_stack_builds_its_own_network_and_asks_torchvision_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _torchvision(monkeypatch, _stock("resnet18"))
    recipe = {"backbone": "layers_net", "layers": "conv(32) pool conv(64) pool dropout(0.3)"}
    model = net.model_for(_FAKE_TORCH, recipe, 5, None, pretrained=False)
    assert calls == []
    assert sorted(model.kids) == ["body", "head"]
    assert [part.kind for part in model.get_submodule("body").order] == [
        "conv",
        "batchnorm",
        "relu",
        "maxpool",
        "conv",
        "batchnorm",
        "relu",
        "maxpool",
        "gap",
        "flatten",
    ]
    head = model.get_submodule("head")
    assert [part.kind for part in head.order] == ["dropout", "linear"]
    assert (head[-1].in_features, head[-1].out_features) == (64, 5)


def test_a_recipe_with_none_of_the_new_fields_takes_the_build_it_always_took() -> None:
    calls: list[tuple[Any, ...]] = []
    recipe = asdict(Recipe(backbone="resnet50", unfreeze="all", epochs=3))
    net.model_for("torch", recipe, 5, "probe", build=lambda *a, **k: calls.append(a))
    assert calls == [("torch", "resnet50", 5, "probe")]


# ─── what a saved file carries ───────────────────────────────────────────────

_TORCH = SimpleNamespace(__version__="2.9.0")


def _meta(**changes: Any) -> dict[str, Any]:
    recipe = asdict(Recipe(backbone="resnet18", unfreeze="all", epochs=3, image_size=160))
    meta = net.saved_meta(
        _TORCH, recipe, task="classification", classes=["cat", "dog", "emu"], outputs=3
    )
    return {**meta, **changes}


def _saved(meta: dict[str, Any]) -> dict[str, Any]:
    return {net.SAVED_KEY: meta, "state_dict": {"fc.weight": np.zeros((3, 4), np.float32)}}


def _tampered(**changes: Any) -> dict[str, Any]:
    """A saved file whose recipe says something no iterate run would write."""
    return _saved(_meta(recipe={**_meta()["recipe"], **changes}))


def test_the_metadata_is_plain_python_and_names_everything_a_loader_needs() -> None:
    meta = _meta()
    assert json.loads(json.dumps(meta)) == meta
    assert meta["format"] == net.SAVED_FORMAT == 1
    assert (meta["task"], meta["classes"], meta["outputs"]) == (
        "classification",
        ["cat", "dog", "emu"],
        3,
    )
    assert (meta["image_size"], meta["recipe"]["backbone"]) == (160, "resnet18")
    assert meta["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert meta["std"] == pytest.approx([0.229, 0.224, 0.225])
    assert (meta["label_centre"], meta["label_spread"]) == (0.0, 1.0)
    assert set(meta["versions"]) == {"iterate", "torch", "torchvision", "python"}
    assert meta["versions"]["torch"] == "2.9.0"


def test_numpy_class_names_and_label_statistics_are_saved_as_plain_values() -> None:
    recipe = asdict(Recipe(backbone="simple_cnn", unfreeze="all", epochs=2, image_size=64))
    classes = list(np.array([3, 5, 8]))
    meta = net.saved_meta(_TORCH, recipe, task="classification", classes=classes, outputs=3)
    assert [type(c) for c in meta["classes"]] == [int, int, int]
    number = net.saved_meta(
        _TORCH,
        recipe,
        task="regression",
        classes=[],
        outputs=1,
        centre=np.float64(41.5),
        spread=np.float32(2.0),
    )
    assert number["classes"] is None
    assert [type(number[k]) for k in ("label_centre", "label_spread")] == [float, float]
    assert net.checked_meta(_saved(number), "net.pt") is number


def test_recipe_values_that_came_from_numpy_are_saved_as_plain_values() -> None:
    """`np.float64` is a float, so a recipe takes one, and `weights_only=True` will not
    open a file that holds one."""
    recipe = asdict(
        Recipe(
            backbone="simple_cnn",
            unfreeze="all",
            epochs=2,
            image_size=64,
            lr=np.logspace(-4, -2, 3)[1],
            label_smoothing=np.float64(0.1),
        )
    )
    meta = net.saved_meta(_TORCH, recipe, task="classification", classes=["a", "b"], outputs=2)
    assert [type(meta["recipe"][k]) for k in ("lr", "label_smoothing")] == [float, float]
    assert meta["recipe"] == recipe
    assert net.checked_meta(_saved(meta), "net.pt") is meta


def test_a_file_a_run_saved_passes_every_check() -> None:
    meta = _meta()
    assert net.checked_meta(_saved(meta), "net.pt") is meta
    topped = _meta(recipe={**meta["recipe"], "drop_stages": 2, "head": [["linear", 64]]})
    assert net.checked_meta(_saved(topped), "net.pt") is topped


@pytest.mark.parametrize(
    ("saved", "told"),
    [
        ([1, 2, 3], "is not a network an iterate image run saved"),
        ({"state_dict": {}}, "is not a network an iterate image run saved"),
        ({net.SAVED_KEY: {"format": "1"}}, "is not a network an iterate image run saved"),
        (_saved(_meta(format=2)), "is saved-network format 2 and this iterate reads format 1"),
        (_saved(_meta(recipe={"backbone": "hf_hub:someone/net"})), "its backbone is not one of"),
        (_saved(_meta(recipe="resnet18")), "its backbone is not one of"),
        (_tampered(drop_stages=99), "its drop_stages is 99, not a whole number from 0 to 2"),
        (_tampered(drop_stages=-1), "its drop_stages is -1"),
        (_tampered(drop_stages=1.5), "its drop_stages is 1.5"),
        (_tampered(drop_stages=True), "its drop_stages is True"),
        (
            _tampered(backbone="simple_cnn", drop_stages=1),
            "simple_cnn has no stages to drop",
        ),
        (_tampered(layers=[["conv", 999999]]), "channels=999999 is outside 4 to 512"),
        (_tampered(head=[["linear", 4096]]), "linear takes one width between 8 and 2048"),
        (_saved(_meta(task="ranking")), "its task is 'ranking'"),
        (_saved(_meta(image_size=100_000)), "its image_size is 100000, outside 32 to 384"),
        (_saved(_meta(image_size=True)), "its image_size is True"),
        (_saved(_meta(image_size="160")), "its image_size is '160'"),
        (_saved(_meta(outputs=0)), "its output count is 0"),
        (_saved(_meta(outputs=4)), "it names 3 classes for 4 outputs"),
        (_saved(_meta(classes=None)), "it names no classes for 3 outputs"),
        (_saved(_meta(task="regression")), "a number model needs one output"),
        (
            _saved(_meta(task="regression", outputs=1, label_centre=float("nan"))),
            "a finite label centre and spread",
        ),
        (_saved(_meta(mean=[0.5, 0.5, 0.5])), "its pixel mean is not the one"),
        (_saved(_meta(std=[0.229, 0.224])), "its pixel std is not the one"),
        ({net.SAVED_KEY: _meta(), "state_dict": {}}, "its weights are not a table of tensors"),
        ({net.SAVED_KEY: _meta(), "state_dict": {"fc.weight": [0.0]}}, "its weights are not"),
        ({net.SAVED_KEY: _meta()}, "its weights are not a table of tensors"),
    ],
)
def test_a_file_that_fails_a_check_is_refused_in_plain_words(saved: Any, told: str) -> None:
    with pytest.raises(ValueError, match=told) as refused:
        net.checked_meta(saved, "/models/net.pt")
    assert "/models/net.pt" in str(refused.value)


# ─── torch itself ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("version", ["2.6.0", "2.9.0+cu121", "2.10.1", "3.0.0a0+git1f2e"])
def test_a_new_enough_torch_is_handed_back(version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert net.torch_at_least("2.6") is fake


@pytest.mark.parametrize("version", ["2.5.1", "1.13.1+cpu", "2.5.0a0+git", "unknown"])
def test_a_torch_that_can_run_code_from_a_file_is_refused(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match=r"needs torch 2\.6 or newer, and this is torch"):
        net.torch_at_least()


def test_no_torch_at_all_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pip install 'iterate-ai\[vision\]'"):
        net.torch_at_least()


@pytest.mark.parametrize(
    ("cuda", "mps", "asked", "picked"),
    [
        (True, True, None, "cuda"),
        (False, True, None, "mps"),
        (False, False, None, "cpu"),
        (True, True, "cpu", "cpu"),
    ],
)
def test_the_device_is_the_one_asked_for_or_the_best_one_there(
    cuda: bool, mps: bool, asked: str | None, picked: str
) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    assert net.pick_device(torch, asked) == picked


# ─── staging a fit's weights ─────────────────────────────────────────────────


class _Weight:
    def detach(self) -> _Weight:
        return self

    def cpu(self) -> _Weight:
        return self


def _model() -> Any:
    return SimpleNamespace(state_dict=lambda: {"fc.weight": _Weight()})


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed"),
        OSError(28, "No space left on device"),
    ],
)
def test_a_full_disk_costs_a_fit_its_weights_file_and_nothing_else(
    tmp_path: Path, error: Exception
) -> None:
    path = tmp_path / "1.pt"

    def save(state: dict[str, Any], to: Path) -> None:
        to.write_bytes(b"half a file")
        raise error

    log: list[str] = []
    dl._stage(SimpleNamespace(save=save), _model(), path, log.append)
    assert not path.exists()
    assert log == [f"this fit has no weights file: {error}"]


def test_a_staged_fit_holds_the_state_on_the_cpu(tmp_path: Path) -> None:
    written: dict[str, Any] = {}

    def save(state: dict[str, Any], to: Path) -> None:
        written.update(state)
        to.write_bytes(b"weights")

    log: list[str] = []
    dl._stage(SimpleNamespace(save=save), _model(), tmp_path / "1.pt", log.append)
    assert list(written) == ["fc.weight"]
    assert (tmp_path / "1.pt").read_bytes() == b"weights"
    assert log == []


def test_a_fit_job_saves_nothing_unless_it_is_told_where() -> None:
    """The baseline, the eval sweep and every stored ceiling build their jobs without
    it, so none of them writes a file."""
    assert {f.name: f.default for f in fields(FitJob)}["save_to"] is None
