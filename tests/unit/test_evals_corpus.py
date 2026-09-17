"""Dataset discovery and content fingerprinting."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import corpus
from evals.config import REPO_ROOT

pytestmark = pytest.mark.unit


def _spec(root: Path, name: str, body: str, *, data: str | None = "a,b\n1,2\n") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "dataset.toml").write_text(body, encoding="utf-8")
    if data is not None:
        (folder / "data.csv").write_text(data, encoding="utf-8")
    return folder


def test_a_dataset_folder_becomes_an_entry(tmp_path: Path) -> None:
    _spec(tmp_path, "churn", 'target = "Churn"\nmetric = "f1"\nsource = "somewhere"\n')

    (dataset,) = corpus.load(tmp_path)

    assert dataset.name == "churn"
    assert dataset.target == "Churn"
    assert dataset.metric == "f1"
    assert dataset.source == "somewhere"
    assert dataset.available


def test_a_missing_csv_is_unavailable_rather_than_an_error(tmp_path: Path) -> None:
    """Most of the corpus is gitignored, so a fresh clone legitimately has almost
    none of it. Discovery must still work."""
    _spec(tmp_path, "diamonds", 'target = "price"\nmetric = "rmse"\n', data=None)

    (dataset,) = corpus.load(tmp_path)

    assert not dataset.available


def test_a_spec_missing_its_target_is_rejected(tmp_path: Path) -> None:
    _spec(tmp_path, "broken", 'metric = "f1"\n')

    with pytest.raises(corpus.BadDatasetSpecError, match="target"):
        corpus.load(tmp_path)


def test_a_data_key_resolves_from_the_repo_root(tmp_path: Path) -> None:
    """How the datasets already sitting in examples/ join the corpus without being
    copied."""
    _spec(
        tmp_path,
        "churn",
        'data = "examples/churn_tabular/data.clean.csv"\ntarget = "Churn"\nmetric = "f1"\n',
        data=None,
    )

    (dataset,) = corpus.load(tmp_path)

    assert dataset.path == REPO_ROOT / "examples/churn_tabular/data.clean.csv"
    assert dataset.available


def test_the_content_hash_follows_the_bytes(tmp_path: Path) -> None:
    folder = _spec(tmp_path, "churn", 'target = "Churn"\nmetric = "f1"\n')
    (dataset,) = corpus.load(tmp_path)
    before = dataset.content_hash()

    (folder / "data.csv").write_text("a,b\n1,3\n", encoding="utf-8")

    assert dataset.content_hash() != before


def test_selecting_an_unknown_dataset_raises(tmp_path: Path) -> None:
    """A typo must not quietly sweep fewer datasets than were asked for."""
    _spec(tmp_path, "churn", 'target = "Churn"\nmetric = "f1"\n')

    with pytest.raises(corpus.BadDatasetSpecError, match="unknown"):
        corpus.select(["chrun"], tmp_path)


def test_selecting_nothing_returns_everything(tmp_path: Path) -> None:
    _spec(tmp_path, "a", 'target = "t"\nmetric = "f1"\n')
    _spec(tmp_path, "b", 'target = "t"\nmetric = "f1"\n')

    assert [d.name for d in corpus.select(None, tmp_path)] == ["a", "b"]


def test_the_shipped_registry_is_loadable_and_names_real_metrics() -> None:
    """The tracked dataset.toml files are the corpus definition, so a typo in one is
    a broken corpus rather than a broken test fixture."""
    from iterate.core.scoring import REGISTRY

    datasets = corpus.load()

    assert datasets, "no dataset specs found under evals/datasets/"
    for dataset in datasets:
        assert dataset.metric in REGISTRY, f"{dataset.name}: unknown metric {dataset.metric}"


def test_a_vision_dataset_says_its_family(tmp_path: Path) -> None:
    _spec(tmp_path, "flowers", 'target = "label"\nmetric = "accuracy"\nfamily = "vision"\n')

    (dataset,) = corpus.load(tmp_path)

    assert dataset.is_vision
    assert not dataset.is_prompt_task


def test_an_unknown_family_is_refused(tmp_path: Path) -> None:
    _spec(tmp_path, "sounds", 'target = "y"\nmetric = "f1"\nfamily = "audio"\n')

    with pytest.raises(corpus.BadDatasetSpecError, match="family 'audio'"):
        corpus.load(tmp_path)


def test_the_two_vision_datasets_are_registered() -> None:
    by_name = {dataset.name: dataset for dataset in corpus.load()}
    for name in ("flowers102", "eurosat"):
        assert by_name[name].is_vision
        assert (by_name[name].target, by_name[name].metric) == ("label", "accuracy")
        assert by_name[name].path == REPO_ROOT / "examples" / name / "data.csv"


def test_a_vision_key_moves_when_an_image_changes_behind_the_same_csv(tmp_path: Path) -> None:
    from PIL import Image

    folder = tmp_path / "tiles"
    (folder / "images").mkdir(parents=True)
    rows = []
    for i in range(4):
        Image.new("RGB", (4, 4), (i * 40, 0, 0)).save(folder / "images" / f"{i}.png")
        rows.append(f"images/{i}.png,c{i % 2}")
    (folder / "dataset.toml").write_text(
        'target = "label"\nmetric = "accuracy"\nfamily = "vision"\n', encoding="utf-8"
    )
    (folder / "data.csv").write_text("image,label\n" + "\n".join(rows) + "\n", encoding="utf-8")

    (dataset,) = corpus.load(tmp_path)
    before = dataset.content_hash()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(folder / "images" / "2.png")

    assert dataset.content_hash() != before


def test_a_family_that_contradicts_the_task_line_is_refused(tmp_path: Path) -> None:
    _spec(tmp_path / "a", "x", 'target = "y"\nmetric = "f1"\nfamily = "prompt"\n')
    with pytest.raises(corpus.BadDatasetSpecError, match="needs a task line"):
        corpus.load(tmp_path / "a")
    _spec(tmp_path / "b", "x", 'target = "y"\nmetric = "f1"\nfamily = "vision"\ntask = "say"\n')
    with pytest.raises(corpus.BadDatasetSpecError, match="makes this a prompt dataset"):
        corpus.load(tmp_path / "b")


def _old_content_hash(path: Path, target: str, *, is_vision: bool) -> str:
    """content_hash as it stood before a holdout could be named, copied verbatim."""
    import hashlib

    import pandas as pd

    from iterate.adapters.data.images import detect_image_column

    def image_digest(csv: Path, target: str) -> str:
        frame = pd.read_csv(csv)
        column = detect_image_column(frame, [c for c in frame.columns if c != target], csv)
        assert column is not None
        digest = hashlib.sha256()
        for value in frame[column.column].astype(str):
            path = Path(value) if Path(value).is_absolute() else column.root / value
            try:
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            except OSError:
                digest.update(b"missing")
        return digest.hexdigest()

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if is_vision:
        digest.update(image_digest(path, target).encode())
    return digest.hexdigest()[:16]


def _tiles(folder: Path, name: str, first: int, count: int) -> None:
    from PIL import Image

    (folder / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(first, first + count):
        Image.new("RGB", (4, 4), (i * 40 % 256, 0, 0)).save(folder / "images" / f"{i}.png")
        rows.append(f"images/{i}.png,c{i % 2}")
    (folder / name).write_text("image,label\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _held(tmp_path: Path, family: str = "", holdout: str = "holdout.csv") -> Path:
    """A spec naming both files by absolute path, which the repo root leaves as they are."""
    folder = tmp_path / "specs" / "split"
    folder.mkdir(parents=True)
    kind = f'family = "{family}"\n' if family else ""
    (folder / "dataset.toml").write_text(
        f'data = "{tmp_path / "train.csv"}"\nholdout = "{tmp_path / holdout}"\n'
        f'target = "label"\nmetric = "accuracy"\n{kind}',
        encoding="utf-8",
    )
    return tmp_path / "specs"


def test_a_holdout_key_resolves_from_the_repo_root(tmp_path: Path) -> None:
    _spec(
        tmp_path,
        "storm",
        'data = "examples/storm/train.csv"\nholdout = "examples/storm/holdout.csv"\n'
        'target = "wind"\nmetric = "rmse"\n',
        data=None,
    )

    (dataset,) = corpus.load(tmp_path)

    assert dataset.path == REPO_ROOT / "examples/storm/train.csv"
    assert dataset.holdout == REPO_ROOT / "examples/storm/holdout.csv"


def test_without_a_holdout_key_there_is_no_holdout(tmp_path: Path) -> None:
    _spec(tmp_path, "churn", 'target = "Churn"\nmetric = "f1"\n')

    (dataset,) = corpus.load(tmp_path)

    assert dataset.holdout is None


@pytest.mark.parametrize(("train", "holdout"), [(True, False), (False, True), (True, True)])
def test_a_split_dataset_is_available_only_with_both_files(
    tmp_path: Path, train: bool, holdout: bool
) -> None:
    for present, name in ((train, "train.csv"), (holdout, "holdout.csv")):
        if present:
            (tmp_path / name).write_text("a,label\n1,x\n", encoding="utf-8")

    (dataset,) = corpus.load(_held(tmp_path))

    assert dataset.available is (train and holdout)
    expected = [
        tmp_path / n for ok, n in ((train, "train.csv"), (holdout, "holdout.csv")) if not ok
    ]
    assert dataset.missing == expected


def test_without_a_holdout_the_key_is_the_one_the_old_code_made(tmp_path: Path) -> None:
    _spec(tmp_path, "churn", 'target = "Churn"\nmetric = "f1"\n', data="a,Churn\n1,2\n3,4\n")
    folder = tmp_path / "tiles"
    folder.mkdir()
    _tiles(folder, "data.csv", 0, 4)
    (folder / "dataset.toml").write_text(
        'target = "label"\nmetric = "accuracy"\nfamily = "vision"\n', encoding="utf-8"
    )

    churn, tiles = corpus.load(tmp_path)

    assert churn.content_hash() == _old_content_hash(churn.path, "Churn", is_vision=False)
    assert tiles.content_hash() == _old_content_hash(tiles.path, "label", is_vision=True)


def test_a_holdout_moves_the_key_when_its_bytes_change(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text("a,label\n1,x\n2,y\n", encoding="utf-8")
    (tmp_path / "holdout.csv").write_text("a,label\n3,x\n", encoding="utf-8")
    (dataset,) = corpus.load(_held(tmp_path))
    before = dataset.content_hash()

    assert before != _old_content_hash(dataset.path, "label", is_vision=False)
    (tmp_path / "holdout.csv").write_text("a,label\n3,y\n", encoding="utf-8")

    assert dataset.content_hash() != before


def test_a_vision_holdout_moves_the_key_when_one_of_its_images_changes(tmp_path: Path) -> None:
    from PIL import Image

    _tiles(tmp_path, "train.csv", 0, 4)
    _tiles(tmp_path, "holdout.csv", 4, 2)
    (dataset,) = corpus.load(_held(tmp_path, "vision"))
    before = dataset.content_hash()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(tmp_path / "images" / "5.png")

    assert dataset.content_hash() != before


def test_a_vision_holdout_in_another_folder_reads_its_image_paths_beside_its_own_csv(
    tmp_path: Path,
) -> None:
    from PIL import Image

    _tiles(tmp_path, "train.csv", 0, 4)
    _tiles(tmp_path / "elsewhere", "holdout.csv", 4, 2)
    (dataset,) = corpus.load(_held(tmp_path, "vision", holdout="elsewhere/holdout.csv"))
    before = dataset.content_hash()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(tmp_path / "images" / "5.png")
    assert dataset.content_hash() == before

    Image.new("RGB", (4, 4), (0, 255, 0)).save(tmp_path / "elsewhere" / "images" / "5.png")
    assert dataset.content_hash() != before


def test_loading_a_split_dataset_keeps_the_users_split(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "a,label\n" + "".join(f"{i},{i % 2}\n" for i in range(8)), encoding="utf-8"
    )
    (tmp_path / "holdout.csv").write_text("a,label\n8,0\n9,1\n10,0\n", encoding="utf-8")
    (dataset,) = corpus.load(_held(tmp_path))

    loaded = corpus.load_data(dataset, task="classification")

    assert loaded.user_split
    assert (loaded.n_train, loaded.n_test) == (8, 3)
    assert sorted(loaded.test_features["a"]) == [8, 9, 10]


def test_loading_without_a_holdout_splits_the_one_file(tmp_path: Path) -> None:
    rows = "".join(f"{i},{i % 2}\n" for i in range(20))
    _spec(tmp_path, "one", 'target = "label"\nmetric = "accuracy"\n', data="a,label\n" + rows)
    (dataset,) = corpus.load(tmp_path)

    loaded = corpus.load_data(dataset)

    assert not loaded.user_split
    assert loaded.n_train + loaded.n_test == 20


class _LoadedError(Exception):
    pass


@pytest.mark.parametrize("module", ["ceilings", "treatments", "prompt_ceilings"])
def test_every_table_and_prompt_sweep_loads_through_load_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    import importlib

    sweeps = importlib.import_module(f"evals.{module}")
    dataset = corpus.Dataset("one", tmp_path / "train.csv", "label", "accuracy")
    seen: list[corpus.Dataset] = []

    def spy(given: corpus.Dataset, **_: object) -> None:
        seen.append(given)
        raise _LoadedError

    monkeypatch.setattr(corpus, "load_data", spy)
    extra = (
        {"task": "t", "target_backend": "ollama", "target_model": "m"}
        if module == "prompt_ceilings"
        else {}
    )
    with pytest.raises(_LoadedError):
        sweeps.sweep(dataset, **extra)
    assert seen == [dataset]
