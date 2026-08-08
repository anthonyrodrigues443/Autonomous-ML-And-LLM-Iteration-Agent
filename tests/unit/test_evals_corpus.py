"""Dataset discovery and content fingerprinting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evals import corpus
from evals.config import REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path

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
