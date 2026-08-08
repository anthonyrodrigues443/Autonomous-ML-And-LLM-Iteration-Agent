"""The adversarial shape corpus — the CI defence against dataset-shape bugs.

Three of these shapes are crashes the v0.4 certification found by running the tool,
after 583 unit tests had missed all three: every fixture in that suite was built the
same way, so the suite was monotonous rather than thin. Each shape below is one real
file the loader could not handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.shapes import SHAPES, Shape, write_all
from iterate.adapters.data.tabular import load_csv, looks_like_classification

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def shape_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_all(tmp_path_factory.mktemp("shapes"))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_the_loader_survives_the_shape(shape: Shape, shape_files: dict[str, Path]) -> None:
    dataset = load_csv(shape_files[shape.name], target=shape.target)

    assert dataset.n_train > 0
    assert dataset.n_test > 0
    assert shape.target not in dataset.features


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_the_task_is_read_correctly(shape: Shape, shape_files: dict[str, Path]) -> None:
    """An integer price read as class labels is how diamonds became unloadable, so
    the task heuristic is asserted per shape rather than assumed."""
    dataset = load_csv(shape_files[shape.name], target=shape.target)

    assert looks_like_classification(dataset.train_target) is shape.expect_classification


def test_the_latin1_file_is_genuinely_not_utf8(shape_files: dict[str, Path]) -> None:
    """Guards the fixture itself. If pandas ever writes this one as UTF-8 the shape
    stops testing anything, and the test above would keep passing regardless."""
    raw = Path(shape_files["latin1_accents"]).read_bytes()

    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_every_shape_records_why_it_exists() -> None:
    for shape in SHAPES:
        assert shape.why, f"{shape.name} has no reason recorded"
