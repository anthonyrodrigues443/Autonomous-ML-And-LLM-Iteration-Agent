"""Adversarial dataset SHAPES — the CI half of the corpus.

Shares the word corpus with the rest of this directory and none of the machinery:
no LLM, no versions, no ceilings, seconds not hours. It answers a different
question. The benchmark corpus asks "is the agent getting better"; this asks "does
the data layer survive a real file".

It exists because of a specific failure. The v0.4 certification found three
crash-class bugs — a non-UTF-8 CSV, boolean columns, an integer regression target —
and 583 unit tests had missed all three, because every fixture in the suite was
built the same way and therefore had the same shape. The tests were not thin, they
were monotonous. Each generator below is one shape that actually broke the tool,
kept small enough to run on every push.

Adding a shape here is the standing fix for that class: when a real file breaks the
loader, its shape becomes a generator, and the corpus gets one dimension less
uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_SEED = 7
_ROWS = 120


@dataclass(frozen=True)
class Shape:
    """One nasty-but-legal dataset."""

    name: str
    target: str
    why: str
    build: Callable[[Path], Path]
    expect_classification: bool


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _latin1_accents(directory: Path) -> Path:
    """A European export encoded latin-1 rather than UTF-8.

    Broke `pd.read_csv`'s UTF-8 assumption with a decoding traceback on a file that
    opens fine in any spreadsheet.

    Accented product names, not a euro sign: latin-1 has no euro codepoint at all.
    A spreadsheet export with a real € is cp1252, where that byte decodes under the
    latin-1 fallback as a control character instead of raising — mangled rather than
    crashed, which is a separate problem this corpus does not yet cover.
    """
    rng = _rng()
    frame = pd.DataFrame(
        {
            "model": [f"Ordinateur Légère Ünité {i}" for i in range(_ROWS)],
            "weight_kg": rng.uniform(1.0, 3.5, _ROWS).round(2),
            "price_euros": rng.uniform(200, 2500, _ROWS).round(2),
        }
    )
    path = directory / "latin1_accents.csv"
    frame.to_csv(path, index=False, encoding="latin-1")
    return path


def _boolean_columns(directory: Path) -> Path:
    """Real booleans in the features and the target, not 0/1 integers."""
    rng = _rng()
    frame = pd.DataFrame(
        {
            "has_wifi": rng.integers(0, 2, _ROWS).astype(bool),
            "is_premium": rng.integers(0, 2, _ROWS).astype(bool),
            "screen_inches": rng.uniform(11.0, 17.0, _ROWS).round(1),
            "churned": rng.integers(0, 2, _ROWS).astype(bool),
        }
    )
    path = directory / "boolean_columns.csv"
    frame.to_csv(path, index=False)
    return path


def _integer_regression_target(directory: Path) -> Path:
    """A count or a price: integer dtype, hundreds of distinct values.

    Read as class labels it becomes hundreds of classes and the stratified split
    raises before the run can start, which is how diamonds became unloadable.
    """
    rng = _rng()
    frame = pd.DataFrame(
        {
            "carat": rng.uniform(0.2, 3.0, _ROWS).round(2),
            "cut": rng.choice(["Fair", "Good", "Ideal"], _ROWS),
            "price": rng.integers(300, 18000, _ROWS),
        }
    )
    path = directory / "integer_regression_target.csv"
    frame.to_csv(path, index=False)
    return path


def _high_cardinality_categorical(directory: Path) -> Path:
    """A near-unique string column, the kind one-hot encoding explodes on."""
    rng = _rng()
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(_ROWS)],
            "city": rng.choice([f"city_{i}" for i in range(_ROWS // 2)], _ROWS),
            "spend": rng.uniform(10, 900, _ROWS).round(2),
            "renewed": rng.choice(["Yes", "No"], _ROWS),
        }
    )
    path = directory / "high_cardinality_categorical.csv"
    frame.to_csv(path, index=False)
    return path


def _missing_values(directory: Path) -> Path:
    """Holes in the features, including one column that is entirely empty."""
    rng = _rng()
    tenure = rng.uniform(0, 72, _ROWS).round(1)
    tenure[rng.integers(0, _ROWS, _ROWS // 4)] = np.nan
    plan = pd.Series(rng.choice(["basic", "pro"], _ROWS))
    plan[rng.integers(0, _ROWS, _ROWS // 5)] = None
    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "plan": plan,
            "never_populated": [None] * _ROWS,
            "churned": rng.choice(["Yes", "No"], _ROWS),
        }
    )
    path = directory / "missing_values.csv"
    frame.to_csv(path, index=False)
    return path


SHAPES: list[Shape] = [
    Shape(
        name="latin1_accents",
        target="price_euros",
        why="non-UTF-8 file (v0.4 certification crash)",
        build=_latin1_accents,
        expect_classification=False,
    ),
    Shape(
        name="boolean_columns",
        target="churned",
        why="bool dtype features and target (v0.4 certification crash)",
        build=_boolean_columns,
        expect_classification=True,
    ),
    Shape(
        name="integer_regression_target",
        target="price",
        why="integer target with many distinct values (v0.4 certification crash)",
        build=_integer_regression_target,
        expect_classification=False,
    ),
    Shape(
        name="high_cardinality_categorical",
        target="renewed",
        why="near-unique string column",
        build=_high_cardinality_categorical,
        expect_classification=True,
    ),
    Shape(
        name="missing_values",
        target="churned",
        why="missing values including an all-null column",
        build=_missing_values,
        expect_classification=True,
    ),
]


def write_all(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return {shape.name: shape.build(directory) for shape in SHAPES}


__all__ = ["SHAPES", "Shape", "write_all"]
