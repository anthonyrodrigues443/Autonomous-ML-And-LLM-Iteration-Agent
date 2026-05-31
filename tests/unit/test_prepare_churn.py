"""Tests for the churn example's data-prep step."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType

_PREPARE = Path(__file__).resolve().parents[2] / "examples" / "churn_tabular" / "prepare.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("churn_prepare", _PREPARE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_drops_id_coerces_charges_encodes_target() -> None:
    prepare = _load()
    raw = pd.DataFrame(
        {
            "customerID": ["a-1", "b-2", "c-3"],
            "tenure": [1, 34, 2],
            "TotalCharges": ["29.85", " ", "108.15"],  # blank string mid-column
            "Churn": ["No", "Yes", "No"],
        }
    )
    cleaned = prepare.clean(raw)

    assert "customerID" not in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 1  # the blank became NaN
    assert sorted(cleaned["Churn"].unique()) == [0, 1]


def test_clean_is_idempotent_on_numeric_target() -> None:
    prepare = _load()
    already = pd.DataFrame({"tenure": [1, 2], "Churn": [0, 1]})
    cleaned = prepare.clean(already)
    assert sorted(cleaned["Churn"].unique()) == [0, 1]
