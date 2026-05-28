"""End-to-end integration: the tabular substrate on the real Telco churn dataset.

Exercises load_csv -> ModelTarget -> model factory -> LocalExecutor on real data,
including graceful failure capture. Marked `integration` (opt-in: run with
`pytest -m integration`) because it reads the committed dataset and trains a few
models, so it is slower than the unit suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from iterate.adapters.compute.local import LocalExecutor
from iterate.schemas.experiment import Candidate

if TYPE_CHECKING:
    from types import ModuleType

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "churn_tabular" / "run.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("churn_run", _EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_churn_substrate_end_to_end() -> None:
    run = _load_example()
    if not run.DATA.exists():
        pytest.skip("Telco churn data.csv not present")

    target = run.build_target()
    executor = LocalExecutor()

    baseline = executor.execute(target)
    assert baseline.succeeded
    assert baseline.metrics is not None
    assert baseline.metrics.primary == "f1"
    assert baseline.duration_seconds is not None

    good = executor.execute(
        target,
        Candidate(
            description="xgboost",
            changes={"model": "xgboost.XGBClassifier", "params": {"n_estimators": 50}},
            rationale="boosted trees",
        ),
    )
    assert good.succeeded
    assert good.metrics is not None

    bad = executor.execute(
        target,
        Candidate(description="broken", changes={"params": {"max_iter": -1}}, rationale="bad"),
    )
    assert not bad.succeeded
    assert bad.error is not None
