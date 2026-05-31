"""End-to-end integration: the tabular substrate on the real Telco churn dataset.

Exercises load_csv -> ModelTarget -> model factory -> LocalExecutor on the
committed prepared data (no LLM), including graceful failure capture. Marked
`integration` (opt-in) because it reads the dataset and trains a few models.
The agentic loop (with a real LLM) is covered in test_agentic_loop_live.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

_CLEAN = Path(__file__).resolve().parents[2] / "examples" / "churn_tabular" / "data.clean.csv"


def _target() -> ModelTarget:
    dataset = load_csv(_CLEAN, target="Churn")
    return ModelTarget(dataset, metric="f1")


@pytest.mark.integration
def test_churn_substrate_end_to_end() -> None:
    if not _CLEAN.exists():
        pytest.skip("data.clean.csv not present; run examples/churn_tabular/prepare.py")

    target = _target()
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
