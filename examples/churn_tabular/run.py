"""Churn tabular example — the substrate end-to-end on a real dataset.

Loads the public Telco Customer Churn dataset, builds a `ModelTarget`, and runs a
re-measured baseline plus a few hand-supplied candidates through the
`LocalExecutor` — the same path the Week-3 agent will drive, except here we supply
the candidates instead of the Proposer. One candidate is deliberately broken to
show the executor capturing a failure rather than crashing.

Run:  python examples/churn_tabular/run.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate, ExperimentResult
from iterate.targets.model import ModelTarget

DATA = Path(__file__).parent / "data.csv"
TARGET = "Churn"
METRIC = "f1"


def prepare(raw: Path = DATA) -> Path:
    """Dataset-specific glue (lives in the example, not the framework).

    Drops the `customerID` identifier, coerces `TotalCharges` (which has ~11 blank
    strings) to numeric, and encodes the `Churn` target `Yes`/`No` -> `1`/`0` so the
    metric panel has encodable binary labels. Writes a cleaned CSV to a temp file and
    returns its path; `ModelTarget` itself stays generic.
    """
    frame = pd.read_csv(raw)
    frame = frame.drop(columns=["customerID"])
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = (frame[TARGET] == "Yes").astype(int)
    out = Path(tempfile.gettempdir()) / "iterate_churn_clean.csv"
    frame.to_csv(out, index=False)
    return out


def build_target(metric: str = METRIC) -> ModelTarget:
    return ModelTarget(load_csv(prepare(), target=TARGET), metric=metric, name="churn-tabular")


CANDIDATES: list[Candidate] = [
    Candidate(
        description="HistGB — more iterations, lower learning rate",
        changes={"params": {"max_iter": 300, "learning_rate": 0.05}},
        rationale="more, smaller boosting steps often generalise better",
    ),
    Candidate(
        description="XGBoost — shallow boosted trees",
        changes={
            "model": "xgboost.XGBClassifier",
            "params": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05},
        },
        rationale="strong tabular baseline; shallow trees curb overfitting",
    ),
    # LightGBM is supported by the model factory but omitted from this demo: its
    # macOS-ARM pip wheel is pathologically slow (~450x; see README). Fine on Linux
    # and in the e2b sandbox, where v0.2 training runs.
    Candidate(
        description="broken proposal (negative max_iter) — should fail gracefully",
        changes={"params": {"max_iter": -1}},
        rationale="demonstrates the executor capturing a bad candidate",
    ),
]


def _show(label: str, result: ExperimentResult) -> None:
    if result.succeeded and result.metrics is not None:
        panel = "  ".join(f"{k}={v:.3f}" for k, v in result.metrics.values.items())
        secs = result.duration_seconds or 0.0
        print(f"{label:<48} f1={result.metrics.primary_value:.3f}   [{panel}]  ({secs:.2f}s)")
    else:
        print(f"{label:<48} FAILED: {result.error}")


def main() -> None:
    target = build_target()
    executor = LocalExecutor()

    baseline = executor.execute(target)
    _show("baseline (HistGradientBoosting, re-measured)", baseline)

    best = baseline
    for candidate in CANDIDATES:
        result = executor.execute(target, candidate)
        _show(candidate.description, result)
        if (
            result.succeeded
            and result.metrics is not None
            and best.metrics is not None
            and result.metrics.primary_value > best.metrics.primary_value
        ):
            best = result

    if best is not baseline and best.metrics is not None and baseline.metrics is not None:
        delta = best.metrics.primary_value - baseline.metrics.primary_value
        print(
            f"\nbest f1 {best.metrics.primary_value:.3f} "
            f"vs baseline {baseline.metrics.primary_value:.3f}  (delta {delta:+.3f})"
        )


if __name__ == "__main__":
    main()
