"""Clean the raw Telco Customer Churn CSV for `iterate run`.

This is **data prep**, not part of `iterate` — standard ML glue that every
workflow needs, kept dataset-specific so the framework (`load_csv` / `ModelTarget`)
stays generic. `iterate` v0.1's job starts *after* this: choosing the model +
hyperparameters. (Auto-discovery + cleaning is a later milestone.)

What it does to the public IBM/Kaggle Telco set:
- drops `customerID` (an identifier, not a feature),
- coerces `TotalCharges` (it has ~11 blank strings) to numeric,
- encodes the `Churn` target `Yes`/`No` -> `1`/`0` (the metric panel needs
  encodable binary labels).

Run:  python examples/churn_tabular/prepare.py
Then: iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data.csv"
CLEAN = HERE / "data.clean.csv"
TARGET = "Churn"


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the Telco-specific cleaning. Pure function so it's easy to test."""
    out = frame.copy()
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # Encode the Yes/No target to 1/0 unless it's already numeric. Tested via
    # "not numeric" rather than "== object" so it works for both the legacy object
    # dtype and pandas' newer Arrow-backed string dtype.
    if TARGET in out.columns and not pd.api.types.is_numeric_dtype(out[TARGET]):
        out[TARGET] = (out[TARGET] == "Yes").astype(int)
    return out


def main() -> None:
    frame = pd.read_csv(RAW)
    cleaned = clean(frame)
    cleaned.to_csv(CLEAN, index=False)
    print(f"wrote {len(cleaned)} rows x {cleaned.shape[1]} cols -> {CLEAN}")


if __name__ == "__main__":
    main()
