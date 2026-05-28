# Churn tabular example

The tabular substrate end-to-end on the public **Telco Customer Churn** dataset
(IBM sample, via Kaggle `blastchar/telco-customer-churn`, 7043 rows). It runs
`load_csv` → `ModelTarget` → model factory → `LocalExecutor`: a re-measured
baseline plus hand-supplied candidates (HistGradientBoosting, XGBoost), including
one deliberately broken candidate to show the executor capturing a failure instead
of crashing. In Week 3 the agent supplies the candidates; here we supply them by hand.

## Run

```bash
python examples/churn_tabular/run.py
```

## Data prep

The dataset needs light, dataset-specific cleaning — done in `run.py`, **not** the
framework, which stays generic:

- drop `customerID` (an identifier, not a feature),
- coerce `TotalCharges` (it has ~11 blank strings) to numeric,
- encode the `Churn` target `Yes`/`No` → `1`/`0` (the metric panel needs encodable
  binary labels).

## On the baseline

We never trust a reported score. The baseline is **re-measured through our own
eval** (the factory's default `HistGradientBoosting`), so every candidate is
compared apples-to-apples against a number we computed the same way.

`data.csv` is the public IBM/Kaggle Telco Customer Churn sample, included for
convenience so the example and its integration test run on clone.

## Known issue: LightGBM on macOS ARM

The model factory supports LightGBM, but it's left out of this demo's candidate
list because the LightGBM 4.6 **prebuilt pip wheel for macOS ARM is pathologically
slow** — ~0.2s/tree of fixed overhead (~450x slower than XGBoost on identical data),
independent of thread settings. It's a known wheel/`libomp` issue, not framework
code, and does **not** reproduce on Linux or in the e2b sandbox (where v0.2 training
runs). For fast LightGBM on local macOS, rebuild it from source against brew's
`libomp` (`uv pip install --no-binary lightgbm lightgbm`); we deliberately do **not**
force that on all installs.
