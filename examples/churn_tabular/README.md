# Churn tabular example — the v0.1 agentic loop

The headline v0.1 demo: the agent autonomously iterates on the public **Telco
Customer Churn** dataset (IBM sample via Kaggle `blastchar/telco-customer-churn`,
7043 rows). It re-measures the baseline through its own eval, then loops — an LLM
proposes a model + hyperparameters from scikit-learn / XGBoost / LightGBM, trains
it leakage-safe, scores it on a sealed holdout, records every attempt, and
iterates to the best it can find.

## Run it

```bash
# 1. Prepare the data (standard ML prep — see note below). One-time.
python examples/churn_tabular/prepare.py

# 2. Run the agent (needs Ollama running with qwen3:14b, or pass --backend).
iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1
```

Representative output (exact models/scores vary run to run — the LLM is
non-deterministic; the loop is what's deterministic):

```
Running on tabular-model; target='Churn', metric=f1

  iteration 1  xgboost.XGBClassifier      f1=0.5800  ↑ +0.0124  ← best
  iteration 2  lightgbm.LGBMClassifier    f1=0.5790  ↓ -0.0010
  iteration 3  sklearn…RandomForest…      f1=0.5810  ↑ +0.0134  ← best

  stopped: max_iterations
  best: RandomForest… (f1=0.5810, +0.0134 vs baseline 0.5676)
```

Useful flags: `--max-iterations N`, `--patience N`, `--until 30m`, `--source
notebook.ipynb` (reconstruct a baseline from your own approach), `--backend
openai-compatible --api-key …` (use a cloud model), `--fresh` (start a new chapter).

## What v0.1 does — and doesn't

v0.1 chooses the **model + its hyperparameters** from scikit-learn, XGBoost, and
LightGBM, and iterates to the best it can find for your prepared dataset. It does
**not** clean your data or use models outside those three libraries — both are
deliberate (sandboxed code-gen for arbitrary models is the next release;
auto-discovery + cleaning come later). See the repo `BUILD_LOG.md` roadmap.

## On data prep

The cleaning below is **standard ML prep, not part of `iterate`** — the same step
any ML workflow needs. It's dataset-specific, so it lives here in the example
rather than in the framework (which stays generic). `prepare.py`:

- drops `customerID` (an identifier, not a feature),
- coerces `TotalCharges` (it has ~11 blank strings) to numeric,
- encodes the `Churn` target `Yes`/`No` → `1`/`0`.

## On the baseline

We never trust a reported score. The baseline is **re-measured through our own
eval** (the factory's default `HistGradientBoosting`, or whatever you seed via
`--source` / prior memory), so every candidate is compared apples-to-apples
against a number we computed the same way.

`data.csv` is the public IBM/Kaggle Telco sample; `data.clean.csv` is its prepared
form (committed so the demo runs on clone).
