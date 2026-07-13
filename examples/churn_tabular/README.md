# Churn tabular example

The headline demo: the agent autonomously iterates on the public **Telco
Customer Churn** dataset (IBM sample via Kaggle `blastchar/telco-customer-churn`,
7043 rows). It re-measures the baseline through its own eval, then loops. In v0.2
each iteration is a cell-by-cell coding session: a Supervisor briefs one
experiment, the coding agent writes and runs real notebook cells in a live kernel
(inspect, transform, fit, validate, submit), and the submission is scored on a
sealed holdout the agent never sees. Every attempt is recorded; the winner ships
as a runnable notebook.

## Run it

```bash
# 1. Prepare the data (standard ML prep, see note below). One-time.
python examples/churn_tabular/prepare.py

# 2. Run the agent (needs Ollama running with gemma4:12b, or pass --backend).
iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1

# The full research journey, one notebook per iteration:
iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1 \
            --notebooks all
```

Representative output (exact levers/scores vary run to run; the loop and its
guards are what is deterministic):

```
coder[iter-02]: cell 5 ok (0.4s, 2/300s budget)
agent loop: iteration 2 'Imbalance Weighting' -> f1=0.6140

  iter   model                            f1    delta
  base   baseline                     0.5676        -
     1   Baseline Model               0.6251  +0.0575
     2   Class Weight Balancing       0.6251  +0.0575
     3   Hyperparameter Tuning        0.6279  +0.0603
     4   Model Swap - XGBoost <- best 0.6312  +0.0636

  stopped: max_iterations
  best: Model Swap - XGBoost (f1=0.6312, +0.0636 vs baseline)
```

Then open `.iterate/runs/<run_id>/best.ipynb`: the winning session as a runnable
notebook, with the hypothesis, every staged cell and its real output, the dead
ends labeled, and the findings.

Useful flags: `--max-iterations N`, `--patience N`, `--until 30m`,
`--notebooks best|all|none`, `--compute e2b` (isolated sandbox), `--source
notebook.ipynb` (reconstruct a baseline from your own approach, read as text and
never executed), `--backend groq --api-key ...` (cloud model), `--fresh` (new
chapter), `--spec` (the v0.1 allow-list path).

## What v0.2 does, and doesn't

The agent **writes its own training code**: any model, any preprocessing, its
call. The harness bounds the process (kernel budgets, sealed holdout, validity
gates, a floor submission), never the solution space. It does **not** clean your
data; auto-discovery and cleaning come later on the roadmap (see the repo
`README.md` roadmap table and `LIMITATIONS.md` for the honest boundary list).

## On data prep

The cleaning below is **standard ML prep, not part of `iterate`**; the same step
any ML workflow needs. It is dataset-specific, so it lives here in the example
rather than in the framework (which stays generic). `prepare.py`:

- drops `customerID` (an identifier, not a feature),
- coerces `TotalCharges` (it has ~11 blank strings) to numeric,
- encodes the `Churn` target `Yes`/`No` to `1`/`0`.

## On the baseline

We never trust a reported score. The baseline is **re-measured through our own
eval** (the factory's default `HistGradientBoosting`, or whatever you seed via
`--source` / prior memory), so every candidate is compared apples-to-apples
against a number we computed the same way.

`data.csv` is the public IBM/Kaggle Telco sample; `data.clean.csv` is its prepared
form (committed so the demo runs on clone).
