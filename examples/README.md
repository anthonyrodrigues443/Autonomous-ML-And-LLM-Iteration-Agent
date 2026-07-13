# Examples

Public-dataset demos that ship with `iterate`.

| Example | Target family | Dataset | Status |
|---|---|---|---|
| `churn_tabular/` | `ModelTarget` | Public Kaggle churn dataset (Telco) | **working, the headline demo** |
| `toxicity_jigsaw/` | `PromptTarget` | Jigsaw Toxic Comment Classification (public) | placeholder, lands with v0.5 (prompt iteration) |
| `intent_clinc150/` | `PromptTarget` | CLINC150 intent classification (public) | placeholder, lands with v0.5 |

The two `PromptTarget` directories are intentionally empty for now: `PromptTarget`
does not exist yet (roadmap v0.5, Week 8). They mark where the pluggability proof
will live so the layout does not churn later.

## Running the working example

```bash
cd examples/churn_tabular
python prepare.py                 # one-time: cleans the raw Kaggle CSV
iterate run --data data.clean.csv --target Churn --metric f1
```

See `churn_tabular/README.md` for what the run produces and what to expect.

## Bringing your own tabular problem

No code needed: any prepared CSV works directly.

```bash
iterate run --data your_data.clean.csv --target <label_column> --metric f1
```

Requirements: one row per sample, the target as a column, categoricals as strings,
and the usual cleaning done (the agent iterates models, it does not clean data for
you). Classification metrics: `f1`, `accuracy`, `precision`, `recall`. Regression:
`rmse`, `mae`, `mse`, `r2`.

Custom `BenchmarkTarget` implementations in your own package (for problems that are
not a flat CSV) are supported at the protocol level (`iterate.targets.base`), but the
CSV path is the supported public interface until the target families grow at v0.5/v0.6.
