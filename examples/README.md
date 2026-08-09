# Examples

Public-dataset demos that ship with `iterate`.

| Example | Target family | Dataset | Status |
|---|---|---|---|
| `churn_tabular/` | `ModelTarget` | Public Kaggle churn dataset (Telco) | **working, the headline demo** |
| `toxicity_jigsaw/` | `PromptTarget` | Jigsaw Toxic Comment Classification (public) | **working**, binary |
| `intent_clinc150/` | `PromptTarget` | CLINC150 intent classification (public) | **working** — but measured at f1_macro 0.989 for a minimal prompt, so it has almost no headroom |
| `hate_speech_davidson/` | `PromptTarget` | Davidson hate / offensive / neither (public) | **working, the prompt example worth running** |
| `sts_benchmark/` | `PromptTarget` | STS-B sentence similarity, scored 0-5 (public) | **working** — the REGRESSION example |

### Which prompt example to run

**`hate_speech_davidson`.** Measured headroom of 0.096 against a 0.586 baseline: the
model genuinely cannot separate hate speech from merely offensive without being told
where the line falls, and 29.5% of the raw rows had the three annotators disagree.
That contested boundary is where prompt wording earns its keep.

`intent_clinc150` is kept because multiclass is worth exercising, but a minimal
prompt already scores 0.989 on it — about one error in a hundred. There is nothing
for prompt iteration to find, so a run there demonstrates the plumbing and not the
capability. Ceilings for all of these live in [evals/RESULTS.md](../evals/RESULTS.md).

## These datasets are also the eval corpus

The tabular CSVs here double as the corpus the internal eval harness measures
versions against, registered in `evals/datasets/*/dataset.toml` by path rather than
copied. Each one has a brute-force ceiling measured against the same sealed split
the agent gets, which is what separates "the agent found nothing" from "there was
nothing to find". See [evals/README.md](../evals/README.md). That harness is
internal tooling, never part of the installed package.

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
