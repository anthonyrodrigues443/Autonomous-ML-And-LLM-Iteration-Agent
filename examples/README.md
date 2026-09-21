# Examples

Public-dataset demos that ship with `iterate`.

| Example | Target family | Dataset | Status |
|---|---|---|---|
| `churn_tabular/` | `ModelTarget` | Public Kaggle churn dataset (Telco) | **working, the headline demo** |
| `toxicity_jigsaw/` | `PromptTarget` | Jigsaw Toxic Comment Classification (public) | **working**, binary |
| `intent_clinc150/` | `PromptTarget` | CLINC150 intent classification (public) | **working** — but measured at f1_macro 0.989 for a minimal prompt, so it has almost no headroom |
| `hate_speech_davidson/` | `PromptTarget` | Davidson hate / offensive / neither (public) | **working, the prompt example worth running** |
| `sts_benchmark/` | `PromptTarget` | STS-B sentence similarity, scored 0-5 (public) | **working** — the REGRESSION example |
| `flowers102/` | `DLModelTarget` | Oxford Flowers102, 102 species (public) | **working, the image example worth running**: the plain CNN baseline scores 0.554 against a measured ceiling of 0.974 |
| `eurosat/` | `DLModelTarget` | EuroSAT satellite land use, 10 classes (public, MIT) | **working**, the image certification dataset: no ImageNet overlap, 0.950 against a 0.986 ceiling |
| `cyclone_wind/` | `DLModelTarget` | NASA storm wind speed from satellite frames (public, CC-BY-4.0) | **working**, the NUMBER-label image example, split by storm: 13.12 knots against an 8.87 ceiling |

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

## Running an image example

Since v0.6 an image run works end to end: a plain CNN baseline, briefed experiments
that fine-tune a pretrained backbone through `fit()` or write their own torch code, a
sealed holdout, and `best.ipynb`.

```bash
cd examples/flowers102
python prepare.py                 # one-time: downloads, verifies, writes images/ and data.csv
iterate run --data data.csv --target label --metric accuracy

cd ../cyclone_wind
python prepare.py                 # about 250 MB: writes train.csv and holdout.csv, split by storm
iterate run --train train.csv --holdout holdout.csv --target label --metric rmse
```

`--target label` is required on the CSV form. Name the metric: left to pick for
itself the agent usually chooses `f1_macro` on the two class datasets, and the number
then does not line up with the accuracy these READMEs and the stored ceilings quote.
An image run needs torch: the harness installs it at the start of the run with your
consent, or install it yourself with `pip install 'iterate-ai[vision]'`. It trains on
your machine, and `--compute e2b` is refused. Expect tens of minutes: the four live
runs so far took 29 to 49 minutes for 2 or 3 iterations on an Apple M5 with a local
gemma4:12b. GPU compatible: Apple (MPS) and NVIDIA (CUDA).

## Bringing your own tabular problem

No code needed: any prepared CSV works directly.

```bash
iterate run --data your_data.clean.csv --target <label_column> --metric f1
```

Requirements: one row per sample, the target as a column, categoricals as strings,
and the usual cleaning done (the agent iterates models, it does not clean data for
you). Classification metrics: `f1`, `accuracy`, `precision`, `recall`. Regression:
`rmse`, `mae`, `mse`, `r2`.

## Bringing your own images

```bash
iterate run --data path/to/image_folder                 # the labels are found by rules and shown to you before anything trains
iterate run --data your_images.csv --target <label_column> --metric accuracy
```

A folder can be laid out however it came: class folders, a label table beside the
images, a train and test pair. A CSV needs one column of image paths, relative to the
CSV's own folder, and one label column holding a class or a number.

Custom `BenchmarkTarget` implementations in your own package (for problems that are
not a CSV or a folder of images) are supported at the protocol level
(`iterate.targets.base`), but the CLI forms above are the supported public interface.
