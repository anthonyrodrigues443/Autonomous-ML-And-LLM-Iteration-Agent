# evals — internal measurement, not a product feature

Answers one question: **is `iterate` getting better from one version to the next?**

Same datasets, same floor model, same budget, one version changed. The output is a
table where every cell is the fraction of *available* gain the agent captured.

## Why this is not shipped

Only the person developing `iterate` asks this question. A user has one version
installed, does not have these datasets, and does not care how v0.2 did. Shipping it
would mean bundling data, documenting it, supporting it on other people's machines,
and growing the package for something zero users need.

So: no `iterate eval` command, nothing under `src/iterate`, nothing in the wheel.
It lives in the repo because a committed results table is evidence, and evidence is
worth being public even when the tool that produced it is not.

The one piece that may graduate into the product later is headroom reporting at the
end of a run, so a user can tell "the agent failed" from "there was nothing to get".
That needs its own design, because measuring a ceiling costs compute the user did
not ask for.

## Commands

```
make eval                                  fill whatever the store is missing
make eval ARGS="--versions dev"            just the new version
make eval ARGS="--datasets churn,diamonds" just those datasets
make eval ARGS="--dry-run"                 print the commands, run nothing
make eval ARGS="--force"                   re-run cells already recorded
make eval-ceilings                         measure the brute-force ceilings (no LLM)
make eval-report                           rebuild RESULTS.md from the store
make eval-list                             what is in the corpus and the store
make eval-shapes                           the fast CI shape checks
```

`sweep` resumes by default. Released versions are frozen, so their numbers never
change and a new version costs one row, not the whole grid.

## Config versus flags

`config.toml` holds everything that changes what a number MEANS: the model, the
iteration budget, patience, repeats. It is tracked, and it is hashed into a
fingerprint stamped on every cell.

Flags only choose which rows to fill. If the budget were a flag, one sweep would run
v0.4 at ten iterations and the next v0.5 at fifteen, the table would show v0.5 ahead,
and the table would be fiction.

Change a condition and the old cells do not vanish or get averaged in. They keep
their old fingerprint and the report says two sets of conditions are present.

## Adding a dataset

```
evals/datasets/<name>/
    dataset.toml     tracked: target, metric, source, notes
    data.csv         gitignored
```

Or point `data` at a path relative to the repo root, which is how the datasets in
`examples/` join the corpus without being copied.

The toml is tracked and the csv is not, so the repo records what the corpus IS
without carrying megabytes of it. Every result stores the content hash of the file
it was measured on: swap the bytes behind a name and the old numbers are correctly
treated as results for something else.

## Ceilings

A ceiling is the best score a brute-force sweep of ordinary models reaches, measured
with no LLM, through the product's own `load_csv` and `ModelTarget` on the same
sealed split the agent gets. Without one, "no candidate beat the baseline" cannot be
read: it means the agent failed, or it means there was nothing to find.

A ceiling is a **lower bound**, not a maximum. It is the best of a fixed list, so the
agent going past it is a real result and nothing here clamps to 100%.

Because it is a lower bound, a stored ceiling never goes DOWN. `put_ceiling` keeps
the better of the old and new value, so re-measuring can only raise the bar. The
first run proved why: the automated sweep found a far better ceiling than the hand
sweep on laptop price, and a slightly worse one on churn. Overwriting would have
thrown away real knowledge in the second case.

**Known gap, `brute_force_sweep_v1` varies the MODEL, not the FEATURES.** The Aug 2
hand sweep varied both, which is why it found small margins on churn, heart and
mobile where this one finds none. On datasets where the win lives in feature
engineering, this ceiling underestimates. Adding feature treatments (target
encoding, interactions, scaling) is `v2` of the sweep, and until then the sweep
version in `method` is what says which kind of ceiling a number is.

## Two kinds of ceiling

A `task` line in a dataset's `dataset.toml` marks it as a PROMPT dataset and selects
a different sweep. Both answer the same question — what is achievable here, with no
LLM deciding — and both are lower bounds.

**Tabular: model families.** Nine estimators through the real `ModelTarget`.

**Prompt: techniques.** Six standard prompt moves, built mechanically from the task
line, the label set and rows sampled from TRAINING data:

```
minimal · define-the-labels · few-shot · reasoning · expert-role · define-plus-few-shot
```

Nothing here is hand-written for a particular dataset, which is what keeps it a fair
floor rather than a target someone tuned. Model families transfer between datasets;
prompt wording does not, so only the FORM is fixed.

Few-shot examples come from training rows only. Taken from the holdout they would
raise the bar using answers the agent is never allowed to see, and every capture
fraction measured against that ceiling would be wrong.

**It cost about 80 minutes for two datasets** (6 techniques x 200 records x 2, at
~3.2s a call on a local 12B). The answer cache makes a re-measure free.

Measured 2026-08-09:

| dataset | baseline | ceiling | best technique | headroom |
|---|---|---|---|---|
| toxicity_jigsaw | 0.8398 | 0.8681 | define-plus-few-shot | 0.028 |
| hate_speech_davidson | 0.5862 | 0.6822 | few-shot | 0.096 |

Two things worth knowing from that first run. `define-the-labels` ALONE scored worse
than minimal on both datasets — telling a small model to be precise about boundaries
without showing it any is actively harmful. And the best technique differs by
dataset, so there is no single prompt shape to hardcode.

## Two corpora, one word

- **the benchmark corpus** (`datasets/`) — real data, ceilings, LLM runs, hours
- **the shape corpus** (`shapes.py`) — tiny synthetic files with nasty shapes, no
  LLM, seconds, runs in CI

The second exists because the v0.4 certification found three crash-class bugs
(non-UTF-8 files, boolean columns, integer regression targets) that 583 unit tests
missed. Not because the tests were thin, but because every fixture in the suite was
built the same way. When a real file breaks the loader, its shape becomes a
generator here.

## Layout

```
config.py     conditions + version list, read from config.toml
corpus.py     dataset discovery and content hashing
store.py      the accumulating sqlite ledger: sweeps, cells, ceilings
ceilings.py   the no-LLM brute-force sweep
runner.py     runs one isolated cell and reads it back
readout.py    parses a memory db with raw SQL so ANY version's db is readable
score.py      headroom-normalised capture
report.py     RESULTS.md, built only from the store
shapes.py     the adversarial shape corpus
run.py        the entrypoint behind the make targets
```

`.work/` (per-cell memory dbs and logs) and `results.db` are gitignored. `RESULTS.md`
is committed.
