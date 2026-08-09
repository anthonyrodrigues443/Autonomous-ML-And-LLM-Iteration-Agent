# STS-B — prompt iteration on a SCORE

The regression example. Two sentences in, a similarity score from 0 to 5 out.

```bash
cd examples/sts_benchmark
python prepare.py
iterate run --data data.csv --target similarity \
  --task "rate how similar these two sentences are, from 0 to 5" \
  --metric pearson
```

## Why this dataset

It is not a contrived example of a scoring task — it **is** the standard one, and
**Pearson and Spearman are its canonical metrics**. So the correlations this release
adds get used on the dataset the field already measures them with.

It is also the first example with **two input columns**, so a record reaches the
model as labelled lines rather than one bare field:

```
sentence_a: A woman is cutting an onion.
sentence_b: A woman is cutting through an onion.
```

## Which metric

`pearson` by default, and the choice is not cosmetic. A prompt that rates everything
three points too high but in exactly the right order scores **pearson 1.000 and rmse
3.000**. For a rating task the first verdict is usually the useful one, because a
constant offset can be calibrated away and the ordering cannot.

`rmse` and `mae` are there when absolute accuracy is what you need. `spearman` and
`kendall` when only rank matters. The Researcher picks when you omit `--metric`.

One guard worth knowing: a prompt that answers the same number to everything is the
likeliest failure of a rating task, and scipy calls the correlation undefined there.
iterate scores it **0.0**, not nan.

## What the prep does

- **Rounds to one decimal.** The source carries float32 noise like `3.799999952316284`,
  which nobody is scoring and which reads as false precision inside a prompt.
- **Fetches through the HuggingFace rows API**, not the parquet file, so a prep script
  needs no parquet engine.
- 800 rows by default. One model call per record per pass, so size decides whether
  the loop finishes.

## Source

[nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue), `stsb` config — the
STS Benchmark from SemEval, as distributed in GLUE. No account needed.
