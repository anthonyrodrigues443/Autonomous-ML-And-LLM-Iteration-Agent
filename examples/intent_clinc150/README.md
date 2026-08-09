# CLINC150 intents — prompt iteration

Multiclass text classification, and the harder of the two prompt examples: 20
possible answers instead of 2. Its real job is surfacing the genericity bugs a
binary task hides.

## Get the data

```bash
cd examples/intent_clinc150
python prepare.py                        # 1200 rows, the 20 most common intents
python prepare.py --intents 50 --rows 2000
```

Downloads 2.5 MB from the authors' GitHub (no account needed) and writes `data.csv`:

```
text,intent
what is your nationality,where_are_you_from
what is the current time,time
set default language to english,change_language
```

The download and the prepared file are gitignored; `prepare.py` is the tracked,
reproducible artifact.

## Run it

```bash
iterate run --data data.csv --target intent \
  --task "classify the user's utterance into one of the listed intents" \
  --metric f1_macro
```

`f1_macro` rather than accuracy: the set is balanced across intents, and macro
averaging means a prompt cannot win by being good at the easy intents alone.

## What the prep does, and why

**The label set is narrowed to 20 by default.** CLINC150 has 150 intents. All of
them means every model call carries a 150-item list of allowed answers — a large
prompt paid once per row, and a lot to ask of a small model. Twenty keeps the task
genuinely hard while leaving the answer list readable. `--intents 150` uses
everything.

**Out-of-scope rows are dropped.** The dataset ships an `oos` class for queries no
intent covers. That is a real and interesting problem, but a different one, and
mixing it in means a run measures two things at once.

**train, val and test are merged.** The only split that should exist is the one
iterate makes, or the holdout stops being comparable to the training rows.

**It is balanced per intent**, so "always answer with the most common one" scores
badly and a gain has to come from reading the utterance.

## Source

[clinc/oos-eval](https://github.com/clinc/oos-eval) — the dataset from "An
Evaluation Dataset for Intent Classification and Out-of-Scope Prediction"
(EMNLP 2019). Also mirrored at
[UCI](https://archive.ics.uci.edu/dataset/570/clinc150). CC BY-SA 3.0.
