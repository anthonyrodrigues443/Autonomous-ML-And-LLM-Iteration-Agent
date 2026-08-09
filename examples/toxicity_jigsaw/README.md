# Toxicity (Jigsaw) — prompt iteration

Binary text classification. The agent writes a prompt that decides whether a
Wikipedia comment is toxic, measures it on training rows, reads what it got wrong,
and rewrites it. The winner is scored once on a sealed holdout.

## Get the data

```bash
cd examples/toxicity_jigsaw
python prepare.py
```

Downloads ~68 MB from Hugging Face (no account needed) and writes `data.csv`:
1500 rows, two columns, balanced 750/750.

```
comment,label
"Eat my asshole. You did nothing when others wrongly accused me...",toxic
"The title isn't. Way to fire off a snide, absolutely useless reply though...",not toxic
```

The raw download and the prepared file are both gitignored. `prepare.py` is the
tracked, reproducible artifact — and the prepared set is 750 genuinely abusive
comments, which is not something to keep in a public repo when a script rebuilds it
in seconds.

## Run it

```bash
iterate run --data data.csv --target label \
  --task "decide whether this Wikipedia comment is toxic" \
  --metric f1
```

Add `--prompt-file my_prompt.txt` to start from a prompt you already use, and
`--target-model` to tune a model other than the one driving the run.

## What the prep does, and why

**Six label columns become one.** The raw file has `toxic`, `severe_toxic`,
`obscene`, `threat`, `insult`, `identity_hate`. A prompt run scores one answer per
record, so it answers the `toxic` question. The other five are dropped rather than
OR'd together, because an OR of six labels is a vaguer question than any of them.

**0/1 becomes words.** The model reads its answer options. `toxic` carries meaning
that `1` does not.

**It is balanced.** The raw set is about 10% toxic, so "always say not toxic" scores
around 0.90 accuracy while being useless. Balanced, a score has to come from
actually reading the comment.

**It is small, and comments are trimmed to 1200 characters.** One model call per
row per pass, several passes per run: size is what decides whether the loop
finishes. The first paragraph settles almost every one of these judgements.

## Source

[thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
on Hugging Face — the Kaggle "Toxic Comment Classification Challenge" training
split. CC0, with the comment text under Wikipedia's CC-BY-SA-3.0.
