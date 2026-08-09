"""Turn the CLINC150 release into an eval set `iterate` can run a prompt against.

    python prepare.py            # download + write data.csv
    python prepare.py --intents 20 --rows 1200

The source is a JSON file of `[text, intent]` pairs across 150 intents. Two things
have to change before it is a sensible prompt eval set.

**The label set gets narrowed by default.** All 150 intents means every single model
call carries a 150-item list of allowed answers, which is a large prompt paid once
per row and a lot to ask of a small model. The default takes the 20 most frequent
intents, which keeps the task genuinely hard (20-way classification) while leaving
the answer list readable. `--intents 150` uses everything.

**Out-of-scope rows are dropped.** CLINC150 ships an `oos` class for queries no
intent covers. It is a real and interesting problem, but it is a different one from
intent classification, and mixing it in means a run measures two things at once.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
HERE = Path(__file__).parent
RAW = HERE / "data_full.json"
OUT = HERE / "data.csv"

# One model call per row, per pass, and a run makes several passes. A few thousand
# rows is the difference between a loop that finishes and one that does not.
DEFAULT_ROWS = 1200
DEFAULT_INTENTS = 20
SEED = 42


def download(force: bool = False) -> Path:
    if RAW.exists() and not force:
        print(f"using cached {RAW.name}")
        return RAW
    print(f"downloading {SOURCE}")
    with urllib.request.urlopen(SOURCE) as response, RAW.open("wb") as handle:
        handle.write(response.read())
    return RAW


def build(rows: int, intents: int, seed: int) -> None:
    import pandas as pd

    payload = json.loads(RAW.read_text(encoding="utf-8"))
    # train + val + test: the split iterate makes is its own, and it must be the
    # only one, or the holdout stops being comparable to the training rows.
    pairs = [pair for key in ("train", "val", "test") for pair in payload.get(key, [])]
    frame = pd.DataFrame(pairs, columns=["text", "intent"])
    frame = frame[frame["intent"] != "oos"]

    keep = [name for name, _ in Counter(frame["intent"]).most_common(intents)]
    frame = frame[frame["intent"].isin(keep)]

    # Even per intent, so "always answer with the most common one" scores badly and
    # a gain has to come from actually reading the utterance.
    #
    # Built by concat rather than `groupby(...).apply(...)`: on current pandas that
    # form consumes the grouping column, so `intent` would silently vanish from the
    # written file and the eval set would have no answers in it.
    per_intent = max(1, rows // len(keep))
    balanced = (
        pd.concat(
            [
                group.sample(n=min(per_intent, len(group)), random_state=seed)
                for _, group in frame.groupby("intent")
            ]
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    balanced.to_csv(OUT, index=False)
    print(f"wrote {OUT} — {len(balanced)} rows, {balanced['intent'].nunique()} intents")
    print(balanced.head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--intents", type=int, default=DEFAULT_INTENTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    build(args.rows, args.intents, args.seed)


if __name__ == "__main__":
    main()
