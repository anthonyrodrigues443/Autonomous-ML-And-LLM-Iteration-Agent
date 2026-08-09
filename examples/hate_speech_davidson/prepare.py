"""Build the Davidson hate-speech eval set: the boundary humans argue about.

    python prepare.py            # download + write data.csv
    python prepare.py --rows 1500 --contested-only

Chosen over an easier dataset for one measured reason. On CLINC intents a minimal
prompt already scores 0.989 — one error in a hundred — so there is nothing for
prompt iteration to find and a run demonstrates nothing. Here **29.5% of rows had
annotator disagreement**: three people read the same tweet and did not agree
whether it was hate speech, merely offensive, or neither.

That is the shape of task where prompt wording earns its keep. A minimal prompt
cannot guess where you draw the line between hate and offensive; a good prompt
states it. "what is the current time" has no such line to draw.

The three annotators' votes are in the raw file, so `--contested-only` keeps just
the rows they split on — a deliberately hard subset for probing where a prompt
actually fails.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SOURCE = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)
HERE = Path(__file__).parent
RAW = HERE / "labeled_data.csv"
OUT = HERE / "data.csv"

DEFAULT_ROWS = 1200
SEED = 42
MAX_CHARS = 600
# The dataset's own encoding of its three classes.
_CLASSES = {0: "hate speech", 1: "offensive", 2: "neither"}


def download(force: bool = False) -> Path:
    if RAW.exists() and not force:
        print(f"using cached {RAW.name}")
        return RAW
    print(f"downloading {SOURCE}")
    with urllib.request.urlopen(SOURCE) as response, RAW.open("wb") as handle:
        handle.write(response.read())
    return RAW


def build(rows: int, seed: int, contested_only: bool) -> None:
    import pandas as pd

    frame = pd.read_csv(RAW)
    frame = frame.dropna(subset=["tweet"])
    frame["tweet"] = frame["tweet"].str.strip().str.slice(0, MAX_CHARS)
    frame = frame[frame["tweet"].str.len() > 0]

    votes = frame[["hate_speech", "offensive_language", "neither"]]
    frame = frame.assign(unanimous=votes.max(axis=1) == frame["count"])
    contested = int((~frame["unanimous"]).sum())
    print(f"{len(frame)} rows, {contested} ({contested / len(frame):.1%}) had annotators disagree")

    if contested_only:
        frame = frame[~frame["unanimous"]]
        print(f"keeping only the {len(frame)} contested rows")

    per_class = max(1, rows // 3)
    # Built by concat rather than groupby().apply(): on current pandas that form
    # consumes the grouping column and the answers vanish from the written file.
    balanced = pd.concat(
        [
            group.sample(n=min(per_class, len(group)), random_state=seed)
            for _, group in frame.groupby("class")
        ]
    ).sample(frac=1, random_state=seed)

    out = pd.DataFrame(
        {
            "tweet": balanced["tweet"].to_numpy(),
            # Words, not 0/1/2: the model reads its answer options, so "hate speech"
            # carries meaning that "0" does not.
            "label": [_CLASSES[int(c)] for c in balanced["class"]],
        }
    )
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} — {len(out)} rows")
    print(out["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--contested-only",
        action="store_true",
        help="keep only rows the three annotators disagreed on",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    build(args.rows, args.seed, args.contested_only)


if __name__ == "__main__":
    main()
