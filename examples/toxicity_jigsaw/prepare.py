"""Turn the Jigsaw toxic-comment release into an eval set `iterate` can run against.

    python prepare.py            # download + write data.csv
    python prepare.py --rows 3000

The raw file is 68 MB of Wikipedia comments with six 0/1 label columns. Four things
change on the way in, and each one is there to stop the eval set flattering a prompt.

**Six label columns become one.** `PromptTarget` scores one answer per record, so
the run answers the `toxic` question. The other five (obscene, threat, insult, …)
are dropped rather than merged: an OR of six labels is a different, vaguer question.

**0/1 becomes words.** The model reads its answer options, so `toxic` / `not toxic`
carries meaning that `1` / `0` does not.

**It is balanced.** The raw set is roughly 10% toxic, where "always answer not
toxic" scores about 0.90 accuracy while being useless. A balanced sample means a
score has to come from reading the comment.

**It is small.** One model call per row, per pass, several passes per run.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SOURCE = (
    "https://huggingface.co/datasets/thesofakillers/"
    "jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv"
)
HERE = Path(__file__).parent
RAW = HERE / "train.raw.csv"
OUT = HERE / "data.csv"

DEFAULT_ROWS = 1500
SEED = 42
# Comments run to thousands of characters. A very long one costs prompt tokens on
# every call it appears in, for a judgement the first paragraph almost always
# settles. Trimmed here rather than by the runtime, so what the model sees is what
# the file says.
MAX_CHARS = 1200


def download(force: bool = False) -> Path:
    if RAW.exists() and not force:
        print(f"using cached {RAW.name}")
        return RAW
    print(f"downloading {SOURCE} (~68 MB)")
    with urllib.request.urlopen(SOURCE) as response, RAW.open("wb") as handle:
        handle.write(response.read())
    return RAW


def build(rows: int, seed: int) -> None:
    import pandas as pd

    frame = pd.read_csv(RAW, usecols=["comment_text", "toxic"])
    frame = frame.dropna(subset=["comment_text"])
    frame["comment_text"] = frame["comment_text"].str.strip().str.slice(0, MAX_CHARS)
    frame = frame[frame["comment_text"].str.len() > 0]

    per_class = max(1, rows // 2)
    # Built by concat rather than `groupby(...).apply(...)`: on current pandas that
    # form consumes the grouping column, and the answers would vanish from the file.
    balanced = pd.concat(
        [
            group.sample(n=min(per_class, len(group)), random_state=seed)
            for _, group in frame.groupby("toxic")
        ]
    ).sample(frac=1, random_state=seed)

    out = pd.DataFrame(
        {
            "comment": balanced["comment_text"].to_numpy(),
            "label": ["toxic" if value == 1 else "not toxic" for value in balanced["toxic"]],
        }
    )
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} — {len(out)} rows")
    print(out["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    build(args.rows, args.seed)


if __name__ == "__main__":
    main()
