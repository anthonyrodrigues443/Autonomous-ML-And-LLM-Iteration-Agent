"""Build the STS-B eval set: the regression example, and the field's own benchmark.

    python prepare.py            # download + write data.csv
    python prepare.py --rows 800

Two sentences in, a similarity score from 0 to 5 out. Chosen because it is not a
contrived example of a scoring task — it IS the standard one, and **Pearson and
Spearman are its canonical metrics**, so the correlations this release adds are
being used on the dataset the field already measures them with.

It also has TWO input columns, which is the first example to exercise multi-column
rendering: a record reaches the model as labelled lines rather than one bare field.

Fetched through the HuggingFace rows API rather than the parquet file, so no parquet
engine is needed for a prep script.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://datasets-server.huggingface.co/rows"
DATASET, CONFIG, SPLIT = "nyu-mll/glue", "stsb", "train"
HERE = Path(__file__).parent
RAW = HERE / "stsb.jsonl"
OUT = HERE / "data.csv"

DEFAULT_ROWS = 800
SEED = 42
_PAGE = 100  # the rows API's maximum per request


def download(rows: int, force: bool = False) -> Path:
    if RAW.exists() and not force:
        print(f"using cached {RAW.name}")
        return RAW
    print(f"downloading {rows} rows of {DATASET}/{CONFIG}")
    collected: list[dict[str, object]] = []
    while len(collected) < rows:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": len(collected),
                "length": min(_PAGE, rows - len(collected)),
            }
        )
        with urllib.request.urlopen(f"{BASE}?{query}") as response:
            page = json.loads(response.read())["rows"]
        if not page:
            break
        collected.extend(row["row"] for row in page)
    RAW.write_text("\n".join(json.dumps(r) for r in collected), encoding="utf-8")
    print(f"  fetched {len(collected)} rows")
    return RAW


def build(rows: int, seed: int) -> None:
    import pandas as pd

    frame = pd.DataFrame([json.loads(line) for line in RAW.read_text().splitlines() if line])
    frame = frame.dropna(subset=["sentence1", "sentence2", "label"])
    out = pd.DataFrame(
        {
            "sentence_a": frame["sentence1"].astype(str).str.strip(),
            "sentence_b": frame["sentence2"].astype(str).str.strip(),
            # Rounded to one decimal: the source carries float32 noise like
            # 3.799999952316284, which is not a distinction anyone is scoring and
            # reads as false precision in a prompt.
            "similarity": frame["label"].astype(float).round(1),
        }
    ).sample(n=min(rows, len(frame)), random_state=seed)

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} — {len(out)} rows")
    print(f"similarity range {out['similarity'].min()} to {out['similarity'].max()}")
    print(out.head(3).to_string(index=False, max_colwidth=40))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(args.rows, force=args.force_download)
    build(args.rows, args.seed)


if __name__ == "__main__":
    main()
