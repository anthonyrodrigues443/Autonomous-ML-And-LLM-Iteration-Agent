"""Build the storm wind eval set: a satellite image in, a wind speed in knots out.

    python prepare.py              # download, verify, write images/, train.csv and holdout.csv
    python prepare.py --every 12   # a smaller set: every 12th frame instead of every 6th

NASA Tropical Cyclone Wind Estimation (Maskey et al., 2020): infrared GOES frames of
494 Atlantic and East Pacific storms, 2000 to 2019, each labelled with the storm's
maximum sustained wind. Served file by file from Source Cooperative, no account
needed. Both label files are checked against pinned sha256 values before a frame is
chosen, and at the default setting the downloaded frames are checked against a
pinned manifest of every file's size and sha256.

The split is by storm, never by frame: frames of one storm are 30 minutes apart and
nearly identical, so a frame split would put the same storm on both sides. The
script keeps every 6th frame of every storm, shuffles the storm ids with a fixed
seed, puts one fifth of the storms in the holdout, and asserts no storm is on both
sides. Every image becomes `images/NNNNN.jpg` under one shuffled index, so no file
name carries a storm id or a time. See README.md for what this set measured.

License: CC-BY-4.0. Cite: M. Maskey, R. Ramachandran, I. Gurung, B. Freitag,
M. Ramasubramanian, J. Miller. "Tropical Cyclone Wind Estimation Competition
Dataset", Version 1.0, Radiant MLHub. https://doi.org/10.34911/rdnt.xs53up
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import os
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = "https://data.source.coop/nasa/tropical-storm-competition/"
LABEL_FILES = {
    "training_set_features.csv": "832eb4611b21d5b68cb24dbecfd5d19605710970c850ad4afdaf14643fa3d5ac",
    "training_set_labels.csv": "fed49dba2d01d56b7d56b626432868ed7a62cd20a044391f315e844189091f44",
}
# sha256 over sorted "images/NNNNN.jpg <bytes> <sha256>" lines, for the default --every.
MANIFEST_SHA256 = "a1569d01d24968775356fc3500c93fa54b414196c74d8d61441b533eb8604c0f"
DEFAULT_EVERY = 6
HOLDOUT_SHARE = 0.2
SEED = 0
WORKERS = 16
# The server refuses Python's default user agent.
AGENT = (
    "iterate-ai-examples/0.6 (github.com/anthonyrodrigues443/Autonomous-ML-And-LLM-Iteration-Agent)"
)
HERE = Path(__file__).parent
IMAGES = HERE / "images"


def _get(url: str, *, check: Callable[[bytes], None] | None = None) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=60) as resp:
                expected = int(resp.headers["Content-Length"])
                body: bytes = resp.read()
            if len(body) != expected:
                raise OSError(f"{url}: got {len(body)} of {expected} bytes")
            if check is not None:
                check(body)
            return body
        except (OSError, http.client.HTTPException):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _decode(body: bytes) -> None:
    with Image.open(io.BytesIO(body)) as im:
        im.load()


def _whole(path: Path) -> bool:
    try:
        _decode(path.read_bytes())
    except OSError:
        return False
    return True


def label_rows() -> list[dict[str, str]]:
    texts: dict[str, str] = {}
    for name, pinned in LABEL_FILES.items():
        path = HERE / name
        if not path.exists():
            print(f"downloading {ROOT}{name} ...")
            path.write_bytes(_get(ROOT + name))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pinned:
            raise SystemExit(
                f"{name}: sha256 {actual} is not the pinned {pinned}; delete it and retry"
            )
        texts[name] = path.read_text(encoding="utf-8")
    features = csv.DictReader(io.StringIO(texts["training_set_features.csv"]))
    labels = csv.DictReader(io.StringIO(texts["training_set_labels.csv"]))
    wind = {r["Image ID"]: r["Wind Speed"] for r in labels}
    return [{**r, "label": wind[r["Image ID"]]} for r in features]


def choose(rows: list[dict[str, str]], every: int) -> list[dict[str, str]]:
    """Every `every`th frame of each storm, the storm split, and the opaque names."""
    by_storm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_storm[row["Storm ID"]].append(row)
    storms = sorted(by_storm)
    kept = [
        frame
        for storm in storms
        for frame in sorted(by_storm[storm], key=lambda r: int(r["Relative Time"]))[::every]
    ]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(storms))
    held = {storms[i] for i in order[: round(len(storms) * HOLDOUT_SHARE)]}
    for row, number in zip(kept, rng.permutation(len(kept)), strict=True):
        row["file"] = f"images/{number:05d}.jpg"
        row["side"] = "holdout" if row["Storm ID"] in held else "train"
    sides = defaultdict(set)
    for row in kept:
        sides[row["side"]].add(row["Storm ID"])
    assert not sides["train"] & sides["holdout"], "a storm is on both sides of the split"
    return kept


def download(kept: list[dict[str, str]]) -> str:
    IMAGES.mkdir(exist_ok=True)

    def one(row: dict[str, str]) -> str:
        path = HERE / row["file"]
        if not _whole(path):
            part = path.with_name(f"{path.name}.part")
            part.write_bytes(_get(f"{ROOT}train/{row['Image ID']}.jpg", check=_decode))
            os.replace(part, path)
        body = path.read_bytes()
        return f"{row['file']} {len(body)} {hashlib.sha256(body).hexdigest()}"

    started = time.monotonic()
    lines = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, line in enumerate(pool.map(one, kept), 1):
            lines.append(line)
            if i % 2000 == 0 or i == len(kept):
                print(f"  {i}/{len(kept)} frames in {time.monotonic() - started:.0f}s", flush=True)
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def write(kept: list[dict[str, str]]) -> None:
    for side in ("train", "holdout"):
        with (HERE / f"{side}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "label"])
            writer.writerows(
                (r["file"], r["label"])
                for r in sorted(kept, key=lambda r: r["file"])
                if r["side"] == side
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--every", type=int, default=DEFAULT_EVERY, help="keep every Nth frame of a storm"
    )
    args = parser.parse_args()
    kept = choose(label_rows(), args.every)
    marker = IMAGES / ".every"
    if marker.exists() and marker.read_text().strip() != str(args.every):
        raise SystemExit(
            f"{IMAGES} holds frames from --every {marker.read_text().strip()}; delete it and retry"
        )
    IMAGES.mkdir(exist_ok=True)
    marker.write_text(str(args.every))
    print(f"fetching {len(kept)} frames; files already here are checked, not downloaded again")
    digest = download(kept)
    if args.every == DEFAULT_EVERY and digest != MANIFEST_SHA256:
        raise SystemExit(
            f"the frames' manifest sha256 {digest} is not the pinned {MANIFEST_SHA256}; "
            f"delete {IMAGES} and rerun"
        )
    write(kept)
    for side in ("train", "holdout"):
        rows = [r for r in kept if r["side"] == side]
        winds = [float(r["label"]) for r in rows]
        storms = len({r["Storm ID"] for r in rows})
        print(
            f"wrote {side}.csv: {len(rows)} frames from {storms} storms, wind {min(winds):.0f} to {max(winds):.0f} kt"
        )


if __name__ == "__main__":
    main()
