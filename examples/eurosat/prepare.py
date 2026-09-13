"""Build the EuroSAT eval set: satellite land use, ten classes.

    python prepare.py                  # download + write images/ and data.csv
    python prepare.py --per-class 500  # a smaller set (default: every tile)

EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 patches at 64 x 64 px, one 95 MB
zip from Zenodo, no account needed. The checksum is verified against the value on
the Zenodo record before anything is extracted.

The layout is flattened: every image becomes `images/NNNNN.jpg` under a shuffled
index and the label lives ONLY in `data.csv`. The archive's own filenames repeat the
class name, so the script asserts that no output name carries a class token before
it copies a single file. See README.md for why this dataset and what it measured.

License: MIT, per the Zenodo record. Cite: Helber, Bischke, Dengel, Borth. "EuroSAT:
A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification." IEEE JSTARS 2019. https://doi.org/10.1109/JSTARS.2019.2918242
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
HERE = Path(__file__).parent
ARCHIVE = HERE / "EuroSAT_RGB.zip"
IMAGES = HERE / "images"
OUT = HERE / "data.csv"

LABELS = {
    "AnnualCrop": "annual crop",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway",
    "Industrial": "industrial",
    "Pasture": "pasture",
    "PermanentCrop": "permanent crop",
    "Residential": "residential",
    "River": "river",
    "SeaLake": "sea or lake",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    if not ARCHIVE.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    actual = _md5(ARCHIVE)
    if actual != MD5:
        raise SystemExit(
            f"{ARCHIVE.name}: md5 {actual} does not match the published {MD5}; delete it and retry"
        )


def sources(per_class: int | None) -> list[tuple[Path, str]]:
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(HERE / ".extract")
    root = HERE / ".extract" / "EuroSAT_RGB"
    rows: list[tuple[Path, str]] = []
    for folder, label in LABELS.items():
        files = sorted((root / folder).glob("*.jpg"))
        if per_class is not None:
            files = files[:per_class]
        rows += [(f, label) for f in files]
    return rows


def flatten(rows: list[tuple[Path, str]]) -> list[tuple[str, str]]:
    """Copy into images/ under opaque shuffled names; return (relative path, label)."""
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    plan = [(rows[old], f"images/{new:05d}.jpg") for new, old in enumerate(order)]
    forbidden = {f.lower() for f in LABELS} | {w for v in LABELS.values() for w in v.split()}
    for _, rel in plan:
        assert not any(tok in rel.lower() for tok in forbidden), f"{rel} leaks its class"
    if IMAGES.exists():
        shutil.rmtree(IMAGES)
    IMAGES.mkdir()
    for (src, _), rel in plan:
        shutil.copyfile(src, HERE / rel)
    return sorted((rel, label) for (_, label), rel in plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=None, help="cap images per class")
    args = parser.parse_args()
    download()
    rows = flatten(sources(args.per_class))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(rows)
    shutil.rmtree(HERE / ".extract", ignore_errors=True)
    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"wrote {OUT} with {len(rows)} rows across {len(counts)} classes:")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
