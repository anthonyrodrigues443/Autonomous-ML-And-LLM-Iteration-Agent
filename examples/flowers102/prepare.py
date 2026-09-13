"""Build the Flowers102 eval set: real photographs, 102 classes.

    python prepare.py                 # download + write images/ and data.csv
    python prepare.py --per-class 40  # a smaller set

Oxford Flowers102 (Nilsback and Zisserman, 2008): 8,189 photographs of 102 flower
species, one archive and one label file from Oxford, no account needed. Both
checksums are verified against the values torchvision's own loader uses before
anything is extracted.

The layout is flattened: every image becomes `images/NNNNN.jpg` under a shuffled
index and the label lives ONLY in `data.csv`, as `class_NNN`, the number Oxford
ships. The script asserts that no output name carries a class token before it
copies a single file. See README.md for why this dataset and what it measured.

License: Oxford states none for the images. This script downloads them for local
use; nothing is redistributed by this repo. Cite: Nilsback, M-E. and Zisserman, A.
"Automated flower classification over a large number of classes." ICVGIP 2008.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FILES = {
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
}
HERE = Path(__file__).parent
IMAGES = HERE / "images"
OUT = HERE / "data.csv"
N_CLASSES = 102


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    for name, md5 in FILES.items():
        target = HERE / name
        if not target.exists():
            print(f"downloading {BASE}{name} ...")
            urllib.request.urlretrieve(BASE + name, target)
        actual = _md5(target)
        if actual != md5:
            raise SystemExit(
                f"{name}: md5 {actual} does not match the published {md5}; delete it and retry"
            )


def sources(per_class: int | None) -> list[tuple[Path, str]]:
    """Every (file, class label) pair, labels read from Oxford's own imagelabels.mat."""
    from scipy.io import loadmat

    labels = loadmat(HERE / "imagelabels.mat", squeeze_me=True)["labels"]
    if len(labels) != 8189 or int(labels.max()) != N_CLASSES:
        raise SystemExit(f"imagelabels.mat has {len(labels)} labels over {labels.max()} classes")
    with tarfile.open(HERE / "102flowers.tgz") as tar:
        tar.extractall(HERE / ".extract", filter="data")
    root = HERE / ".extract" / "jpg"
    rows: list[tuple[Path, str]] = []
    taken: dict[int, int] = {}
    for index, label in enumerate(labels, start=1):
        klass = int(label)
        if per_class is not None and taken.get(klass, 0) >= per_class:
            continue
        taken[klass] = taken.get(klass, 0) + 1
        rows.append((root / f"image_{index:05d}.jpg", f"class_{klass:03d}"))
    return rows


def flatten(rows: list[tuple[Path, str]]) -> list[tuple[str, str]]:
    """Copy into images/ under opaque shuffled names; return (relative path, label)."""
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    plan = [(rows[old], f"images/{new:05d}.jpg") for new, old in enumerate(order)]
    for _, rel in plan:
        assert "class" not in rel, f"{rel} leaks its origin"
        assert "image_" not in rel, f"{rel} leaks its origin"
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
    print(f"wrote {OUT} with {len(rows)} rows across {len(counts)} classes")
    print(f"  smallest class: {min(counts.values())} images, largest: {max(counts.values())}")


if __name__ == "__main__":
    main()
