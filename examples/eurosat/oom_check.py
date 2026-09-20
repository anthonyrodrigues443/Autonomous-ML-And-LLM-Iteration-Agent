"""Prove a fit too big for the GPU ends as a clean failed result, not a crash.

Works with a plain `pip install 'iterate-ai[vision]'`. Run it in the folder where
EuroSAT's prepare.py wrote data.csv, or pass the CSV's path:
    python oom_check.py                      # the real check, reads ./data.csv
    python oom_check.py path/to/data.csv     # the same, CSV elsewhere
    python oom_check.py --dry                # build everything, skip the fit
"""

import sys
from pathlib import Path

import pandas as pd

from iterate.adapters.data.images import prepare_images
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate
from iterate.targets.dl import DLModelTarget, Recipe

paths = [a for a in sys.argv[1:] if not a.startswith("--")]
source = (
    Path(paths[0])
    if paths
    else next(
        (p for p in (Path("data.csv"), Path("examples/eurosat/data.csv")) if p.exists()),
        Path("data.csv"),
    )
)
if not source.exists():
    raise SystemExit(
        f"{source} not found: run EuroSAT's prepare.py first, or pass the path to its data.csv"
    )
subset = source.with_name("oom_check.csv")
pd.read_csv(source).groupby("label").head(200).to_csv(subset, index=False)

loaded = load_csv(subset, target="label", task="classification")
prepared = prepare_images(loaded, subset)
target = DLModelTarget(
    prepared.dataset,
    column=prepared.column.column,
    metric="accuracy",
    image_size=prepared.image_size,
    profile=prepared.profile,
)
changes = {
    "backbone": "convnext_tiny",
    "unfreeze": "all",
    "epochs": 1,
    "image_size": 224,
    "batch_size": 256,
}
Recipe.from_changes(changes, task="classification")
print(
    "device:",
    target.device,
    "| rows:",
    prepared.dataset.n_train,
    "train,",
    prepared.dataset.n_test,
    "holdout",
)
if "--dry" in sys.argv:
    print("dry run: everything built, fit skipped")
    raise SystemExit(0)

result = target.run(
    Candidate(description="oom check", changes=changes, rationale="prove the capture")
)
print("score:", result.metrics.primary_value if result.metrics else None)
print("error:", result.error)
print(
    "PASS: out of memory was caught cleanly"
    if result.error and "memory" in result.error
    else "NOT an out-of-memory result"
)
