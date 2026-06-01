"""The code-gen contract — how a generated training script is run and scored.

v0.2 lets the agent write its own model code instead of naming an installed
estimator. To keep that safe and comparable, the agent does NOT write file I/O:
it provides one function,

    def train_and_predict(X_train, y_train, X_holdout):
        # any imports, preprocessing, model — returns one prediction per X_holdout row
        return predictions

and we wrap it in a fixed harness that loads the data and writes the predictions.
The harness owns the plumbing (correct by construction); the LLM owns only the
modelling. The sealed holdout stays sealed: the script receives `X_holdout`
*features* but never the labels — we hold those and score the returned
predictions through `core.scoring`, identically to the spec path.

A code-candidate is an ordinary `Candidate` whose ``changes = {"code": "<the
train_and_predict source>"}``; the executor routes on the presence of ``"code"``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from iterate.core.scoring import direction, score, task_for_metric
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from iterate.adapters.data.tabular import TabularDataset

# File names exchanged with the runner's working directory.
TRAIN_CSV = "train.csv"
HOLDOUT_CSV = "holdout.csv"
META_JSON = "meta.json"
PREDICTIONS_CSV = "predictions.csv"

# The harness around the LLM's train_and_predict. Loads inputs, calls the
# function, writes predictions. The LLM source is inserted between the two halves.
_PREAMBLE = f"""\
import json
import pandas as pd

with open({META_JSON!r}) as _f:
    _meta = json.load(_f)
_target = _meta["target"]

_train = pd.read_csv({TRAIN_CSV!r})
X_train = _train.drop(columns=[_target])
y_train = _train[_target]
X_holdout = pd.read_csv({HOLDOUT_CSV!r})  # FEATURES ONLY — labels are held back

# ─── agent-provided train_and_predict below ───
"""

_POSTAMBLE = f"""
# ─── harness: run the agent's function and write predictions ───
_preds = train_and_predict(X_train, y_train, X_holdout)
pd.Series(list(_preds)).to_csv({PREDICTIONS_CSV!r}, index=False, header=False)
"""


def is_code_candidate(changes: dict[str, Any]) -> bool:
    """True if a candidate's ``changes`` carries generated code rather than a spec."""
    return isinstance(changes.get("code"), str) and bool(changes["code"].strip())


def assemble_script(code: str) -> str:
    """Wrap the agent's `train_and_predict` source in the I/O harness."""
    return _PREAMBLE + code.strip() + "\n" + _POSTAMBLE


def build_inputs(dataset: TabularDataset) -> dict[str, bytes]:
    """The files handed to the runner: train (with target), holdout FEATURES, meta.

    The holdout target is never written — it stays host-side for scoring, so the
    sealed holdout cannot leak through the sandbox boundary.
    """
    train = dataset.train_features.copy()
    train[dataset.target] = dataset.train_target.to_numpy()
    meta = {
        "target": dataset.target,
        "task": task_for_metric_safe(dataset),
        "features": list(dataset.features),
    }
    return {
        TRAIN_CSV: train.to_csv(index=False).encode(),
        HOLDOUT_CSV: dataset.test_features.to_csv(index=False).encode(),
        META_JSON: json.dumps(meta).encode(),
    }


def task_for_metric_safe(dataset: TabularDataset) -> str:
    # The dataset doesn't carry the metric; the task is conveyed for the LLM's
    # benefit. We infer it from the target dtype the same way the data adapter does.
    return "classification" if _looks_classification(dataset) else "regression"


def _looks_classification(dataset: TabularDataset) -> bool:
    target = dataset.train_target
    # Few distinct values or non-float dtype -> treat as classification.
    return target.nunique() <= 20 or not _is_float(target)


def _is_float(series: object) -> bool:
    dtype = getattr(series, "dtype", None)
    return bool(getattr(dtype, "kind", "") == "f")


def score_predictions(
    dataset: TabularDataset, predictions_csv: bytes | None, *, metric: str, experiment_id: str
) -> ExperimentResult:
    """Score a script's predictions against the held-back holdout labels.

    A missing/empty/wrong-length predictions file is a captured failure (a
    non-success `ExperimentResult`), never an exception — same contract as a bad
    spec candidate.
    """
    if not predictions_csv:
        return _failed(experiment_id, "no predictions file produced")
    raw = predictions_csv.decode(errors="replace").strip()
    if not raw:
        return _failed(experiment_id, "predictions file was empty")
    preds = raw.splitlines()
    expected = dataset.n_test
    if len(preds) != expected:
        return _failed(experiment_id, f"expected {expected} predictions, got {len(preds)}")
    y_pred = _coerce(preds, classification=_looks_classification(dataset))
    task = task_for_metric(metric)
    metrics = Metrics(
        values=score(task, dataset.test_target.to_numpy(), y_pred),
        primary=metric,
        direction=direction(metric),
        n_samples=expected,
    )
    return ExperimentResult(experiment_id=experiment_id, metrics=metrics)


def _coerce(preds: list[str], *, classification: bool) -> list[int | float | str]:
    if classification:
        # Keep labels as-is unless they're clean ints (match common 0/1 targets).
        out: list[int | float | str] = []
        for raw in preds:
            value = raw.strip()
            out.append(int(float(value)) if _is_intlike(value) else value)
        return out
    return [float(p) for p in preds]


def _is_intlike(value: str) -> bool:
    try:
        f = float(value)
    except ValueError:
        return False
    return f.is_integer()


def _failed(experiment_id: str, reason: str) -> ExperimentResult:
    return ExperimentResult(experiment_id=experiment_id, error=f"code-gen contract: {reason}")


__all__ = [
    "HOLDOUT_CSV",
    "META_JSON",
    "PREDICTIONS_CSV",
    "TRAIN_CSV",
    "assemble_script",
    "build_inputs",
    "is_code_candidate",
    "score_predictions",
]
