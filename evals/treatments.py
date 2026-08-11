"""Ceiling sweep v2: varying the FEATURE TREATMENT, not just the estimator.

`ceilings.py` sweeps nine model families through the spec path, and on three
datasets it returned a ceiling exactly equal to the baseline — churn, heart and
mobile all measured at zero headroom. The v0.4 hand sweep had put churn at 1.6%
and mobile at 2.1%, and the difference is not that one of them is wrong. The spec
path fixes the preprocessing and varies only the estimator, so a margin that lives
in how the columns are ENCODED is invisible to it however many models it tries.

That gap is what left carry-in 7 ("the agent misses thin margins") unanswerable:
you cannot ask whether a margin was missed until you know whether it exists.

So this sweeps the other axis, and it does so through the agent's own code path —
`build_code_job` writes the same sealed holdout the agent gets, the script runs in
the same runner, and `score_code_job` applies the same ruler. A treatment that
wins here is a thing the agent could actually have written.

Same two rules as v1. A ceiling is a LOWER BOUND, nothing clamps, and `put_ceiling`
keeps the better of the two sweeps, so v2 can only ever raise a dataset's bar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evals.store import Ceiling
from iterate.adapters.compute.runner import LocalCodeRunner
from iterate.adapters.data.tabular import load_csv
from iterate.core.scoring import direction as metric_direction
from iterate.core.scoring import task_for_metric
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    from evals.corpus import Dataset

METHOD = "feature_treatment_sweep_v2"

# Generous: a treatment that fits 500 trees on 50k rows is still a treatment a
# person would try, and a timeout here silently lowers the ceiling.
CELL_TIMEOUT = 900.0

# Shared by every treatment below. Splitting columns by dtype rather than by name
# is what keeps the list dataset-agnostic — the same eight treatments run on churn,
# on diamonds and on anything else in the corpus without being retuned.
_HEADER = """
import numpy as np
import pandas as pd

def _split(X):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return cat, num

def _ordinal(tr, ho, cat):
    from sklearn.preprocessing import OrdinalEncoder
    tr, ho = tr.copy(), ho.copy()
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[cat] = enc.fit_transform(tr[cat].astype(str))
        ho[cat] = enc.transform(ho[cat].astype(str))
    return tr, ho
"""

# `_out` adapts one fitted estimator to the harness contract: labels for a
# regressor, (labels, probabilities) for a classifier that can produce them.
_OUT = """
def _out(model, ho, task):
    preds = model.predict(ho)
    if task == "classification" and hasattr(model, "predict_proba"):
        return preds, model.predict_proba(ho)
    return preds
"""

_CLASSIFIER = "from sklearn.ensemble import HistGradientBoostingClassifier as _HGB"
_REGRESSOR = "from sklearn.ensemble import HistGradientBoostingRegressor as _HGB"


def _treatments(task: str) -> dict[str, str]:
    """The fixed list, in the order a competent person reaches for them.

    Deliberately the standard encodings and framings rather than a search: a
    ceiling built from a hundred variants would measure patience, and the point of
    the number is to say what was ACHIEVABLE, not what was findable given a week.
    """
    est = _CLASSIFIER if task == "classification" else _REGRESSOR
    common = _HEADER + _OUT + est + f'\n_TASK = "{task}"\n'

    bodies: dict[str, str] = {
        # The corpus baseline treatment, repeated here so v2's numbers are
        # internally comparable without reaching across to the v1 table.
        "ordinal-encoding": """
def train_and_predict(X_train, y_train, X_holdout):
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    return _out(_HGB(random_state=0).fit(tr, y_train), ho, _TASK)
""",
        # Boosting told which columns are categorical splits on them natively
        # instead of on an arbitrary integer ordering.
        "native-categorical": """
def train_and_predict(X_train, y_train, X_holdout):
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    mask = [c in cat for c in tr.columns]
    if not any(mask):
        mask = None
    return _out(_HGB(random_state=0, categorical_features=mask).fit(tr, y_train), ho, _TASK)
""",
        # Out-of-fold so the encoding cannot see a row's own answer. The in-fold
        # version is the classic leak, and it would inflate this ceiling with a
        # score no honest agent could reach.
        "target-encoding": """
def train_and_predict(X_train, y_train, X_holdout):
    from sklearn.model_selection import KFold
    cat, _ = _split(X_train)
    tr, ho = X_train.copy(), X_holdout.copy()
    y = pd.Series(np.asarray(y_train))
    if _TASK == "classification":
        codes, _uniq = pd.factorize(y)
        y = pd.Series(codes, dtype=float)
    prior = y.mean()
    for c in cat:
        oof = pd.Series(np.full(len(tr), prior, dtype=float), index=tr.index)
        for fit_idx, out_idx in KFold(5, shuffle=True, random_state=0).split(tr):
            means = y.iloc[fit_idx].groupby(tr[c].iloc[fit_idx].astype(str).values).mean()
            oof.iloc[out_idx] = (
                tr[c].iloc[out_idx].astype(str).map(means).fillna(prior).to_numpy()
            )
        full = y.groupby(tr[c].astype(str).values).mean()
        tr[c] = oof.to_numpy()
        ho[c] = ho[c].astype(str).map(full).fillna(prior).to_numpy()
    return _out(_HGB(random_state=0).fit(tr, y_train), ho, _TASK)
""",
        # How often a level occurs is sometimes the whole signal (rare plan codes,
        # rare cities) and ordinal encoding destroys it.
        "frequency-encoding": """
def train_and_predict(X_train, y_train, X_holdout):
    cat, _ = _split(X_train)
    tr, ho = X_train.copy(), X_holdout.copy()
    for c in cat:
        counts = tr[c].astype(str).value_counts()
        tr[c] = tr[c].astype(str).map(counts).fillna(0).to_numpy()
        ho[c] = ho[c].astype(str).map(counts).fillna(0).to_numpy()
    return _out(_HGB(random_state=0).fit(tr, y_train), ho, _TASK)
""",
        # Trees cannot build a ratio or a product on their own; they can only
        # approximate one with splits. Handing over the top numeric pairs is the
        # cheapest version of the feature engineering a person would do.
        "numeric-interactions": """
def train_and_predict(X_train, y_train, X_holdout):
    cat, num = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    top = sorted(num, key=lambda c: tr[c].std(), reverse=True)[:6]
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            tr[f"{a}_x_{b}"] = tr[a] * tr[b]
            ho[f"{a}_x_{b}"] = ho[a] * ho[b]
            denom_tr, denom_ho = tr[b].replace(0, np.nan), ho[b].replace(0, np.nan)
            tr[f"{a}_over_{b}"] = tr[a] / denom_tr
            ho[f"{a}_over_{b}"] = ho[a] / denom_ho
    return _out(_HGB(random_state=0).fit(tr, y_train), ho, _TASK)
""",
        # Three seeds and three depths averaged. Not a feature treatment, but it is
        # the move that most reliably buys the last fraction of a percent, which is
        # exactly the size of margin carry-in 7 is about.
        "seed-ensemble": """
def train_and_predict(X_train, y_train, X_holdout):
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    fitted = [
        _HGB(random_state=s, max_leaf_nodes=n).fit(tr, y_train)
        for s, n in ((0, 31), (1, 63), (2, 15))
    ]
    if _TASK == "classification" and hasattr(fitted[0], "predict_proba"):
        proba = np.mean([m.predict_proba(ho) for m in fitted], axis=0)
        classes = fitted[0].classes_
        return classes[np.argmax(proba, axis=1)], proba
    return np.mean([m.predict(ho) for m in fitted], axis=0)
""",
    }

    if task == "classification":
        bodies["balanced-classes"] = """
def train_and_predict(X_train, y_train, X_holdout):
    from sklearn.utils.class_weight import compute_sample_weight
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    w = compute_sample_weight("balanced", y_train)
    return _out(_HGB(random_state=0).fit(tr, y_train, sample_weight=w), ho, _TASK)
"""
        # Ranking metrics like average_precision read the probabilities directly,
        # so how well calibrated they are is not a cosmetic concern — it is the
        # score. Two of the corpus datasets are scored this way.
        bodies["calibrated"] = """
def train_and_predict(X_train, y_train, X_holdout):
    from sklearn.calibration import CalibratedClassifierCV
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    model = CalibratedClassifierCV(_HGB(random_state=0), method="isotonic", cv=5)
    model.fit(tr, y_train)
    return _out(model, ho, _TASK)
"""
    else:
        # A right-skewed target is the standard reason a price model looks
        # hopeless; predicting the log and exponentiating back is the standard fix.
        bodies["log-target"] = """
def train_and_predict(X_train, y_train, X_holdout):
    cat, _ = _split(X_train)
    tr, ho = _ordinal(X_train, X_holdout, cat)
    y = np.asarray(y_train, dtype=float)
    if y.min() < 0:
        return _out(_HGB(random_state=0).fit(tr, y_train), ho, _TASK)
    model = _HGB(random_state=0).fit(tr, np.log1p(y))
    return np.expm1(model.predict(ho))
"""

    return {name: common + body for name, body in bodies.items()}


@dataclass(frozen=True)
class TreatmentResult:
    """One treatment in the sweep."""

    name: str
    score: float | None
    seconds: float
    error: str = ""


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate > incumbent if direction == "maximize" else candidate < incumbent


def sweep(
    dataset: Dataset,
    *,
    on_progress: Callable[[TreatmentResult], None] | None = None,
) -> tuple[Ceiling, list[TreatmentResult]]:
    """Score every treatment through the agent's code path; best one is the ceiling."""
    loaded = load_csv(dataset.path, target=dataset.target)
    target = ModelTarget(loaded, metric=dataset.metric, name=dataset.name)
    direction = metric_direction(dataset.metric)
    task = task_for_metric(dataset.metric)
    runner = LocalCodeRunner()

    baseline_value: float | None = None
    try:
        outcome = target.baseline()
        if outcome.metrics is not None:
            baseline_value = outcome.metrics.primary_value
    except Exception:  # a ceiling without a baseline is still a ceiling
        pass

    results: list[TreatmentResult] = []
    best: float | None = None
    best_name = ""

    for name, code in _treatments(task).items():
        started = time.monotonic()
        try:
            job = target.build_code_job(
                Candidate(description=name, changes={"code": code}, rationale="ceiling sweep")
            )
            run = runner.run(
                job.script, inputs=job.inputs, outputs=job.outputs, timeout=CELL_TIMEOUT
            )
            scored = target.score_code_job(run, f"treatment-{name}")
            value = scored.metrics.primary_value if scored.metrics is not None else None
            error = scored.error or ""
        except Exception as exc:
            # One treatment that cannot run on this dataset must not cost the other
            # seven: the ceiling is the best of what actually executed, and the
            # detail column records what did not.
            value, error = None, f"{type(exc).__name__}: {exc}"

        result = TreatmentResult(name, value, time.monotonic() - started, error.strip()[:300])
        results.append(result)
        if on_progress:
            on_progress(result)
        if value is not None and (best is None or _better(value, best, direction)):
            best, best_name = value, name

    if best is None:
        raise RuntimeError(f"{dataset.name}: no treatment in the sweep produced a score")

    return (
        Ceiling(
            dataset=dataset.name,
            dataset_hash=dataset.content_hash(),
            metric=dataset.metric,
            ceiling=best,
            direction=direction,
            baseline=baseline_value,
            method=f"{METHOD} (best: {best_name}, {len(results)} treatments)",
            measured_at=datetime.now(UTC).isoformat(),
            detail=json.dumps(
                [
                    {
                        "treatment": r.name,
                        "score": r.score,
                        "seconds": round(r.seconds, 1),
                        "error": r.error,
                    }
                    for r in results
                ]
            ),
        ),
        results,
    )


__all__ = ["CELL_TIMEOUT", "METHOD", "TreatmentResult", "sweep"]
