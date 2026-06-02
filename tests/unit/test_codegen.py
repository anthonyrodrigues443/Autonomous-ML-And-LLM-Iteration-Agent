"""Tests for the code-gen contract.

The headline test runs a hand-written `train_and_predict` end to end through the
real LocalCodeRunner (no LLM, no e2b) and scores it, proving the whole contract
path before the CodeProposer (Day 4) generates real functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.compute.runner import LocalCodeRunner
from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen

if TYPE_CHECKING:
    from pathlib import Path


def _classification_csv(tmp_path: Path) -> Path:
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return path


# A hand-written agent function: one-hot the categoricals, fit LogisticRegression.
_GOOD_FN = """
def train_and_predict(X_train, y_train, X_holdout):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    Xtr = pd.get_dummies(X_train)
    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
    model = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    return model.predict(Xho)
"""


def test_is_code_candidate() -> None:
    assert codegen.is_code_candidate({"code": "def train_and_predict(): ..."})
    assert not codegen.is_code_candidate({"model": "xgboost.XGBClassifier"})
    assert not codegen.is_code_candidate({"code": "   "})


def test_inputs_never_include_holdout_labels(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    inputs = codegen.build_inputs(ds)
    holdout = inputs[codegen.HOLDOUT_CSV].decode()
    assert "churn" not in holdout.splitlines()[0]  # target column absent from holdout
    # The train file DOES carry the target (the script trains on it).
    assert "churn" in inputs[codegen.TRAIN_CSV].decode().splitlines()[0]


def test_end_to_end_through_local_runner(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    script = codegen.assemble_script(_GOOD_FN)
    run = LocalCodeRunner().run(
        script,
        inputs=codegen.build_inputs(ds),
        outputs=[codegen.PREDICTIONS_CSV],
        timeout=60,
    )
    assert run.succeeded, run.stderr
    result = codegen.score_predictions(
        ds,
        run.outputs.get(codegen.PREDICTIONS_CSV),
        metric="f1",
        experiment_id="e1",
    )
    assert result.succeeded
    assert result.metrics is not None
    assert result.metrics.primary == "f1"
    assert 0.0 <= result.metrics.primary_value <= 1.0
    assert result.metrics.n_samples == ds.n_test


def test_wrong_length_predictions_is_a_captured_failure(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    result = codegen.score_predictions(ds, b"1\n0\n1\n", metric="f1", experiment_id="e2")
    assert not result.succeeded
    assert result.error is not None
    assert "expected" in result.error


def test_missing_predictions_is_a_captured_failure(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    result = codegen.score_predictions(ds, None, metric="f1", experiment_id="e3")
    assert not result.succeeded
    assert "no predictions" in (result.error or "")


def test_required_imports_maps_and_filters_stdlib() -> None:
    code = (
        "def train_and_predict(a, b, c):\n"
        "    import json, os\n"  # stdlib — filtered out
        "    import numpy as np\n"  # name == package
        "    import sklearn.ensemble\n"  # dotted -> top level, mapped name
        "    from xgboost import XGBClassifier\n"  # from-import
        "    import cv2\n"  # aliased package name
        "    return []\n"
    )
    assert codegen.required_imports(code) == ["numpy", "opencv-python", "scikit-learn", "xgboost"]


def test_required_imports_ignores_relative_imports() -> None:
    code = "def train_and_predict(a, b, c):\n    from . import helpers\n    return []\n"
    assert codegen.required_imports(code) == []


def test_required_imports_of_unparseable_code_is_empty() -> None:
    assert codegen.required_imports("def train_and_predict(:\n") == []


def test_validate_accepts_a_well_formed_function() -> None:
    assert codegen.validate_train_and_predict(_GOOD_FN) is None


def test_validate_rejects_syntax_error() -> None:
    assert "did not parse" in (codegen.validate_train_and_predict("def f(:\n") or "")


def test_validate_rejects_missing_entry_point() -> None:
    reason = codegen.validate_train_and_predict("def other(a, b, c):\n    return []\n")
    assert reason is not None
    assert codegen.ENTRY_POINT in reason


def test_validate_rejects_wrong_arity() -> None:
    reason = codegen.validate_train_and_predict("def train_and_predict(a, b):\n    return []\n")
    assert reason is not None
    assert "X_train" in reason


def test_validate_allows_varargs() -> None:
    code = "def train_and_predict(*args, **kwargs):\n    return []\n"
    assert codegen.validate_train_and_predict(code) is None


def test_raising_function_is_captured_by_the_runner(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    bad_fn = "def train_and_predict(X_train, y_train, X_holdout):\n    raise ValueError('boom')\n"
    run = LocalCodeRunner().run(
        codegen.assemble_script(bad_fn),
        inputs=codegen.build_inputs(ds),
        outputs=[codegen.PREDICTIONS_CSV],
        timeout=60,
    )
    assert not run.succeeded
    assert "boom" in run.stderr
    # And no predictions file means scoring also reports a failure.
    result = codegen.score_predictions(
        ds, run.outputs.get(codegen.PREDICTIONS_CSV), metric="f1", experiment_id="e4"
    )
    assert not result.succeeded
