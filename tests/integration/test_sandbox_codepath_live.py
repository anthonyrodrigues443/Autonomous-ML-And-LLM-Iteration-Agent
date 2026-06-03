"""Live integration: the full code path against a real e2b sandbox.

The first true sandboxed run — `SandboxExecutor(E2BCodeRunner())` runs a generated
`train_and_predict` (with a non-preinstalled import, to exercise install-on-demand)
in real e2b and scores it through our contract. No LLM here; the CodeProposer's
own live test (real qwen3 writing the function) is a separate, heavier check.

Skips unless E2B_API_KEY is set (the e2b SDK ships in core).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.compute.runner import E2BCodeRunner
from iterate.adapters.compute.sandbox import SandboxExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from pathlib import Path

_HAS_KEY = bool(os.environ.get("E2B_API_KEY"))

# Imports xgboost — not in the base e2b image — so the run exercises install-on-demand.
_XGB_FN = """
def train_and_predict(X_train, y_train, X_holdout):
    import pandas as pd
    from xgboost import XGBClassifier
    print("rows:", len(X_train))
    Xtr = pd.get_dummies(X_train)
    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
    return XGBClassifier(n_estimators=50).fit(Xtr, y_train).predict(Xho)
"""


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="needs E2B_API_KEY")
def test_code_candidate_runs_in_real_e2b(tmp_path: Path) -> None:
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    target = ModelTarget(load_csv(path, target="churn"), metric="f1")

    candidate = Candidate(description="xgboost", changes={"code": _XGB_FN}, rationale="r")
    result = SandboxExecutor(E2BCodeRunner()).execute(target, candidate)

    assert result.succeeded, result.error
    assert result.metrics is not None
    assert result.logs is not None
    assert "rows:" in result.logs
