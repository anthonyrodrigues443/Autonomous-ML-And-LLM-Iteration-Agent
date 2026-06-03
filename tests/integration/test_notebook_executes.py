"""Live integration: a rendered notebook actually runs top to bottom.

Proves the deliverable is genuinely runnable — executes the winner notebook through
a real Jupyter kernel and checks it produces a score. Needs a kernel (the dev
`ipykernel`), so it's opt-in like the other integration tests.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.deliver.notebook import build_notebook, save_notebook
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path

_HAS_NBCLIENT = importlib.util.find_spec("nbclient") is not None

_FN = (
    "def train_and_predict(X_train, y_train, X_holdout):\n"
    "    import pandas as pd\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    Xtr = pd.get_dummies(X_train)\n"
    "    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)\n"
    "    return LogisticRegression(max_iter=1000).fit(Xtr, y_train).predict(Xho)\n"
)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_NBCLIENT, reason="needs a Jupyter kernel (dev extra: nbclient/ipykernel)")
def test_best_notebook_runs_and_prints_a_score(tmp_path: Path) -> None:
    from nbclient import NotebookClient

    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    csv = tmp_path / "clf.csv"
    frame.to_csv(csv, index=False)

    exp = Experiment(
        candidate=Candidate(description="logreg", changes={"code": _FN}, rationale="r"),
        target="tabular-model",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e1",
            metrics=Metrics(values={"f1": 0.8}, primary="f1", direction="maximize"),
        ),
    )
    nb = build_notebook(exp, data_path=str(csv), target="churn", metric="f1")
    path = save_notebook(nb, tmp_path / "best.ipynb")

    executed = NotebookClient(nb, timeout=120).execute()  # raises if any cell errors
    save_notebook(executed, path)
    outputs = [
        out.get("text", "")
        for cell in executed.cells
        for out in cell.get("outputs", [])
    ]
    assert any("f1" in text for text in outputs)  # the score cell printed the metric dict
