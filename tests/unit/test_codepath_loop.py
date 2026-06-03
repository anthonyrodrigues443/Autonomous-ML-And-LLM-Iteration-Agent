"""Full agentic loop on the code path, end to end, with no LLM and no e2b.

Wires the REAL pieces — `ModelTarget` + `SandboxExecutor(LocalCodeRunner())` + the
`Orchestrator` + in-memory `Memory` — and drives them with a fake proposer that
emits real `train_and_predict` code candidates. Proves code candidates run, score,
and that a run's printed stdout is captured on the result for the next proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from iterate.adapters.compute.runner import LocalCodeRunner
from iterate.adapters.compute.sandbox import SandboxExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.core.memory import InMemoryMemory
from iterate.core.orchestrator import Orchestrator
from iterate.core.proposer import ProposerError
from iterate.core.terminator import MaxIterations
from iterate.schemas.experiment import Candidate

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.schemas.experiment import ExperimentResult
    from iterate.targets.model import ModelTarget


def _good_fn(label: str) -> str:
    return f"""
def train_and_predict(X_train, y_train, X_holdout):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    print("attempt {label}; train rows", len(X_train))
    Xtr = pd.get_dummies(X_train)
    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
    return LogisticRegression(max_iter=1000).fit(Xtr, y_train).predict(Xho)
"""


class _FakeCodeProposer:
    def __init__(self, codes: list[str]) -> None:
        self._codes = list(codes)
        self.seen_history: list[list[Any]] = []

    def propose(
        self,
        *,
        data_summary: str,
        baseline: ExperimentResult,
        current_model: str = "",
        history: list[Any] | None = None,
    ) -> Candidate:
        self.seen_history.append(list(history or []))
        if not self._codes:
            raise ProposerError("no more canned code")
        code = self._codes.pop(0)
        return Candidate(description="logreg", changes={"code": code}, rationale="r")


def _target(tmp_path: Path) -> ModelTarget:
    from iterate.targets.model import ModelTarget

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
    return ModelTarget(load_csv(path, target="churn"), metric="f1")


def test_full_loop_runs_code_candidates_and_captures_output(tmp_path: Path) -> None:
    target = _target(tmp_path)
    proposer = _FakeCodeProposer([_good_fn("one"), _good_fn("two")])
    orch = Orchestrator(
        target,
        proposer,  # type: ignore[arg-type]
        SandboxExecutor(LocalCodeRunner()),
        MaxIterations(2),
        InMemoryMemory(),
        data_summary="x",
        baseline_model="base",
    )

    result = orch.run()

    assert result.stopped_because == "max_iterations"
    assert len(result.history) == 2
    ran = [e for e in result.history if e.result and e.result.succeeded]
    assert len(ran) == 2  # both code candidates ran and scored
    # the script's stdout was captured on the result (fed back to the next proposal)
    assert all("train rows" in (e.result.logs or "") for e in ran if e.result)
    # the second proposal saw the first attempt in its history
    assert len(proposer.seen_history[1]) == 1
