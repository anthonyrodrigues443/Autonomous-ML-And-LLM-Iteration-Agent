"""Tests for the ComputeBackend protocol + its implementations.

The headline test runs a real code candidate end to end through
`SandboxExecutor(LocalCodeRunner())` + `ModelTarget` (no LLM, no e2b key),
proving the executor routes, runs, and scores the code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.compute.base import ComputeBackend, SupportsCodeGen
from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.compute.runner import LocalCodeRunner, RunResult
from iterate.adapters.compute.sandbox import SandboxExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.schemas.experiment import ExperimentResult


_GOOD_FN = """
def train_and_predict(X_train, y_train, X_holdout):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    print("train shape", X_train.shape)
    Xtr = pd.get_dummies(X_train)
    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
    return LogisticRegression(max_iter=1000).fit(Xtr, y_train).predict(Xho)
"""


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


def _target(tmp_path: Path) -> ModelTarget:
    return ModelTarget(load_csv(_classification_csv(tmp_path), target="churn"), metric="f1")


def test_local_executor_satisfies_the_protocol() -> None:
    assert isinstance(LocalExecutor(), ComputeBackend)


def test_sandbox_executor_satisfies_the_protocol() -> None:
    assert isinstance(SandboxExecutor(LocalCodeRunner()), ComputeBackend)


def test_model_target_supports_code_gen(tmp_path: Path) -> None:
    assert isinstance(_target(tmp_path), SupportsCodeGen)


def test_sandbox_runs_a_code_candidate_end_to_end(tmp_path: Path) -> None:
    target = _target(tmp_path)
    candidate = Candidate(description="logreg", changes={"code": _GOOD_FN}, rationale="r")
    result = SandboxExecutor(LocalCodeRunner()).execute(target, candidate)
    assert result.succeeded, result.error
    assert result.metrics is not None
    assert result.metrics.primary == "f1"
    assert result.duration_seconds is not None
    # stdout the script printed is surfaced for the next proposal to read.
    assert result.logs is not None
    assert "train shape" in result.logs


def test_sandbox_runs_baseline_in_process(tmp_path: Path) -> None:
    result = SandboxExecutor(LocalCodeRunner()).execute(_target(tmp_path), None)
    assert result.succeeded, result.error
    assert result.experiment_id == "baseline"


def test_sandbox_runs_spec_candidate_in_process(tmp_path: Path) -> None:
    candidate = Candidate(
        description="rf", changes={"model": "sklearn.ensemble.RandomForestClassifier"}, rationale="r"
    )
    result = SandboxExecutor(LocalCodeRunner()).execute(_target(tmp_path), candidate)
    assert result.succeeded, result.error


def test_sandbox_captures_a_crashing_code_candidate(tmp_path: Path) -> None:
    bad = "def train_and_predict(X_train, y_train, X_holdout):\n    raise ValueError('boom')\n"
    candidate = Candidate(description="bad", changes={"code": bad}, rationale="r")
    result = SandboxExecutor(LocalCodeRunner()).execute(_target(tmp_path), candidate)
    assert not result.succeeded
    assert result.error is not None
    assert "boom" in result.error  # the traceback tail is fed back


def test_sandbox_rejects_code_for_a_non_codegen_target() -> None:
    candidate = Candidate(description="x", changes={"code": _GOOD_FN}, rationale="r")
    result = SandboxExecutor(LocalCodeRunner()).execute(_PlainTarget(), candidate)
    assert not result.succeeded
    assert "does not support code" in (result.error or "")


def test_sandbox_captures_a_runner_that_cannot_boot(tmp_path: Path) -> None:
    candidate = Candidate(description="x", changes={"code": _GOOD_FN}, rationale="r")
    result = SandboxExecutor(_ExplodingRunner()).execute(_target(tmp_path), candidate)
    assert not result.succeeded
    assert "RuntimeError" in (result.error or "")


# ─── stubs ──────────────────────────────────────────────────────────────────


class _PlainTarget:
    name = "plain"

    def baseline(self) -> ExperimentResult:  # pragma: no cover - not reached
        raise NotImplementedError

    def run(self, candidate: Candidate) -> ExperimentResult:  # pragma: no cover
        raise NotImplementedError


class _ExplodingRunner:
    def run(
        self,
        script: str,
        *,
        inputs: dict[str, bytes],
        outputs: list[str],
        timeout: float,
        packages: list[str] | None = None,
    ) -> RunResult:
        raise RuntimeError("sandbox failed to boot")
