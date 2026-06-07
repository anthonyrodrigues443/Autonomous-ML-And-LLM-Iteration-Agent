"""Tests for the notebook deliverable renderer + the CLI's best/all/none wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nbformat

from iterate.deliver.notebook import build_notebook, save_notebook, slug
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path

_FN = (
    "def train_and_predict(X_train, y_train, X_holdout):\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    return LogisticRegression().fit(X_train, y_train).predict(X_holdout)\n"
)


def _experiment(
    *, code: str | None = None, model: str | None = None, score: float = 0.8, iteration: int = 2
) -> Experiment:
    changes = {"code": code} if code is not None else {"model": model, "params": {"max_depth": 4}}
    return Experiment(
        candidate=Candidate(description="logreg one-hot baseline", changes=changes, rationale="why"),
        target="tabular-model",
        hypothesis="h",
        status="completed",
        iteration=iteration,
        result=ExperimentResult(
            experiment_id="e1",
            metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize"),
        ),
    )


def _sources(nb: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in nb.cells)


def test_code_notebook_is_valid_and_runnable_shaped() -> None:
    nb = build_notebook(
        _experiment(code=_FN), data_path="churn.csv", target="churn", metric="f1",
        baseline_score=0.70, is_best=True,
    )
    nbformat.validate(nb)  # schema-valid
    text = _sources(nb)
    assert "train_and_predict" in text  # the winning code is present
    assert "load_csv('churn.csv', target='churn')" in text  # reproduces the exact split
    assert "task_for_metric('f1')" in text  # scored by the same ruler
    assert "0.8000" in text  # the reported score in the header
    assert "vs baseline 0.7000" in text


def test_leaderboard_cell_lists_every_attempt_and_failures() -> None:
    ok = _experiment(code=_FN, score=0.82, iteration=2)
    failed = Experiment(
        candidate=Candidate(description="xgb attempt", changes={"code": _FN}, rationale="r"),
        target="t",
        hypothesis="h",
        status="failed",
        iteration=1,
        result=ExperimentResult(experiment_id="e", error="code script failed:\nKeyError: 'age'"),
    )
    nb = build_notebook(
        ok, data_path="d.csv", target="y", metric="f1", baseline_score=0.70,
        is_best=True, leaderboard=[failed, ok],
    )
    text = _sources(nb)
    assert "What was tried" in text
    assert "| base | baseline | 0.7000 | — |" in text
    assert "0.8200" in text  # the successful attempt's score
    assert "KeyError" in text  # the failure + its reason are shown


def test_spec_notebook_rebuilds_through_model_target() -> None:
    nb = build_notebook(
        _experiment(model="xgboost.XGBClassifier"), data_path="d.csv", target="y", metric="f1"
    )
    nbformat.validate(nb)
    text = _sources(nb)
    assert "ModelTarget" in text
    assert "xgboost.XGBClassifier" in text


def test_failed_experiment_notebook_notes_the_failure() -> None:
    exp = Experiment(
        candidate=Candidate(description="bad", changes={"code": _FN}, rationale="r"),
        target="t",
        hypothesis="h",
        status="failed",
        result=ExperimentResult(experiment_id="e", error="code script failed:\nKeyError: 'age'"),
    )
    nb = build_notebook(exp, data_path="d.csv", target="y", metric="f1")
    assert "FAILED" in _sources(nb)


def test_slug_is_filesystem_safe() -> None:
    assert slug("XGBoost (max_depth=4): curb overfit!") == "xgboost-max-depth-4-curb-overfit"
    assert slug("") == "experiment"


def test_session_notebook_attaches_real_outputs_to_cells() -> None:
    from iterate.deliver.notebook import build_session_notebook

    cells = [
        # a cell with REAL captured outputs (a stream + an execute_result)
        {
            "code": "print('hi'); 6*7",
            "stdout": "hi\n",
            "error": None,
            "source": "agent",
            "outputs": [
                {"type": "stream", "name": "stdout", "text": "hi\n"},
                {"type": "execute_result", "data": {"text/plain": "42"}, "execution_count": 1},
            ],
        },
        # a cell with only an error string (no structured outputs) → fallback
        {"code": "boom()", "stdout": "", "error": "NameError: name 'boom' is not defined",
         "source": "agent", "outputs": []},
    ]
    nb = build_session_notebook(cells, title="logreg session", metric="f1", score=0.61, baseline_score=0.57)
    nbformat.validate(nb)  # schema-valid executed notebook
    assert "f1 = 0.6100" in nb.cells[0].source  # header
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 2
    assert code_cells[0].execution_count == 1
    # the real captured outputs are attached to the cell (not markdown notes)
    kinds = [o.output_type for o in code_cells[0].outputs]
    assert "stream" in kinds
    assert "execute_result" in kinds
    # the error cell falls back to a synthesized error output
    assert code_cells[1].outputs[0].output_type == "error"
    assert "boom" in "\n".join(code_cells[1].outputs[0].traceback)


def test_save_round_trips(tmp_path: Path) -> None:
    nb = build_notebook(_experiment(code=_FN), data_path="d.csv", target="y", metric="f1")
    path = save_notebook(nb, tmp_path / "sub" / "best.ipynb")
    assert path.exists()
    nbformat.read(path, as_version=4)  # reads back clean


# ─── CLI wiring: best / all / none ────────────────────────────────────────


def _run_result(experiments: list[Experiment]) -> object:
    from iterate.core.orchestrator import RunResult

    baseline = ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": 0.70}, primary="f1", direction="maximize"),
    )
    return RunResult(
        baseline=baseline,
        history=experiments,
        best=experiments[-1] if experiments else None,
        stopped_because="max_iterations",
        run_id="run1",
    )


def test_write_notebooks_best_emits_only_winner(tmp_path: Path) -> None:
    from iterate.cli import _write_notebooks

    result = _run_result([_experiment(code=_FN), _experiment(code=_FN, score=0.82)])
    _write_notebooks(result, mode="best", run_dir=tmp_path, data_path="d.csv", target="y", metric="f1")  # type: ignore[arg-type]
    assert (tmp_path / "best.ipynb").exists()
    assert not (tmp_path / "notebooks").exists()


def test_save_best_model_code_winner_creates_run_dir(tmp_path: Path) -> None:
    # Regression: a code winner skips target.save_model (which used to mkdir), so
    # _save_best_model must create the run dir itself before writing best.json.
    from iterate.cli import _save_best_model

    result = _run_result([_experiment(code=_FN, score=0.82)])
    run_dir = tmp_path / "runs" / "abc"  # does not exist yet
    _save_best_model(None, result, "f1", run_dir / "best_model.joblib")  # type: ignore[arg-type]
    best_json = run_dir / "best.json"
    assert best_json.exists()
    import json

    saved = json.loads(best_json.read_text())
    assert saved["code"] == _FN
    assert saved["score"] == 0.82


def test_write_notebooks_all_emits_one_per_experiment(tmp_path: Path) -> None:
    from iterate.cli import _write_notebooks

    result = _run_result(
        [_experiment(code=_FN, iteration=1), _experiment(model="xgboost.XGBClassifier", iteration=2)]
    )
    _write_notebooks(result, mode="all", run_dir=tmp_path, data_path="d.csv", target="y", metric="f1")  # type: ignore[arg-type]
    journey = list((tmp_path / "notebooks").glob("*.ipynb"))
    assert len(journey) == 2
    assert (tmp_path / "best.ipynb").exists()
