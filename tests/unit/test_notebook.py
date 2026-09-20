"""Tests for the notebook deliverable renderer + the CLI's best/all/none wiring."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

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


def test_string_traceback_in_captured_outputs_renders_and_validates() -> None:
    # cells recorded before the e2b traceback normalization carry the traceback as
    # ONE string; the schema wants an array, and rendering a finished run must not
    # crash on it (it did, at the end of the first live e2b run).
    from iterate.deliver.notebook import build_session_notebook

    cells = [{
        "code": "boom()", "stdout": "", "error": "ValueError: bad", "source": "agent",
        "outputs": [{
            "type": "error", "ename": "ValueError", "evalue": "bad",
            "traceback": "Traceback (most recent call last)\nValueError: bad",
        }],
    }]
    nb = build_session_notebook(cells, title="t", metric="f1", score=0.6)
    nbformat.validate(nb)  # would raise without the coercion
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells[0].outputs[0].traceback == [
        "Traceback (most recent call last)", "ValueError: bad",
    ]


def test_session_notebook_carries_the_honesty_note_in_the_header() -> None:
    # run 20 i3: a duplicate's header read "+0.0321 vs baseline" although the
    # submission was byte-identical to an earlier one — the machine verdict must
    # ride the header so the delta cannot mislead.
    from iterate.deliver.notebook import build_session_notebook

    cells = [{"code": "x = 1", "stdout": "", "error": None, "source": "agent", "outputs": []}]
    nb = build_session_notebook(
        cells, title="dup", metric="f1", score=0.5997, baseline_score=0.5676,
        honesty_note="this submission is byte-identical to an earlier experiment's",
    )
    nbformat.validate(nb)
    assert "byte-identical to an earlier experiment" in nb.cells[0].source
    # without a note the header stays as before
    plain = build_session_notebook(cells, title="t", metric="f1", score=0.6)
    assert "Note:" not in plain.cells[0].source


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


def test_session_notebook_renders_thinking_as_markdown_before_the_cell() -> None:
    from iterate.deliver.notebook import build_session_notebook

    cells = [
        # preamble-style cell without thinking → no reasoning block
        {"code": "load()", "stdout": "ok", "error": None, "source": "preamble", "outputs": []},
        # an agent cell carrying the model's reasoning trace
        {
            "code": "X_tr = prepare(X_train)",
            "stdout": "",
            "error": None,
            "source": "agent",
            "outputs": [],
            "thinking": "The profile shows skew in tenure.\n\nI will log-transform it first.",
        },
    ]
    nb = build_session_notebook(cells, title="t", metric="f1")
    nbformat.validate(nb)
    md = [c.source for c in nb.cells if c.cell_type == "markdown"]
    # header + exactly ONE reasoning block (the no-thinking cell adds none)
    reasoning = [m for m in md if "Model reasoning" in m]
    assert len(reasoning) == 1
    assert "> The profile shows skew in tenure." in reasoning[0]
    assert "> I will log-transform it first." in reasoning[0]
    # the reasoning block sits immediately BEFORE the code cell it produced
    kinds = [(c.cell_type, "Model reasoning" in c.source) for c in nb.cells]
    idx = kinds.index(("markdown", True))
    assert nb.cells[idx + 1].cell_type == "code"
    assert "prepare(X_train)" in nb.cells[idx + 1].source


def test_session_notebook_renders_hypothesis_and_findings() -> None:
    from iterate.deliver.notebook import build_session_notebook

    cells = [{"code": "x=1", "stdout": "ok", "error": None, "source": "agent", "outputs": []}]
    digest = {
        "what_helped": ["target encoding: 0.55 -> 0.61"],
        "what_hurt": ["power transform: no change"],
        "data_insights": ["27% positive class"],
        "val_trail": "0.55 -> 0.61",
        "takeaway": "Add threshold tuning next.",
    }
    nb = build_session_notebook(
        cells, title="t", metric="f1", score=0.61,
        hypothesis="so far: one-hot scored 0.55; target-encode the high-cardinality columns",
        findings=digest,
    )
    nbformat.validate(nb)
    md = [c.source for c in nb.cells if c.cell_type == "markdown"]
    # hypothesis right after the header, carrying the supervisor brief verbatim
    assert "## Hypothesis" in md[1]
    assert "target-encode the high-cardinality columns" in md[1]
    # findings is the LAST cell: helped/hurt/insights/trail/takeaway all rendered
    assert nb.cells[-1].cell_type == "markdown"
    last = nb.cells[-1].source
    assert "## Findings" in last
    assert "target encoding: 0.55 -> 0.61" in last
    assert "power transform: no change" in last
    assert "27% positive class" in last
    assert "**Takeaway:** Add threshold tuning next." in last


def test_session_notebook_skips_hypothesis_and_findings_when_absent() -> None:
    from iterate.deliver.notebook import build_session_notebook

    cells = [{"code": "x=1", "stdout": "", "error": None, "source": "agent", "outputs": []}]
    nb = build_session_notebook(cells, title="t", metric="f1")
    nbformat.validate(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "## Hypothesis" not in text
    assert "## Findings" not in text


def test_save_round_trips(tmp_path: Path) -> None:
    nb = build_notebook(_experiment(code=_FN), data_path="d.csv", target="y", metric="f1")
    path = save_notebook(nb, tmp_path / "sub" / "best.ipynb")
    assert path.exists()
    nbformat.read(path, as_version=4)  # reads back clean


# ─── progress bars in captured outputs ────────────────────────────────────

_DOWNLOADING = 'Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to x.pth'
_TRAINING_LINES = [
    "epoch 1/3 loss=0.5033 train_acc=0.8443 22s",
    "epoch 2/3 loss=0.2073 train_acc=0.9329 19s",
    "epoch 3/3 loss=0.0829 train_acc=0.9734 20s",
    'FIT {"backbone": "resnet18", "image_size": 64, "epochs_run": 3, "val": 0.9737}',
    'MODEL {"model": "efficientnet_b0", "image_size": 224, "epochs": 2, "val": 0.9478}',
    'SUBMITTED {"backbone": "resnet18", "image_size": 64, "val": 0.9737}',
    "KEPT the earlier submission: its val f1_macro 0.9804 is not beaten by 0.9737",
]


def _stream(name: str, text: str) -> dict[str, Any]:
    return {"type": "stream", "name": name, "text": text}


def _session(captured: list[dict[str, Any]]) -> nbformat.NotebookNode:
    from iterate.deliver.notebook import build_session_notebook

    cell = {"code": "fit()", "stdout": "", "error": None, "source": "agent", "outputs": captured}
    nb = build_session_notebook([cell], title="t", metric="f1")
    nbformat.validate(nb)
    return nb


def _outputs(captured: list[dict[str, Any]]) -> list[Any]:
    return list(next(c for c in _session(captured).cells if c.cell_type == "code").outputs)


def test_training_lines_survive_in_the_written_notebook_and_tick_lines_do_not(
    tmp_path: Path,
) -> None:
    download = [_stream("stderr", f"\r{i / 10:.1f}%") for i in range(1, 1001)]
    batches = [_stream("stderr", f"\r{i * 10:3d}%|{'#' * i:<10}| {i}/10") for i in range(1, 11)]
    first, *rest = _TRAINING_LINES
    captured = [
        _stream("stdout", _DOWNLOADING + "\n"),
        *download,
        _stream("stderr", "\n"),
        _stream("stdout", first + "\n"),
        *batches,
        _stream("stderr", "\n"),
        *[_stream("stdout", line + "\n") for line in rest],
    ]
    path = save_notebook(_session(captured), tmp_path / "best.ipynb")
    written = nbformat.read(path, as_version=4)
    nbformat.validate(written)
    outputs = next(c for c in written.cells if c.cell_type == "code").outputs

    assert [o.name for o in outputs] == ["stdout", "stderr", "stdout", "stderr", "stdout"]
    printed = "".join(o.text for o in outputs if o.name == "stdout")
    assert printed.splitlines() == [_DOWNLOADING, *_TRAINING_LINES]
    bars = "".join(o.text for o in outputs if o.name == "stderr")
    assert bars.splitlines() == ["100.0%", "100%|##########| 10/10"]
    assert not any("\r" in o.text for o in outputs)


def test_a_progress_bar_that_ends_each_tick_with_a_carriage_return_settles_too() -> None:
    ticks = [_stream("stdout", f"{i}%\r") for i in (10, 20, 30)]
    (only,) = _outputs([*ticks, _stream("stdout", "done\nepoch 1/1 loss=0.5\n")])
    assert only.text == "done\nepoch 1/1 loss=0.5\n"


def test_a_windows_line_ending_is_left_alone() -> None:
    text = "epoch 1/2 loss=0.5\r\nepoch 2/2 loss=0.4\r\r\nFIT {}\r\n"
    (only,) = _outputs([_stream("stdout", text)])
    assert only.text == text


def test_streams_join_only_with_a_neighbour_of_the_same_name() -> None:
    result = {"type": "execute_result", "data": {"text/plain": "1"}, "execution_count": 1}
    outputs = _outputs(
        [
            _stream("stdout", "one\n"),
            _stream("stderr", "warn\n"),
            _stream("stdout", "two\n"),
            _stream("stdout", "three\n"),
            result,
            _stream("stdout", "four\n"),
        ]
    )
    assert [(o.output_type, o.get("name"), o.get("text")) for o in outputs] == [
        ("stream", "stdout", "one\n"),
        ("stream", "stderr", "warn\n"),
        ("stream", "stdout", "two\nthree\n"),
        ("execute_result", None, None),
        ("stream", "stdout", "four\n"),
    ]


def test_settling_a_cell_leaves_its_captured_record_alone() -> None:
    captured = [_stream("stderr", "\r1%"), _stream("stderr", "\r2%\n")]
    (only,) = _outputs(captured)
    assert only.text == "2%\n"
    assert captured == [_stream("stderr", "\r1%"), _stream("stderr", "\r2%\n")]


def test_one_long_unbroken_line_after_a_progress_bar_is_kept_whole() -> None:
    blob = "A" * 400_000
    (only,) = _outputs([_stream("stdout", "\r50%\r100%\n"), _stream("stdout", blob + "\n")])
    assert only.text == "100%\n" + blob + "\n"


def test_a_hundred_thousand_stream_outputs_in_one_cell_join_in_under_a_second() -> None:
    from iterate.deliver.notebook import build_session_notebook

    lines = [f"{i:>8} " + "x" * 50 + "\n" for i in range(100_000)]
    captured = [_stream("stdout", line) for line in lines]
    cell = {"code": "fit()", "stdout": "", "error": None, "source": "agent", "outputs": captured}

    # CPU time, not wall time: other work on the machine must not move this number.
    started = time.process_time()
    nb = build_session_notebook([cell], title="t", metric="f1")
    spent = time.process_time() - started

    (only,) = next(c for c in nb.cells if c.cell_type == "code").outputs
    assert only.text == "".join(lines)
    assert spent < 1.0


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


def test_delivered_notebook_scoring_cell_handles_the_proba_contract() -> None:
    """The cell calls itself "the same ruler iterate used", so it has to unpack the
    same 2-tuple contract; otherwise a roc_auc notebook prints a panel with no
    roc_auc in it."""
    from iterate.deliver.notebook import _score_code_cell

    cell = _score_code_cell("roc_auc")
    assert "isinstance(_out, tuple)" in cell
    assert "y_proba=probabilities" in cell


def test_the_load_cell_of_a_users_split_names_both_files() -> None:
    from iterate.deliver.notebook import _load_cell

    cell = _load_cell("train.csv", "y", "holdout.csv")
    assert "load_split('train.csv', 'holdout.csv', target='y')" in cell
    assert "load_csv" not in cell
    plain = _load_cell("data.csv", "y")
    assert "load_csv('data.csv', target='y')" in plain
    assert "load_split" not in plain
