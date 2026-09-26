"""Tests for the `iterate run` CLI command + its helpers.

Heavy on mocking — we want to verify *wiring*, not exercise the real LLM/data
path (the live integration test for the loop comes at Day 6).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from iterate import cli as cli_module
from iterate.cli import (
    _archive_memory_db,
    _check_baseline_divergence,
    _default_baseline_model,
    _parse_duration,
    _read_source,
    app,
)
from iterate.schemas.experiment import Candidate, ExperimentResult, Metrics

# Error panels wrap at the terminal width, which can split a phrase a test looks for.
runner = CliRunner(env={"COLUMNS": "1000"})


# ─── Helpers ─────────────────────────────────────────────────────────────


def test_parse_duration_handles_h_m_s_combinations() -> None:
    assert _parse_duration("30s") == 30
    assert _parse_duration("15m") == 900
    assert _parse_duration("2h") == 7200
    assert _parse_duration("1h30m") == 5400
    assert _parse_duration("90") == 90  # bare number → seconds


def test_parse_duration_rejects_garbage() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_duration("forever")
    with pytest.raises(typer.BadParameter):
        _parse_duration("5x")


def test_default_baseline_model_is_task_aware() -> None:
    assert "Classifier" in _default_baseline_model("f1")
    assert "Classifier" in _default_baseline_model("accuracy")
    assert "Regressor" in _default_baseline_model("rmse")
    assert "Regressor" in _default_baseline_model("r2")


def test_read_source_returns_text_for_plain_files(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# my notes\nused CatBoost", encoding="utf-8")
    assert "CatBoost" in _read_source(path)


def test_read_source_walks_notebook_cells(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Churn baseline\n"]},
            {"cell_type": "code", "source": "import catboost\nm = catboost.CatBoostClassifier()\n"},
        ]
    }
    path = tmp_path / "approach.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    text = _read_source(path)
    assert "Churn baseline" in text
    assert "CatBoostClassifier" in text
    assert "```python" in text


def test_archive_memory_db_renames_existing_file(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    db.write_text("fake-db-bytes")

    archived = _archive_memory_db(db)

    assert archived is not None
    assert not db.exists()
    assert archived.exists()
    assert ".bak" in archived.name


def test_archive_memory_db_noop_when_missing(tmp_path: Path) -> None:
    assert _archive_memory_db(tmp_path / "does-not-exist.db") is None


def test_baseline_divergence_warning_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    # Within tolerance — silent.
    _check_baseline_divergence(reported=0.78, measured=0.80)
    out, _ = capsys.readouterr()
    assert "warning" not in out.lower()

    # Outside tolerance — warns.
    _check_baseline_divergence(reported=0.78, measured=0.60)
    out, _ = capsys.readouterr()
    assert "warning" in out.lower()


def test_baseline_divergence_safe_against_zero_reported() -> None:
    _check_baseline_divergence(reported=0, measured=0.5)  # must not divide by zero


# ─── CLI flag validation ────────────────────────────────────────────────


def _write_tiny_csv(path: Path) -> None:
    # Need enough rows for stratified train/test split with both classes in each.
    n = 60
    frame = pd.DataFrame({"feat": list(range(n)), "churn": [i % 2 for i in range(n)]})
    frame.to_csv(path, index=False)


def test_run_rejects_baseline_without_source(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)

    result = runner.invoke(
        app,
        ["run", "--data", str(data), "--target", "churn", "--metric", "f1", "--baseline", "0.7"],
    )
    assert result.exit_code != 0
    assert "--baseline requires --source" in (result.stderr or result.stdout)


def test_run_rejects_unknown_metric(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)

    result = runner.invoke(
        app,
        ["run", "--data", str(data), "--target", "churn", "--metric", "bleu"],
    )
    assert result.exit_code != 0
    assert "unknown metric" in (result.stderr or result.stdout)


def test_run_rejects_cloud_backend_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    # Make sure no env-var key sneaks in
    for env_attr in (
        "openai_api_key",
        "groq_api_key",
        "together_api_key",
        "deepseek_api_key",
    ):
        monkeypatch.setattr(cli_module.get_settings(), env_attr, None, raising=False)

    result = runner.invoke(
        app,
        ["run", "--data", str(data), "--target", "churn", "--metric", "f1", "--backend", "groq"],
    )
    assert result.exit_code != 0
    assert "requires --api-key" in (result.stderr or result.stdout)


# ─── End-to-end wiring (with heavy mocking) ──────────────────────────────


class _StubProposer:
    def propose(self, **_: Any) -> Candidate:
        return Candidate(description="x", changes={"model": "stub.M"}, rationale="r")


def _stub_run_orchestrator(
    monkeypatch: pytest.MonkeyPatch, *, best_model: str | None = None
) -> dict[str, Any]:
    """Patch the heavyweight bits so `iterate run` returns instantly. Capture the wiring.

    If ``best_model`` is given, the stubbed run returns a RunResult with a winning
    Experiment using that model — so the model-save path is exercised end to end.
    """
    captured: dict[str, Any] = {}

    # Fake LLM client — never called because we also stub Proposer.
    class _FakeClient:
        @property
        def model(self) -> str:
            return "fake"

        def chat(self, *a: Any, **kw: Any) -> Any:
            raise AssertionError("LLM client should not be invoked in this test")

    def _fake_build_client(name: str, **kw: Any) -> _FakeClient:
        captured["backend"] = name
        captured["client_kwargs"] = kw
        return _FakeClient()

    def _fake_proposer_ctor(client: Any) -> _StubProposer:
        return _StubProposer()

    def _fake_orchestrator_run(self: Any) -> Any:
        captured["data_summary"] = self._data_summary
        captured["baseline_model"] = self._initial_model
        captured["baseline_candidate"] = self._baseline_candidate
        captured["memory_type"] = type(self._memory).__name__
        from iterate.core.orchestrator import RunResult
        from iterate.schemas.experiment import Candidate, Experiment

        baseline = ExperimentResult(
            experiment_id="b",
            metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=100),
        )
        best: Experiment | None = None
        if best_model is not None:
            best = Experiment(
                candidate=Candidate(
                    description=f"winner {best_model}", changes={"model": best_model}, rationale="r"
                ),
                target="tabular-model",
                hypothesis="x",
                status="completed",
                result=ExperimentResult(
                    experiment_id="e1",
                    metrics=Metrics(
                        values={"f1": 0.8}, primary="f1", direction="maximize", n_samples=100
                    ),
                ),
            )
        return RunResult(
            baseline=baseline,
            history=[best] if best else [],
            best=best,
            stopped_because="max_iterations",
            run_id="testrun",
        )

    # `run()` imports these lazily (`from … import X`), so patch them at their
    # source modules — patching cli_module wouldn't be seen by the local import.
    import iterate.core.orchestrator as orch_module
    import iterate.core.proposer as proposer_module
    import iterate.llm.factory as factory_module

    monkeypatch.setattr(factory_module, "build_client", _fake_build_client)
    monkeypatch.setattr(proposer_module, "Proposer", _fake_proposer_ctor)
    monkeypatch.setattr(orch_module.Orchestrator, "run", _fake_orchestrator_run)
    return captured


def test_run_uses_default_baseline_when_no_source_and_empty_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory_db = tmp_path / "memory.db"

    captured = _stub_run_orchestrator(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(memory_db),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["baseline_candidate"] is None
    assert "Classifier" in captured["baseline_model"]
    assert captured["backend"] == "ollama"


def test_run_with_fresh_archives_existing_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory_db = tmp_path / "memory.db"
    memory_db.write_text("existing-data")  # pre-existing db

    _stub_run_orchestrator(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(memory_db),
            "--fresh",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Original db got archived (renamed), so the original path no longer exists
    # until SqliteMemory creates a fresh one.
    # After the run, a new db exists at the path AND a .bak sibling exists.
    siblings = list(tmp_path.iterdir())
    assert any(".bak" in p.name for p in siblings), siblings


def test_run_uses_prior_best_from_memory_as_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When memory has a prior best and we're not --fresh, that becomes the baseline."""
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory_db = tmp_path / "memory.db"

    # Pre-seed a real SqliteMemory at that path with a prior best.
    from iterate.core.memory import SqliteMemory
    from iterate.schemas.experiment import Experiment

    seed_mem = SqliteMemory(memory_db)
    rid = seed_mem.start_run(
        "tabular-model",
        ExperimentResult(
            experiment_id="b0",
            metrics=Metrics(values={"f1": 0.6}, primary="f1", direction="maximize", n_samples=100),
        ),
    )
    prior = Experiment(
        candidate=Candidate(
            description="prior winner",
            changes={"model": "xgboost.XGBClassifier"},
            rationale="r",
        ),
        target="tabular-model",
        hypothesis="prior winner",
        status="completed",
        result=ExperimentResult(
            experiment_id="e1",
            metrics=Metrics(values={"f1": 0.79}, primary="f1", direction="maximize", n_samples=100),
        ),
    )
    seed_mem.record(rid, prior)
    seed_mem.finish_run(rid, "max_iterations")
    seed_mem.close()

    captured = _stub_run_orchestrator(monkeypatch)
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(memory_db),
        ],
    )
    assert result.exit_code == 0, result.stdout
    cand = captured["baseline_candidate"]
    assert cand is not None
    assert cand.changes["model"] == "xgboost.XGBClassifier"
    assert "memory" in result.stdout.lower()


def test_in_memory_memory_used_when_fresh_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI still uses SqliteMemory; --fresh archives the OLD db; new db
    is written. The Memory type observed by the orchestrator is SqliteMemory."""
    # This test verifies the memory_type is SqliteMemory after archive (not
    # the in-memory implementation, just a fresh sqlite file).
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory_db = tmp_path / "memory.db"

    captured = _stub_run_orchestrator(monkeypatch)
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(memory_db),
            "--fresh",
        ],
    )
    assert result.exit_code == 0
    assert captured["memory_type"] == "SqliteMemory"


def test_run_saves_best_model_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The winning model + a best.json sidecar are written where --output points."""
    import joblib

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    out = tmp_path / "models" / "best_model.joblib"

    _stub_run_orchestrator(monkeypatch, best_model="sklearn.linear_model.LogisticRegression")
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(tmp_path / "memory.db"),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    sidecar = out.with_name("best.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["model"] == "sklearn.linear_model.LogisticRegression"
    assert meta["metric"] == "f1"
    # The saved artifact is a usable fitted pipeline.
    pipeline = joblib.load(out)
    assert hasattr(pipeline, "predict")


# ─── Code-path wiring: --think applies to the coder only ─────────────────


def _stub_run_supervised(
    monkeypatch: pytest.MonkeyPatch, *, invoke_hook: bool = False
) -> dict[str, Any]:
    """Stub the code path's heavy bits; capture which client each agent received.

    With ``invoke_hook``, the fake loop calls ``on_experiment`` once with a finished
    cells-experiment and returns an EMPTY history — so any notebook on disk can only
    have come from the incremental hook, never the end-of-run writer."""
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, think: bool) -> None:
            self.think = think

        @property
        def model(self) -> str:
            return "fake"

        def chat(self, *a: Any, **kw: Any) -> Any:
            raise AssertionError("LLM client should not be invoked in this test")

    def _fake_build_client(name: str, **kw: Any) -> _FakeClient:
        return _FakeClient(think=kw.get("think", False))

    def _fake_run_supervised(
        *,
        target: Any,
        dataset: Any,
        supervisor: Any,
        make_coder: Any,
        terminator: Any,
        memory: Any,
        data_summary: str,
        summarizer: Any = None,
        on_experiment: Any = None,
        controller: Any = None,
        # **kwargs so adding a loop parameter does not break every CLI test; the
        # ones these tests assert on are named explicitly above.
        **kwargs: Any,
    ) -> Any:
        from iterate.core.orchestrator import RunResult
        from iterate.schemas.experiment import Candidate, Experiment

        coder = make_coder()
        captured["coder"] = coder
        captured["kernel"] = coder._kernel
        captured["dataset"] = dataset
        captured["supervisor_client"] = supervisor._client
        captured["coder_client"] = coder._client
        captured["controller"] = controller
        captured["researcher"] = kwargs.get("researcher")
        baseline = ExperimentResult(
            experiment_id="b",
            metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=100),
        )
        if invoke_hook and on_experiment is not None:
            exp = Experiment(
                candidate=Candidate(
                    description="probe attempt",
                    changes={
                        "cells": [
                            {
                                "code": "x=1",
                                "stdout": "ok",
                                "error": None,
                                "source": "agent",
                                "outputs": [],
                                "thinking": None,
                            }
                        ]
                    },
                    rationale="r",
                ),
                target="tabular-model",
                hypothesis="h",
                status="completed",
                iteration=1,
                result=ExperimentResult(
                    experiment_id="e",
                    metrics=Metrics(
                        values={"f1": 0.8}, primary="f1", direction="maximize", n_samples=100
                    ),
                ),
            )
            on_experiment(experiment=exp, baseline=baseline, is_best=True, run_id="t")
        return RunResult(
            baseline=baseline,
            history=[],
            best=None,
            stopped_because="max_iterations",
            run_id="t",
        )

    # run() imports these lazily — patch at the source modules.
    import iterate.core.agent_loop as loop_module
    import iterate.llm.factory as factory_module

    monkeypatch.setattr(factory_module, "build_client", _fake_build_client)
    monkeypatch.setattr(loop_module, "run_supervised", _fake_run_supervised)
    return captured


def test_think_applies_to_the_coder_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    captured = _stub_run_supervised(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--code",
            "--think",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    # the supervisor must NEVER think (single-tool-call role); only the coder does
    assert captured["supervisor_client"].think is False
    assert captured["coder_client"].think is True
    assert captured["supervisor_client"] is not captured["coder_client"]
    # no terminal attached under the test runner -> no interactive controller
    assert captured["controller"] is None


def test_without_think_both_agents_share_one_no_think_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    captured = _stub_run_supervised(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--code",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["coder_client"] is captured["supervisor_client"]  # same instance
    assert captured["coder_client"].think is False


def test_each_iteration_notebook_is_saved_the_moment_it_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "runs"))
    get_settings.cache_clear()
    _stub_run_supervised(monkeypatch, invoke_hook=True)
    try:
        result = runner.invoke(
            app,
            [
                "run",
                "--data",
                str(data),
                "--target",
                "churn",
                "--metric",
                "f1",
                "--code",
                "--notebooks",
                "all",
                "--memory",
                str(tmp_path / "m.db"),
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.stdout
    run_dir = tmp_path / "runs" / "t"
    # the stubbed loop returned an EMPTY history, so these files can only have been
    # written by the per-iteration hook — i.e. while the run was still going.
    assert list((run_dir / "notebooks").glob("iter_01_*.ipynb")), "incremental save missing"
    assert (run_dir / "best.ipynb").exists()  # best.ipynb tracks the best-so-far


def test_the_winner_notebook_gets_the_files_its_first_cell_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run All has failed at cell 1 since v0.2: the session read its three files from
    the kernel's folder, which is deleted when the run ends. They land beside the
    notebook now, as the same bytes the kernel was started with."""
    from iterate.config import get_settings
    from iterate.core import codegen

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "runs"))
    get_settings.cache_clear()
    captured = _stub_run_supervised(monkeypatch, invoke_hook=True)
    try:
        result = runner.invoke(
            app,
            [
                "run",
                "--data",
                str(data),
                "--target",
                "churn",
                "--metric",
                "f1",
                "--code",
                "--memory",
                str(tmp_path / "m.db"),
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.stdout
    run_dir = tmp_path / "runs" / "t"
    started = codegen.build_inputs(captured["dataset"])
    started.update(captured["coder"]._extra_inputs or {})
    assert started  # a table run adds none of its own, so this is build_inputs alone
    for name, sent in started.items():
        assert (run_dir / name).read_bytes() == sent, name
    assert "churn" not in (run_dir / codegen.HOLDOUT_CSV).read_text().splitlines()[0]


def test_a_prompt_winner_gets_the_rows_the_loop_scored_and_no_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt run swaps the dataset for a smaller-holdout one before the loop, so
    the delivered files have to come from that one and not from a second read."""
    import pandas as pd

    from iterate.config import get_settings
    from iterate.core import codegen

    data = tmp_path / "d.csv"
    pd.DataFrame(
        {"text": [f"comment {i}" for i in range(60)], "label": [i % 2 for i in range(60)]}
    ).to_csv(data, index=False)
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "runs"))
    get_settings.cache_clear()
    captured = _stub_run_supervised(monkeypatch, invoke_hook=True)
    try:
        result = runner.invoke(
            app,
            [
                "run",
                "--data",
                str(data),
                "--target",
                "label",
                "--task",
                "say whether the comment is toxic",
                "--loop-holdout",
                "4",
                "--memory",
                str(tmp_path / "m.db"),
                "--plain",
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.stdout
    run_dir = tmp_path / "runs" / "t"
    started = codegen.build_inputs(captured["dataset"])
    for name in (codegen.TRAIN_CSV, codegen.HOLDOUT_CSV):
        assert (run_dir / name).read_bytes() == started[name], name
    held = pd.read_csv(run_dir / codegen.HOLDOUT_CSV)
    assert "label" not in held.columns
    assert len(held) == 4  # the loop's slice, not the full holdout
    meta = json.loads((run_dir / codegen.META_JSON).read_text())
    assert meta["target_model"]
    assert meta["target_backend"]


def test_a_run_that_is_refused_leaves_no_run_folder_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `.gitignore` marker mkdirs the folder, so it has to sit below the last
    refusal: the e2b key check is the one that can fire after the data is loaded."""
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    dot = tmp_path / "dot"
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(dot / "runs"))
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["run", "--data", str(data), "--target", "churn", "--compute", "e2b", "--plain"],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    assert "E2B API key" in result.output
    assert not dot.exists()


def test_research_is_on_by_default_and_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-research must genuinely skip the specialist, so an offline or
    hurry-up run never waits on a literature API."""
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)

    captured = _stub_run_supervised(monkeypatch)
    result = CliRunner().invoke(
        app, ["run", "--data", str(data), "--target", "churn", "--metric", "f1"]
    )
    assert result.exit_code == 0, result.stdout
    assert captured["researcher"] is not None

    captured = _stub_run_supervised(monkeypatch)
    result = CliRunner().invoke(
        app,
        ["run", "--data", str(data), "--target", "churn", "--metric", "f1", "--no-research"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["researcher"] is None


# ─── one input we split, or two the user split ────────────────────────────


def _two_csvs(tmp_path: Path) -> tuple[Path, Path]:
    frame = pd.DataFrame({"f1": range(40), "y": [i % 2 for i in range(40)]})
    train, holdout = tmp_path / "train.csv", tmp_path / "holdout.csv"
    frame.to_csv(train, index=False)
    frame.to_csv(holdout, index=False)
    return train, holdout


def _plain(output: str) -> str:
    """Typer draws the error inside a wrapped box; read it back as one line."""
    stripped = "".join(ch for ch in output if ch not in "│╭╰─╮╯")
    return " ".join(stripped.split())


def test_run_needs_data_or_both_of_train_and_holdout(tmp_path: Path) -> None:
    train, _ = _two_csvs(tmp_path)
    result = runner.invoke(app, ["run", "--target", "y"])
    assert result.exit_code != 0
    assert "both --train and --holdout" in _plain(result.output)
    result = runner.invoke(app, ["run", "--train", str(train), "--target", "y"])
    assert result.exit_code != 0
    assert "both --train and --holdout" in _plain(result.output)


def test_run_refuses_data_together_with_a_users_split(tmp_path: Path) -> None:
    train, holdout = _two_csvs(tmp_path)
    result = runner.invoke(
        app, ["run", "--data", str(train), "--holdout", str(holdout), "--target", "y"]
    )
    assert result.exit_code != 0
    assert "one form, not both" in _plain(result.output)


def test_the_same_file_for_train_and_holdout_is_refused(tmp_path: Path) -> None:
    train, _ = _two_csvs(tmp_path)
    result = runner.invoke(
        app, ["run", "--train", str(train), "--holdout", str(train), "--target", "y"]
    )
    assert result.exit_code != 0
    assert "the same file" in _plain(result.output)


def test_a_users_split_is_refused_on_the_spec_lane(tmp_path: Path) -> None:
    train, holdout = _two_csvs(tmp_path)
    result = runner.invoke(
        app,
        ["run", "--train", str(train), "--holdout", str(holdout), "--target", "y", "--spec"],
    )
    assert result.exit_code != 0
    assert "fast lane" in _plain(result.output)


def test_a_users_split_reaches_the_loop_unsplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, holdout = _two_csvs(tmp_path)
    frame = pd.DataFrame({"f1": range(100, 112), "y": [i % 2 for i in range(12)]})
    frame.to_csv(holdout, index=False)
    captured = _stub_run_supervised(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--train",
            str(train),
            "--holdout",
            str(holdout),
            "--target",
            "y",
            "--metric",
            "f1",
            "--code",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    dataset = captured["dataset"]
    assert dataset.user_split is True
    assert (dataset.n_train, dataset.n_test) == (40, 12)
    assert sorted(dataset.test_features["f1"]) == list(range(100, 112))
    assert "your split: 40 train rows, 12 holdout rows" in _plain(result.output)


def test_an_explicit_metric_names_the_task_for_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "ratings.csv"
    pd.DataFrame({"f1": range(60), "rating": [1 + i % 5 for i in range(60)]}).to_csv(
        data, index=False
    )
    captured = _stub_run_supervised(monkeypatch)
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "rating",
            "--metric",
            "rmse",
            "--code",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["dataset"].task == "regression"
    assert "read as" not in _plain(result.output)  # nothing guessed, nothing announced


# ─── every local kernel is confined where the machine can (Sprint 4 Day 5) ───


def _local_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sandbox: bool, extra: list[str]
) -> tuple[Any, str]:
    captured, output = _captured_local_run(tmp_path, monkeypatch, sandbox=sandbox, extra=extra)
    return captured["kernel"], output


def _captured_local_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sandbox: bool, extra: list[str]
) -> tuple[dict[str, Any], str]:
    from iterate.adapters.compute import confine
    from iterate.config import get_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(confine, "sandbox_available", lambda: sandbox)
    get_settings.cache_clear()
    captured = _stub_run_supervised(monkeypatch)
    try:
        result = runner.invoke(app, ["run", "--compute", "local", "--no-research", *extra])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    return captured, _plain(result.output)


def test_a_local_run_confines_its_kernel_to_what_the_run_gave_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.adapters.compute.kernel import LocalKernel

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory = tmp_path / "elsewhere" / "m.db"
    kernel, output = _local_run(
        tmp_path,
        monkeypatch,
        sandbox=True,
        extra=["--data", "d.csv", "--target", "churn", "--metric", "f1", "--memory", str(memory)],
    )
    assert isinstance(kernel, LocalKernel)
    confinement = kernel._confinement
    assert confinement.weights == tmp_path / "cache" / "iterate" / "weights"
    assert confinement.files == ()
    assert set(confinement.protected) == {
        (tmp_path / ".iterate").resolve(),
        memory.resolve(),
        data.resolve(),
    }
    assert kernel._target_key is None
    assert (
        "cells are confined: they open their own folder, and model weights under "
        "~/cache/iterate/weights, nothing else"
    ) in output


def test_a_local_run_without_a_sandbox_says_its_cells_are_not_confined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tiny_csv(tmp_path / "d.csv")
    _, output = _local_run(
        tmp_path,
        monkeypatch,
        sandbox=False,
        extra=["--data", "d.csv", "--target", "churn", "--metric", "f1"],
    )
    assert "cells are NOT confined on this machine" in output
    assert "cells are confined" not in output


def test_the_weights_folder_is_shown_from_home(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from iterate.cli import _home_tilde

    monkeypatch.setenv("HOME", "/Users/tony")
    assert _home_tilde(Path("/Users/tony/.cache/iterate/weights")) == "~/.cache/iterate/weights"
    assert _home_tilde(Path("/Users/tonyx/weights")) == "/Users/tonyx/weights"


def test_a_prompt_kernel_gets_the_answer_cache_and_the_key_from_the_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = ["text,label"] + [
        f"comment {i},{'toxic' if i % 3 == 0 else 'clean'}" for i in range(30)
    ]
    (tmp_path / "eval.csv").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-only-in-the-project\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ITERATE_TARGET_API_KEY", raising=False)
    kernel, output = _local_run(
        tmp_path,
        monkeypatch,
        sandbox=True,
        extra=[
            "--data",
            "eval.csv",
            "--target",
            "label",
            "--metric",
            "f1",
            "--task",
            "say whether the comment is toxic",
            "--target-backend",
            "groq",
            "--target-model",
            "llama-3.3-70b",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert kernel._target_key == "gsk-only-in-the-project"
    assert kernel._confinement.files == ((tmp_path / ".iterate" / "prompt-answers.db").resolve(),)
    assert "they open their own folder, the answer cache, and model weights" in output


def test_a_table_run_and_a_prompt_run_hand_the_coder_no_network_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`keep_model` is the image family's alone: without it the coder never asks the
    kernel for a network, so these two families run exactly as they did."""
    import tempfile

    from iterate.core import coder as coder_module

    real, built = coder_module.CodingAgent, []
    monkeypatch.setattr(
        coder_module, "CodingAgent", lambda *a, **kw: built.append(kw) or real(*a, **kw)
    )
    monkeypatch.setattr(
        tempfile, "mkdtemp", lambda *a, **kw: pytest.fail("only an image run stages a network")
    )
    _write_tiny_csv(tmp_path / "d.csv")
    lines = ["text,label"] + [
        f"comment {i},{'toxic' if i % 3 == 0 else 'clean'}" for i in range(30)
    ]
    (tmp_path / "eval.csv").write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    table = ["--data", "d.csv", "--target", "churn", "--metric", "f1"]
    prompt = [
        *["--data", "eval.csv", "--target", "label", "--metric", "f1"],
        *["--task", "say whether the comment is toxic"],
        *["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
    ]
    for argv in (table, prompt):
        built.clear()
        extra = [*argv, "--memory", str(tmp_path / "m.db")]
        captured, _ = _captured_local_run(tmp_path, monkeypatch, sandbox=False, extra=extra)
        (kw,) = built
        assert "keep_model" not in kw
        assert captured["coder"]._keep_model is None
    assert built[0]["family"] == "prompt"


# ─── cells never install; the harness installs for local runs with consent ───


def test_a_local_run_with_install_consent_gives_each_session_an_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.adapters.compute import deps

    _write_tiny_csv(tmp_path / "d.csv")
    run = ["--data", "d.csv", "--target", "churn", "--metric", "f1"]
    captured, _ = _captured_local_run(
        tmp_path, monkeypatch, sandbox=False, extra=[*run, "--install"]
    )
    coder = captured["coder"]
    assert coder._install is True
    assert isinstance(coder._installer, deps.Installer)
    assert coder._installer._pending.parent == tmp_path / "cache" / "iterate" / "installs"
    captured, _ = _captured_local_run(tmp_path, monkeypatch, sandbox=False, extra=run)
    assert (captured["coder"]._install, captured["coder"]._installer) == (False, None)


def _saved(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from iterate.adapters.compute import deps

    ran: list[str] = []

    def install_pending(self: Any) -> list[str]:
        ran.append("install_pending")
        return ["installed sktime 1.1.0 saved by an earlier run (pandas 3.0.3 -> 2.3.3)"]

    monkeypatch.setattr(deps.Installer, "pending", lambda self: [{"package": "sktime"}])
    monkeypatch.setattr(deps.Installer, "install_pending", install_pending)
    return ran


def test_saved_installs_run_first_in_a_run_with_install_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tiny_csv(tmp_path / "d.csv")
    ran = _saved(monkeypatch)
    run = ["--data", "d.csv", "--target", "churn", "--metric", "f1", "--install"]
    _, output = _captured_local_run(tmp_path, monkeypatch, sandbox=True, extra=run)
    assert ran == ["install_pending"]
    line = "installs: installed sktime 1.1.0 saved by an earlier run (pandas 3.0.3 -> 2.3.3)"
    assert output.index(line) < output.index("cells are confined")


def test_saved_installs_wait_for_install_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tiny_csv(tmp_path / "d.csv")
    ran = _saved(monkeypatch)
    run = ["--data", "d.csv", "--target", "churn", "--metric", "f1"]
    _, output = _captured_local_run(tmp_path, monkeypatch, sandbox=True, extra=run)
    assert ran == []
    assert "installs: sktime saved by an earlier run, waiting for --install" in output


def test_an_e2b_run_leaves_saved_installs_alone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ran = _saved(monkeypatch)
    cli_module._install_saved_packages(install=True, compute="e2b")
    assert ran == []
    assert "installs" not in capsys.readouterr().out


# ─── the winner leaves with its serving price (Sprint 5 Day 1) ───────────


def test_a_spec_winner_gets_a_serving_block_in_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """best.json carries the winner's serving profile, and the summary prints it."""
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    out = tmp_path / "models" / "best_model.joblib"

    _stub_run_orchestrator(monkeypatch, best_model="sklearn.linear_model.LogisticRegression")
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(tmp_path / "memory.db"),
            "--output",
            str(out),
            "--requests-per-hour",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.stdout
    serving = json.loads(out.with_name("best.json").read_text())["serving"]
    assert serving["requests_per_hour"] == 5000
    assert serving["chosen"]["host"]["kind"] == "cpu"
    assert serving["usd_per_1k_requests"] > 0
    assert serving["basis"][0].startswith("linear pipeline (LogisticRegression)")
    assert "serving: about $" in _plain(result.output)
    assert "prices: aws shipped" in _plain(result.output)


def test_requests_per_hour_must_be_at_least_one(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--requests-per-hour",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "requests-per-hour" in (result.stderr or result.stdout)


# ─── live prices: the flags and the two commands (Sprint 5 Day 2) ─────────


def test_region_needs_a_cloud_and_the_cloud_must_be_one_of_three(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    base = ["run", "--data", str(data), "--target", "churn", "--metric", "f1"]

    result = runner.invoke(app, [*base, "--region", "us-east-1"])
    assert result.exit_code != 0
    assert "--region needs --cloud" in (result.stderr or result.stdout)

    result = runner.invoke(app, [*base, "--cloud", "ibm"])
    assert result.exit_code != 0
    assert "--cloud must be aws | gcp | azure" in (result.stderr or result.stdout)


def test_a_spec_winner_priced_on_one_cloud_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    out = tmp_path / "models" / "best_model.joblib"

    _stub_run_orchestrator(monkeypatch, best_model="sklearn.linear_model.LogisticRegression")
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "churn",
            "--metric",
            "f1",
            "--spec",
            "--memory",
            str(tmp_path / "memory.db"),
            "--output",
            str(out),
            "--cloud",
            "azure",
            "--region",
            "eastus",
        ],
    )
    assert result.exit_code == 0, result.stdout
    serving = json.loads(out.with_name("best.json").read_text())["serving"]
    assert serving["chosen"]["host"]["cloud"] == "azure"
    assert [cost["host"]["cloud"] for cost in serving["by_cloud"]] == ["azure"]
    assert "prices: azure shipped" in _plain(result.output)


def test_prices_show_lists_every_cloud_and_the_timm_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = runner.invoke(app, ["prices", "show"])

    assert result.exit_code == 0, result.output
    assert "aws us-east-1: shipped rows" in result.output
    assert "gcp us-central1: shipped rows" in result.output
    assert "azure eastus: shipped rows" in result.output
    assert "timm:" in result.output


def test_prices_refresh_for_gcp_touches_no_network_and_says_the_shipped_rows_stand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.core import prices as prices_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    fetched: list[str] = []
    monkeypatch.setattr(prices_mod, "refresh_timm", lambda **kw: fetched.append("timm"))

    result = runner.invoke(app, ["prices", "refresh", "--cloud", "gcp"])

    assert result.exit_code == 0, result.output
    assert "gcp: shipped rows stand" in result.output
    assert fetched == ["timm"]


def test_prices_refresh_refuses_a_region_without_a_cloud() -> None:
    result = runner.invoke(app, ["prices", "refresh", "--region", "eastus"])

    assert result.exit_code != 0
    assert "--region needs --cloud" in (result.stderr or result.stdout)
