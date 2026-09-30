"""Tests for the `iterate run` CLI command + its helpers.

Heavy on mocking — we want to verify *wiring*, not exercise the real LLM/data
path (the live integration test for the loop comes at Day 6).
"""

from __future__ import annotations

import json
import re
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
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

# Error panels wrap at the terminal width, which can split a phrase a test looks for.
pytestmark = pytest.mark.unit

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
        "iterate_backend_api_key",
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
    monkeypatch: pytest.MonkeyPatch,
    *,
    invoke_hook: bool = False,
    stopped_because: str = "max_iterations",
    refusals: tuple[tuple[str, float, str], ...] = (),
) -> dict[str, Any]:
    """Stub the code path's heavy bits; capture which client each agent received.

    With ``invoke_hook``, the fake loop calls ``on_experiment`` once with a finished
    cells-experiment and returns an EMPTY history — so any notebook on disk can only
    have come from the incremental hook, never the end-of-run writer. ``refusals`` are
    recorded on the wall the CLI handed the loop, as the Supervisor would."""
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
        # The first client a run builds is the harness's.
        captured.setdefault("harness", {"backend": name, **kw})
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
        captured["wall"] = kwargs.get("wall")
        for what, usd, kind in refusals:
            captured["wall"].record_refusal(what, usd, kind)
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
            stopped_because=stopped_because,
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


# ─── two sets of model settings: the harness and the model under test (v0.7 Day 5) ───

_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "ITERATE_BACKEND_API_KEY",
    "ITERATE_TARGET_API_KEY",
)


def _prompt_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    *,
    env: dict[str, str] | None = None,
    saved: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """A prompt run with no provider key in the environment but the ones given, in a
    folder of its own. Returns the CLI result and what the prompt target was built
    with; the run is stubbed, so nothing is called."""
    from iterate import userconfig
    from iterate.config import get_settings

    lines = ["text,label"] + [
        f"comment {i},{'toxic' if i % 3 == 0 else 'clean'}" for i in range(30)
    ]
    (tmp_path / "eval.csv").write_text("\n".join(lines), encoding="utf-8")
    # Each call starts from a folder no run has touched, with a memory from a run before
    # and --fresh: a refusal below the archive would move it.
    import shutil

    shutil.rmtree(tmp_path / ".iterate", ignore_errors=True)
    for left in tmp_path.glob("m*"):
        left.unlink()
    (tmp_path / "m.db").write_bytes(_THE_RUN_BEFORE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in _PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if saved is not None:
        userconfig.save_user_config(
            saved.get("harness", {"backend": "ollama"}),
            allowed=saved.get("allowed"),
            providers=saved.get("providers", {}),
        )
    built: dict[str, Any] = {}
    real = cli_module._build_prompt_target

    def spy(dataset: Any, **kwargs: Any) -> Any:
        built.update(kwargs)
        return real(dataset, **kwargs)

    monkeypatch.setattr(cli_module, "_build_prompt_target", spy)
    captured = _stub_run_supervised(monkeypatch)
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            [
                "run",
                *["--data", "eval.csv", "--target", "label", "--metric", "f1"],
                *["--task", "say whether the comment is toxic"],
                *["--no-research", "--plain", "--fresh", "--memory", str(tmp_path / "m.db")],
                *extra,
            ],
        )
    finally:
        get_settings.cache_clear()
    built["kernel"] = captured.get("kernel")
    built["harness"] = captured.get("harness")
    return result, built


_THE_RUN_BEFORE = b"the run before"


def _refused(result: Any, tmp_path: Path) -> str:
    """The refusal's text, from a run that wrote nothing and moved nothing."""
    assert result.exit_code != 0, result.output
    assert not (tmp_path / ".iterate").exists()
    assert sorted(p.name for p in tmp_path.glob("m*")) == ["m.db"]
    assert (tmp_path / "m.db").read_bytes() == _THE_RUN_BEFORE
    return " ".join(_plain(result.output).split())


def test_the_model_under_test_never_takes_the_harness_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--backend", "ollama", "--base-url", "http://gpu-box:11434"],
            *["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        ],
        env={"GROQ_API_KEY": "gsk-env"},
    )
    assert result.exit_code == 0, result.output
    assert built["base_url"] == "https://api.groq.com/openai/v1"
    assert (built["api_key"], built["kernel"]._target_key) == ("gsk-env", "gsk-env")
    assert (
        "model under test: groq llama-3.3-70b at https://api.groq.com/openai/v1, "
        "key from GROQ_API_KEY" in " ".join(_plain(result.output).split())
    )


def test_the_prompt_runs_on_ollama_whatever_runs_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named with --target-backend or not at all: a harness on a server of yours keeps
    its address and key, and the model under test is Ollama's."""
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--backend", "openai-compatible", "--base-url", "http://gpu-box:8000/v1"],
            *["--api-key", "box-key", "--model", "my-llama", "--target-model", "gemma4:12b"],
        ],
    )
    assert result.exit_code == 0, result.output
    assert (built["backend"], built["model"]) == ("ollama", "gemma4:12b")
    assert built["kernel"]._target_key is None
    harness = built["harness"]
    assert (harness["backend"], harness["model"]) == ("openai-compatible", "my-llama")
    assert (harness["base_url"], harness["api_key"]) == ("http://gpu-box:8000/v1", "box-key")
    assert "model under test: ollama gemma4:12b" in " ".join(_plain(result.output).split())


def test_a_harness_not_on_ollama_names_the_ollama_model_the_prompt_is_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path, monkeypatch, ["--backend", "openai", "--api-key", "k", "--model", "gpt-4o-mini"]
    )
    text = _refused(result, tmp_path)
    assert (
        "the model under test runs on Ollama unless --target-backend names another provider, "
        "and the harness's model on openai means nothing to Ollama"
    ) in text


def test_a_harness_found_through_the_environment_keeps_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ITERATE_BACKEND_URL` is how `.env.example` points the harness at a server."""
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--backend", "openai-compatible", "--model", "my-llama", "--target-model", "g"],
        env={
            "ITERATE_BACKEND_URL": "http://gpu-box:8000/v1",
            "ITERATE_BACKEND_API_KEY": "box-key",
        },
    )
    assert result.exit_code == 0, result.output
    assert built["harness"]["api_key"] == "box-key"
    assert built["backend"] == "ollama"


def test_the_model_under_test_has_an_address_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "vllm", "--target-model", "my-llama"],
            *["--target-base-url", "http://gpu-box:8000/v1"],
        ],
    )
    assert result.exit_code == 0, result.output
    # The host is told there is no key, so it never looks for one in the environment.
    assert (built["base_url"], built["api_key"]) == ("http://gpu-box:8000/v1", "not-needed")
    assert built["kernel"]._target_key is None
    assert "at http://gpu-box:8000/v1, no key sent" in " ".join(_plain(result.output).split())


def test_an_ollama_model_under_test_is_handed_its_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settled on the host and written into meta.json, so the cell calls the host the
    baseline called whatever its own environment says."""
    result, built = _prompt_run(
        tmp_path, monkeypatch, ["--model", "gemma4:12b"], env={"OLLAMA_HOST": "http://box:11434"}
    )
    assert result.exit_code == 0, result.output
    assert (built["backend"], built["base_url"]) == ("ollama", "http://box:11434")


def test_a_key_saved_for_a_provider_reaches_the_model_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "openai", "--target-model", "gpt-4o-mini"],
        saved={
            "allowed": ["openai"],
            "providers": {"openai": SavedProvider("openai", api_key="sk-saved-for-prompts")},
        },
    )
    assert result.exit_code == 0, result.output
    assert built["kernel"]._target_key == "sk-saved-for-prompts"
    text = " ".join(_plain(result.output).split())
    assert "key from the saved config; prompt providers allowed: openai" in text
    assert "sk-saved-for-prompts" not in text


def test_a_provider_outside_the_list_is_refused_before_its_key_is_looked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.llm import factory
    from iterate.userconfig import SavedProvider

    looked_up: list[str] = []
    real = factory.own_key_for
    monkeypatch.setattr(
        factory,
        "own_key_for",
        lambda name, *a, **kw: looked_up.append(name) or real(name, *a, **kw),
    )
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        env={"GROQ_API_KEY": "gsk-env"},
        saved={
            "allowed": ["openai"],
            "providers": {"openai": SavedProvider("openai", api_key="sk-saved")},
        },
    )
    text = _refused(result, tmp_path)
    assert "served by groq, which is not among the prompt providers you allow (openai)" in text
    assert "Pass --providers groq for this run" in text
    assert "groq" not in looked_up


def test_a_runs_own_list_replaces_the_saved_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    saved = {
        "allowed": ["openai"],
        "providers": {"openai": SavedProvider("openai", api_key="sk-saved")},
    }
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b", "--providers", "groq"],
        env={"GROQ_API_KEY": "gsk-env"},
        saved=saved,
    )
    assert result.exit_code == 0, result.output
    assert built["api_key"] == "gsk-env"

    # It replaces the list, it does not add to it: what was saved is now outside.
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "openai", "--target-model", "gpt-4o-mini", "--providers", "groq"],
        env={"GROQ_API_KEY": "gsk-env"},
        saved=saved,
    )
    text = _refused(result, tmp_path)
    assert "served by openai, which is not among the prompt providers you allow (groq)" in text


def test_every_allowed_provider_needs_its_key_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
            *["--providers", "groq,openai,vllm"],
        ],
        env={"GROQ_API_KEY": "gsk-env", "ITERATE_BACKEND_API_KEY": "the-harness-key"},
    )
    assert _refused(result, tmp_path).endswith(
        "these allowed prompt providers are not ready: openai (no key, set OPENAI_API_KEY), "
        "vllm (no base URL saved). Save them with `iterate setup`, or pass --providers groq "
        "to allow only the model under test"
    )


def test_the_model_under_test_with_no_key_is_refused_and_never_takes_the_harness_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        env={"ITERATE_BACKEND_API_KEY": "the-harness-key", "OPENAI_API_KEY": "sk-other"},
    )
    assert "groq has no key" in _refused(result, tmp_path)


def test_the_exported_target_key_is_the_model_under_tests_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
            *["--providers", "groq,openai"],
        ],
        env={"ITERATE_TARGET_API_KEY": "gsk-for-this-run"},
    )
    text = _refused(result, tmp_path)
    assert "are not ready: openai (no key, set OPENAI_API_KEY)." in text
    assert "groq (" not in text


def test_a_key_the_provider_refuses_stops_the_run_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openai import AuthenticationError

    def refuses(provider: Any, timeout: float, model: str | None) -> None:
        response = type("R", (), {"request": None, "status_code": 401, "headers": {}})()
        raise AuthenticationError("bad key", response=response, body=None)

    monkeypatch.setattr("iterate.llm.factory._list_models", refuses)
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        env={"GROQ_API_KEY": "gsk-wrong"},
    )
    text = _refused(result, tmp_path)
    assert "api.groq.com refused the groq key from GROQ_API_KEY" in text
    assert "gsk-wrong" not in text


def test_a_prompt_run_on_e2b_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path, monkeypatch, ["--compute", "e2b"], env={"E2B_API_KEY": "e2b-key"}
    )
    text = _refused(result, tmp_path)
    assert "a prompt run cannot use --compute e2b" in text
    assert "Pass --compute local" in text


def test_a_model_under_test_named_apart_needs_its_own_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--model", "gemma4:12b", "--target-backend", "groq"],
        env={"GROQ_API_KEY": "gsk-env"},
    )
    assert "--target-backend groq needs --target-model" in _refused(result, tmp_path)


@pytest.mark.parametrize(
    ("extra", "said"),
    [
        (
            ["--target-backend", "grok", "--target-model", "m"],
            "grok: not a provider iterate knows. Choose from ollama, anthropic, deepseek, groq",
        ),
        (["--providers", "openia"], "openia: not a provider iterate knows. Choose from"),
        (
            ["--target-backend", "vllm", "--target-model", "m"],
            "vllm is a server you run, so it has no address of its own",
        ),
        (
            [
                *["--target-backend", "vllm", "--target-model", "m"],
                *["--target-base-url", "http://me:pw@gpu:8000/v1"],
            ],
            "carries a user name, a password or a token",
        ),
        (
            [
                *["--target-backend", "vllm", "--target-model", "m"],
                *["--target-base-url", "https://gpu.test/v1?api_key=pw@in-the-query"],
            ],
            "carries a user name, a password or a token",
        ),
        (
            [
                *["--target-backend", "vllm", "--target-model", "m"],
                *["--target-base-url", "https://gpu.test:abc/v1"],
            ],
            "cannot be read: check its port",
        ),
        (
            [
                *["--target-backend", "vllm", "--target-model", "m"],
                *["--target-base-url", "gpu.test/v1"],
            ],
            "has to start with http:// or https://",
        ),
    ],
)
def test_a_model_under_test_that_cannot_be_called_is_refused_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], said: str
) -> None:
    result, _ = _prompt_run(tmp_path, monkeypatch, extra)
    text = _refused(result, tmp_path)
    assert said in text
    assert "pw@" not in text


def test_a_companys_address_with_a_port_on_it_is_still_that_company(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named groq and aimed at OpenAI's host: it is openai, which the list leaves out,
    and the Groq key stays where it is."""
    from iterate.userconfig import SavedProvider

    for address in ("https://api.openai.com:443/v1", "https://API.openai.com./v1"):
        result, _ = _prompt_run(
            tmp_path,
            monkeypatch,
            [
                *["--target-backend", "openai-compatible", "--target-model", "gpt-4o-mini"],
                *["--target-base-url", address],
            ],
            saved={
                "allowed": ["groq"],
                "providers": {"groq": SavedProvider("groq", api_key="gsk-saved-for-groq")},
            },
        )
        text = _refused(result, tmp_path)
        assert "served by openai, which is not among the prompt providers you allow" in text


def test_a_company_reached_in_the_clear_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "openai", "--target-model", "gpt-4o-mini"],
            *["--target-base-url", "http://api.openai.com/v1"],
        ],
        env={"OPENAI_API_KEY": "sk-env"},
    )
    assert "openai takes its key over https only" in _refused(result, tmp_path)


def test_a_server_saved_under_its_own_name_at_a_companys_address_runs_under_either_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    saved = {
        "allowed": ["openai-compatible"],
        "providers": {
            "openai-compatible": SavedProvider(
                "openai-compatible", api_key="sk-saved", base_url="https://api.openai.com/v1"
            )
        },
    }
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "openai-compatible", "--target-model", "gpt-4o-mini"],
        saved=saved,
    )
    assert result.exit_code == 0, result.output
    assert (built["base_url"], built["api_key"]) == ("https://api.openai.com/v1", "sk-saved")
    assert "model under test: openai gpt-4o-mini" in " ".join(_plain(result.output).split())


def test_a_key_saved_for_a_server_you_run_goes_to_that_server_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    saved = {
        "allowed": ["vllm"],
        "providers": {
            "vllm": SavedProvider("vllm", api_key="key-of-server-a", base_url="http://a.test/v1")
        },
    }
    target = ["--target-backend", "vllm", "--target-model", "m"]
    result, built = _prompt_run(tmp_path, monkeypatch, target, saved=saved)
    assert result.exit_code == 0, result.output
    assert (built["base_url"], built["api_key"]) == ("http://a.test/v1", "key-of-server-a")

    result, built = _prompt_run(
        tmp_path, monkeypatch, [*target, "--target-base-url", "http://b.test/v1"], saved=saved
    )
    assert result.exit_code == 0, result.output
    assert (built["base_url"], built["api_key"]) == ("http://b.test/v1", "not-needed")


def test_with_no_list_saved_the_key_saved_for_the_provider_named_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        saved={
            "allowed": None,
            "providers": {
                "groq": SavedProvider("groq", api_key="gsk-saved"),
                "openai": SavedProvider("openai", api_key="sk-never-read"),
            },
        },
    )
    assert result.exit_code == 0, result.output
    assert built["api_key"] == "gsk-saved"
    assert "prompt providers allowed" not in _plain(result.output)


def test_with_a_list_saved_the_prompts_default_ollama_has_to_be_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness saved on groq keeps its key; the model under test is Ollama unless a
    provider is named, and the list says where records may go."""
    from iterate.userconfig import SavedProvider

    saved = {
        "harness": {"backend": "groq", "api_key": "gsk-saved-harness"},
        "allowed": ["groq"],
        "providers": {"groq": SavedProvider("groq", base_url="https://gateway.test/v1")},
    }
    result, _ = _prompt_run(
        tmp_path, monkeypatch, ["--model", "llama-3.3-70b", "--target-model", "g"], saved=saved
    )
    assert "served by ollama, which is not among the prompt providers you allow (groq)" in (
        _refused(result, tmp_path)
    )

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--model", "llama-3.3-70b", "--target-model", "g", "--providers", "ollama"],
        saved=saved,
    )
    assert result.exit_code == 0, result.output
    assert built["harness"]["api_key"] == "gsk-saved-harness"
    assert built["backend"] == "ollama"


def test_a_harness_aimed_at_a_company_takes_that_companys_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`openai-compatible` names a wire, not a company: aimed at Groq it reads Groq's
    key, and OpenAI's stays home."""
    harness = ["--backend", "openai-compatible", "--model", "llama-3.3-70b"]
    harness += ["--base-url", "https://api.groq.com/openai/v1"]
    result, _ = _prompt_run(tmp_path, monkeypatch, harness, env={"OPENAI_API_KEY": "sk-openai"})
    assert "requires --api-key" in _refused(result, tmp_path)

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [*harness, "--target-model", "g"],
        env={"OPENAI_API_KEY": "sk-openai", "GROQ_API_KEY": "gsk-env"},
    )
    assert result.exit_code == 0, result.output
    assert built["harness"]["api_key"] == "gsk-env"


def test_every_allowed_provider_is_settled_with_its_key_in_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    settled: list[Any] = []
    real = cli_module._model_under_test
    monkeypatch.setattr(
        cli_module, "_model_under_test", lambda **kw: settled.append(real(**kw)) or settled[-1]
    )
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "llama-3.3-70b"],
        env={"GROQ_API_KEY": "gsk-env", "ITERATE_TARGET_API_KEY": "gsk-this-run"},
        saved={
            "allowed": ["openai", "groq", "ollama"],
            "providers": {"openai": SavedProvider("openai", api_key="sk-saved")},
        },
    )
    assert result.exit_code == 0, result.output
    (found,) = settled
    assert (found.backend, found.model, found.listed) == (
        "groq",
        "llama-3.3-70b",
        ("openai", "groq", "ollama"),
    )
    assert {p.name: p.api_key for p in found.allowed} == {
        "groq": "gsk-this-run",
        "openai": "sk-saved",
        "ollama": None,
    }


def test_the_help_names_every_provider_iterate_knows() -> None:
    from iterate.llm import factory

    text = " ".join(_plain(runner.invoke(app, ["run", "--help"]).output).split())
    for name in factory.known_providers():
        assert name in text


def test_a_model_ollama_does_not_have_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness on a cloud with --target-model and no --target-backend asks Ollama:
    one that is not running, or has not pulled the model, stops the run up front."""
    monkeypatch.setattr("iterate.llm.factory._ollama_models", lambda host, timeout: ["g:1b"])
    harness = ["--backend", "openai", "--api-key", "k", "--model", "gpt-4o"]
    result, _ = _prompt_run(tmp_path, monkeypatch, [*harness, "--target-model", "gpt-4o-mini"])
    assert "Ollama at http://localhost:11434 has no model gpt-4o-mini" in _refused(result, tmp_path)

    def down(host: str, timeout: float) -> list[str]:
        raise ConnectionError("refused")

    monkeypatch.setattr("iterate.llm.factory._ollama_models", down)
    result, _ = _prompt_run(tmp_path, monkeypatch, [*harness, "--target-model", "g:1b"])
    assert "no Ollama server answers at http://localhost:11434" in _refused(result, tmp_path)


def test_claude_allowed_beside_the_model_under_test_is_installed_with_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from iterate.adapters.compute import deps
    from iterate.userconfig import SavedProvider

    real = pytest.importorskip("anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    consents: list[bool] = []

    def installs(*, consent: bool, **kwargs: Any) -> str:
        consents.append(consent)
        if consent:
            sys.modules["anthropic"] = real
            return ""
        return "the anthropic package is not installed: pip install 'iterate-ai[anthropic]' (or pass --install)"

    monkeypatch.setattr(deps, "ensure_anthropic", installs)
    saved = {
        "allowed": ["ollama", "anthropic"],
        "providers": {"anthropic": SavedProvider("anthropic", api_key="sk-ant-saved")},
    }
    result, _ = _prompt_run(tmp_path, monkeypatch, ["--model", "g"], saved=saved)
    assert "(or pass --install)" in _refused(result, tmp_path)
    result, built = _prompt_run(tmp_path, monkeypatch, ["--model", "g", "--install"], saved=saved)
    assert consents == [False, True]
    assert result.exit_code == 0, result.output
    assert built["backend"] == "ollama"


def test_an_install_this_python_still_cannot_import_is_not_called_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from iterate.adapters.compute import deps

    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setattr(deps, "ensure_anthropic", lambda **kwargs: "")
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5", "--install"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    text = _refused(result, tmp_path)
    assert "pip reported the install done, but this Python cannot import anthropic" in text
    assert "Anthropic's library is in" not in text


def test_the_harness_key_never_goes_to_the_model_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even on the harness's own company at its own address: the model under test is
    named apart, and takes the key of its provider or none."""
    harness = ["--backend", "openai", "--api-key", "sk-the-harness-key", "--model", "gpt-4o-mini"]
    target = ["--target-backend", "openai", "--target-model", "gpt-4o-mini"]
    for address in ("https://gateway.test/v1", "https://api.openai.com/v1/"):
        result, _ = _prompt_run(
            tmp_path, monkeypatch, [*harness, *target, "--target-base-url", address]
        )
        text = _refused(result, tmp_path)
        assert "openai has no key" in text
        assert "sk-the-harness-key" not in text


def test_a_provider_the_run_does_not_call_is_not_mended_by_a_flag_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path, monkeypatch, ["--model", "gemma4:12b", "--providers", "ollama,vllm"]
    )
    text = _refused(result, tmp_path)
    assert "not ready: vllm (no base URL saved)" in text
    assert "--target-base-url" not in text
    assert "pass --providers ollama" in text


def test_named_apart_on_the_harnesss_own_backend_it_still_takes_nothing_of_the_harnesss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model name is the backend's and carries over. The key and the address are the
    harness's and do not: by the flag, or saved."""
    from iterate.userconfig import SavedProvider

    named = ["--backend", "groq", "--model", "llama-3.3-70b", "--target-backend", "groq"]
    result, _ = _prompt_run(tmp_path, monkeypatch, [*named, "--api-key", "gsk-the-harness-key"])
    assert "groq has no key" in _refused(result, tmp_path)

    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--model", "llama-3.3-70b", "--target-backend", "groq"],
        saved={"harness": {"backend": "groq", "api_key": "gsk-saved-harness-key"}},
    )
    assert "groq has no key" in _refused(result, tmp_path)

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [*named, "--api-key", "gsk-the-harness-key"],
        saved={"allowed": None, "providers": {"groq": SavedProvider("groq", api_key="gsk-own")}},
    )
    assert result.exit_code == 0, result.output
    assert (built["model"], built["api_key"]) == ("llama-3.3-70b", "gsk-own")


def test_named_apart_with_no_model_name_anywhere_it_takes_the_default_the_harness_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--backend", "groq", "--target-backend", "groq"],
        env={"GROQ_API_KEY": "gsk-env", "ITERATE_MODEL": "model-from-the-environment"},
    )
    assert result.exit_code == 0, result.output
    assert built["model"] == "model-from-the-environment"


def test_named_apart_it_does_not_follow_the_harness_to_its_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before v0.7 it did, so the run that relied on it is told what to pass, not moved
    to another server without a word."""
    flags = ["--backend", "ollama", "--base-url", "http://gpu-box:11434", "--model", "gemma4:12b"]
    result, _ = _prompt_run(tmp_path, monkeypatch, [*flags, "--target-backend", "ollama"])
    text = _refused(result, tmp_path)
    assert "Pass --target-base-url http://gpu-box:11434, or drop --target-backend" in text

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [*flags, "--target-backend", "ollama", "--target-base-url", "http://gpu-box:11434"],
    )
    assert result.exit_code == 0, result.output
    assert built["base_url"] == "http://gpu-box:11434"


def test_a_saved_list_that_cannot_be_read_refuses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read as no list, it would let the run call a provider the user ruled out."""
    from iterate import userconfig

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_text("[prompt]\nallowed = 5\n", encoding="utf-8")
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "m"],
        env={"GROQ_API_KEY": "gsk-env"},
    )
    assert "has to be a list of names" in _refused(result, tmp_path)


def test_a_saved_file_that_is_not_toml_refuses_any_run_by_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate import userconfig

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    monkeypatch.chdir(tmp_path)
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('backend = "groq\n', encoding="utf-8")
    result = runner.invoke(
        app, ["run", "--data", str(data), "--target", "churn", "--metric", "f1", "--plain"]
    )
    assert result.exit_code == 2
    assert "is not valid TOML" in " ".join(_plain(result.output).split())
    assert not (tmp_path / ".iterate").exists()


def test_a_saved_key_with_no_backend_beside_it_is_the_runs_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--backend", "groq", "--target-model", "g"],
        saved={"harness": {"model": "saved-model", "api_key": "gsk-saved"}},
    )
    assert result.exit_code == 0, result.output
    assert (built["harness"]["model"], built["harness"]["api_key"]) == ("saved-model", "gsk-saved")


def test_a_name_on_the_list_does_not_cover_a_company_it_is_aimed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list names where records may go. `vllm` on it is the server saved as vllm,
    not OpenAI reached through the name."""
    from iterate.llm import factory
    from iterate.userconfig import SavedProvider

    looked_up: list[str] = []
    real = factory.own_key_for
    monkeypatch.setattr(
        factory,
        "own_key_for",
        lambda name, *a, **kw: looked_up.append(name) or real(name, *a, **kw),
    )
    saved = {
        "allowed": ["vllm"],
        "providers": {
            "vllm": SavedProvider("vllm", api_key="key-of-server-a", base_url="http://a.test/v1")
        },
    }
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "vllm", "--target-model", "gpt-4o-mini"],
            *["--target-base-url", "https://api.openai.com/v1"],
        ],
        env={"OPENAI_API_KEY": "sk-env"},
        saved=saved,
    )
    text = _refused(result, tmp_path)
    assert "served by openai, which is not among the prompt providers you allow (vllm)" in text
    assert "Pass --providers openai for this run" in text
    assert "openai" not in looked_up


def test_aimed_at_a_company_by_a_flag_the_saved_key_of_your_server_is_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    saved = {
        "allowed": None,
        "providers": {
            "vllm": SavedProvider("vllm", api_key="key-of-server-a", base_url="http://a.test/v1")
        },
    }
    flags = ["--target-backend", "vllm", "--target-model", "gpt-4o-mini"]
    flags += ["--target-base-url", "https://api.openai.com/v1"]
    result, _ = _prompt_run(tmp_path, monkeypatch, flags, saved=saved)
    assert "openai has no key" in _refused(result, tmp_path)

    result, built = _prompt_run(
        tmp_path, monkeypatch, flags, env={"OPENAI_API_KEY": "sk-env"}, saved=saved
    )
    assert result.exit_code == 0, result.output
    assert (built["api_key"], built["kernel"]._target_key) == ("sk-env", "sk-env")


def test_an_ollama_model_under_test_aimed_at_a_company_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "ollama", "--target-model", "m"],
            *["--target-base-url", "https://api.openai.com/v1"],
        ],
        env={"OPENAI_API_KEY": "sk-env"},
    )
    text = _refused(result, tmp_path)
    assert "is openai's address, not an Ollama server. Pass --target-backend openai" in text


def test_claude_is_the_model_under_test_with_its_own_key_at_its_own_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("anthropic")
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env", "ITERATE_BACKEND_API_KEY": "harness-key"},
    )
    assert result.exit_code == 0, result.output
    assert built["base_url"] == "https://api.anthropic.com"
    assert (built["api_key"], built["kernel"]._target_key) == ("sk-ant-env", "sk-ant-env")
    assert (
        "model under test: anthropic claude-haiku-4-5 at https://api.anthropic.com, "
        "key from ANTHROPIC_API_KEY" in " ".join(_plain(result.output).split())
    )


@pytest.mark.parametrize("backend", ["anthropic", "mistral"])
def test_a_harness_that_cannot_run_the_loop_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--backend", backend, "--model", "m"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    text = _refused(result, tmp_path)
    said = {
        "anthropic": "anthropic runs as the model under test of a prompt run "
        "(--target-backend anthropic); as the harness that runs the loop it comes in v1.0",
        "mistral": "--backend mistral: not a backend iterate knows. Choose from ollama,",
    }
    assert said[backend] in text


@pytest.mark.parametrize(
    ("model", "said"),
    [
        ("claude-opus-5-5", "thinks before it answers"),
        ("claude-sonnet-5-5", "thinks before it answers"),
        ("claude-fable-5-1", "thinks before it answers"),
        ("claude-opus-4-8", "is not a Claude model iterate knows how to ask yet"),
    ],
)
def test_a_claude_model_a_prompt_run_cannot_ask_is_refused_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, said: str
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", model],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    text = _refused(result, tmp_path)
    assert f"{model} {said}" in text
    assert "Pick claude-haiku-4-5, claude-sonnet-5 or claude-opus-5" in text


def test_a_claude_model_the_key_is_not_served_is_refused_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str | None] = []

    def serves(provider: Any, timeout: float, model: str | None) -> list[str]:
        asked.append(model)
        return []

    monkeypatch.setattr("iterate.llm.factory._claude_serves", serves)
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5-20990101"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    text = _refused(result, tmp_path)
    assert asked == ["claude-haiku-4-5-20990101"]
    assert "anthropic does not serve claude-haiku-4-5-20990101 to this key" in text


def test_claude_with_install_consent_gets_its_library_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As an image run gets torch: with --install the library is installed, not asked
    for."""
    import sys

    from iterate.adapters.compute import deps

    real = pytest.importorskip("anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    consents: list[bool] = []

    def installs(*, consent: bool, **kwargs: Any) -> str:
        consents.append(consent)
        sys.modules["anthropic"] = real
        return ""

    monkeypatch.setattr(deps, "ensure_anthropic", installs)
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5", "--install"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    assert consents == [True]
    assert result.exit_code == 0, result.output
    assert built["backend"] == "anthropic"
    assert "installs: Anthropic's library is in" in _plain(result.output)


def test_claude_without_its_library_is_refused_with_the_command_that_adds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5"],
        env={"ANTHROPIC_API_KEY": "sk-ant-env"},
    )
    text = _refused(result, tmp_path)
    assert (
        "the anthropic package is not installed: pip install 'iterate-ai[anthropic]' "
        "(or pass --install)"
    ) in text


def test_an_ollama_harness_aimed_at_a_company_is_told_both_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no --target-backend the model under test is the Ollama harness's own, at its
    address: an address that is a company's needs the company named, and a model."""
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--backend", "ollama", "--model", "gpt-4o-mini"],
            *["--base-url", "https://api.openai.com/v1"],
        ],
        env={"OPENAI_API_KEY": "sk-env"},
    )
    text = _refused(result, tmp_path)
    assert "Pass --target-backend openai --target-model gpt-4o-mini" in text


def test_a_saved_backend_that_cannot_run_the_loop_is_named_as_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _prompt_run(tmp_path, monkeypatch, [], saved={"harness": {"backend": "grok"}})
    text = _refused(result, tmp_path)
    assert "the saved backend grok (change it with `iterate setup`): not a backend" in text


def test_a_server_saved_at_claudes_address_needs_no_claude_library_for_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is called with the OpenAI client its name says, so a groq run it is allowed
    beside goes on as before Day 6."""
    import sys

    from iterate.userconfig import SavedProvider

    monkeypatch.setitem(sys.modules, "anthropic", None)
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", "groq", "--target-model", "qwen/qwen3.8-27b"],
        env={"GROQ_API_KEY": "gsk-env"},
        saved={
            "allowed": ["groq", "openai-compatible"],
            "providers": {
                "openai-compatible": SavedProvider(
                    "openai-compatible", api_key="k", base_url="https://api.anthropic.com/v1"
                )
            },
        },
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("extra", "said"),
    [
        (
            [
                *["--target-backend", "openai-compatible", "--target-model", "m"],
                *["--target-base-url", "https://api.anthropic.com/v1"],
            ],
            "is anthropic's address, and iterate calls anthropic with its own client, not "
            "openai-compatible's. Pass --target-backend anthropic",
        ),
        (
            [
                *["--target-backend", "anthropic", "--target-model", "claude-haiku-4-5"],
                *["--target-base-url", "https://api.groq.com/openai/v1"],
            ],
            "is groq's address, and iterate calls groq with its own client, not "
            "anthropic's. Pass --target-backend groq",
        ),
    ],
)
def test_a_client_aimed_at_a_company_it_cannot_speak_to_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], said: str
) -> None:
    result, _ = _prompt_run(
        tmp_path,
        monkeypatch,
        extra,
        env={"ANTHROPIC_API_KEY": "sk-ant-env", "GROQ_API_KEY": "gsk-env"},
    )
    assert said in _refused(result, tmp_path)


def test_a_key_saved_over_https_is_not_sent_over_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    saved = {
        "allowed": ["vllm"],
        "providers": {
            "vllm": SavedProvider("vllm", api_key="key-a", base_url="https://llm.corp.test/v1")
        },
    }
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--target-backend", "vllm", "--target-model", "m"],
            *["--target-base-url", "http://llm.corp.test/v1"],
        ],
        saved=saved,
    )
    assert result.exit_code == 0, result.output
    assert built["api_key"] == "not-needed"

    harness = ["--backend", "vllm", "--base-url", "https://llm.corp.test/v1", "--model", "x"]
    harness += ["--api-key", "the-harness-key", "--target-backend", "vllm"]
    result, built = _prompt_run(
        tmp_path, monkeypatch, [*harness, "--target-base-url", "http://llm.corp.test/v1"]
    )
    assert result.exit_code == 0, result.output
    assert built["api_key"] == "not-needed"


def test_named_apart_it_is_called_at_the_address_saved_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness has an address of its own and so has the provider: nothing was taken
    from the harness, so nothing is refused."""
    from iterate.userconfig import SavedProvider

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        [
            *["--backend", "ollama", "--base-url", "http://gpu-box:11434", "--model", "gemma4:12b"],
            *["--target-backend", "ollama"],
        ],
        saved={
            "allowed": None,
            "providers": {"ollama": SavedProvider("ollama", base_url="http://under-test:11434")},
        },
    )
    assert result.exit_code == 0, result.output
    assert built["base_url"] == "http://under-test:11434"


def test_the_harness_model_on_ollama_is_under_test_where_the_harness_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.userconfig import SavedProvider

    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--model", "gemma4:12b"],
        saved={
            "allowed": None,
            "providers": {"ollama": SavedProvider("ollama", base_url="http://saved-box:11434")},
        },
    )
    assert result.exit_code == 0, result.output
    assert built["base_url"] == "http://localhost:11434"


def test_the_provider_of_the_model_under_test_is_read_whatever_its_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--target-backend", " GROQ", "--target-model", "llama-3.3-70b"],
        env={"GROQ_API_KEY": "gsk-env"},
    )
    assert result.exit_code == 0, result.output
    assert (built["backend"], built["api_key"]) == ("groq", "gsk-env")


def test_a_refusal_names_the_flag_its_value_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for flags, said in (
        (["--target-backend", "grok", "--target-model", "m"], "--target-backend grok: not a"),
        (["--providers", "openia"], "--providers openia: not a provider"),
        (
            [
                *["--target-backend", "vllm", "--target-model", "m"],
                *["--target-base-url", "gpu.test/v1"],
            ],
            "--target-base-url gpu.test/v1 has to start with http:// or https://",
        ),
        (
            ["--backend", "ollama", "--model", "m", "--base-url", "http://me:pw@gpu.test:11434"],
            "--base-url http://gpu.test:11434 carries a user name, a password or a token",
        ),
    ):
        result, _ = _prompt_run(tmp_path, monkeypatch, flags)
        text = _refused(result, tmp_path)
        assert said in text
        assert "pw@" not in text


def test_a_harness_the_environment_aims_at_a_company_takes_that_companys_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ITERATE_BACKEND_URL` is the address the client calls when no flag gives one, so
    it is the address the key is picked by."""
    result, built = _prompt_run(
        tmp_path,
        monkeypatch,
        ["--backend", "openai-compatible", "--model", "gpt-4o-mini", "--target-model", "g"],
        env={"ITERATE_BACKEND_URL": "https://api.openai.com/v1", "OPENAI_API_KEY": "sk-env"},
    )
    assert result.exit_code == 0, result.output
    assert built["harness"]["api_key"] == "sk-env"


def test_a_harness_with_no_key_is_told_which_variables_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases: tuple[tuple[list[str], dict[str, Any] | None, str], ...] = (
        (["--backend", "groq"], None, "GROQ_API_KEY or ITERATE_BACKEND_API_KEY in the environment"),
        (
            ["--backend", "openai-compatible", "--base-url", "http://gpu.test/v1"],
            None,
            "requires --api-key, or ITERATE_BACKEND_API_KEY in the environment "
            "(OPENAI_API_KEY is not read for a server you run)",
        ),
        (
            ["--backend", "openai"],
            {"harness": {"backend": "groq", "api_key": "gsk-saved"}},
            "The key saved is for groq, so it is not sent here",
        ),
        (
            ["--backend", "vllm", "--base-url", "http://b.test/v1"],
            {"harness": {"backend": "vllm", "api_key": "k", "base_url": "http://a.test/v1"}},
            "The key saved is for vllm at http://a.test/v1, so it is not sent here",
        ),
    )
    for flags, saved, said in cases:
        result, _ = _prompt_run(tmp_path, monkeypatch, [*flags, "--model", "m"], saved=saved)
        assert said in _refused(result, tmp_path)


def test_the_flags_of_a_prompt_run_are_refused_on_a_run_that_is_not_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    monkeypatch.chdir(tmp_path)
    _stub_run_supervised(monkeypatch)
    for flag in (["--providers", "openai"], ["--target-base-url", "http://gpu:8000/v1"]):
        result = runner.invoke(
            app, ["run", "--data", str(data), "--target", "churn", "--metric", "f1", *flag]
        )
        assert result.exit_code != 0
        assert "this run has no --task" in " ".join(_plain(result.output).split())


def test_a_key_saved_for_the_harness_is_not_sent_to_another_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved key is the saved backend's. `--backend groq` on a file saved for openai
    finds groq's own key, or none."""
    from iterate import userconfig
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    for name in _PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    userconfig.save_user_config(
        {"backend": "openai", "model": "gpt-4o", "api_key": "sk-saved-for-openai"}
    )
    seen: list[dict[str, Any]] = []
    captured = _stub_run_supervised(monkeypatch)
    import iterate.llm.factory as factory_module

    fake = factory_module.build_client
    monkeypatch.setattr(
        factory_module, "build_client", lambda name, **kw: seen.append(kw) or fake(name, **kw)
    )
    argv = ["run", "--data", str(data), "--target", "churn", "--metric", "f1", "--no-research"]
    argv += ["--plain", "--memory", str(tmp_path / "m.db"), "--backend", "groq"]
    get_settings.cache_clear()
    try:
        refused = runner.invoke(app, argv)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-env")
        get_settings.cache_clear()
        ran = runner.invoke(app, argv)
    finally:
        get_settings.cache_clear()
    assert refused.exit_code != 0
    assert "requires --api-key" in _plain(refused.output)
    assert ran.exit_code == 0, ran.output
    assert captured["kernel"] is not None
    assert {kw["api_key"] for kw in seen} == {"gsk-env"}
    assert {kw["model"] for kw in seen} == {None}


def test_an_e2b_run_with_no_key_keeps_the_memory_it_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal sits above the archive: `--fresh` on a run that cannot start moves
    nothing."""
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    memory = tmp_path / "m.db"
    memory.write_bytes(b"the run before")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            [
                *["run", "--data", str(data), "--target", "churn", "--metric", "f1"],
                *["--compute", "e2b", "--fresh", "--plain", "--memory", str(memory)],
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    assert "E2B API key" in result.output
    assert memory.read_bytes() == b"the run before"
    assert sorted(p.name for p in tmp_path.glob("m*")) == ["m.db"]


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
    from iterate.core import prices as prices_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # The background refresh is wired, and a unit test never reaches for the network.
    started: list[str | None] = []
    monkeypatch.setattr(
        prices_mod, "refresh_azure_in_background", lambda region=None: started.append(region)
    )
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
    assert started == ["eastus"]
    serving = json.loads(out.with_name("best.json").read_text())["serving"]
    assert serving["chosen"]["host"]["cloud"] == "azure"
    assert [cost["host"]["cloud"] for cost in serving["by_cloud"]] == ["azure"]
    assert "prices: azure shipped" in _plain(result.output)
    assert "no price list cached for azure eastus" in _plain(result.output)


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


# ─── the serving budget is a wall (Sprint 5 Day 3) ──────────────────────────


@pytest.mark.parametrize("budget", ["0", "-5"])
def test_serving_budget_must_be_above_zero(tmp_path: Path, budget: str) -> None:
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
            "--serving-budget",
            budget,
        ],
    )
    assert result.exit_code != 0
    assert f"--serving-budget is dollars a month above zero, got {budget}" in _plain(result.output)


def test_the_size_of_the_starting_prompt_is_read_for_the_estimate(tmp_path: Path) -> None:
    prompt = tmp_path / "p.txt"
    prompt.write_text("x" * 2000)
    assert cli_module._prompt_chars(None, prompt) == 0  # not a prompt run
    assert cli_module._prompt_chars("say whether", None) == len("say whether")
    assert cli_module._prompt_chars("say whether", prompt) == len("say whether") + 2000
    assert cli_module._prompt_chars("say whether", tmp_path / "missing.txt") == len("say whether")


def _baseline_price(family: str, rate: int, **facts_kw: Any) -> tuple[float, str]:
    """What the shipped table says the family's baseline costs, so the tests own no
    dollar figure the next snapshot refresh would move."""
    from iterate.core import serving

    facts = serving.baseline_facts(family, **facts_kw)
    chosen = serving.profile(facts, rate, serving.load_prices()).chosen
    assert chosen is not None
    return chosen.usd_per_month, chosen.host.label


def test_a_table_run_whose_baseline_is_over_the_budget_is_refused_before_a_folder_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even the default boosting pipeline costs one small box a month; a budget under that
    is refused like a wrong metric, with the price, the budget and the table's floor."""
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    dot = tmp_path / "dot"
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(dot / "runs"))
    get_settings.cache_clear()
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
                "--serving-budget",
                "5",
                "--plain",
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    usd, host = _baseline_price("tabular", 1000, n_features=1)
    assert usd > 5  # the fixture's premise: one small box costs more than the budget
    assert (
        f"the baseline alone (the default boosting pipeline on 1 features) costs about "
        f"${usd:,.2f} a month at 1,000 requests an hour on {host}, above your --serving-budget "
        f"of $5; the cheapest machine that could hold it is {host} at ${usd:,.2f} a month, so "
        "no request rate fits this budget. Raise the budget"
    ) in _plain(result.output)
    assert "lower --requests-per-hour" not in _plain(result.output)
    assert not dot.exists()


def test_a_prompt_run_whose_starting_prompt_is_over_the_budget_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    dot = tmp_path / "dot"
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(dot / "runs"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-for-the-model-under-test")
    get_settings.cache_clear()
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
                "--task",
                "say whether the customer churns",
                "--backend",
                "openai",
                "--api-key",
                "k",
                "--target-backend",
                "openai",
                "--target-model",
                "gpt-4o-mini",
                "--serving-budget",
                "1",
                "--requests-per-hour",
                "50000",
                "--plain",
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    text = _plain(result.output)
    usd, host = _baseline_price(
        "prompt",
        50000,
        provider="openai",
        model="gpt-4o-mini",
        prompt_chars=len("say whether the customer churns"),
    )
    assert (
        f"the baseline alone (the starting prompt on gpt-4o-mini) costs about ${usd:,.2f} a "
        f"month at 50,000 requests an hour on {host}, above your --serving-budget of $1. Raise "
        "the budget, or lower --requests-per-hour"
    ) in text
    assert "cheapest machine" not in text  # an API has no machine floor to quote
    assert not dot.exists()


def test_the_prompt_baseline_is_priced_at_the_size_of_the_starting_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2,000-character prompt file is not one token: the pre-run price reads it, the
    way the wall reads the winner's own prompt after the run."""
    from iterate.config import get_settings

    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    prompt = tmp_path / "p.txt"
    prompt.write_text("x" * 2000)
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "dot" / "runs"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-for-the-model-under-test")
    get_settings.cache_clear()
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
                "--task",
                "say whether the customer churns",
                "--prompt-file",
                str(prompt),
                "--backend",
                "openai",
                "--api-key",
                "k",
                "--target-backend",
                "openai",
                "--target-model",
                "gpt-4o-mini",
                "--serving-budget",
                "1000",
                "--requests-per-hour",
                "50000",
                "--plain",
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    usd, _ = _baseline_price(
        "prompt",
        50000,
        provider="openai",
        model="gpt-4o-mini",
        prompt_chars=len("say whether the customer churns") + 2000,
    )
    assert usd > 1000
    assert f"costs about ${usd:,.2f} a month" in _plain(result.output)


def test_a_winner_within_the_budget_says_so_on_the_serving_line_and_in_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            "--serving-budget",
            "1000",
        ],
    )
    assert result.exit_code == 0, result.stdout
    serving = json.loads(out.with_name("best.json").read_text())["serving"]
    assert serving["budget_usd_per_month"] == 1000.0
    assert serving["within_budget"] is True
    assert "within the $1,000 serving budget, prices:" in _plain(result.output)


def test_the_summary_names_what_the_wall_turned_away_and_how_far_each_got(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iterate.core import serving
    from iterate.core.orchestrator import RunResult

    monkeypatch.setenv("COLUMNS", "200")  # a wide console, so the table row stays on one line

    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=100),
    )
    over = Experiment(
        candidate=Candidate(
            description="a wide forest", changes={"over_budget": 36.79}, rationale="r"
        ),
        target="t",
        hypothesis="h",
        status="completed",
        iteration=1,
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.9}, primary="f1", direction="maximize", n_samples=100),
        ),
    )
    wall = serving.Wall(requests_per_hour=1000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("swap the backbone to convnext_tiny", 36.79, "off the line")
    wall.record_refusal("swap the backbone to convnext_tiny", 36.79, "brief refused")
    wall.record_refusal("a wide forest", 36.79, "trained", budget=30.0)
    wall.budget = None  # lifted later: the refusals it made still print
    result = RunResult(baseline=baseline, history=[over], best=None, stopped_because="patience")
    with cli_module.console.capture() as captured:
        cli_module._render_summary(result, "f1", wall=wall)
    text = _plain(captured.get())
    assert "no candidate within the serving budget beat the baseline" in text
    # The table cell may wrap, so the marker is checked in order, not as one string.
    assert re.search(r"a wide forest.*over the serving budget: \$37 a month", text)
    assert (
        "over the serving budget this run, taken off the line before a brief: swap the "
        "backbone to convnext_tiny ($37 against $20)"
    ) in text
    assert (
        "over the serving budget this run, briefed anyway and refused: swap the backbone to "
        "convnext_tiny ($37 against $20)"
    ) in text
    assert (
        "over the serving budget this run, trained, over the budget, never the winner: a wide "
        "forest ($37 against $30)"
    ) in text


# ─── the Researcher and the Pricer go back and forth (Sprint 5 Day 4) ──────


def _over_budget_result(stopped_because: str) -> Any:
    from iterate.core.orchestrator import RunResult

    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=100),
    )
    return RunResult(baseline=baseline, history=[], best=None, stopped_because=stopped_because)


def test_a_run_the_wall_stopped_says_what_nothing_fitted_and_how_to_open_it() -> None:
    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=400_000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("swap the backbone to convnext_tiny", 36.79, "off the line")
    wall.record_refusal("train at 128 px", 24.53, "off the line")
    with cli_module.console.capture() as captured:
        cli_module._render_summary(_over_budget_result("over_budget"), "f1", wall=wall)
    text = _plain(captured.get())

    assert "stopped: over_budget" in text
    assert (
        "nothing the run could try next fits the serving budget, $20 a month at 400,000 "
        "requests an hour. To open it, raise --serving-budget, lower --requests-per-hour, "
        'or run it on a terminal and type "budget $N" when it asks.'
    ) in text
    assert (
        "no candidate within the serving budget beat the baseline; what the wall held back "
        "was never tried."
    ) in text
    assert (
        "over the serving budget this run, taken off the line before a brief: swap the "
        "backbone to convnext_tiny ($37 against $20), train at 128 px ($25 against $20)"
    ) in text


def test_a_run_refused_only_off_the_line_closes_on_the_budget_not_on_the_baseline() -> None:
    """Nothing trained over the budget, yet the wall shaped the run: the closing line
    says so instead of reading as a plain loss to the baseline."""
    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=1000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("swap the backbone to convnext_tiny", 36.79, "off the line")
    with cli_module.console.capture() as captured:
        cli_module._render_summary(_over_budget_result("patience"), "f1", wall=wall)
    text = _plain(captured.get())

    assert "stopped: patience" in text
    assert "what the wall held back was never tried" in text
    assert "no candidate beat the baseline." not in text
    assert "To open it" not in text


def test_a_run_the_wall_never_touched_closes_as_before() -> None:
    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=1000, prices=serving.load_prices(), budget=20.0)
    with cli_module.console.capture() as captured:
        cli_module._render_summary(_over_budget_result("patience"), "f1", wall=wall)
    text = _plain(captured.get())

    assert "no candidate beat the baseline." in text
    assert "serving budget" not in text


def test_the_over_budget_line_reads_the_wall_the_cli_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scripted run the wall emptied ends on the budget and rate it was given, with the
    refusals the loop made on that same wall listed under it."""
    data = tmp_path / "d.csv"
    _write_tiny_csv(data)
    captured = _stub_run_supervised(
        monkeypatch,
        stopped_because="over_budget",
        refusals=(("swap the backbone to convnext_tiny", 36.79, "off the line"),),
    )

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
            "--serving-budget",
            "20",
        ],
    )
    text = _plain(result.output)

    assert result.exit_code == 0, result.stdout
    assert captured["controller"] is None
    assert "stopped: over_budget" in text
    assert "fits the serving budget, $20 a month at 1,000 requests an hour" in text
    assert "swap the backbone to convnext_tiny ($37 against $20)" in text


def test_an_entry_the_pricer_could_not_size_is_listed_as_not_priced() -> None:
    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=400_000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("own model: google/vit-base-patch16-224", None, "off the line")
    with cli_module.console.capture() as captured:
        cli_module._render_summary(_over_budget_result("over_budget"), "f1", wall=wall)
    text = _plain(captured.get())

    assert "own model: google/vit-base-patch16-224 (not priced, under $20)" in text


def _scored(changes: dict[str, Any], f1: float) -> Any:
    return Experiment(
        candidate=Candidate(description="a try", changes=changes, rationale="r"),
        target="t",
        hypothesis="h",
        status="completed",
        iteration=1,
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": f1}, primary="f1", direction="maximize", n_samples=100),
        ),
    )


def test_a_stamped_try_that_lost_to_the_baseline_is_not_said_to_have_beaten_it() -> None:
    from dataclasses import replace

    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=400_000, prices=serving.load_prices(), budget=20.0)
    lost = _scored({"model": "x", "over_budget": 36.79}, 0.6)
    wall.record_refusal("a try", 36.79, "trained")
    result = replace(_over_budget_result("max_iterations"), history=[lost])
    with cli_module.console.capture() as captured:
        cli_module._render_summary(result, "f1", wall=wall)
    text = _plain(captured.get())
    assert "the ones that beat it" not in text
    assert "no candidate beat the baseline." in text


def test_a_network_held_back_then_trained_after_a_raise_is_not_called_never_tried() -> None:
    from dataclasses import replace

    from iterate.core import serving

    wall = serving.Wall(requests_per_hour=400_000, prices=serving.load_prices(), budget=20.0)
    wall.record_refusal("swap to convnext_tiny", 36.79, "off the line", network="convnext_tiny")
    wall.budget, wall.moves = 80.0, 1
    ran = _scored({"recipe": {"backbone": "convnext_tiny", "image_size": 64}}, 0.6)
    result = replace(_over_budget_result("max_iterations"), history=[ran])
    with cli_module.console.capture() as captured:
        cli_module._render_summary(result, "f1", wall=wall)
    text = _plain(captured.get())
    assert "never tried" not in text
    assert "no candidate beat the baseline." in text


def test_the_summary_shows_the_common_records_comparison_it_banked_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored, the winner's score is over fewer records and reads as worse than the
    baseline; the bracket says what the loop weighed."""
    from iterate.core.agent_loop import COMPARED_ON_COMMON
    from iterate.core.orchestrator import RunResult

    monkeypatch.setenv("COLUMNS", "200")
    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=20),
    )
    winner = Experiment(
        candidate=Candidate(
            description="shorter prompt",
            changes={COMPARED_ON_COMMON: [0.6875, 0.625]},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        iteration=1,
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(
                values={"f1": 0.6875}, primary="f1", direction="maximize", n_samples=16
            ),
        ),
    )
    result = RunResult(baseline=baseline, history=[winner], best=winner, stopped_because="patience")
    with cli_module.console.capture() as captured:
        cli_module._render_summary(result, "f1")
    text = " ".join(_plain(captured.get()).split())
    assert "(0.6875 vs 0.6250 on common records)" in text
    assert "the loop weighed it against the best so far on the records both got answers for" in text
