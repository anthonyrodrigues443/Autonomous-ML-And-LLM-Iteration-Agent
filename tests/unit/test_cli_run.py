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

runner = CliRunner()


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
    frame = pd.DataFrame(
        {"feat": list(range(n)), "churn": [i % 2 for i in range(n)]}
    )
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


def _stub_run_orchestrator(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the heavyweight bits so `iterate run` returns instantly. Capture the wiring."""
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
        # Return a minimal RunResult that the summary renderer can swallow.
        baseline = ExperimentResult(
            experiment_id="b",
            metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize", n_samples=100),
        )
        from iterate.core.orchestrator import RunResult

        return RunResult(baseline=baseline, history=[], best=None, stopped_because="max_iterations")

    monkeypatch.setattr(cli_module, "build_client", _fake_build_client)
    monkeypatch.setattr(cli_module, "Proposer", _fake_proposer_ctor)
    monkeypatch.setattr(cli_module.Orchestrator, "run", _fake_orchestrator_run)
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
            "--memory",
            str(memory_db),
            "--fresh",
        ],
    )
    assert result.exit_code == 0
    assert captured["memory_type"] == "SqliteMemory"
