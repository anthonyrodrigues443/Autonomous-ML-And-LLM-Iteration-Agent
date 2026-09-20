"""`iterate run` on images, up to the loop: both input forms, the refusals and their
order, torch at the start, the task line, the metric step, what the coder and the
supervisor are given, and the monitor report in the run folder. The loop entry is
stubbed; nothing trains."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from iterate import cli as cli_module
from iterate.adapters.compute import deps
from iterate.cli import app
from iterate.core import codegen
from iterate.core import researcher as researcher_module
from iterate.deliver import saved_model
from iterate.schemas.llm import ChatResponse, ToolCall
from tests.unit.image_fixtures import class_tree, png, stub_image_run

pytestmark = pytest.mark.unit

runner = CliRunner(env={"COLUMNS": "1000"})
_RESEARCH = researcher_module.Researcher.research


def _plain(output: str) -> str:
    return " ".join("".join(ch for ch in output if ch not in "│╭╰─╮╯").split())


@pytest.fixture(autouse=True)
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "dot" / "runs"))
    monkeypatch.setenv("ITERATE_MEMORY_DB", str(tmp_path / "dot" / "memory.db"))
    cli_module.get_settings.cache_clear()
    yield stub_image_run(monkeypatch, tmp_path)  # type: ignore[misc]
    cli_module.get_settings.cache_clear()


def _csv(root: Path, labels: list[Any], name: str = "data.csv") -> Path:
    for i in range(len(labels)):
        png(root / "images" / f"{i:03d}.png", i, klass=i % 7)
    pd.DataFrame(
        {"image": [f"images/{i:03d}.png" for i in range(len(labels))], "label": labels}
    ).to_csv(root / name, index=False)
    return root / name


def _watch_torch(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Records every ensure_torch call without installing anything."""
    asked: list[bool] = []
    monkeypatch.setattr(deps, "installed", lambda: {"numpy": "2"})
    monkeypatch.setattr(deps, "ensure_torch", lambda *, consent: asked.append(consent) or "")
    return asked


def test_every_form_reaches_the_loop_on_the_vision_target(
    tmp_path: Path, calls: list[dict[str, Any]]
) -> None:
    csv = _csv(tmp_path / "flat", ["a", "b", "c"] * 8)
    frame = pd.read_csv(csv)
    frame.iloc[:18].to_csv(tmp_path / "flat" / "fit.csv", index=False)
    frame.iloc[18:].to_csv(tmp_path / "flat" / "held.csv", index=False)
    folder = class_tree(tmp_path / "pets", per_class=8)
    forms = {
        "csv": ["--data", str(csv), "--target", "label"],
        "pair": [
            "--train",
            str(tmp_path / "flat" / "fit.csv"),
            "--holdout",
            str(tmp_path / "flat" / "held.csv"),
            "--target",
            "label",
        ],
        "folder": ["--data", str(folder)],
    }
    for name, argv in forms.items():
        calls.clear()
        result = runner.invoke(app, ["run", *argv, "--metric", "accuracy", "--plain"])
        assert result.exit_code == 0, (name, result.output)
        (kw,) = calls
        assert type(kw["target"]).__name__ == "DLModelTarget", name
        assert kw["family"] == kw["supervisor"]._family == kw["critic"]._family == "vision"
        assert kw["researcher"]._family == "vision"
        assert "Training images per class" in kw["data_summary"]
        assert "Data checks: Data checks" not in kw["data_summary"], name
        if name == "folder":
            assert "Data checks before the run" in kw["data_summary"]
        # A number run must never be offered a class-only lever, so the supervisor is
        # told the task and the size a lever would change.
        assert kw["supervisor"]._task == "classification", name
        assert kw["supervisor"]._image_size == kw["target"]._image_size
        assert kw["supervisor"]._image_width > 0
        conf = dict(
            zip(
                kw["make_coder"].__code__.co_freevars,
                (c.cell_contents for c in kw["make_coder"].__closure__),
                strict=True,
            )
        )["confinement"]
        assert conf.reads == (Path(kw["dataset"].train_features["image"].iloc[0]).parent,)


@pytest.mark.parametrize(
    ("flags", "says"),
    [
        (["--compute", "e2b"], "Pass --compute local"),
        (["--task", "say what it shows"], "Drop --task"),
        (["--spec"], "Drop --spec"),
    ],
)
def test_a_flag_an_image_run_cannot_take_is_refused_before_anything_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    flags: list[str],
    says: str,
) -> None:
    asked = _watch_torch(monkeypatch)
    memory = tmp_path / "dot" / "memory.db"
    memory.parent.mkdir(parents=True)
    memory.write_text("x")
    csv = _csv(tmp_path / "flat", ["a", "b"] * 6)
    folder = class_tree(tmp_path / "pets", per_class=6)
    for argv in (["--data", str(csv), "--target", "label"], ["--data", str(folder)]):
        result = runner.invoke(app, ["run", *argv, *flags, "--fresh"])
        assert result.exit_code == 2, result.output
        assert says in _plain(result.output)
    assert sorted(p.name for p in memory.parent.iterdir()) == ["memory.db"]
    assert not (tmp_path / "dot" / "data").exists()
    assert asked == []
    assert calls == []


def test_torch_is_checked_before_the_link_and_needs_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(deps, "installed", lambda: {"numpy": "2"})
    asked: list[bool] = []
    monkeypatch.setattr(
        deps, "ensure_torch", lambda *, consent: asked.append(consent) or "no torch"
    )
    folder = class_tree(tmp_path / "pets", per_class=6)
    result = runner.invoke(app, ["run", "--data", str(folder), "--no-install"])
    assert result.exit_code == 2
    assert asked == [False]
    assert "no torch" in _plain(result.output)
    assert not (tmp_path / "dot" / "data").exists()

    monkeypatch.setattr(deps, "ensure_torch", lambda *, consent: asked.append(consent) or "")
    result = runner.invoke(app, ["run", "--data", str(folder), "--install", "--metric", "accuracy"])
    assert result.exit_code == 0, result.output
    assert asked == [False, True]
    text = _plain(result.output)
    assert text.index("torch and torchvision for this image run") < text.index("linked:")
    assert calls


def test_a_name_or_a_key_that_is_wrong_is_refused_before_the_link_and_before_an_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]
) -> None:
    asked = _watch_torch(monkeypatch)
    monkeypatch.setattr(cli_module, "_resolved_api_key_from_env", lambda settings, backend: None)
    folder = class_tree(tmp_path / "pets", per_class=6)
    typo = runner.invoke(app, ["run", "--data", str(folder), "--install", "--metric", "accuarcy"])
    assert typo.exit_code == 2
    assert "unknown metric 'accuarcy'" in _plain(typo.output)
    nokey = runner.invoke(app, ["run", "--data", str(folder), "--install", "--backend", "groq"])
    assert nokey.exit_code == 2
    assert "requires --api-key" in _plain(nokey.output)
    assert "linked:" not in _plain(nokey.output)
    assert not (tmp_path / "dot" / "data").exists()
    assert asked == []
    assert calls == []


def test_the_run_says_how_it_read_the_labels(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    csv = _csv(tmp_path / "ids", [i % 25 for i in range(125)])
    result = runner.invoke(app, ["run", "--data", str(csv), "--target", "label", "--no-research"])
    assert result.exit_code == 0, result.output
    text = _plain(result.output)
    assert "labels read as numbers (25 distinct integer values)" in text
    # The hint to flip the task with --metric is gone: taking it skips the metric step.
    assert "reads them as classes" not in text
    assert calls[0]["dataset"].task == "regression"
    calls.clear()
    result = runner.invoke(
        app, ["run", "--data", str(csv), "--target", "label", "--metric", "accuracy"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["dataset"].task == "classification"


def test_numbered_class_folders_stay_classes_after_the_link(
    tmp_path: Path, calls: list[dict[str, Any]]
) -> None:
    for k in range(25):
        for i in range(5):
            png(tmp_path / "ids" / str(k) / f"{i}.png", i, klass=k)
    result = runner.invoke(app, ["run", "--data", str(tmp_path / "ids"), "--no-research"])
    assert result.exit_code == 0, result.output
    assert calls[0]["dataset"].task == "classification"
    assert "labels read as classes (25 distinct integer values)" in _plain(result.output)


def test_labels_a_metric_cannot_score_are_refused_before_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]
) -> None:
    asked = _watch_torch(monkeypatch)
    fit = _csv(tmp_path / "s", ["a", "b", "c"] * 6, "fit.csv")
    frame = pd.read_csv(fit)
    frame[frame.label != "c"].iloc[:6].to_csv(tmp_path / "s" / "held.csv", index=False)
    whole = ["--data", str(fit), "--target", "label"]
    jaccard = runner.invoke(app, ["run", *whole, "--metric", "jaccard", "--install"])
    assert jaccard.exit_code == 2
    assert "--metric jaccard cannot score a 3-class label" in _plain(jaccard.output)
    binary = runner.invoke(app, ["run", *whole, "--average", "binary"])
    assert binary.exit_code == 2
    assert "average='binary' needs a two-class target" in _plain(binary.output)
    short = runner.invoke(
        app,
        [
            "run",
            "--train",
            str(fit),
            "--holdout",
            str(tmp_path / "s" / "held.csv"),
            "--target",
            "label",
            "--metric",
            "accuracy",
        ],
    )
    assert short.exit_code == 2
    assert "the holdout holds 2 of the 3 training classes" in _plain(short.output)

    four = _csv(tmp_path / "q", ["a", "b", "c", "d"] * 6, "fit.csv")
    held = pd.read_csv(four)
    held[held.label != "d"].iloc[:9].to_csv(tmp_path / "q" / "held.csv", index=False)
    auc = runner.invoke(
        app,
        [
            "run",
            "--train",
            str(four),
            "--holdout",
            str(tmp_path / "q" / "held.csv"),
            "--target",
            "label",
            "--metric",
            "roc_auc",
        ],
    )
    assert auc.exit_code == 2
    assert "the holdout has no image of 1 training class(es)" in _plain(auc.output)
    # Every one of them is a label fact, so none of them reached an install.
    assert asked == []
    assert calls == []


def test_a_split_that_cannot_be_made_is_a_message_not_a_traceback(
    tmp_path: Path, calls: list[dict[str, Any]]
) -> None:
    csv = _csv(tmp_path / "ids", [i % 25 for i in range(120)])
    result = runner.invoke(
        app, ["run", "--data", str(csv), "--target", "label", "--metric", "accuracy"]
    )
    assert result.exit_code == 2, result.output
    assert "the data could not be split as given" in _plain(result.output)
    assert calls == []


class _Papers:
    name = "fake"

    def __init__(self, cache_dir: Any = None) -> None:
        pass

    def search(self, query: str, *, limit: int = 5) -> list[Any]:
        return []


class _Chooser:
    def __init__(self, metric: str) -> None:
        self.metric = metric
        self.seen: list[tuple[str, list[Any], list[Any]]] = []

    @property
    def model(self) -> str:
        return "fake"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        tool = tools[0]
        self.seen.append((tool.name, list(messages), list(tools)))
        args = (
            {"queries": ["image metrics"]}
            if tool.name == "plan_queries"
            else {"metric": self.metric, "why": "w"}
        )
        return ChatResponse(
            model="fake", tool_calls=[ToolCall(id="1", name=tool.name, arguments=args)]
        )


def test_the_metric_step_speaks_images_offers_what_scores_and_chooses_without_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]
) -> None:
    from iterate.core import setup

    monkeypatch.setattr(researcher_module.Researcher, "research", _RESEARCH)
    monkeypatch.setattr(researcher_module, "OpenAlexClient", _Papers)
    monkeypatch.setattr(researcher_module, "ArxivClient", _Papers)
    chooser = _Chooser("balanced_accuracy")
    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: chooser)
    csv = _csv(tmp_path / "flat", ["a", "b", "c"] * 8)
    result = runner.invoke(app, ["run", "--data", str(csv), "--target", "label", "--plain"])
    assert result.exit_code == 0, result.output
    # No suggestion call: this pass picks a ruler, and the loop's Researcher suggests.
    assert [name for name, _, _ in chooser.seen] == ["plan_queries", "choose_setup"]
    _, messages, tools = chooser.seen[1]
    text = " ".join(m.content for m in messages)
    assert "IMAGE" in text
    assert "TABULAR" not in text
    assert "scikit-learn" not in text
    assert "no papers were found" in text
    enum = tools[0].parameters["properties"]["metric"]["enum"]
    assert enum == setup.offered_for_images(calls[0]["dataset"])
    assert "jaccard" not in enum
    assert calls[0]["target"]._metric == "balanced_accuracy"
    assert "metric: balanced_accuracy" in _plain(result.output)


def test_a_table_run_whose_search_finds_nothing_keeps_the_default_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing a ruler with no papers is the image lane's call. A table run takes the
    deterministic default there, as it did before images."""
    monkeypatch.setattr(researcher_module.Researcher, "research", _RESEARCH)
    monkeypatch.setattr(researcher_module, "OpenAlexClient", _Papers)
    monkeypatch.setattr(researcher_module, "ArxivClient", _Papers)
    chooser = _Chooser("mean_absolute_error")
    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: chooser)
    table = tmp_path / "table.csv"
    pd.DataFrame({"x": range(60), "label": [float(i) for i in range(60)]}).to_csv(
        table, index=False
    )
    result = runner.invoke(app, ["run", "--data", str(table), "--target", "label", "--plain"])
    assert result.exit_code == 0, result.output
    assert [name for name, _, _ in chooser.seen] == ["plan_queries"]
    assert "mean_absolute_error" not in _plain(result.output)


def test_the_monitor_report_lands_in_the_run_folder_at_the_first_experiment_and_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.core import agent_loop
    from iterate.core.orchestrator import RunResult
    from iterate.schemas.experiment import ExperimentResult

    runs = tmp_path / "dot" / "runs"
    seen: list[bool] = []

    def loop(**kw: Any) -> RunResult:
        kw["on_experiment"](experiment=None, baseline=None, is_best=False, run_id="r1")
        seen.append((runs / "r1" / "monitor.json").is_file())
        return RunResult(
            baseline=ExperimentResult(experiment_id="baseline", error="stub"),
            history=[],
            best=None,
            stopped_because="baseline_failed",
            run_id="r2",
        )

    monkeypatch.setattr(agent_loop, "run_supervised", loop)
    folder = class_tree(tmp_path / "pets", per_class=8)
    argv = ["run", "--data", str(folder), "--metric", "accuracy", "--notebooks", "none"]
    assert runner.invoke(app, argv).exit_code == 0
    assert seen == [True]
    (linked,) = (tmp_path / "dot" / "data").glob("pets-*")
    assert (runs / "r1" / "monitor.json").read_bytes() == (linked / "monitor.json").read_bytes()
    assert json.loads((runs / "r1" / "monitor.json").read_text())["findings"]

    def quiet(**kw: Any) -> RunResult:
        return RunResult(
            baseline=ExperimentResult(experiment_id="baseline", error="stub"),
            history=[],
            best=None,
            stopped_because="supervisor",
            run_id="r9",
        )

    monkeypatch.setattr(agent_loop, "run_supervised", quiet)
    assert runner.invoke(app, argv).exit_code == 0
    assert (runs / "r9" / "monitor.json").is_file()


@pytest.mark.skipif(
    not hasattr(codegen, "vision_session_preamble"),
    reason="the kernel-session lane's image session has not landed yet",
)
def test_make_coder_carries_every_image_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]
) -> None:
    """A fit gets 540 s, so the tabular 120 s cell timeout would kill every fine-tune,
    the tabular prefix would leave the last cell's device memory in place, and without
    the image summary the coder reads the images as a table of file names."""
    from iterate.core import coder as coder_module

    built: list[dict[str, Any]] = []
    monkeypatch.setattr(coder_module, "CodingAgent", lambda *a, **kw: built.append(kw) or object())
    folder = class_tree(tmp_path / "pets", per_class=8)
    result = runner.invoke(app, ["run", "--data", str(folder), "--metric", "accuracy", "--plain"])
    assert result.exit_code == 0, result.output
    calls[0]["make_coder"]()
    (kw,) = built
    assert kw["family"] == "vision"
    assert kw["preamble"] == codegen.vision_session_preamble()
    assert kw["floor_cell"] == codegen.vision_fallback_baseline()
    assert kw["cell_prefix"] == codegen.VISION_CELL_PREFIX
    assert kw["cell_timeout"] == 750.0
    assert kw["deadline_seconds"] == 2700.0
    assert kw["wall_ceiling_seconds"] == 5400.0
    assert kw["data_summary"] == calls[0]["data_summary"]
    assert kw["floor_carries_code"] is False
    assert codegen.META_JSON in kw["extra_inputs"]
    assert kw["keep_model"].name == saved_model.BEST_MODEL


# ─── the winner's network: staged by the coder, settled by the hook, delivered at the end ───

_FIT = {"backbone": "resnet18", "image_size": 64, "epochs": 3, "val": 0.9}


def _run_with_a_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    network: bytes | None,
    recipe: dict[str, Any] | None = None,
    wins: bool = True,
    extra: tuple[str, ...] = (),
) -> tuple[Any, Path]:
    """The real `iterate run` around a loop that stages `network` the way a session's
    coder does, calls the hook, and returns one finished try."""
    from iterate.core import agent_loop
    from iterate.core import coder as coder_module
    from iterate.core.orchestrator import RunResult
    from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

    def scored(name: str, value: float) -> ExperimentResult:
        metrics = Metrics(
            values={"accuracy": value}, primary="accuracy", direction="maximize", n_samples=24
        )
        return ExperimentResult(experiment_id=name, metrics=metrics)

    staging: list[Path] = []

    def loop(**kw: Any) -> RunResult:
        slot = kw["make_coder"]()["keep_model"]
        staging.append(slot.parent)
        assert slot.parent.is_dir()
        if network is not None:
            slot.write_bytes(network)
        kw["on_experiment"](experiment=None, baseline=None, is_best=wins, run_id="r1")
        changes = {"code": "fit()", "cells": [], "recipe": recipe or _FIT}
        tried = Experiment(
            candidate=Candidate(description="a fine-tune", changes=changes, rationale="r"),
            target="dl-model",
            hypothesis="h",
            status="completed",
            iteration=1,
            result=scored("e1", 0.9),
        )
        return RunResult(
            baseline=scored("baseline", 0.5),
            history=[tried],
            best=tried if wins else None,
            stopped_because="supervisor",
            run_id="r1",
        )

    monkeypatch.setattr(agent_loop, "run_supervised", loop)
    monkeypatch.setattr(coder_module, "CodingAgent", lambda *a, **kw: kw)
    folder = class_tree(tmp_path / "pets", per_class=8)
    argv = ["run", "--data", str(folder), "--metric", "accuracy", "--notebooks", "none", "--plain"]
    result = runner.invoke(app, [*argv, *extra])
    (made,) = staging
    return result, made


def test_the_winners_network_lands_in_the_run_folder_with_its_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, staging = _run_with_a_winner(tmp_path, monkeypatch, network=b"weights")
    assert result.exit_code == 0, result.output
    kept = tmp_path / "dot" / "runs" / "r1" / saved_model.BEST_MODEL
    assert kept.read_bytes() == b"weights"
    assert not os.access(kept, os.W_OK) or os.geteuid() == 0
    best = json.loads(kept.with_name("best.json").read_text())
    assert best["recipe"] == _FIT
    assert best["artifact_path"] == str(kept)
    assert best["score"] == 0.9
    assert "from iterate.vision import load" in _plain(result.output)
    assert "no network was saved" not in _plain(result.output)
    assert not staging.exists()


def test_output_moves_the_network_and_best_json_goes_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "models" / "pets.pt"
    result, _ = _run_with_a_winner(
        tmp_path, monkeypatch, network=b"weights", extra=("--output", str(out))
    )
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == b"weights"
    assert json.loads(out.with_name("best.json").read_text())["artifact_path"] == str(out)
    run_dir = tmp_path / "dot" / "runs" / "r1"
    assert not (run_dir / saved_model.BEST_MODEL).exists()
    assert not (run_dir / "best.json").exists()


def test_a_winner_that_is_the_agents_own_network_says_nothing_was_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = {"model": "timm-vit", "epochs": 4, "val": 0.9}
    result, _ = _run_with_a_winner(tmp_path, monkeypatch, network=None, recipe=own)
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "dot" / "runs" / "r1"
    assert not (run_dir / saved_model.BEST_MODEL).exists()
    best = json.loads((run_dir / "best.json").read_text())
    assert (best["artifact_path"], best["recipe"]) == (None, own)
    said = _plain(result.output)
    assert "no network was saved: the winner is the agent's own network" in said
    assert "saved best model" not in said


def test_a_fit_winner_whose_file_never_arrived_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _run_with_a_winner(tmp_path, monkeypatch, network=None)
    assert result.exit_code == 0, result.output
    assert "no network was saved: the winning try left no network file" in _plain(result.output)


def test_a_run_no_try_won_says_why_there_is_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, staging = _run_with_a_winner(tmp_path, monkeypatch, network=b"a loser", wins=False)
    assert result.exit_code == 0, result.output
    assert "no network was saved: no try beat the baseline" in _plain(result.output)
    assert not (tmp_path / "dot" / "runs" / "r1" / saved_model.BEST_MODEL).exists()
    assert not staging.exists()


def test_a_network_that_cannot_be_moved_is_a_warning_with_the_path_and_the_hook_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def full(*a: Any, **kw: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(saved_model, "settle", full)
    with caplog.at_level("WARNING"):
        result, _ = _run_with_a_winner(tmp_path, monkeypatch, network=b"weights")
    assert result.exit_code == 0, result.output
    kept = tmp_path / "dot" / "runs" / "r1" / saved_model.BEST_MODEL
    assert any(str(kept) in r.getMessage() and "No space" in r.getMessage() for r in caplog.records)
    assert (kept.parent / "monitor.json").is_file()


def test_the_staging_folder_goes_even_when_the_loop_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.core import agent_loop
    from iterate.core import coder as coder_module

    staging: list[Path] = []

    def loop(**kw: Any) -> Any:
        slot = kw["make_coder"]()["keep_model"]
        slot.write_bytes(b"weights")
        staging.append(slot.parent)
        raise KeyboardInterrupt

    monkeypatch.setattr(agent_loop, "run_supervised", loop)
    monkeypatch.setattr(coder_module, "CodingAgent", lambda *a, **kw: kw)
    folder = class_tree(tmp_path / "pets", per_class=8)
    result = runner.invoke(app, ["run", "--data", str(folder), "--metric", "accuracy", "--plain"])
    assert result.exit_code != 0
    (made,) = staging
    assert not made.exists()


def test_an_image_run_host_never_loads_lightgbm(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "flat", ["a", "b"] * 6)
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import pytest
        from typer.testing import CliRunner
        from iterate.cli import app
        from tests.unit.image_fixtures import stub_image_run
        stub_image_run(pytest.MonkeyPatch(), Path({str(tmp_path)!r}))
        result = CliRunner().invoke(app, ["run", "--data", {str(csv)!r}, "--target", "label", "--plain"])
        assert result.exit_code == 0, result.output
        print("lightgbm" in sys.modules)
        """
    )
    env = {"ITERATE_RUNS_DIR": str(tmp_path / "runs"), "ITERATE_MEMORY_DB": str(tmp_path / "m.db")}
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
        timeout=180,
    )
    assert out.stdout.strip().splitlines()[-1:] == ["False"], out.stderr[-2000:]
