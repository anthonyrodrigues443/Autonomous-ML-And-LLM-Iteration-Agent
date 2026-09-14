"""Tests for `iterate run` on a folder of images: the flags, the refusals, the folder
it writes, and the stop it makes until the vision target exists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from iterate import cli as cli_module
from iterate.cli import app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()


def _plain(output: str) -> str:
    stripped = "".join(ch for ch in output if ch not in "│╭╰─╮╯")
    return " ".join(stripped.split())


def _png(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), (seed * 9 % 255, 60, 120)).save(path)


def _class_tree(root: Path, *, per_class: int = 6, seed: int = 0) -> Path:
    for c in ("cat", "dog", "emu"):
        for i in range(per_class):
            _png(root / c / f"{c}_{seed + i}.png", seed + i)
    return root


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ITERATE_RUNS_DIR", str(tmp_path / "dot" / "runs"))
    cli_module.get_settings.cache_clear()
    yield  # type: ignore[misc]
    cli_module.get_settings.cache_clear()


def _data_dir(tmp_path: Path) -> Path:
    return tmp_path / "dot" / "data"


def test_a_folder_needs_no_target_and_stops_after_linking(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    result = runner.invoke(app, ["run", "--data", str(source)])
    assert result.exit_code == 0, result.output
    text = _plain(result.output)
    assert "labels: the folder names (3 classes)" in text
    assert "split: none given, 18 images split here 80/20" in text
    assert "linked:" in text
    assert "this run stops here" in text
    roots = list(_data_dir(tmp_path).iterdir())
    assert len(roots) == 1
    assert (roots[0] / "train.csv").exists()
    assert (roots[0] / "holdout" / "cat").is_dir()


def test_a_csv_still_needs_a_target(tmp_path: Path) -> None:
    frame = pd.DataFrame({"f1": range(20), "y": [i % 2 for i in range(20)]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    result = runner.invoke(app, ["run", "--data", str(tmp_path / "t.csv")])
    assert result.exit_code != 0
    assert "--target is required for a CSV" in _plain(result.output)


def test_folder_flags_are_refused_on_a_csv(tmp_path: Path) -> None:
    frame = pd.DataFrame({"f1": range(20), "y": [i % 2 for i in range(20)]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    result = runner.invoke(
        app, ["run", "--data", str(tmp_path / "t.csv"), "--target", "y", "--yes"]
    )
    assert result.exit_code != 0
    assert "describe a folder of images" in _plain(result.output)


def test_a_folder_and_a_file_cannot_be_mixed(tmp_path: Path) -> None:
    a = _class_tree(tmp_path / "train")
    (tmp_path / "h.csv").write_text("image,label\n")
    result = runner.invoke(app, ["run", "--train", str(a), "--holdout", str(tmp_path / "h.csv")])
    assert result.exit_code != 0
    assert "folders for every input, or files for every input" in _plain(result.output)


def test_key_without_labels_is_refused(tmp_path: Path) -> None:
    a = _class_tree(tmp_path / "a")
    result = runner.invoke(app, ["run", "--data", str(a), "--key", "id"])
    assert result.exit_code != 0
    assert "pass --labels too" in _plain(result.output)


def test_labels_with_two_folders_is_refused(tmp_path: Path) -> None:
    a = _class_tree(tmp_path / "train")
    b = _class_tree(tmp_path / "test", seed=50)
    (tmp_path / "labels.csv").write_text("id,label\n1,a\n")
    result = runner.invoke(
        app,
        ["run", "--train", str(a), "--holdout", str(b), "--labels", str(tmp_path / "labels.csv")],
    )
    assert result.exit_code != 0
    assert "--labels goes with --data" in _plain(result.output)


def test_two_folders_are_the_users_split(tmp_path: Path) -> None:
    a = _class_tree(tmp_path / "train", per_class=5)
    b = _class_tree(tmp_path / "test", per_class=2, seed=100)
    result = runner.invoke(app, ["run", "--train", str(a), "--holdout", str(b)])
    assert result.exit_code == 0, result.output
    root = next(_data_dir(tmp_path).iterdir())
    assert len(pd.read_csv(root / "train.csv")) == 15
    assert len(pd.read_csv(root / "holdout.csv")) == 6
    assert (root / "raw_files" / "train").is_dir()
    assert (root / "raw_files" / "holdout").is_dir()


def test_a_task_mismatch_between_the_two_folders_is_refused(tmp_path: Path) -> None:
    a = _class_tree(tmp_path / "train")
    b = tmp_path / "test"
    for i in range(12):
        _png(b / "img" / f"{i}.png", i)
    pd.DataFrame(
        {"id": [str(i) for i in range(12)], "score": [1.0 + i * 0.37 for i in range(12)]}
    ).to_csv(b / "scores.csv", index=False)
    result = runner.invoke(app, ["run", "--train", str(a), "--holdout", str(b)])
    assert result.exit_code != 0
    assert "reads as classification and --holdout as regression" in _plain(result.output)


def test_a_folder_the_rules_cannot_read_is_refused_with_the_reason(tmp_path: Path) -> None:
    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    result = runner.invoke(app, ["run", "--data", str(tmp_path)])
    assert result.exit_code != 0
    assert "unpack it to images first" in _plain(result.output)


def _partial(tmp_path: Path) -> Path:
    root = tmp_path / "ids"
    for i in range(19):
        _png(root / "images" / f"{i}.jpg", i)
    pd.DataFrame({"id": [str(i) for i in range(20)], "label": ["a", "b"] * 10}).to_csv(
        root / "labels.csv", index=False
    )
    return root


def test_partial_coverage_needs_yes_without_a_terminal(tmp_path: Path) -> None:
    root = _partial(tmp_path)
    refused = runner.invoke(app, ["run", "--data", str(root)])
    assert refused.exit_code != 0
    assert "re-run with --yes" in _plain(refused.output)
    accepted = runner.invoke(app, ["run", "--data", str(root), "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert "95.0% of the table's rows found their image" in _plain(accepted.output)


def test_the_pause_asks_on_a_terminal_and_a_no_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    root = _partial(tmp_path)
    monkeypatch.setattr(cli_module, "_stdin_owns_tty", lambda: True)
    asked: list[str] = []

    def no(prompt: str, default: bool = False) -> bool:
        asked.append(prompt)
        return False

    monkeypatch.setattr(typer, "confirm", no)
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 1
    assert asked == ["Continue with this plan?"]
    assert not _data_dir(tmp_path).exists()
    monkeypatch.setattr(typer, "confirm", lambda prompt, default=False: True)
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 0, result.output
    assert "note: 95.0% of rows resolved to an image; the rest are dropped" in _plain(result.output)


def test_a_csv_of_image_paths_stops_before_the_loop(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    for i in range(12):
        _png(root / "images" / f"{i:03d}.png", i)
    pd.DataFrame(
        {"image": [f"images/{i:03d}.png" for i in range(12)], "label": ["a", "b"] * 6}
    ).to_csv(root / "data.csv", index=False)
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(root / "data.csv"),
            "--target",
            "label",
            "--memory",
            str(tmp_path / "m.db"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "this CSV holds image paths" in _plain(result.output)
