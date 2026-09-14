"""Tests for `iterate run` on a folder of images: the flags, the refusals, the folder
it writes, the Linker where the rules stop, the conversation at the pause, the plan
remembered from an earlier yes, and the stop it makes until the vision target exists."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from iterate import cli as cli_module
from iterate.adapters.data import linking
from iterate.cli import app
from iterate.core.linker import Proposal

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


def _answers(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[str]:
    """A terminal that answers the pause in order; returns the prompts it was asked."""
    import typer

    asked: list[str] = []
    queue = list(replies)

    def prompt(text: str, default: str = "", show_default: bool = True) -> str:
        asked.append(text)
        return queue.pop(0) if queue else "no"

    monkeypatch.setattr(cli_module, "_stdin_owns_tty", lambda: True)
    monkeypatch.setattr(typer, "prompt", prompt)
    return asked


def test_the_pause_asks_on_a_terminal_and_a_no_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _partial(tmp_path)
    asked = _answers(monkeypatch, ["no"])
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 1
    assert asked == ["yes to continue, no to stop, or say what to change"]
    assert not _data_dir(tmp_path).exists()
    _answers(monkeypatch, ["", "y"])  # an empty line is asked again, not read as anything
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 0, result.output
    assert "note: 95.0% of rows resolved to an image; the rest are dropped" in _plain(result.output)


# ─── the Linker ──────────────────────────────────────────────────────────────


class _NeverBuilt:
    def __init__(self, client: object) -> None:
        raise AssertionError("the Linker must not be built for this folder")


class _FakeLinker:
    """Scripted: each entry is the target column to pick, or None for no plan. Every
    plan it returns is a real measured plan, exactly as the real one's would be."""

    script: ClassVar[list[str | None]] = []
    calls: ClassVar[list[dict[str, Any]]] = []
    built: ClassVar[int] = 0

    def __init__(self, client: object) -> None:
        type(self).built += 1

    def propose(
        self,
        inv: linking.Inventory,
        *,
        refusal: str = "",
        notes: Any = (),
        previous: Any = None,
    ) -> Proposal:
        cls = type(self)
        cls.calls.append(
            {"root": inv.root, "refusal": refusal, "notes": list(notes), "previous": previous}
        )
        target = cls.script.pop(0) if cls.script else None
        if target is None:
            return Proposal(None, "the model gave no answer in the tool's shape")
        try:
            plan_ = linking.plan_from_choice(
                inv,
                table=inv.tables[0],
                key_column="file",
                key_to_file="basename",
                target_column=target,
                source="agent",
            )
        except linking.LinkError as exc:
            return Proposal(None, str(exc))
        return Proposal(plan_, f"picked {target}")


@pytest.fixture
def fake_linker(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLinker]:
    _FakeLinker.script = []
    _FakeLinker.calls = []
    _FakeLinker.built = 0
    monkeypatch.setattr("iterate.core.linker.Linker", _FakeLinker)
    return _FakeLinker


def _ambiguous(root: Path, *, n: int = 24) -> Path:
    for i in range(n):
        _png(root / "img" / f"{i:03d}.jpg", i)
    pd.DataFrame(
        {
            "file": [f"{i:03d}.jpg" for i in range(n)],
            "region": ["north", "south", "east"] * (n // 3),
            "species_code": ["x", "y"] * (n // 2),
            "photographer": [f"person {i}" for i in range(n)],
        }
    ).to_csv(root / "meta.csv", index=False)
    return root


def test_a_rules_folder_never_builds_the_linker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("iterate.core.linker.Linker", _NeverBuilt)
    source = _class_tree(tmp_path / "pets")
    result = runner.invoke(app, ["run", "--data", str(source)])
    assert result.exit_code == 0, result.output


def test_a_structural_refusal_never_calls_the_linker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("iterate.core.linker.Linker", _NeverBuilt)
    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    result = runner.invoke(app, ["run", "--data", str(tmp_path)])
    assert result.exit_code != 0
    assert "unpack it to images first" in _plain(result.output)


def test_an_ambiguous_folder_goes_to_the_linker_and_needs_a_yes_in_a_script(
    tmp_path: Path, fake_linker: type[_FakeLinker]
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["species_code"]
    refused = runner.invoke(app, ["run", "--data", str(root)])
    assert refused.exit_code != 0
    text = _plain(refused.output)
    assert "target 'species_code'" in text
    assert "linked by: agent" in text
    assert "the Linker: picked species_code" in text
    assert "re-run with --yes" in text
    assert fake_linker.calls[0]["refusal"].endswith("one folder per class")
    assert not _data_dir(tmp_path).exists()

    fake_linker.script = ["species_code"]
    accepted = runner.invoke(app, ["run", "--data", str(root), "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert "linked:" in _plain(accepted.output)
    plans = list((_data_dir(tmp_path) / "plans").iterdir())
    assert len(plans) == 1
    assert plans[0].name.startswith("birds-")


def test_a_remembered_plan_never_goes_through_the_linker_twice(
    tmp_path: Path, fake_linker: type[_FakeLinker]
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["species_code"]
    assert runner.invoke(app, ["run", "--data", str(root), "--yes"]).exit_code == 0
    assert fake_linker.built == 1

    again = runner.invoke(app, ["run", "--data", str(root)])  # no --yes, no terminal
    assert again.exit_code == 0, again.output
    assert fake_linker.built == 1
    text = _plain(again.output)
    assert "linked by: agent" in text
    assert "remembered from an earlier yes; delete" in text


def test_an_edited_table_forgets_the_plan(tmp_path: Path, fake_linker: type[_FakeLinker]) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["species_code", "region"]
    assert runner.invoke(app, ["run", "--data", str(root), "--yes"]).exit_code == 0
    frame = pd.read_csv(root / "meta.csv")
    frame["species_code"] = ["y", "x"] * 12
    frame.to_csv(root / "meta.csv", index=False)
    again = runner.invoke(app, ["run", "--data", str(root), "--yes"])
    assert again.exit_code == 0, again.output
    assert fake_linker.built == 2
    assert "target 'region'" in _plain(again.output)


def test_a_correction_goes_back_to_the_linker_and_a_yes_ends_it(
    tmp_path: Path, fake_linker: type[_FakeLinker], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["region", "species_code"]
    asked = _answers(monkeypatch, ["the label is species_code, not region", "yes"])
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 0, result.output
    assert len(asked) == 2
    second = fake_linker.calls[1]
    assert second["notes"] == ["the label is species_code, not region"]
    assert second["previous"].target_column == "region"
    text = _plain(result.output)
    assert "target 'region'" in text
    assert "target 'species_code'" in text
    assert "the Linker: picked species_code" in text
    ws = next(p for p in _data_dir(tmp_path).iterdir() if p.name != "plans")
    assert sorted(q.name for q in (ws / "train").iterdir()) == ["x", "y"]


def test_a_no_at_the_pause_writes_nothing(
    tmp_path: Path, fake_linker: type[_FakeLinker], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["region"]
    _answers(monkeypatch, ["no"])
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 1
    assert not _data_dir(tmp_path).exists()


def test_a_correction_the_linker_cannot_use_is_asked_again(
    tmp_path: Path, fake_linker: type[_FakeLinker], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["region", None, "species_code"]
    _answers(monkeypatch, ["something unclear", "use species_code", "y"])
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code == 0, result.output
    text = _plain(result.output)
    assert "the Linker could not turn that into a plan: the model gave no answer" in text
    assert "target 'species_code'" in text
    assert fake_linker.calls[2]["notes"] == ["something unclear", "use species_code"]


def test_five_corrections_and_no_yes_is_a_refusal(
    tmp_path: Path, fake_linker: type[_FakeLinker], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = [
        "region",
        "species_code",
        "region",
        "species_code",
        "region",
        "species_code",
    ]
    _answers(monkeypatch, ["again"] * 6)
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code != 0
    assert "5 corrections and no plan you accepted" in _plain(result.output)
    assert len(fake_linker.calls) == 6  # the first look and five corrections
    assert not _data_dir(tmp_path).exists()


def test_the_linker_failing_at_the_first_look_is_a_refusal_with_both_reasons(
    tmp_path: Path, fake_linker: type[_FakeLinker]
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = [None]
    result = runner.invoke(app, ["run", "--data", str(root)])
    assert result.exit_code != 0
    text = _plain(result.output)
    assert "no rule links them" in text
    assert "the Linker could not settle it either: the model gave no answer" in text


def test_the_pair_form_takes_the_first_folders_choice_for_the_second(
    tmp_path: Path, fake_linker: type[_FakeLinker]
) -> None:
    a = _ambiguous(tmp_path / "train")
    b = _ambiguous(tmp_path / "test", n=12)
    fake_linker.script = ["species_code"]
    result = runner.invoke(app, ["run", "--train", str(a), "--holdout", str(b), "--yes"])
    assert result.exit_code == 0, result.output
    assert len(fake_linker.calls) == 1
    text = _plain(result.output)
    assert "note: columns taken from the other folder's meta.csv" in text
    assert "split: yours, from the folders, 24 train / 12 holdout" in text
    assert "linked by: agent" in text
    root = next(p for p in _data_dir(tmp_path).iterdir() if p.name != "plans")
    assert len(pd.read_csv(root / "holdout.csv")) == 12


def test_flags_still_beat_a_remembered_plan_and_the_linker(
    tmp_path: Path, fake_linker: type[_FakeLinker]
) -> None:
    root = _ambiguous(tmp_path / "birds")
    fake_linker.script = ["species_code"]
    assert runner.invoke(app, ["run", "--data", str(root), "--yes"]).exit_code == 0
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(root),
            "--labels",
            str(root / "meta.csv"),
            "--key",
            "file",
            "--target",
            "region",
        ],
    )
    assert result.exit_code == 0, result.output
    text = _plain(result.output)
    assert "target 'region'" in text
    assert "linked by: flags" in text
    assert fake_linker.built == 1


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
