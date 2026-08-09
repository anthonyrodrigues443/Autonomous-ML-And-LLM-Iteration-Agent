"""Building the command for one cell, and gating flags by version."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evals.config import Conditions
from evals.corpus import Dataset
from evals.runner import cell_dir, command_for, parse_version, supports

pytestmark = pytest.mark.unit

CONDITIONS = Conditions(
    model="gemma4:12b",
    backend="ollama",
    max_iterations=10,
    patience=3,
    repeats=3,
    timeout_minutes=90,
    sweep_threads=4,
)

DATASET = Dataset(
    name="churn",
    path=Path("examples/churn_tabular/data.clean.csv"),
    target="Churn",
    metric="average_precision",
)


def test_dev_runs_the_working_tree() -> None:
    argv = command_for("dev", DATASET, CONDITIONS)

    assert argv[:2] == ["uv", "run"]
    assert "--with" not in argv


def test_a_released_version_runs_in_its_own_environment() -> None:
    """--no-project matters: without it uv resolves this repo's package and the
    working tree shadows the version under test."""
    argv = command_for("0.4.0", DATASET, CONDITIONS)

    assert "--no-project" in argv
    assert "iterate-ai==0.4.0" in argv


def test_the_metric_is_always_explicit() -> None:
    """From v0.4 the agent may pick its own metric. Letting it would mean two
    versions scoring on different rulers, so the harness always names one."""
    for version in ("0.1.3", "0.3.1", "0.4.0", "dev"):
        argv = command_for(version, DATASET, CONDITIONS)
        assert argv[argv.index("--metric") + 1] == "average_precision"


def test_the_budget_is_pinned_on_every_version() -> None:
    argv = command_for("dev", DATASET, CONDITIONS)

    assert argv[argv.index("--max-iterations") + 1] == "10"
    assert argv[argv.index("--patience") + 1] == "3"
    assert argv[argv.index("--model") + 1] == "gemma4:12b"


def test_flags_a_version_predates_are_left_out() -> None:
    assert "--plain" not in command_for("0.2.1", DATASET, CONDITIONS)
    assert "--plain" in command_for("0.3.1", DATASET, CONDITIONS)


def test_the_data_path_is_absolute() -> None:
    """The subprocess runs from the repo root but the path must survive any cwd."""
    argv = command_for("dev", DATASET, CONDITIONS)

    assert Path(argv[argv.index("--data") + 1]).is_absolute()


def test_dev_sorts_above_every_release() -> None:
    assert parse_version("dev") > parse_version("0.9.9")
    assert parse_version("0.4.0") > parse_version("0.3.1")


def test_a_malformed_version_raises() -> None:
    with pytest.raises(ValueError, match=r"x\.y\.z"):
        parse_version("0.4")


def test_supports_defaults_to_yes_for_an_unmapped_flag() -> None:
    assert supports("0.1.0", "--data")


def test_each_cell_gets_its_own_directory(tmp_path: Path) -> None:
    """Sharing one would let a previous cell's memory carry over as the next cell's
    baseline, and the grid would measure the order the cells ran in."""
    first = cell_dir("dev", "churn", 1, tmp_path)
    second = cell_dir("dev", "churn", 2, tmp_path)
    other_version = cell_dir("0.4.0", "churn", 1, tmp_path)

    assert len({first, second, other_version}) == 3


def test_a_released_version_uses_the_distribution_name() -> None:
    """`iterate-ai` is the distribution; `iterate` is only the console script and
    the import. Getting it wrong failed cell 1 of the first grid with "no version of
    iterate==0.1.3", which reads like a missing release rather than a wrong name."""
    argv = command_for("0.1.3", DATASET, CONDITIONS)

    assert "iterate-ai==0.1.3" in argv
    assert "iterate==0.1.3" not in argv


def test_a_run_that_never_created_a_database_is_recorded_not_raised(tmp_path: Path) -> None:
    """`run_cell` promises never to raise so a long sweep survives one bad cell.
    sqlite3.Error is not an OSError, so a run that died before creating its database
    raised straight through and killed a 24-cell sweep on the first one.
    """
    from evals.runner import CellSpec, run_cell

    dataset = Dataset(
        name="missing",
        path=tmp_path / "nope.csv",
        target="y",
        metric="f1",
    )
    spec = CellSpec(version="dev", dataset=dataset, repeat=1)

    # No CSV, so the subprocess fails instantly and writes no memory.db.
    (tmp_path / "nope.csv").write_text("a,y\n1,x\n", encoding="utf-8")
    record = run_cell(
        spec,
        replace(CONDITIONS, timeout_minutes=1),
        work_dir=tmp_path / "work",
        repo_root=tmp_path,
    )

    assert record.status != "ok"
    assert record.error  # the reason is recorded rather than thrown
