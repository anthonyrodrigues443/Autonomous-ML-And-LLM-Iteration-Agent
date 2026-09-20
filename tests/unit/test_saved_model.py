"""The host side of an image run's saved network: which file the run folder keeps after
each experiment, and the move `--output` asks for. No torch; the files are marker bytes."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from iterate.deliver import saved_model
from iterate.deliver.saved_model import BEST_MODEL, deliver, settle

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _slot(tmp_path: Path, data: bytes | None) -> Path:
    staged = tmp_path / "staging" / BEST_MODEL
    staged.parent.mkdir(exist_ok=True)
    if data is not None:
        staged.write_bytes(data)
    return staged


def _writable(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_a_new_best_moves_its_network_into_the_run_folder(tmp_path: Path) -> None:
    staged, best = _slot(tmp_path, b"first"), tmp_path / "runs" / "r1" / BEST_MODEL
    settle(staged, best, is_best=True)
    assert best.read_bytes() == b"first"
    assert not staged.exists()
    assert not _writable(best)


def test_a_better_try_replaces_a_read_only_best(tmp_path: Path) -> None:
    best = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), best, is_best=True)
    settle(_slot(tmp_path, b"second"), best, is_best=True)
    assert best.read_bytes() == b"second"
    assert not _writable(best)
    assert [p.name for p in best.parent.iterdir()] == [BEST_MODEL]


def test_a_try_that_is_not_the_best_is_dropped_and_the_best_stays(tmp_path: Path) -> None:
    best = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), best, is_best=True)
    staged = _slot(tmp_path, b"a loser")
    settle(staged, best, is_best=False)
    assert best.read_bytes() == b"first"
    assert not staged.exists()


def test_a_new_best_that_left_no_network_takes_the_old_one_away(tmp_path: Path) -> None:
    """An own-model winner: the old file is another try's, and best.json names this one."""
    best = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), best, is_best=True)
    settle(_slot(tmp_path, None), best, is_best=True)
    assert not best.exists()


def test_nothing_staged_and_nothing_kept_is_not_an_error(tmp_path: Path) -> None:
    best = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, None), best, is_best=False)
    settle(_slot(tmp_path, None), best, is_best=True)
    assert not best.parent.exists()


def test_a_copy_that_stops_part_way_leaves_the_earlier_best_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    best = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), best, is_best=True)
    staged = _slot(tmp_path, b"second")

    def stopped(src: Path, dst: Path) -> None:
        dst.write_bytes(b"sec")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", stopped)
    with pytest.raises(OSError, match="No space"):
        settle(staged, best, is_best=True)
    assert best.read_bytes() == b"first"
    assert [p.name for p in best.parent.iterdir()] == [BEST_MODEL]


def test_output_moves_the_network_and_keeps_it_read_only(tmp_path: Path) -> None:
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"winner"), kept, is_best=True)
    wanted = tmp_path / "models" / "flowers.pt"
    assert deliver(kept, wanted) == wanted
    assert wanted.read_bytes() == b"winner"
    assert not _writable(wanted)
    assert not kept.exists()


def test_the_default_output_is_the_file_where_it_already_is(tmp_path: Path) -> None:
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"winner"), kept, is_best=True)
    assert deliver(kept, kept) == kept
    assert deliver(kept, kept.parent / "." / BEST_MODEL) == kept.parent / "." / BEST_MODEL
    assert kept.read_bytes() == b"winner"


def test_a_winner_with_no_network_never_claims_a_file_already_at_output(tmp_path: Path) -> None:
    wanted = tmp_path / "models" / "flowers.pt"
    wanted.parent.mkdir()
    wanted.write_bytes(b"an earlier run's")
    assert deliver(tmp_path / "runs" / "r1" / BEST_MODEL, wanted) is None
    assert wanted.read_bytes() == b"an earlier run's"


def test_the_host_side_loads_no_torch() -> None:
    script = (
        "import sys; import iterate.deliver.saved_model; "
        "print('torch' in sys.modules, 'numpy' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.split() == ["False", "False"]
    assert saved_model.__all__ == ["BEST_MODEL", "deliver", "settle"]
