"""The host side of an image run's saved network: which file the run folder keeps after
each experiment, and the move `--output` asks for. No torch; the files are marker bytes."""

from __future__ import annotations

import hashlib
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


def test_a_network_the_winners_recipe_vouches_for_is_delivered(tmp_path: Path) -> None:
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"winner"), kept, is_best=True)
    assert deliver(kept, kept, sha256=hashlib.sha256(b"winner").hexdigest()) == kept
    assert kept.read_bytes() == b"winner"


@pytest.mark.parametrize("recorded", [hashlib.sha256(b"the winner").hexdigest(), ""])
def test_another_trys_network_is_never_delivered_as_the_winners(
    tmp_path: Path, recorded: str
) -> None:
    """A settle that failed leaves the earlier best in the run folder. `""` is a winner
    whose recipe.json names no network at all."""
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"an earlier best"), kept, is_best=True)
    wanted = tmp_path / "models" / "flowers.pt"
    assert deliver(kept, wanted, sha256=recorded) is None
    assert not kept.exists()
    assert not wanted.exists()


def test_a_volume_that_refuses_chmod_still_delivers_the_latest_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(self: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        raise PermissionError(1, "Operation not permitted", str(self))

    monkeypatch.setattr(type(tmp_path), "chmod", refused)
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), kept, is_best=True)
    staged = _slot(tmp_path, b"second")
    settle(staged, kept, is_best=True)
    assert not staged.exists()
    wanted = tmp_path / "models" / "flowers.pt"
    assert deliver(kept, wanted, sha256=hashlib.sha256(b"second").hexdigest()) == wanted
    assert wanted.read_bytes() == b"second"
    assert not kept.exists()


def test_a_read_only_network_is_removed_where_unlink_refuses_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows: unlink fails on a file with no write bit."""
    unlink = type(tmp_path).unlink

    def windows_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.exists() and not _writable(self):
            raise PermissionError(13, "Access is denied", str(self))
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(tmp_path), "unlink", windows_unlink)
    kept = tmp_path / "runs" / "r1" / BEST_MODEL
    settle(_slot(tmp_path, b"first"), kept, is_best=True)
    settle(_slot(tmp_path, b"second"), kept, is_best=True)
    wanted = tmp_path / "models" / "flowers.pt"
    assert deliver(kept, wanted) == wanted
    assert wanted.read_bytes() == b"second"
    assert not kept.exists()

    settle(_slot(tmp_path, b"third"), kept, is_best=True)
    assert deliver(kept, kept, sha256="not the winner's") is None
    assert not kept.exists()

    settle(_slot(tmp_path, b"fourth"), kept, is_best=True)
    settle(_slot(tmp_path, None), kept, is_best=True)
    assert not kept.exists()


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
