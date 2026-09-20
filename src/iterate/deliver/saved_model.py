"""The image run's saved network on the host: one staged file per session, and the run
folder keeps only the best experiment's. No torch here; `iterate.vision` opens the file.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

BEST_MODEL = "best_model.pt"
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def settle(staged: Path, best: Path, *, is_best: bool) -> None:
    """Called after every experiment. A new best's network replaces the old one, a new
    best that left none takes the old one away, and any other try's file is dropped."""
    if not is_best:
        staged.unlink(missing_ok=True)
    elif staged.exists():
        _place(staged, best)
    else:
        _remove(best)


def deliver(kept: Path, wanted: Path, *, sha256: str | None = None) -> Path | None:
    """The run folder's network, moved to where `--output` asked for it. None when the
    winner left no file: a file already at `wanted` is then some other run's. `sha256` is
    what the winner's recipe.json recorded, and a file it does not match is removed: a
    settle that failed leaves an earlier best's network in the run folder."""
    if not kept.exists():
        return None
    if sha256 is not None and _sha256(kept) != sha256:
        _remove(kept)
        return None
    if not (wanted.exists() and wanted.samefile(kept)):
        _place(kept, wanted)
    return wanted


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    try:
        # A copy, then a rename inside one folder: the staging slot can sit on another
        # volume, and a stop part way must leave the earlier file whole.
        shutil.copyfile(src, part)
        if dst.exists():
            # Windows will not replace a read-only file.
            _chmod(dst, _READ_ONLY | stat.S_IWUSR)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
    # Read-only: a re-run notebook cell that writes best_model.pt fails loudly instead
    # of replacing the scored file.
    _chmod(dst, _READ_ONLY)
    _remove(src)


def _remove(path: Path) -> None:
    # Windows will not unlink a read-only file.
    _chmod(path, _READ_ONLY | stat.S_IWUSR)
    path.unlink(missing_ok=True)


def _chmod(path: Path, mode: int) -> None:
    # Some volumes refuse chmod, and the mode is never worth the network.
    with contextlib.suppress(OSError):
        path.chmod(mode)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


__all__ = ["BEST_MODEL", "deliver", "settle"]
