"""The image run's saved network on the host: one staged file per session, and the run
folder keeps only the best experiment's. No torch here; `iterate.vision` opens the file.
"""

from __future__ import annotations

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
        best.unlink(missing_ok=True)


def deliver(kept: Path, wanted: Path) -> Path | None:
    """The run folder's network, moved to where `--output` asked for it. None when the
    winner left no file: a file already at `wanted` is then some other run's."""
    if not kept.exists():
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
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
    # Read-only: a re-run notebook cell that writes best_model.pt fails loudly instead
    # of replacing the scored file.
    dst.chmod(_READ_ONLY)
    src.unlink()


__all__ = ["BEST_MODEL", "deliver", "settle"]
