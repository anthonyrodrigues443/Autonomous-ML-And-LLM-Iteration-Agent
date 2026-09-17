"""Whether a path is inside the data the user gave.

A run reads a data file only when its physical path, every link on the way followed,
sits under the physical path of the folder the user named. Nothing here opens a file.
"""

from __future__ import annotations

import functools
import os
import shlex
import sys
from pathlib import Path


class Paths:
    """Physical paths under one root. The folder lookups are cached, so a CSV of many
    rows in one folder costs one realpath of that folder and one lstat a row."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.path.realpath(root)
        self._real_dir = functools.cache(os.path.realpath)
        self._dir_within = functools.cache(self._uncached_within)

    def physical(self, value: str) -> str:
        """Where ``value`` leads, joined to the root when relative, links followed."""
        path = os.path.join(self.root, value)
        head, name = os.path.split(path)
        if name in ("", ".", ".."):
            return os.path.realpath(path)
        real = os.path.join(self._real_dir(head), name)
        return os.path.realpath(real) if os.path.islink(real) else real

    def within(self, real: str) -> bool:
        """True when a physical path sits under the root, the root itself excluded."""
        return real != self.root and self._dir_within(os.path.dirname(real))

    def link_out(self, value: str) -> str | None:
        """The first link along ``value`` that leads outside the root, or None when the
        value leaves without one, by ``..`` or an absolute path."""
        rel = os.path.relpath(value, self.root) if os.path.isabs(value) else value
        probe, depth = self.root, 0
        for part in Path(rel).parts:
            depth += -1 if part == ".." else 1
            if depth < 0:
                return None
            probe = os.path.join(probe, part)
            if os.path.islink(probe):
                real = os.path.realpath(probe)
                if real != self.root and not self.within(real):
                    return probe
        return None

    def _uncached_within(self, real_dir: str) -> bool:
        if real_dir == self.root or real_dir.startswith(self.root.rstrip(os.sep) + os.sep):
            return True
        # APFS and NTFS fold case, so one folder can be spelled two ways.
        try:
            top = os.stat(self.root)
        except OSError:
            return False
        probe = real_dir
        while True:
            try:
                st = os.stat(probe)
            except OSError:
                st = None
            if st is not None and (st.st_dev, st.st_ino) == (top.st_dev, top.st_ino):
                return True
            parent = os.path.dirname(probe)
            if parent == probe:
                return False
            probe = parent


def copy_command(folder: str) -> str:
    """A pasteable copy of ``folder`` beside it with every link replaced by its target."""
    flags = "-RLc" if sys.platform == "darwin" else "-RL --reflink=auto"
    return f"cp {flags} {shlex.quote(folder)} {shlex.quote(folder.rstrip(os.sep) + '-copy')}"


__all__ = ["Paths", "copy_command"]
