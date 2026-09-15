"""Tiny PNG fixtures for the folder tests. The class is folded into the colour, so a
class tree is clean by construction: no two classes share an image's bytes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

CLASSES = ("cat", "dog", "emu")


def png(path: Path, seed: int = 0, *, klass: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), ((seed * 9 + klass * 71) % 255, (60 + klass * 40) % 255, 120)).save(
        path
    )


def class_tree(root: Path, *, per_class: int = 4, seed: int = 0) -> Path:
    for k, c in enumerate(CLASSES):
        for i in range(per_class):
            png(root / c / f"{c}_{seed + i}.png", seed + i, klass=k)
    return root
