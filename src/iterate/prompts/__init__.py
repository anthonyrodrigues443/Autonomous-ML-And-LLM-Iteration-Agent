"""Centralized prompt registry for the iterate agent.

All prompt wording — system prompts, user templates, retry nudges, tool descriptions
— lives in ``prompts.yaml``. Modules import the loaded mapping via
``from iterate.prompts import PROMPTS`` and reference structured keys such as
``PROMPTS["proposer"]["system"]``. Keeping wording out of code lets us iterate on
prompts without touching logic (and lets a reviewer see every prompt in one place).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"

with _PROMPTS_PATH.open(encoding="utf-8") as _f:
    PROMPTS: dict[str, Any] = cast("dict[str, Any]", yaml.safe_load(_f))

__all__ = ["PROMPTS"]
