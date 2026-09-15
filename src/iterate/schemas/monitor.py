"""The data report: what the deterministic checks found before a run.

Made by the monitor, shown at the pause, saved beside the plan as ``monitor.json``
and folded into the supervisor's data summary through ``brief``, which carries
counts only. A finding may name a holdout file to the person and on disk, never to
a model.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from iterate.schemas.link import Split  # noqa: TC001  (pydantic reads it at class build)

Severity = Literal["warn", "note", "pass"]
Check = Literal["coverage", "labels", "twins", "lookalikes", "sources", "floors", "groups"]
RANK: dict[Severity, int] = {"warn": 0, "note": 1, "pass": 2}
WORD: dict[Severity, str] = {"warn": "warn", "note": "note", "pass": "passed"}
BRIEF: dict[Check, str] = {
    "coverage": "{count} images have no row or cannot be read",
    "labels": "{count} training images carry more than one label",
    "twins": "{count} holdout images ({share}) are byte-identical to a training image",
    "lookalikes": "{count} holdout images ({share}) look like a training image at a coarse hash",
    "sources": "a second label source disagrees on {count} images",
    "floors": "{count} classes are lopsided or on one side of the split only",
    "groups": "a column groups the rows and {share} of holdout rows share a value with train",
}
EXAMPLES_SHOWN = 3
EXAMPLES = 3  # kept on a finding for the terminal
DETAILS = 1_000  # kept on a finding for disk


class Finding(BaseModel):
    """One check's result. ``examples`` are shown at the pause, ``details`` only on
    disk. ``count`` is what the brief may carry; for labels it is taken on the
    training side and the table only, never inside the holdout. ``share`` is of the
    holdout for twins, lookalikes and groups, of the compared rows for sources, of
    all images for coverage."""

    model_config = ConfigDict(extra="forbid")

    check: Check
    severity: Severity
    summary: str
    way_out: str = ""
    count: int = Field(default=0, ge=0)
    share: float | None = Field(default=None, ge=0.0, le=1.0)
    examples: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)


class DataReport(BaseModel):
    """The seven findings, always, so a pass is a line a person reads. ``dropped``
    lists the holdout images left out because their bytes sit in training: by the
    split itself when the split is ours, at the person's request when it is theirs."""

    model_config = ConfigDict(extra="forbid")

    version: str
    images: int
    train_rows: int
    holdout_rows: int
    split: Split
    seconds: float
    findings: list[Finding]
    dropped: list[str] = Field(default_factory=list)

    @property
    def verdict(self) -> Severity:
        return min((f.severity for f in self.findings), key=RANK.__getitem__, default="pass")

    @property
    def needs_a_look(self) -> bool:
        return self.verdict == "warn"

    def render(self) -> str:
        """The lines under the link block: warns, notes, the dropped line, then one
        line for everything that passed."""
        counts = Counter(f.severity for f in self.findings)
        head = ", ".join(f"{counts[s]} {WORD[s]}" for s in RANK if counts[s])
        lines = [f"checks: {head} ({self.images:,} images read in {self.seconds:.1f} s)"]
        for f in sorted(self.findings, key=lambda f: RANK[f.severity]):
            if f.severity == "pass":
                continue
            line = f"{f.severity}: {f.summary}"
            if f.examples:
                line += "; e.g. " + ", ".join(f.examples[:EXAMPLES_SHOWN])
            if f.way_out:
                line += f". {f.way_out}"
            lines.append(line)
        if self.dropped:
            how = (
                "left out of the holdout because their bytes are in training"
                if self.split == "ours"
                else "dropped from your holdout at your request"
            )
            lines.append(
                f"dropped: {len(self.dropped)} holdout images {how}; listed in monitor.json"
            )
        passed = [f.summary for f in self.findings if f.severity == "pass"]
        if passed:
            lines.append("passed: " + "; ".join(passed))
        return "\n".join(lines)

    def brief(self) -> str:
        """For the supervisor: a fixed template per check filled from count and share
        only; a finding with no count is named and nothing more. Nothing here reads a
        summary, an example or a detail, so no file name and no label can reach a
        model through it."""
        told = []
        for f in self.findings:
            if f.severity == "pass":
                continue
            share = f"{f.share:.1%}" if f.share is not None else "?"
            line = (
                BRIEF[f.check].format(count=f.count, share=share)
                if f.count
                else f"{f.check}: see the report"
            )
            told.append(f"{line} ({f.severity})")
        if self.dropped:
            told.append(
                f"{len(self.dropped)} holdout images left out as byte copies of training images"
            )
        passed = [f.check for f in self.findings if f.severity == "pass"]
        out = "Data checks before the run (host-run, deterministic): " + (
            "; ".join(told) or "nothing found"
        )
        return out + (f". Passed: {', '.join(passed)}." if passed else ".")


__all__ = ["BRIEF", "DETAILS", "EXAMPLES", "RANK", "Check", "DataReport", "Finding", "Severity"]
