"""The deterministic experiment dossier — what a finished session can be known to
have done, without asking an LLM.

The Summarizer already turns a session into an `ExperimentDigest`, but that record
is LLM-authored: a degraded or failed summarizer call loses the whole experiment's
knowledge, and every field is a claim rather than an observation. This module
extracts the part that needs no judgment — what the session printed, what it
instantiated, what it scored, how it failed — so that record always exists.

The hard rule here is that a dossier NEVER invents. Every fact it carries is a line
the session actually printed or a value computed from the cell records, quoted or
counted, never paraphrased. That is what makes it safe as the Summarizer's input
(v0.4 Day 4) and as its fallback: a fallback that could hallucinate would be worse
than no fallback at all.

Deliberately NOT wired into the supervisor's planning prompt. A same-model A/B in
June measured additive supervisor context regressing the floor model (best 0.6325
to ~0.61, collapsing onto one lever 5-6 of 10 iterations) and the feature was
reverted; the banked lesson is deterministic machinery over prompt nudges. This
record earns its keep by feeding the Summarizer and the ledger, not by being read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from iterate.core import codegen

# A shape tuple as pandas/numpy prints it: (1234, 15). The single most reliable
# data fact in agent stdout — models print shapes constantly and in one format.
_SHAPE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")
_FLOAT = re.compile(r"-?\d+\.\d+")
# Words that mark a printed line as an observation about the DATA rather than
# progress chatter. Matched on the line, not the code, so it reflects what the
# session actually learned.
_DATA_WORDS = (
    "shape",
    "loaded",
    "dtype",
    "missing",
    "null",
    "nan",
    "balance",
    "class",
    "unique",
    "cardinality",
    "rows",
    "columns",
)
_MAX_FACTS = 12
_MAX_FAILURES = 6
_LINE_CAP = 160


@dataclass(frozen=True)
class Dossier:
    """One finished experiment, as observed rather than as described."""

    techniques: list[str] = field(default_factory=list)
    data_facts: list[str] = field(default_factory=list)
    val_trail: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    score: float | None = None
    cells_run: int = 0
    cells_errored: int = 0
    cells_timed_out: int = 0

    @property
    def clean(self) -> bool:
        """No errored or timed-out cells — the session ran straight through."""
        return self.cells_errored == 0 and self.cells_timed_out == 0

    def render(self) -> str:
        """A capped text block, for the Summarizer's input and its fallback.

        Not for the planning prompt: see the module docstring on the June
        regression. Callers that want this in front of the supervisor should have a
        same-model before/after measurement to point at.
        """
        lines: list[str] = []
        if self.techniques:
            lines.append("used: " + ", ".join(self.techniques))
        if self.data_facts:
            lines.append("observed: " + " | ".join(self.data_facts))
        if self.val_trail:
            lines.append("validation: " + " -> ".join(f"{v:.4f}" for v in self.val_trail))
        if self.failures:
            lines.append("failed: " + " | ".join(self.failures))
        shape = f"{self.cells_run} cells"
        if self.cells_errored:
            shape += f", {self.cells_errored} errored"
        if self.cells_timed_out:
            shape += f", {self.cells_timed_out} timed out"
        lines.append("session: " + shape)
        return "\n".join(lines)


def _cell_dicts(experiment: Any) -> list[dict[str, Any]]:
    cells = experiment.candidate.changes.get("cells")
    if not isinstance(cells, list):
        return []
    return [c for c in cells if isinstance(c, dict)]


def _error_signature(error: str) -> str:
    """The last non-empty traceback line — the `ExceptionType: message` — so two
    cells that failed the same way collapse to one entry. Mirrors the coder's own
    breaker signature so the two agree on what "the same failure" means."""
    lines = [ln.strip() for ln in (error or "").splitlines() if ln.strip()]
    return lines[-1][:_LINE_CAP] if lines else ""


def _data_facts(cells: list[dict[str, Any]]) -> list[str]:
    """Lines the session printed that state something about the data.

    Quoted verbatim (trimmed), never rewritten. A line qualifies if it carries a
    shape tuple or names a data property — which keeps ordinary progress chatter
    ("fitting model", "done") out without needing to understand the line.
    """
    facts: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        for raw in (cell.get("stdout") or "").splitlines():
            line = raw.strip()
            if not line or len(line) > _LINE_CAP:
                continue
            low = line.lower()
            if not (_SHAPE.search(line) or any(w in low for w in _DATA_WORDS)):
                continue
            # A validation score is a result, not a data fact; it has its own field.
            if "val" in low or "score" in low:
                continue
            if line in seen:
                continue
            seen.add(line)
            facts.append(line)
    return facts[:_MAX_FACTS]


def _val_trail(cells: list[dict[str, Any]]) -> list[float]:
    """Validation scores the session printed, in order, consecutive repeats dropped."""
    trail: list[float] = []
    for cell in cells:
        for raw in (cell.get("stdout") or "").splitlines():
            low = raw.lower()
            if "val" not in low and "score" not in low:
                continue
            found = _FLOAT.findall(raw)
            if not found:
                continue
            value = float(found[-1])
            if not trail or trail[-1] != value:
                trail.append(value)
    return trail[-8:]


def build(experiment: Any) -> Dossier:
    """Distil one finished experiment into its observed record. Never raises: a
    malformed or half-written session yields a thinner dossier, not an exception,
    because this is the path that has to work when everything else degraded."""
    cells = _cell_dicts(experiment)
    agent_cells = [c for c in cells if c.get("source") == "agent"]

    code = experiment.candidate.changes.get("code")
    techniques = codegen.components_used(code) if isinstance(code, str) else []

    failures: list[str] = []
    for cell in agent_cells:
        signature = _error_signature(cell.get("error") or "")
        if signature and signature not in failures:
            failures.append(signature)

    result = getattr(experiment, "result", None)
    score = (
        result.metrics.primary_value
        if result is not None and result.succeeded and result.metrics is not None
        else None
    )

    return Dossier(
        techniques=techniques,
        data_facts=_data_facts(agent_cells),
        val_trail=_val_trail(agent_cells),
        failures=failures[:_MAX_FAILURES],
        score=score,
        cells_run=len(agent_cells),
        cells_errored=sum(1 for c in agent_cells if c.get("error")),
        cells_timed_out=sum(1 for c in agent_cells if c.get("timed_out")),
    )


__all__ = ["Dossier", "build"]
