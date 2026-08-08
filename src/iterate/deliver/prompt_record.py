"""`prompts.yaml` — the deliverable of a prompt run.

For a tabular run the artifact you leave with is a fitted model. For a prompt run
it is the prompt, and a notebook is the wrong place to keep it: it shows the journey
across dozens of cells but cannot tell you which one held the winner.

The HARNESS writes this file, never the agent. An agent writing its own scoreboard
can mislabel which prompt was best or drift from what it actually ran; the harness
already holds every score, so it owns the record. Same reason `best.json` is
host-written, and the same rule as the dossier: it may be incomplete, it may not be
wrong.

`best: true` is set here and respects the Critic. An experiment whose score was
rejected for a proven leak can never be marked best, because that score is not a
result. Rejected versions still appear, with their reason, since a prompt that
looked good and was not is worth being able to see.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

from iterate.core import codegen
from iterate.core.critic import REJECTED
from iterate.core.prompting import Prompt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iterate.schemas.experiment import Experiment

PROMPTS_YAML = "prompts.yaml"

_HEADER = (
    "Every prompt this run tried, in order, with the score it reached on the sealed\n"
    "holdout. The one marked best is the one to put into production.\n"
    "\n"
    "{input} in a user_template is replaced with the record's input columns.\n"
    "{column_name} is replaced with that one column.\n"
)


class _Block(str):
    """A string that dumps as a YAML block scalar, so prompts stay readable."""


def _represent_block(dumper: yaml.Dumper, data: _Block) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.add_representer(_Block, _represent_block, Dumper=yaml.SafeDumper)


def _submitted_prompt(experiment: Experiment) -> Prompt | None:
    """The prompt the session actually submitted, read from its own artifact.

    Taken from `prompt.json`, which `submit()` writes in the same call that writes
    the predictions — so the prompt recorded here is provably the prompt that
    produced the score recorded next to it.
    """
    result = experiment.result
    if result is None:
        return None
    raw = result.artifacts.get(codegen.PROMPT_JSON)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return Prompt.from_dict(payload) if isinstance(payload, dict) else None


def _score_of(experiment: Experiment) -> float | None:
    result = experiment.result
    if result is None or result.metrics is None:
        return None
    return result.metrics.primary_value


def _rejection(experiment: Experiment) -> str:
    return str(experiment.candidate.changes.get(REJECTED) or "")


def _best_index(entries: list[dict[str, Any]], direction: str) -> int | None:
    scored = [
        (index, entry["score"])
        for index, entry in enumerate(entries)
        if entry.get("score") is not None and not entry.get("rejected")
    ]
    if not scored:
        return None
    picker = max if direction == "maximize" else min
    return picker(scored, key=lambda pair: pair[1])[0]


def build(
    *,
    task: str,
    metric: str,
    direction: str,
    model_under_test: str,
    baseline_prompt: Prompt,
    baseline_score: float | None,
    history: Sequence[Experiment],
) -> str:
    """Render the whole record. Pure, so it is testable without a run."""
    entries: list[dict[str, Any]] = [
        {
            "version": "v0",
            "kind": "baseline",
            "score": baseline_score,
            "changed": "the starting prompt, measured like any other candidate",
            "system": _Block(baseline_prompt.system),
            "user_template": _Block(baseline_prompt.user_template),
        }
    ]

    for position, experiment in enumerate(history, start=1):
        prompt = _submitted_prompt(experiment)
        if prompt is None:
            continue
        rejected = _rejection(experiment)
        entry: dict[str, Any] = {
            "version": f"v{position}",
            "score": _score_of(experiment),
            "changed": experiment.candidate.description,
            "system": _Block(prompt.system),
            "user_template": _Block(prompt.user_template),
        }
        if rejected:
            entry["rejected"] = rejected
        entries.append(entry)

    best = _best_index(entries, direction)
    for index, entry in enumerate(entries):
        entry["best"] = index == best

    document = {
        "task": task,
        "metric": metric,
        "direction": direction,
        "model_under_test": model_under_test,
        "versions": entries,
    }
    body = str(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100))
    return "".join(f"# {line}\n" for line in _HEADER.splitlines()) + "\n" + body


def write(run_dir: Path, **kwargs: Any) -> Path:
    """Write `prompts.yaml` into a run directory and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PROMPTS_YAML
    path.write_text(build(**kwargs), encoding="utf-8")
    return path


__all__ = ["PROMPTS_YAML", "build", "write"]
