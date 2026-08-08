"""What a prompt IS, how a row gets into it, and the baseline it starts from.

The prompt is the artifact a `PromptTarget` run optimises, the way a pipeline is
what a `ModelTarget` run optimises. It is deliberately two plain strings — a system
message and a user template with placeholders — rather than a structured bundle of
levers. Few-shot examples, output rules and role framing are all things the agent
writes INTO those strings, so the solution space stays open and the harness never
has to grow a field every time a prompting technique is invented.

Two rules here are load-bearing.

*The baseline must not be a strawman.* A deliberately terrible starting prompt
scores near zero and makes every later change look like a triumph, which is the
same failure as measuring against a ceiling that is too low. So the baseline states
the task, states the allowed answers, asks for one, and stops: the simplest prompt a
competent person writes in ten seconds.

*Substitution happens once.* A record's own text routinely contains braces, and a
naive `str.format` (or a second pass over already-substituted text) would try to
interpret them as placeholders — turning a user's data into a crash or, worse, into
prompt structure. Every placeholder is resolved in a single regex pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# `{input}` renders every input column. `{some_column}` renders one of them.
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_ .-]*)\}")
ALL_INPUTS = "input"

# Longer than this and a single cell of user data would dominate the prompt. Trimmed
# with a visible marker rather than silently, so a truncated record cannot look like
# a complete one the model simply got wrong.
_MAX_FIELD_CHARS = 4000


@dataclass(frozen=True)
class Prompt:
    """A system message plus a user template. The unit that gets iterated."""

    system: str
    user_template: str

    def rendered_for(self, row: Mapping[str, Any], columns: Sequence[str]) -> str:
        return render(self.user_template, row, columns)

    def as_dict(self) -> dict[str, str]:
        return {"system": self.system, "user_template": self.user_template}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Prompt:
        return cls(
            system=str(payload.get("system") or ""),
            user_template=str(payload.get("user_template") or ""),
        )


def _clip(text: str) -> str:
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return (
        text[:_MAX_FIELD_CHARS] + f"\n[... trimmed, {len(text) - _MAX_FIELD_CHARS} more characters]"
    )


def render_value(value: Any) -> str:
    """One field as prompt text.

    A column holding a JSON object becomes `key: value` lines rather than raw JSON.
    Users arrive with both shapes — several columns, or one column carrying a JSON
    record — and the model reads labelled fields far better than it reads a brace
    soup, so the two shapes are made to look the same by the time they reach it.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return _clip(text)
        if isinstance(parsed, dict):
            return _clip("\n".join(f"{k}: {v}" for k, v in parsed.items()))
    return _clip(text)


def render_all(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    """Every input column as labelled lines. What `{input}` expands to.

    A single column is rendered bare: labelling a lone `text` field with "text:" adds
    noise to every call for no information.
    """
    usable = [c for c in columns if c in row]
    if len(usable) == 1:
        return render_value(row[usable[0]])
    return "\n".join(f"{name}: {render_value(row[name])}" for name in usable)


def render(template: str, row: Mapping[str, Any], columns: Sequence[str]) -> str:
    """Fill a template's placeholders from one record, in a single pass.

    Unknown placeholders are left exactly as written. That is deliberate: a prompt
    saying `{format}` as English is far more likely than the agent inventing a column
    name, and blanking it would quietly delete an instruction.
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == ALL_INPUTS:
            return render_all(row, columns)
        if name in row:
            return render_value(row[name])
        return match.group(0)

    return _PLACEHOLDER.sub(substitute, template)


def baseline_prompt(task: str, labels: Sequence[str] | None = None) -> Prompt:
    """The starting point every later prompt is measured against.

    Intentionally plain: no examples, no reasoning instructions, no role play, no
    output lecture. It names the task, names the allowed answers so the reply is
    usable at all, and asks for one. Anything more and the baseline stops being a
    floor and starts being a competitor.
    """
    system = task.strip() or "Answer the question about the record."
    if labels:
        system += "\n\nAnswer with exactly one of: " + ", ".join(str(label) for label in labels)
    return Prompt(system=system, user_template="{input}")


def describe(prompt: Prompt) -> str:
    """A short, stable description for the experiment record and the ledger."""
    system_words = len(prompt.system.split())
    return f"prompt ({system_words} words system, template {prompt.user_template[:40]!r})"


__all__ = [
    "ALL_INPUTS",
    "Prompt",
    "baseline_prompt",
    "describe",
    "render",
    "render_all",
    "render_value",
]
