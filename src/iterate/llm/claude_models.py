"""What each family of Claude models takes in a request, and which a prompt run calls.

Kept apart from the client so a run can be refused before the Anthropic SDK, an
optional extra, is imported.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    """What a family of Claude models takes in a request."""

    # Sent through extra_body: a named argument is a TypeError in the SDK's 1.x line.
    temperature: bool = False
    # Sent as {"type": "disabled"}: these think by default.
    disable_thinking: bool = False
    # tool_choice {"type": "tool"}. Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1
    # answer it with a 400.
    force_tool: bool = False


# By the longest name a model's ID starts with. None: the family thinks before it
# answers and a prompt run cannot yet stop it, while an answer is allowed 64 tokens.
_FAMILIES: dict[str, Rules | None] = {
    "claude-haiku-4-5": Rules(temperature=True, force_tool=True),
    "claude-sonnet-5": Rules(disable_thinking=True, force_tool=True),
    "claude-opus-5": Rules(disable_thinking=True, force_tool=True),
    # Its lowest setting, between_tools, is not tried yet.
    "claude-sonnet-5-5": None,
    "claude-opus-5-5": None,
    "claude-fable-5": None,
    "claude-mythos-5": None,
    "claude-mythos-preview": None,
}

RUNS = tuple(family for family, rules in _FAMILIES.items() if rules is not None)


def _family(model: str) -> str | None:
    matches = [family for family in _FAMILIES if model.startswith(family)]
    return max(matches, key=len) if matches else None


def rules_for(model: str) -> Rules:
    family = _family(model)
    rules = _FAMILIES[family] if family is not None else None
    return rules if rules is not None else Rules()


def refused(model: str) -> str | None:
    """Why a prompt run does not call this Claude model, or None when it does."""
    runs = f"{', '.join(RUNS[:-1])} or {RUNS[-1]}"
    family = _family(model)
    if family is None:
        return f"{model} is not a Claude model iterate knows how to ask yet. Pick {runs}"
    if _FAMILIES[family] is None:
        return (
            f"{model} thinks before it answers, and a prompt run allows an answer 64 "
            f"tokens, which the thinking can use up. Pick {runs}"
        )
    return None


__all__ = ["RUNS", "Rules", "refused", "rules_for"]
