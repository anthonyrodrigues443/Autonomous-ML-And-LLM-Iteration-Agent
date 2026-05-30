"""Tests for the centralized prompt registry."""

from __future__ import annotations

from iterate.prompts import PROMPTS


def test_prompts_loads_proposer_block() -> None:
    proposer = PROMPTS["proposer"]
    # System carries the placeholders the Proposer formats in.
    for placeholder in ("{metric}", "{direction}", "{current_model}"):
        assert placeholder in proposer["system"]
    # User template carries its own placeholders.
    for placeholder in (
        "{data_summary}",
        "{current_model}",
        "{metric}",
        "{score}",
        "{direction}",
        "{history_section}",
    ):
        assert placeholder in proposer["user_template"]
    # Tool wording is keyed cleanly.
    tool = proposer["tool"]
    assert tool["name"] == "propose_candidate"
    assert "model" in tool["fields"]
    assert "rationale" in tool["fields"]
    # Retry nudge is present.
    assert "propose_candidate" in proposer["retry_nudge"]
