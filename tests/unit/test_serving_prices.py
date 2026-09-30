"""The shipped price snapshot: dated, sourced, and consistent with the estimator."""

from __future__ import annotations

from datetime import date

import pytest

from iterate.core import prices, serving
from iterate.targets import net

pytestmark = pytest.mark.unit


def test_the_snapshot_ships_and_is_dated() -> None:
    table = prices.shipped()

    assert date.fromisoformat(table.snapshot_date) <= date.today()
    assert table.hosts
    assert table.api_models


def test_every_row_says_where_it_came_from_and_when() -> None:
    table = prices.shipped()

    for row in [*table.hosts, *table.api_models]:
        assert row.source.startswith("https://"), row.name
        assert date.fromisoformat(row.read_on) <= date.fromisoformat(table.snapshot_date)


def test_each_cloud_has_a_cpu_box_and_a_gpu_box() -> None:
    table = prices.shipped()

    for cloud in ("aws", "gcp", "azure"):
        assert any(h.cloud == cloud for h in table.machines("cpu")), cloud
        assert any(h.cloud == cloud for h in table.machines("gpu")), cloud


def test_a_gpu_rate_is_only_claimed_where_a_plain_torch_number_was_published() -> None:
    table = prices.shipped()
    rated = {h.name for h in table.machines("gpu") if h.resnet50_224_ms is not None}

    assert rated == {"g4dn.xlarge", "NC4as_T4_v3", "n1-standard-4 + T4"}


def test_the_reference_latencies_cover_the_backbones_and_the_estimator_families() -> None:
    ref = prices.shipped().cpu_reference

    assert set(ref.backbone_ms) == set(serving.BACKBONE_SIZES)
    assert all(set(sizes) == {"64", "128", "224"} for sizes in ref.backbone_ms.values())
    assert set(ref.tabular_ms) == set(serving.ESTIMATOR_FAMILIES)
    assert ref.tabular_margin == 2.0


def test_the_backbone_table_matches_the_vision_runner() -> None:
    assert set(serving.BACKBONE_SIZES) == set(net.BACKBONES)
    widths = {name: stages[-1][1] for name, stages in net.STAGES.items()}
    assert widths == serving.FEATURES


def test_api_rows_are_the_models_a_prompt_run_can_name() -> None:
    table = prices.shipped()

    assert table.api("openai", "gpt-4o-mini") is not None
    assert table.api("OpenAI", "GPT-4o-mini") is not None
    # Enterprise only now, with no public price.
    assert table.api("groq", "llama-3.3-70b-versatile") is None
    assert table.api("groq", "openai/gpt-oss-20b") is not None
    assert {row.cloud for row in table.api_models} == {
        "openai",
        "together",
        "deepseek",
        "groq",
        "anthropic",
    }


def test_claude_is_priced_under_every_name_the_api_takes() -> None:
    """The lookup is by the name typed, and Haiku 4.5 answers to two."""
    table = prices.shipped()
    alias = table.api("anthropic", "claude-haiku-4-5")
    dated = table.api("anthropic", "claude-haiku-4-5-20251001")

    assert alias is not None
    assert dated is not None
    assert (alias.usd_per_1m_in, alias.usd_per_1m_out) == (1.0, 5.0)
    assert (dated.usd_per_1m_in, dated.usd_per_1m_out) == (1.0, 5.0)


def test_a_shipped_price_is_dated_by_the_day_its_own_rows_were_read() -> None:
    """The table is dated by its newest row; a machine read days before must not look
    as new as that."""
    table = prices.shipped()

    assert table.read_on("aws") == "2026-09-25"
    assert (
        table.provenance()
        == "aws shipped 2026-09-25, azure shipped 2026-09-25, gcp shipped 2026-09-25"
    )
    measured = '{"system": "s", "user_template": "{t}", "tokens_in_per_record": 600.0}'
    facts = serving.facts_from_prompt(measured, provider="openai", model="gpt-4o-mini")
    assert serving.profile(facts, 1000, table).prices_as_of == "openai shipped 2026-09-25"
    assert table.api("anthropic", "claude-haiku-4-5").read_on == table.snapshot_date  # type: ignore[union-attr]


def test_a_claude_prompt_is_estimated_with_the_tool_prompt_anthropic_adds() -> None:
    """Measured, the tokens include it; estimated from the prompt's text, a short
    prompt on Claude would be priced at a fraction of its cost."""
    facts = serving.facts_from_prompt(
        None, provider="anthropic", model="claude-haiku-4-5", prompt_chars=400
    )
    assert facts.tokens_in == 100 + 588
    assert "588 of them the tool prompt Anthropic adds to every call" in facts.basis[0]
    other = serving.facts_from_prompt(
        None, provider="openai", model="gpt-4o-mini", prompt_chars=400
    )
    assert other.tokens_in == 100
