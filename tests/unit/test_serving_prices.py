"""The shipped price snapshot: dated, sourced, and consistent with the estimator."""

from __future__ import annotations

from datetime import date

import pytest

from iterate.core import serving
from iterate.targets import net

pytestmark = pytest.mark.unit


def test_the_snapshot_ships_and_is_dated() -> None:
    prices = serving.load_prices()

    assert date.fromisoformat(prices.snapshot_date) <= date.today()
    assert prices.hosts
    assert prices.api_models


def test_every_row_says_where_it_came_from_and_when() -> None:
    prices = serving.load_prices()

    for row in [*prices.hosts, *prices.api_models]:
        assert row.source.startswith("https://"), row.name
        assert date.fromisoformat(row.read_on) <= date.fromisoformat(prices.snapshot_date)


def test_each_cloud_has_a_cpu_box_and_a_gpu_box() -> None:
    prices = serving.load_prices()

    for cloud in ("aws", "gcp", "azure"):
        assert any(h.cloud == cloud for h in prices.machines("cpu")), cloud
        assert any(h.cloud == cloud for h in prices.machines("gpu")), cloud


def test_a_gpu_rate_is_only_claimed_where_a_plain_torch_number_was_published() -> None:
    prices = serving.load_prices()
    rated = {h.name for h in prices.machines("gpu") if h.resnet50_224_ms is not None}

    assert rated == {"g4dn.xlarge", "NC4as_T4_v3", "n1-standard-4 + T4"}


def test_the_reference_latencies_cover_the_backbones_and_the_estimator_families() -> None:
    ref = serving.load_prices().cpu_reference

    assert set(ref.backbone_ms) == set(serving.BACKBONE_SIZES)
    assert all(set(sizes) == {"64", "128", "224"} for sizes in ref.backbone_ms.values())
    assert set(ref.tabular_ms) == set(serving.ESTIMATOR_FAMILIES)
    assert ref.tabular_margin == 2.0


def test_the_backbone_table_matches_the_vision_runner() -> None:
    assert set(serving.BACKBONE_SIZES) == set(net.BACKBONES)
    widths = {name: stages[-1][1] for name, stages in net.STAGES.items()}
    assert widths == serving.FEATURES


def test_api_rows_are_the_models_a_prompt_run_can_name() -> None:
    prices = serving.load_prices()

    assert prices.api("openai", "gpt-4o-mini") is not None
    assert prices.api("OpenAI", "GPT-4o-mini") is not None
    assert prices.api("groq", "llama-3.3-70b-versatile") is None
    assert all(row.cloud in {"openai", "together", "deepseek"} for row in prices.api_models)
