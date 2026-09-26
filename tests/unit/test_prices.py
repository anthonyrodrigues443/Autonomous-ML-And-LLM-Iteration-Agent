"""Live prices: the reducers on fixture rows, the cache, and the merge with the shipped
file. No network here; the fetchers run against stub clients."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from iterate.core import prices, serving

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _own_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


_AWS_COLUMNS = [
    "SKU",
    "TermType",
    "Unit",
    "PricePerUnit",
    "Currency",
    "Location",
    "Instance Type",
    "Current Generation",
    "vCPU",
    "Memory",
    "Tenancy",
    "Operating System",
    "CapacityStatus",
    "GPU",
    "GPU Memory",
    "Pre Installed S/W",
    "Instance Family",
]


def _aws(**over: str) -> dict[str, str]:
    row = {
        "SKU": "ABC123",
        "TermType": "OnDemand",
        "Unit": "Hrs",
        "PricePerUnit": "0.0208",
        "Currency": "USD",
        "Location": "US East (N. Virginia)",
        "Instance Type": "t3.small",
        "Current Generation": "Yes",
        "vCPU": "2",
        "Memory": "2 GiB",
        "Tenancy": "Shared",
        "Operating System": "Linux",
        "CapacityStatus": "Used",
        "GPU": "",
        "GPU Memory": "",
        "Pre Installed S/W": "NA",
        "Instance Family": "General purpose",
    }
    row.update(over)
    return row


# ─── AWS: what the reducer keeps and drops ───────────────────────────────


def test_aws_keeps_on_demand_linux_shared_current_rows_and_nothing_else() -> None:
    rows = [
        _aws(),
        _aws(**{"Operating System": "Windows", "PricePerUnit": "0.03"}),
        _aws(Tenancy="Dedicated", PricePerUnit="0.03"),
        _aws(TermType="Reserved", PricePerUnit="0.01"),
        _aws(**{"Pre Installed S/W": "SQL Std", "PricePerUnit": "0.5"}),
        _aws(CapacityStatus="UnusedCapacityReservation", PricePerUnit="0.01"),
        _aws(Unit="GB", PricePerUnit="0.09"),
        _aws(**{"Current Generation": "No", "Instance Type": "t2.small", "PricePerUnit": "0.02"}),
        _aws(PricePerUnit="0"),
    ]
    hosts = prices.reduce_aws(rows, region="us-east-1", read_on="2026-09-26")

    assert [(h.name, h.usd_per_hour, h.kind) for h in hosts] == [("t3.small", 0.0208, "cpu")]
    assert hosts[0].memory_gb == 2.0
    assert hosts[0].region == "us-east-1"
    assert hosts[0].source == prices.AWS_CSV_URL.format(region="us-east-1")


def test_aws_keeps_the_cheapest_row_per_instance_type() -> None:
    hosts = prices.reduce_aws(
        [_aws(PricePerUnit="0.03"), _aws(PricePerUnit="0.0208")], region="r", read_on="d"
    )

    assert [h.usd_per_hour for h in hosts] == [0.0208]


def test_aws_gpu_rows_get_their_vram_and_only_the_t4_gets_a_rate() -> None:
    rows = [
        _aws(
            **{
                "Instance Type": "g4dn.xlarge",
                "vCPU": "4",
                "Memory": "16 GiB",
                "GPU": "1",
                "GPU Memory": "16 GB",
                "PricePerUnit": "0.526",
                "Instance Family": "GPU instance",
            }
        ),
        _aws(
            **{
                "Instance Type": "g5.xlarge",
                "vCPU": "4",
                "Memory": "16 GiB",
                "GPU": "1",
                "GPU Memory": "",
                "PricePerUnit": "1.006",
                "Instance Family": "GPU instance",
            }
        ),
        _aws(
            **{
                "Instance Type": "inf1.xlarge",
                "vCPU": "4",
                "Memory": "8 GiB",
                "GPU": "1",
                "GPU Memory": "",
                "PricePerUnit": "0.3",
                "Instance Family": "Machine Learning ASIC Instances",
            }
        ),
    ]
    hosts = {h.name: h for h in prices.reduce_aws(rows, region="r", read_on="d")}

    assert hosts["g4dn.xlarge"].kind == "gpu"
    assert hosts["g4dn.xlarge"].vram_gb == 16
    assert hosts["g4dn.xlarge"].resnet50_224_ms == prices.T4_RESNET50_224_MS
    # No GPU memory in the row: the family table supplies it, and no rate.
    assert hosts["g5.xlarge"].vram_gb == 24
    assert hosts["g5.xlarge"].resnet50_224_ms is None
    # An accelerator this table does not know cannot be sized, so it is dropped.
    assert "inf1.xlarge" not in hosts


class _Lines:
    def __init__(self, text: str) -> None:
        self._text = text

    def iter_lines(self) -> Any:
        return iter(self._text.splitlines())

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> _Lines:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_the_aws_file_opens_with_five_metadata_lines_before_the_header() -> None:
    header = ",".join(f'"{c}"' for c in _AWS_COLUMNS)
    body = ",".join(f'"{v}"' for v in _aws().values())
    text = "\n".join(
        [
            '"FormatVersion","v1.0"',
            '"Disclaimer","x"',
            '"Publication Date","d"',
            '"Version","1"',
            '"OfferCode","AmazonEC2"',
            header,
            body,
        ]
    )

    rows = list(prices._stream_csv_rows(_Lines(text)))  # type: ignore[arg-type]

    assert rows[0]["Instance Type"] == "t3.small"
    assert rows[0]["TermType"] == "OnDemand"


class _Client:
    """A stub for httpx.Client: GET answers by URL prefix, stream hands back lines."""

    def __init__(self, pages: list[dict[str, Any]] | None = None, csv_text: str = "") -> None:
        self._pages = pages or []
        self._csv = csv_text
        self.calls: list[str] = []

    def get(self, url: str, params: dict[str, str] | None = None) -> Any:
        self.calls.append(url)
        if url.startswith(prices.TIMM_CSV_URL):
            return _Response(text=self._csv)
        page = self._pages[len([c for c in self.calls if "prices.azure.com" in c]) - 1]
        return _Response(payload=page)

    def stream(self, method: str, url: str) -> _Lines:
        self.calls.append(url)
        return _Lines(self._csv)

    def close(self) -> None:
        return None


class _Response:
    def __init__(self, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_refresh_aws_streams_the_file_reduces_it_and_writes_the_cache() -> None:
    header = ",".join(f'"{c}"' for c in _AWS_COLUMNS)
    body = ",".join(f'"{v}"' for v in _aws().values())
    text = "\n".join(['"a","b"'] * 5 + [header, body])
    logged: list[str] = []

    path = prices.refresh_aws("us-east-1", client=_Client(csv_text=text), log=logged.append)  # type: ignore[arg-type]

    cached = prices.read_cache("aws", "us-east-1")
    assert path.exists()
    assert cached is not None
    assert [h.name for h in cached.hosts] == ["t3.small"]
    assert cached.date == prices.today()
    assert logged[-1].startswith("aws us-east-1: 1 machines")


# ─── Azure: specs from the SKU name, rows from the API ────────────────────


@pytest.mark.parametrize(
    ("sku", "expected"),
    [
        ("Standard_D2s_v5", (2, 8.0, None)),
        ("Standard_D2ls_v5", (2, 4.0, None)),
        ("Standard_D2as_v5", (2, 8.0, None)),
        ("Standard_DC2s_v3", (2, 8.0, None)),
        ("Standard_E4s_v5", (4, 32.0, None)),
        ("Standard_F2s_v2", (2, 4.0, None)),
        ("Standard_B2s", (2, 4.0, None)),
        ("Standard_B1ls", (1, 0.5, None)),
        ("Standard_NC4as_T4_v3", (4, 28.0, "T4")),
        ("Standard_NC64as_T4_v3", (64, 448.0, "T4")),
        ("Standard_NV36ads_A10_v5", (36, 439.9, "A10")),
        ("Standard_NC24ads_A100_v4", (24, 220.1, "A100")),
        ("Standard_NC40ads_H100_v5", (40, 320.0, "H100")),
    ],
)
def test_azure_specs_follow_the_naming_convention(
    sku: str, expected: tuple[int, float, str | None]
) -> None:
    spec = prices.azure_specs(sku)

    assert spec is not None
    assert (spec.vcpu, spec.memory_gb, spec.gpu) == expected


@pytest.mark.parametrize(
    "sku",
    [
        "Standard_NV6ads_A10_v5",  # a sixth of a GPU
        "Standard_NC6s_v3",  # no accelerator token in the name
        "Standard_M128ms",  # a family the table does not know
        "Standard_E4-2s_v5",  # constrained cores
        "Standard_B2als_v2",  # the B v2 sizes are not in the table
        "Standard_NC128ds_xl_RTXPRO6000BSE_v6",  # an accelerator the table does not know
        "Basic_A1",
    ],
)
def test_azure_specs_refuse_what_the_table_cannot_size(sku: str) -> None:
    assert prices.azure_specs(sku) is None


def _azure(**over: Any) -> dict[str, Any]:
    item = {
        "armSkuName": "Standard_D2s_v5",
        "skuName": "D2s v5",
        "productName": "Virtual Machines Dsv5 Series",
        "type": "Consumption",
        "unitOfMeasure": "1 Hour",
        "retailPrice": 0.096,
    }
    item.update(over)
    return item


def test_azure_reducer_keeps_linux_hourly_pay_as_you_go_and_the_cheapest_meter() -> None:
    items = [
        _azure(),
        _azure(retailPrice=0.11),
        _azure(skuName="D2s v5 Spot", retailPrice=0.02),
        _azure(skuName="D2s v5 Low Priority", retailPrice=0.03),
        _azure(productName="Virtual Machines Dsv5 Series Windows", retailPrice=0.19),
        _azure(unitOfMeasure="1 GB", retailPrice=0.001),
        _azure(type="DevTestConsumption", retailPrice=0.05),
        _azure(armSkuName="Standard_M128ms", skuName="M128ms", retailPrice=13.3),
        _azure(armSkuName="Standard_NC4as_T4_v3", skuName="NC4as T4 v3", retailPrice=0.526),
    ]
    hosts, unknown = prices.reduce_azure(items, region="eastus", read_on="2026-09-26")

    by_name = {h.name: h for h in hosts}
    assert set(by_name) == {"D2s_v5", "NC4as_T4_v3"}
    assert by_name["D2s_v5"].usd_per_hour == 0.096
    assert by_name["D2s_v5"].memory_gb == 8.0
    assert by_name["NC4as_T4_v3"].kind == "gpu"
    assert by_name["NC4as_T4_v3"].resnet50_224_ms == prices.T4_RESNET50_224_MS
    assert unknown == 1
    assert "specs read from the SKU name" in (by_name["D2s_v5"].note or "")


def test_refresh_azure_walks_the_pages_and_writes_the_cache() -> None:
    pages = [
        {"Items": [_azure()], "NextPageLink": "https://prices.azure.com/api/retail/prices?page=2"},
        {
            "Items": [_azure(armSkuName="Standard_B2s", skuName="B2s", retailPrice=0.0416)],
            "NextPageLink": None,
        },
    ]
    client = _Client(pages=pages)
    logged: list[str] = []

    prices.refresh_azure("eastus", client=client, log=logged.append)  # type: ignore[arg-type]

    cached = prices.read_cache("azure", "eastus")
    assert cached is not None
    assert sorted(h.name for h in cached.hosts) == ["B2s", "D2s_v5"]
    assert len([c for c in client.calls if "prices.azure.com" in c]) == 2
    assert "2 pages" in logged[-1]


def test_the_background_refresh_does_nothing_while_the_cache_is_fresh() -> None:
    prices.write_cache("azure", "eastus", [], prices.AZURE_API_URL)

    assert prices.is_stale("azure", "eastus") is False
    assert prices.refresh_azure_in_background("eastus") is None


def test_a_cache_written_yesterday_is_stale() -> None:
    path = prices.cache_path("azure", "eastus")
    path.parent.mkdir(parents=True)
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    path.write_text(json.dumps({"date": yesterday, "url": "u", "hosts": []}))

    assert prices.is_stale("azure", "eastus") is True


# ─── timm: every network's size ──────────────────────────────────────────


def test_timm_reducer_keeps_batch_one_rows_with_their_size_and_time() -> None:
    text = (
        "model,infer_img_size,infer_samples_per_sec,infer_step_time,infer_batch_size,"
        "param_count,infer_gmacs,infer_macts\n"
        "resnet18,224,126.23,7.907,1,11.69,1.82,2.48\n"
        "resnet50,224,67.2,14.865,1,25.56,4.11,11.11\n"
        "big,224,1.0,1000.0,256,100.0,50.0,60.0\n"
    )
    rows = prices.reduce_timm(csv.DictReader(text.splitlines()))

    assert [(r.model, r.img_size, r.param_count_m, r.gmacs, r.ms_batch1_cpu) for r in rows] == [
        ("resnet18", 224, 11.69, 1.82, 7.907),
        ("resnet50", 224, 25.56, 4.11, 14.865),
    ]


def test_the_shipped_timm_table_covers_the_pinned_backbones_at_batch_one() -> None:
    sizes, source = prices.timm_sizes()

    assert source.kind == "shipped"
    resnet50 = [r for r in sizes["resnet50"] if r.img_size == 224]
    assert resnet50[0].ms_batch1_cpu == 14.865
    assert resnet50[0].param_count_m == 25.56
    assert sum(len(v) for v in sizes.values()) > 1000


def test_refresh_timm_writes_the_cache_and_the_cache_is_read_first() -> None:
    text = (
        "model,infer_img_size,infer_samples_per_sec,infer_step_time,infer_batch_size,"
        "param_count,infer_gmacs,infer_macts\nonly_one,160,1,2.0,1,3.0,0.4,0.5\n"
    )
    prices.refresh_timm(client=_Client(csv_text=text))  # type: ignore[arg-type]

    sizes, source = prices.timm_sizes()
    assert source.kind == "refreshed"
    assert list(sizes) == ["only_one"]


# ─── the merge: cache over shipped, said per cloud ───────────────────────


def test_load_reads_the_cache_for_a_cloud_that_has_one_and_the_shipped_file_otherwise() -> None:
    fresh = prices.reduce_aws([_aws()], region="us-east-1", read_on=prices.today())
    prices.write_cache("aws", "us-east-1", fresh, "u")

    table = prices.load(("aws", "gcp"))

    assert [h.name for h in table.hosts if h.cloud == "aws"] == ["t3.small"]
    assert len([h for h in table.hosts if h.cloud == "gcp"]) == 4
    assert table.sources["aws"].kind == "refreshed"
    assert table.sources["gcp"].kind == "shipped"
    assert table.provenance(("aws", "gcp")) == (
        f"aws refreshed {prices.today()} (us-east-1), gcp shipped {table.snapshot_date} (us-central1)"
    )


def test_with_no_cache_everything_is_shipped_and_dated() -> None:
    table = prices.load()

    assert {h.cloud for h in table.hosts} == {"aws", "gcp"}
    assert (
        table.provenance()
        == f"aws shipped {table.snapshot_date} (us-east-1), gcp shipped {table.snapshot_date} (us-central1)"
    )


def test_the_profile_line_says_which_list_it_used() -> None:
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 128})
    line = serving.profile(facts, 1000, serving.load_prices(("aws",))).render()[0]

    assert line.endswith(f"prices: aws shipped {prices.shipped().snapshot_date} (us-east-1)")


def test_describe_names_every_cloud_and_the_timm_table() -> None:
    lines = prices.describe()

    assert lines[0].startswith("aws us-east-1: shipped rows")
    assert lines[1].startswith("gcp us-central1: shipped rows")
    assert lines[2].startswith("azure eastus: shipped rows")
    assert lines[3].startswith("timm:")
    assert "api models" in lines[4]


# ─── the Pricer with live lists ──────────────────────────────────────────


def test_a_network_the_researcher_names_is_sized_from_timms_table() -> None:
    facts = serving.facts_from_model_name("convnext_small", 224)

    sizes, _ = prices.timm_sizes()
    row = next(r for r in sizes["convnext_small"] if r.img_size == 224)
    assert facts.weights == int(row.param_count_m * 1e6)
    assert facts.multiply_adds == int(row.gmacs * 1e9)
    assert facts.reference_ms == pytest.approx(row.ms_batch1_cpu * 49.83 / 14.865)
    assert facts.basis[0].startswith("convnext_small at 224 px, sized from timm's published table")


def test_a_timm_name_with_a_prefix_and_another_size_is_scaled() -> None:
    facts = serving.facts_from_model_name("timm/resnet18", 64)

    sizes, _ = prices.timm_sizes()
    nearest = min(sizes["resnet18"], key=lambda r: abs(r.img_size - 64))
    assert facts.backbone == "resnet18"
    assert facts.multiply_adds == int(nearest.gmacs * 1e9 * (64 / nearest.img_size) ** 2)


def test_a_name_timm_never_measured_is_unpriced_with_the_reason() -> None:
    profile = serving.profile(serving.facts_from_model_name("my_net", 224), 1000, prices.load())

    assert (
        profile.unpriced_because
        == "my_net is not in timm's published table, so its size is not known"
    )


def test_a_timm_sized_network_prices_with_the_compiled_caveat() -> None:
    facts = serving.facts_from_model_name("convnext_small", 224)
    profile = serving.profile(facts, 1000, prices.load())

    assert profile.chosen is not None
    assert any("compiled" in line and "can be slower" in line for line in profile.basis)


def test_a_bigger_cpu_box_serves_one_worker_per_two_vcpus() -> None:
    table = prices.load(("aws",))
    ref = table.cpu_reference
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 224})
    two = next(h for h in table.hosts if h.name == "t3.small")
    four = two.model_copy(update={"name": "c6i.xlarge", "vcpu": 4, "usd_per_hour": 0.17})

    on_two = serving._cost_on(two, facts, 1000, ref)
    on_four = serving._cost_on(four, facts, 1000, ref)
    assert on_two is not None
    assert on_four is not None
    assert on_four.capacity_per_hour == pytest.approx(2 * on_two.capacity_per_hour, abs=2)


def test_the_agents_own_network_is_priced_from_timms_table_when_the_name_is_there() -> None:
    facts = serving.facts_from_recipe({"model": "timm/convnext_small", "image_size": 224})
    unknown = serving.facts_from_recipe({"model": "hf_hub:timm/vit_base"})

    assert facts.unpriced_because is None
    assert facts.basis[0].startswith("the agent's own network: convnext_small at 224 px")
    assert unknown.unpriced_because is not None
    assert "the agent's own network" in unknown.unpriced_because
    assert "not in timm's published table" in unknown.unpriced_because


# ─── what review found, kept out for good ────────────────────────────────


@pytest.mark.parametrize(
    "sku", ["Standard_D11_v2", "Standard_D3_v2", "Standard_D2", "Standard_D14_v2"]
)
def test_azure_d_v1_and_v2_names_carry_a_size_code_not_a_vcpu_count_so_they_are_refused(
    sku: str,
) -> None:
    assert prices.azure_specs(sku) is None


def test_azure_f_v6_has_more_memory_per_vcpu_than_older_f() -> None:
    assert prices.azure_specs("Standard_F2s_v2") == prices.AzureSpec(vcpu=2, memory_gb=4.0)
    assert prices.azure_specs("Standard_F2as_v6") == prices.AzureSpec(vcpu=2, memory_gb=8.0)
    assert prices.azure_specs("Standard_F2ams_v6") == prices.AzureSpec(vcpu=2, memory_gb=16.0)


def test_aws_fractional_gpus_and_non_gpu_accelerators_never_land_in_the_cpu_pool() -> None:
    rows = [
        _aws(
            **{
                "Instance Type": "g6f.large",
                "GPU": "0.125",
                "GPU Memory": "3 GB",
                "Instance Family": "GPU instance",
                "PricePerUnit": "0.3",
            }
        ),
        _aws(
            **{
                "Instance Type": "dl1.24xlarge",
                "GPU": "",
                "Instance Family": "Machine Learning ASIC Instances",
                "PricePerUnit": "13.1",
            }
        ),
        _aws(
            **{
                "Instance Type": "f1.2xlarge",
                "GPU": "",
                "Instance Family": "FPGA Instances",
                "PricePerUnit": "1.65",
            }
        ),
        _aws(),
    ]
    hosts = prices.reduce_aws(rows, region="r", read_on="d")

    assert [h.name for h in hosts] == ["t3.small"]


def test_aws_rows_without_the_family_column_are_still_read() -> None:
    row = _aws()
    del row["Instance Family"]

    assert [h.name for h in prices.reduce_aws([row], region="r", read_on="d")] == ["t3.small"]


def test_the_aws_header_is_found_by_content_when_the_metadata_grows() -> None:
    header = ",".join(f'"{c}"' for c in _AWS_COLUMNS)
    body = ",".join(f'"{v}"' for v in _aws().values())
    text = "\n".join(['"a","b"'] * 7 + [header, body])

    rows = list(prices._stream_csv_rows(_Lines(text)))  # type: ignore[arg-type]

    assert rows[0]["Instance Type"] == "t3.small"


def test_a_refresh_that_reduces_to_nothing_raises_and_leaves_the_old_cache() -> None:
    good = prices.reduce_aws([_aws()], region="us-east-1", read_on="d")
    prices.write_cache("aws", "us-east-1", good, "u")
    empty = "\n".join(['"a","b"'] * 5 + [",".join(f'"{c}"' for c in _AWS_COLUMNS)])

    with pytest.raises(ValueError, match="no machines"):
        prices.refresh_aws("us-east-1", client=_Client(csv_text=empty))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no machines"):
        prices.refresh_azure("EastUS", client=_Client(pages=[{"Items": [], "NextPageLink": None}]))  # type: ignore[arg-type]

    cached = prices.read_cache("aws", "us-east-1")
    assert cached is not None
    assert [h.name for h in cached.hosts] == ["t3.small"]
    assert prices.read_cache("azure", "EastUS") is None


def test_a_cache_with_an_unreadable_or_future_date_is_stale_and_never_read() -> None:
    path = prices.cache_path("azure", "eastus")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"date": "yesterday", "url": "u", "hosts": []}))
    assert prices.read_cache("azure", "eastus") is None
    assert prices.is_stale("azure", "eastus") is True

    tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    path.write_text(json.dumps({"date": tomorrow, "url": "u", "hosts": []}))
    assert prices.is_stale("azure", "eastus") is True


def test_cache_writes_are_atomic() -> None:
    path = prices.write_cache("aws", "us-east-1", [], "u")

    assert path.exists()
    assert not path.with_suffix(".json.part").exists()


def test_the_background_refresh_runs_when_the_cache_is_stale_and_swallows_any_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def boom(region: str | None = None, **kw: Any) -> None:
        calls.append(str(region))
        raise AttributeError("a page that was not a dict")

    monkeypatch.setattr(prices, "refresh_azure", boom)
    thread = prices.refresh_azure_in_background("eastus")

    assert thread is not None
    thread.join(timeout=5)
    assert calls == ["eastus"]


def test_the_shipped_fallback_names_the_region_it_is_for() -> None:
    table = prices.load(("aws",), "eu-west-1")

    assert table.sources["aws"].kind == "shipped"
    assert table.sources["aws"].region == "us-east-1"
    assert table.provenance(("aws",)).endswith("(us-east-1)")


def test_refresh_keeps_going_when_one_list_fails_and_says_so_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[str] = []
    monkeypatch.setattr(prices, "refresh_timm", lambda **kw: logged.append("timm ok"))
    monkeypatch.setattr(
        prices, "refresh_azure", lambda *a, **kw: (_ for _ in ()).throw(OSError("down"))
    )
    monkeypatch.setattr(prices, "refresh_aws", lambda *a, **kw: logged.append("aws ok"))

    with pytest.raises(ValueError, match="azure: OSError: down"):
        prices.refresh(None, None, log=logged.append)

    assert "aws ok" in logged
    assert any(line.startswith("azure: not refreshed") for line in logged)


def test_describe_lists_every_cached_region_not_only_the_default() -> None:
    prices.write_cache(
        "aws", "eu-west-1", prices.reduce_aws([_aws()], region="eu-west-1", read_on="d"), "u"
    )

    lines = prices.describe()

    assert any(line.startswith("aws eu-west-1: 1 machines, refreshed") for line in lines)


def test_a_timm_name_with_a_pretrained_tag_is_the_same_network() -> None:
    tagged = serving.facts_from_model_name("timm/resnet50.a1_in1k", 224)
    plain = serving.facts_from_model_name("resnet50", 224)

    assert tagged.weights == plain.weights
    assert tagged.backbone == "resnet50"


def test_a_one_vcpu_box_serves_half_the_reference_rate_and_the_line_says_so() -> None:
    table = prices.load(("aws",))
    ref = table.cpu_reference
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 224})
    two = next(h for h in table.hosts if h.name == "t3.small")
    one = two.model_copy(update={"name": "t3.micro", "vcpu": 1, "usd_per_hour": 0.0104})

    on_two = serving._cost_on(two, facts, 1000, ref)
    on_one = serving._cost_on(one, facts, 1000, ref)
    assert on_two is not None
    assert on_one is not None
    assert on_one.capacity_per_hour == on_two.capacity_per_hour // 2
    assert "half a worker on 1 vCPU" in serving._capacity_basis(on_one, facts, ref)[0]


def test_the_tabular_basis_line_keeps_the_measured_time_on_a_bigger_box() -> None:
    table = prices.load(("aws",))
    ref = table.cpu_reference
    facts = serving.facts_from_code(None, "HistGradientBoostingClassifier()", n_features=5)
    two = next(h for h in table.hosts if h.name == "t3.small")
    four = two.model_copy(update={"name": "c6i.xlarge", "vcpu": 4, "usd_per_hour": 0.17})

    cost = serving._cost_on(four, facts, 1000, ref)
    assert cost is not None
    line = serving._capacity_basis(cost, facts, ref)[0]
    assert line.startswith("one row predicts in about 4.5 ms")
    assert "scaled to 2 workers on 4 vCPUs" in line


def test_a_table_winner_needs_a_gigabyte_so_a_half_gigabyte_box_is_out() -> None:
    table = prices.load(("aws",))
    nano = next(h for h in table.hosts if h.name == "t3.small").model_copy(
        update={"name": "t3.nano", "memory_gb": 0.5, "usd_per_hour": 0.0052}
    )
    table = table.model_copy(update={"hosts": [*table.hosts, nano]})
    facts = serving.facts_from_code(None, "LogisticRegression()", n_features=3)

    profile = serving.profile(facts, 1000, table)
    assert profile.chosen is not None
    assert profile.chosen.host.name != "t3.nano"
