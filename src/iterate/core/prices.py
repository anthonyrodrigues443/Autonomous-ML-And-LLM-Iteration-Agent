"""Live prices: the clouds' own lists, fetched, reduced and cached, with the shipped
snapshot as the fallback that prints its date.

Three clouds, three truths. AWS publishes its price list per region as a file of some
300 MB, so it is fetched only by `iterate prices refresh`, streamed line by line and
reduced on the way to a few kilobytes: on-demand, Linux, shared tenancy, current
generation. Azure's retail price API is open and small, so it is walked page by page,
at a refresh or in the background at the start of a run when the cache is a day old;
it carries no vCPU or memory, so specs are read from the SKU name through Azure's own
naming convention. GCP's catalog needs a key, so its shipped rows stand. timm's
published benchmark table gives every network's size at batch 1 and is fetched the same
way. Nothing here asks a model anything; every row keeps its source and the day it was
read, and `load` says per cloud whether the Pricer is reading a refreshed list or the
shipped one.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import TYPE_CHECKING, Any

import httpx

from iterate import userconfig
from iterate.schemas.serving import Host, Kind, Prices, Source

if TYPE_CHECKING:
    from pathlib import Path

PRICES_FILE = "serving_prices.json"
TIMM_FILE = "timm_sizes.csv"
AWS_CSV_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region}/index.csv"
)
AZURE_API_URL = "https://prices.azure.com/api/retail/prices"
TIMM_CSV_URL = (
    "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/"
    "benchmark-infer-fp32-nchw-pt240-cpu-i9_10940x-dynamo.csv"
)
DEFAULT_CLOUDS: tuple[str, ...] = ("aws", "gcp")
CLOUDS: tuple[str, ...] = ("aws", "gcp", "azure")
DEFAULT_REGIONS: dict[str, str] = {"aws": "us-east-1", "gcp": "us-central1", "azure": "eastus"}
STALE_AFTER = timedelta(days=1)
TIMEOUT = httpx.Timeout(120.0, connect=15.0)
# The one GPU with a published plain-PyTorch batch-1 number (NVIDIA's ResNet-50 v1.5
# README): 10.7 ms. Every other GPU prices one box with its capacity unsaid.
T4_RESNET50_224_MS = 10.7
# AWS's list carries a GPU count and, for most types, the GPU memory; the GPU model is
# read from the instance family. Only the T4 families get the published rate.
AWS_GPU_FAMILIES: dict[str, tuple[str, float]] = {
    "g4dn": ("T4", 16),
    "g5": ("A10G", 24),
    "g6": ("L4", 24),
    "g6e": ("L40S", 48),
    "gr6": ("L4", 24),
    "p3": ("V100", 16),
    "p4d": ("A100", 40),
    "p4de": ("A100", 80),
    "p5": ("H100", 80),
}
# Azure's API carries no specs. Memory per vCPU follows the family letter, and the
# GPU follows the accelerator token in the name. Families not listed are dropped and
# counted, never guessed.
AZURE_GB_PER_VCPU: dict[str, float] = {"D": 4.0, "DC": 4.0, "E": 8.0, "EC": 8.0, "F": 2.0}
AZURE_B_SERIES_GB: dict[str, float] = {
    "B1ls": 0.5,
    "B1s": 1,
    "B1ms": 2,
    "B2s": 4,
    "B2ms": 8,
    "B4ms": 16,
    "B8ms": 32,
    "B12ms": 48,
    "B16ms": 64,
    "B20ms": 80,
}
# (family, accelerator) -> (GPU, VRAM in GB per GPU, the vCPU counts that carry a whole GPU)
AZURE_GPUS: dict[tuple[str, str], tuple[str, float, tuple[int, ...]]] = {
    ("NC", "T4"): ("T4", 16, (4, 8, 16, 64)),
    ("NV", "A10"): ("A10", 24, (36, 72)),
    ("NC", "A100"): ("A100", 80, (24, 48, 96)),
    ("NC", "H100"): ("H100", 94, (40, 80)),
}
_AZURE_SKU = re.compile(
    r"^Standard_(?P<family>[A-Z]+)(?P<vcpu>\d+)(?P<constrained>-\d+)?(?P<letters>[a-z]*)"
    r"(?:_(?P<accel>[A-Z][A-Za-z0-9]*?))?(?:_v(?P<version>\d+))?$"
)
_GB = re.compile(r"([\d.]+)\s*Gi?B", re.IGNORECASE)

Log = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


@dataclass(frozen=True)
class Cached:
    hosts: list[Host]
    date: str
    url: str
    region: str


@dataclass(frozen=True)
class TimmRow:
    model: str
    img_size: int
    param_count_m: float
    gmacs: float
    ms_batch1_cpu: float


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def cache_root() -> Path:
    return userconfig.cache_dir() / "prices"


def cache_path(cloud: str, region: str) -> Path:
    return cache_root() / f"{cloud}-{region}.json"


def timm_cache_path() -> Path:
    return cache_root() / "timm_sizes.json"


def shipped() -> Prices:
    """The snapshot shipped with this version of the package."""
    text = resources.files("iterate.core").joinpath(PRICES_FILE).read_text(encoding="utf-8")
    return Prices.model_validate_json(text)


def region_for(cloud: str, region: str | None) -> str:
    return region or DEFAULT_REGIONS[cloud]


def load(clouds: Sequence[str] | None = None, region: str | None = None) -> Prices:
    """The cache where it exists, the shipped file where it does not, and a source per
    cloud so the profile can say which it used."""
    base = shipped()
    hosts: list[Host] = []
    sources: dict[str, Source] = {}
    for cloud in clouds or DEFAULT_CLOUDS:
        where = region_for(cloud, region)
        cached = read_cache(cloud, where)
        if cached is not None:
            hosts.extend(cached.hosts)
            sources[cloud] = Source(
                kind="refreshed", date=cached.date, url=cached.url, region=where
            )
        else:
            hosts.extend(h for h in base.hosts if h.cloud == cloud)
            sources[cloud] = Source(kind="shipped", date=base.snapshot_date)
    return base.model_copy(update={"hosts": hosts, "sources": sources})


# ─── the cache ───────────────────────────────────────────────────────────


def read_cache(cloud: str, region: str) -> Cached | None:
    path = cache_path(cloud, region)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        hosts = [Host.model_validate(row) for row in raw["hosts"]]
        return Cached(hosts=hosts, date=str(raw["date"]), url=str(raw["url"]), region=region)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_cache(cloud: str, region: str, hosts: Sequence[Host], url: str) -> Path:
    path = cache_path(cloud, region)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "cloud": cloud,
        "region": region,
        "date": today(),
        "url": url,
        "hosts": [h.model_dump() for h in hosts],
    }
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path


def cache_age(cloud: str, region: str) -> timedelta | None:
    cached = read_cache(cloud, region)
    if cached is None:
        return None
    return datetime.now(UTC).date() - datetime.fromisoformat(cached.date).date()


def is_stale(cloud: str, region: str) -> bool:
    age = cache_age(cloud, region)
    return age is None or age >= STALE_AFTER


# ─── AWS: a 300 MB file, streamed and reduced ────────────────────────────


def reduce_aws(rows: Iterable[Mapping[str, str]], *, region: str, read_on: str) -> list[Host]:
    """On-demand, Linux, shared tenancy, nothing pre-installed, current generation, priced
    by the hour in dollars; the cheapest row per instance type when there are several."""
    url = AWS_CSV_URL.format(region=region)
    best: dict[str, Host] = {}
    for r in rows:
        if (r.get("TermType"), r.get("Operating System"), r.get("Tenancy")) != (
            "OnDemand",
            "Linux",
            "Shared",
        ):
            continue
        if r.get("Pre Installed S/W") != "NA" or r.get("CapacityStatus") != "Used":
            continue
        if r.get("Unit") != "Hrs" or r.get("Currency", "USD") != "USD":
            continue
        if r.get("Current Generation") != "Yes" or not r.get("Instance Type"):
            continue
        try:
            price = float(r.get("PricePerUnit") or 0)
            vcpu = int(r.get("vCPU") or 0)
        except ValueError:
            continue
        memory = _gb(r.get("Memory"))
        if price <= 0 or vcpu <= 0 or memory is None:
            continue
        name = str(r["Instance Type"])
        gpus = _int(r.get("GPU"))
        kind: Kind = "gpu" if gpus else "cpu"
        vram: float | None = None
        rate: float | None = None
        if gpus:
            family = name.split(".")[0]
            known = AWS_GPU_FAMILIES.get(family)
            if known is None:
                continue
            model, table_vram = known
            vram = _gb(r.get("GPU Memory")) or table_vram
            # The list gives the memory of all the GPUs together; one model runs on one.
            if gpus > 1 and vram > table_vram * 1.5:
                vram = vram / gpus
            if model == "T4":
                rate = T4_RESNET50_224_MS
        host = Host(
            cloud="aws",
            name=name,
            kind=kind,
            region=region,
            vcpu=vcpu,
            memory_gb=memory,
            vram_gb=vram,
            usd_per_hour=price,
            resnet50_224_ms=rate,
            source=url,
            read_on=read_on,
            note=f"{r.get('Location', region)}, on-demand Linux, from AWS's own price list",
        )
        if name not in best or price < (best[name].usd_per_hour or 0):
            best[name] = host
    return sorted(best.values(), key=lambda h: (h.usd_per_hour or 0, h.name))


def _stream_csv_rows(response: httpx.Response, *, skip: int = 5) -> Iterator[dict[str, str]]:
    """AWS's file opens with five metadata lines before the header."""
    lines = response.iter_lines()
    for _ in range(skip):
        next(lines, None)
    yield from csv.DictReader(lines)


def refresh_aws(
    region: str | None = None, *, client: httpx.Client | None = None, log: Log = _quiet
) -> Path:
    where = region_for("aws", region)
    url = AWS_CSV_URL.format(region=where)
    log(f"aws {where}: fetching AWS's own price list ({url}), about 300 MB, streamed")
    own = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    try:
        with own.stream("GET", url) as response:
            response.raise_for_status()
            hosts = reduce_aws(_stream_csv_rows(response), region=where, read_on=today())
    finally:
        if client is None:
            own.close()
    path = write_cache("aws", where, hosts, url)
    log(f"aws {where}: {len(hosts)} machines, refreshed {today()}")
    return path


# ─── Azure: an open API, specs from the SKU name ─────────────────────────


@dataclass(frozen=True)
class AzureSpec:
    vcpu: int
    memory_gb: float
    gpu: str | None = None
    vram_gb: float | None = None


def azure_specs(sku: str) -> AzureSpec | None:
    """What Azure's naming convention says about a SKU, or None for a family this table
    does not know or a fraction of a GPU."""
    found = _AZURE_SKU.match(sku)
    if found is None or found.group("constrained"):
        return None
    family, vcpu = found.group("family"), int(found.group("vcpu"))
    letters, accel = found.group("letters") or "", found.group("accel")
    if family == "B" and not found.group("version"):
        memory = AZURE_B_SERIES_GB.get(f"B{vcpu}{letters}")
        return None if memory is None else AzureSpec(vcpu=vcpu, memory_gb=memory)
    if accel:
        gpu = AZURE_GPUS.get((family, accel))
        if gpu is None or vcpu not in gpu[2]:
            return None
        model, vram, _ = gpu
        return AzureSpec(vcpu=vcpu, memory_gb=vcpu * 4.0, gpu=model, vram_gb=vram)
    per_vcpu = AZURE_GB_PER_VCPU.get(family)
    if per_vcpu is None:
        return None
    if "l" in letters and family.startswith("D"):
        per_vcpu = 2.0
    return AzureSpec(vcpu=vcpu, memory_gb=vcpu * per_vcpu)


def reduce_azure(
    items: Iterable[Mapping[str, Any]], *, region: str, read_on: str
) -> tuple[list[Host], int]:
    """Linux, pay as you go, priced by the hour; Spot, Low Priority and Windows rows
    dropped; the cheapest meter per SKU. Returns the hosts and how many SKUs were dropped
    because their family is not in the spec table."""
    best: dict[str, float] = {}
    for item in items:
        sku = str(item.get("armSkuName") or "")
        name = str(item.get("skuName") or "")
        if not sku.startswith("Standard_") or item.get("type") != "Consumption":
            continue
        if item.get("unitOfMeasure") != "1 Hour" or "Windows" in str(item.get("productName")):
            continue
        if "Spot" in name or "Low Priority" in name:
            continue
        price = float(item.get("retailPrice") or 0)
        if price <= 0:
            continue
        best[sku] = min(best.get(sku, price), price)
    hosts: list[Host] = []
    unknown = 0
    for sku, price in best.items():
        spec = azure_specs(sku)
        if spec is None:
            unknown += 1
            continue
        hosts.append(
            Host(
                cloud="azure",
                name=sku.removeprefix("Standard_"),
                kind="gpu" if spec.gpu else "cpu",
                region=region,
                vcpu=spec.vcpu,
                memory_gb=spec.memory_gb,
                vram_gb=spec.vram_gb,
                usd_per_hour=price,
                resnet50_224_ms=T4_RESNET50_224_MS if spec.gpu == "T4" else None,
                source=AZURE_API_URL,
                read_on=read_on,
                note=f"{region}, pay as you go Linux, from Azure's retail price API; "
                "specs read from the SKU name",
            )
        )
    return sorted(hosts, key=lambda h: (h.usd_per_hour or 0, h.name)), unknown


def _azure_pages(client: httpx.Client, region: str) -> Iterator[list[dict[str, Any]]]:
    params = {
        "$filter": "serviceName eq 'Virtual Machines' and "
        f"armRegionName eq '{region}' and priceType eq 'Consumption'"
    }
    response = client.get(AZURE_API_URL, params=params)
    response.raise_for_status()
    page = response.json()
    yield list(page.get("Items") or [])
    for _ in range(200):
        link = page.get("NextPageLink")
        if not link:
            return
        response = client.get(link)
        response.raise_for_status()
        page = response.json()
        yield list(page.get("Items") or [])


def refresh_azure(
    region: str | None = None, *, client: httpx.Client | None = None, log: Log = _quiet
) -> Path:
    where = region_for("azure", region)
    own = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    items: list[dict[str, Any]] = []
    pages = 0
    try:
        for page in _azure_pages(own, where):
            items.extend(page)
            pages += 1
    finally:
        if client is None:
            own.close()
    hosts, unknown = reduce_azure(items, region=where, read_on=today())
    path = write_cache("azure", where, hosts, AZURE_API_URL)
    log(
        f"azure {where}: {len(hosts)} machines from Azure's retail price API, {pages} pages, "
        f"refreshed {today()}; {unknown} SKUs dropped, their families are not in the spec table"
    )
    return path


def refresh_azure_in_background(region: str | None = None) -> threading.Thread | None:
    """Started at the top of a run when the Azure cache is a day old: the run ends an hour
    later and prices against today. Silent, daemon, never blocks, and a failure leaves the
    old cache or the shipped rows in place."""
    where = region_for("azure", region)
    if not is_stale("azure", where):
        return None

    def work() -> None:
        try:
            refresh_azure(where)
        except (httpx.HTTPError, OSError, ValueError):
            return

    thread = threading.Thread(target=work, name="iterate-prices-azure", daemon=True)
    thread.start()
    return thread


# ─── timm: every network's size at batch 1 ───────────────────────────────


def reduce_timm(rows: Iterable[Mapping[str, str]]) -> list[TimmRow]:
    out: list[TimmRow] = []
    for r in rows:
        try:
            if int(r.get("infer_batch_size") or 1) != 1:
                continue
            out.append(
                TimmRow(
                    model=str(r["model"]),
                    img_size=int(r["infer_img_size"]),
                    param_count_m=float(r["param_count"]),
                    gmacs=float(r["infer_gmacs"]),
                    ms_batch1_cpu=float(r["infer_step_time"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return out


def refresh_timm(*, client: httpx.Client | None = None, log: Log = _quiet) -> Path:
    own = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    try:
        response = own.get(TIMM_CSV_URL)
        response.raise_for_status()
        rows = reduce_timm(csv.DictReader(response.text.splitlines()))
    finally:
        if client is None:
            own.close()
    path = timm_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"date": today(), "url": TIMM_CSV_URL, "rows": [row.__dict__ for row in rows]}),
        encoding="utf-8",
    )
    log(f"timm: {len(rows)} networks from timm's published table, refreshed {today()}")
    return path


def timm_sizes() -> tuple[dict[str, list[TimmRow]], Source]:
    """Every network timm measured, by name, from the cache when it exists and the
    shipped copy when it does not, with where it came from."""
    path = timm_cache_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = [TimmRow(**row) for row in raw["rows"]]
            return _by_name(rows), Source(kind="refreshed", date=str(raw["date"]), url=raw["url"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    text = resources.files("iterate.core").joinpath(TIMM_FILE).read_text(encoding="utf-8")
    rows = []
    for r in csv.DictReader(text.splitlines()):
        try:
            rows.append(
                TimmRow(
                    model=r["model"],
                    img_size=int(r["img_size"]),
                    param_count_m=float(r["param_count_m"]),
                    gmacs=float(r["gmacs"]),
                    ms_batch1_cpu=float(r["ms_batch1_cpu"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return _by_name(rows), Source(kind="shipped", date=shipped().snapshot_date, url=TIMM_CSV_URL)


def _by_name(rows: Iterable[TimmRow]) -> dict[str, list[TimmRow]]:
    out: dict[str, list[TimmRow]] = {}
    for row in rows:
        out.setdefault(row.model, []).append(row)
    return out


# ─── the two commands ────────────────────────────────────────────────────


def refresh(cloud: str | None, region: str | None, *, log: Log) -> None:
    """`iterate prices refresh`: every cloud, or one, then timm."""
    for name in (cloud,) if cloud else CLOUDS:
        if name == "aws":
            refresh_aws(region, log=log)
        elif name == "azure":
            refresh_azure(region, log=log)
        else:
            rows = [h for h in shipped().hosts if h.cloud == "gcp"]
            log(
                f"gcp: shipped rows stand ({len(rows)} machines, {shipped().snapshot_date}); "
                "GCP's catalog needs a key, which this version does not take"
            )
    refresh_timm(log=log)


def describe() -> list[str]:
    """`iterate prices show`: what the Pricer would read right now, per cloud."""
    base = shipped()
    lines: list[str] = []
    for cloud in CLOUDS:
        region = DEFAULT_REGIONS[cloud]
        cached = read_cache(cloud, region)
        rows = [h for h in base.hosts if h.cloud == cloud]
        if cached is not None:
            lines.append(
                f"{cloud} {region}: {len(cached.hosts)} machines, refreshed {cached.date} "
                f"from {cached.url}"
            )
        else:
            lines.append(
                f"{cloud} {region}: shipped rows ({len(rows)} machines, {base.snapshot_date}); "
                f"run `iterate prices refresh --cloud {cloud}`"
                if cloud != "gcp"
                else f"{cloud} {region}: shipped rows ({len(rows)} machines, "
                f"{base.snapshot_date}); GCP's catalog needs a key"
            )
    sizes, source = timm_sizes()
    lines.append(
        f"timm: {sum(len(v) for v in sizes.values())} networks, {source.kind} {source.date}"
    )
    lines.append(
        f"api models: {len(base.api_models)} rows, shipped {base.snapshot_date} "
        "(no cloud publishes these as a feed)"
    )
    return lines


def _gb(value: str | None) -> float | None:
    if not value:
        return None
    found = _GB.search(value)
    return float(found.group(1)) if found else None


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


__all__ = [
    "AWS_CSV_URL",
    "AZURE_API_URL",
    "CLOUDS",
    "DEFAULT_CLOUDS",
    "DEFAULT_REGIONS",
    "TIMM_CSV_URL",
    "AzureSpec",
    "Cached",
    "TimmRow",
    "azure_specs",
    "cache_path",
    "describe",
    "is_stale",
    "load",
    "read_cache",
    "reduce_aws",
    "reduce_azure",
    "reduce_timm",
    "refresh",
    "refresh_aws",
    "refresh_azure",
    "refresh_azure_in_background",
    "refresh_timm",
    "region_for",
    "shipped",
    "timm_sizes",
    "write_cache",
]
