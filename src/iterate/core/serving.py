"""The Pricer: what a run's winner costs to serve, from facts the run already has.

Pure arithmetic over the dated snapshot in `serving_prices.json`. It never reads an
abstract or asks a model, so it cannot invent a price, and it never imports torch or
scikit-learn, so it runs on the host after every family. Three facts builders read the
winner: a table pipeline by the classes it named, an image recipe by its weights and
multiply-adds through the layer grammar, a prompt by its model and the tokens it used.
`profile` turns facts and a request rate into the cheapest machine or API, the monthly
cost, and a basis line for every number.
"""

from __future__ import annotations

import json
import math
import re
from importlib import resources
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.schemas.serving import (
    CpuReference,
    Host,
    HostCost,
    Kind,
    Prices,
    ServingFacts,
    ServingProfile,
)
from iterate.targets import layers

if TYPE_CHECKING:
    from collections.abc import Mapping

HOURS_PER_MONTH = 730
DEFAULT_REQUESTS_PER_HOUR = 1000
FLOAT_BYTES = 4
GB = 2**30
# A torch process serving one network keeps about this much beside the weights.
RUNTIME_GB = 1.0
# A local LLM at 4-bit: about 0.6 GB per billion weights, plus room for the context.
LLM_GB_PER_BILLION = 0.6
LLM_HEADROOM_GB = 2.0
PRICES_FILE = "serving_prices.json"

# Weights and multiply-adds per image at 224 px: torchvision's model table, read
# 2026-09-25, and the weights checked against the built network on torchvision 0.29.
BACKBONE_SIZES: dict[str, tuple[int, int]] = {
    "resnet18": (11_689_512, 1_814_000_000),
    "resnet50": (25_557_032, 4_089_000_000),
    "convnext_tiny": (28_589_128, 4_456_000_000),
}
# The feature width each backbone ends on: the last entry of `net.STAGES`, copied so this
# module never imports the vision runner.
FEATURES: dict[str, int] = {"resnet18": 512, "resnet50": 2048, "convnext_tiny": 768}
REFERENCE_BACKBONE = "resnet50"
REFERENCE_SIZE = 224

# What a class name says about how the pipeline predicts, slowest family first, because
# the heaviest family named on the winning cell prices the winner.
ESTIMATOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "tree_ensemble": ("RandomForest", "ExtraTrees", "Bagging", "DecisionTree"),
    "boosting": ("HistGradientBoosting", "GradientBoosting", "AdaBoost", "LGBM", "XGB", "CatBoost"),
    "nearest_neighbours": ("KNeighbors", "RadiusNeighbors"),
    "svm": ("SVC", "SVR", "NuSVC", "NuSVR"),
    "mlp": ("MLPClassifier", "MLPRegressor"),
    "linear": (
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGD",
        "LinearSVC",
        "LinearSVR",
        "BayesianRidge",
        "Perceptron",
        "PassiveAggressive",
        "Huber",
        "Quantile",
        "PoissonRegressor",
        "GammaRegressor",
        "TweedieRegressor",
    ),
}
_STACKS = ("Stacking", "Voting")
_LLM_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def load_prices() -> Prices:
    """The snapshot shipped with this version of the package."""
    text = resources.files("iterate.core").joinpath(PRICES_FILE).read_text(encoding="utf-8")
    return Prices.model_validate_json(text)


# ─── facts: what the run knows about its winner ─────────────────────────────


def facts_from_code(
    cells: list[Mapping[str, Any]] | None, code: str | None, *, n_features: int | None
) -> ServingFacts:
    """A table winner, by the estimator classes on its last working cell."""
    named = _winning_components(cells, code)
    families = sorted({family for name in named if (family := estimator_family(name))})
    if not families:
        return ServingFacts(
            family="tabular",
            components=named,
            n_features=n_features,
            basis=["the winning code names no estimator this version prices"],
            unpriced_because="the winning code names no estimator this version prices",
        )
    heaviest = min(families, key=_family_rank)
    shape = f" on {n_features} features" if n_features else ""
    return ServingFacts(
        family="tabular",
        components=named,
        estimator_family=heaviest,
        n_features=n_features,
        basis=[f"{heaviest.replace('_', ' ')} pipeline ({', '.join(named)}){shape}"],
    )


def facts_from_recipe(recipe: Mapping[str, Any]) -> ServingFacts:
    """An image winner, by the recipe its SUBMITTED line carried."""
    if recipe.get("model"):
        name = str(recipe["model"])
        why = f"the winner is the agent's own network ({name}); its size is not known"
        return ServingFacts(family="vision", backbone=name, basis=[why], unpriced_because=why)
    backbone = str(recipe.get("backbone") or "resnet18")
    size = int(recipe.get("image_size") or REFERENCE_SIZE)
    if backbone == "simple_cnn":
        return _stack_facts(backbone, layers.SIMPLE_CNN, size, "the plain CNN baseline")
    if backbone == "layers_net":
        spec = layers.parse(recipe.get("layers"))
        if spec is None:
            why = "the recipe names a layer stack but carries none"
            return ServingFacts(
                family="vision", backbone=backbone, basis=[why], unpriced_because=why
            )
        return _stack_facts(backbone, spec, size, f"your own stack {layers.text(spec)}")
    if backbone not in BACKBONE_SIZES:
        why = f"{backbone} is not a backbone this version prices"
        return ServingFacts(family="vision", backbone=backbone, basis=[why], unpriced_because=why)
    weights, macs_224 = BACKBONE_SIZES[backbone]
    macs = int(macs_224 * (size / REFERENCE_SIZE) ** 2)
    what = backbone
    head = layers.parse(recipe.get("head"), "head")
    if head is not None:
        width = FEATURES[backbone]
        weights += layers.count_weights(head, features=width)
        macs += layers.count_macs(head, size, "head", features=width)
        what = f"{backbone} with head {layers.text(head, 'head')}"
    return _sized_facts(backbone, size, weights, macs, what)


def _stack_facts(backbone: str, spec: layers.Spec, size: int, what: str) -> ServingFacts:
    return _sized_facts(
        backbone, size, layers.count_weights(spec), layers.count_macs(spec, size), what
    )


def _sized_facts(backbone: str, size: int, weights: int, macs: int, what: str) -> ServingFacts:
    return ServingFacts(
        family="vision",
        backbone=backbone,
        image_size=size,
        weights=weights,
        multiply_adds=macs,
        basis=[
            f"{what} at {size} px",
            f"{weights / 1e6:.1f}M weights ({_megabytes(weights * FLOAT_BYTES)}), "
            f"{macs / 1e9:.2f} billion multiply-adds an image",
        ],
    )


def _megabytes(size: int) -> str:
    mb = size / 2**20
    return f"{mb:.1f} MB" if mb < 10 else f"{mb:.0f} MB"


def facts_from_prompt(
    prompt_json: str | None, *, provider: str, model: str, prompt_chars: int = 0
) -> ServingFacts:
    """A prompt winner, by the model under test and the tokens it spent per record."""
    record: dict[str, Any] = {}
    if prompt_json:
        try:
            record = json.loads(prompt_json)
        except ValueError:
            record = {}
    measured = int(record.get("records_measured") or 0)
    tokens_in = record.get("tokens_in_per_record")
    tokens_out = record.get("tokens_out_per_record")
    parameters = _llm_parameters(model) if provider == "ollama" else None
    if provider == "ollama" and parameters is None:
        why = f"{model} runs locally and its size is not in its name, so no GPU can be sized"
        return ServingFacts(
            family="prompt", provider=provider, model=model, basis=[why], unpriced_because=why
        )
    if measured and tokens_in is not None and tokens_out is not None:
        basis = [
            f"{float(tokens_in):.0f} tokens in and {float(tokens_out):.0f} out per record, "
            f"measured on {measured} holdout records"
        ]
        return ServingFacts(
            family="prompt",
            provider=provider,
            model=model,
            tokens_in=float(tokens_in),
            tokens_out=float(tokens_out),
            records_measured=measured,
            parameters=parameters,
            basis=basis,
        )
    guess_in = max(1.0, prompt_chars / 4)
    return ServingFacts(
        family="prompt",
        provider=provider,
        model=model,
        tokens_in=guess_in,
        tokens_out=8.0,
        records_measured=0,
        parameters=parameters,
        basis=[
            f"about {guess_in:.0f} tokens in and 8 out per record, estimated from the prompt "
            "text because every holdout answer came from the cache"
        ],
    )


# ─── the profile ────────────────────────────────────────────────────────────


def profile(facts: ServingFacts, requests_per_hour: int, prices: Prices) -> ServingProfile:
    """The cheapest way to serve the winner at the rate, or why it cannot be priced."""
    if facts.unpriced_because:
        return _unpriced(facts, requests_per_hour, prices, facts.unpriced_because)
    if facts.family == "prompt" and facts.provider != "ollama":
        return _api_profile(facts, requests_per_hour, prices)
    needed_gb = _memory_gb(facts)
    costs = [
        cost
        for host in _machines_for(facts, prices)
        if _fits(host, needed_gb)
        and (cost := _cost_on(host, facts, requests_per_hour, prices.cpu_reference)) is not None
    ]
    if not costs:
        return _unpriced(
            facts,
            requests_per_hour,
            prices,
            f"no machine in the price table fits {needed_gb:.1f} GB",
        )
    cheapest_per_cloud: dict[str, HostCost] = {}
    for cost in sorted(costs, key=lambda cost: cost.usd_per_month):
        cheapest_per_cloud.setdefault(cost.host.cloud, cost)
    by_cloud = list(cheapest_per_cloud.values())
    chosen = by_cloud[0]
    per_1k = chosen.usd_per_month / (requests_per_hour * HOURS_PER_MONTH) * 1000
    return ServingProfile(
        requests_per_hour=requests_per_hour,
        chosen=chosen,
        by_cloud=by_cloud,
        usd_per_1k_requests=per_1k,
        prices_as_of=prices.snapshot_date,
        basis=[*facts.basis, *_capacity_basis(chosen, facts, prices.cpu_reference)],
    )


def estimator_family(class_name: str) -> str | None:
    """The family a scikit-learn style class name belongs to, or None for a preprocessor."""
    for family, prefixes in ESTIMATOR_FAMILIES.items():
        if any(class_name.startswith(prefix) for prefix in prefixes):
            return family
    return None


def _winning_components(cells: list[Mapping[str, Any]] | None, code: str | None) -> list[str]:
    """The classes on the last cell that ran without error and named an estimator, since
    `code` joins every cell the session wrote, losers and dead ends included."""
    for cell in reversed(cells or []):
        if cell.get("error"):
            continue
        named = codegen.components_used(str(cell.get("code") or ""))
        if any(estimator_family(name) or name.startswith(_STACKS) for name in named):
            return named
    return codegen.components_used(code or "")


def _family_rank(family: str) -> int:
    return list(ESTIMATOR_FAMILIES).index(family)


def _llm_parameters(model: str) -> int | None:
    found = _LLM_SIZE.search(model)
    return int(float(found.group(1)) * 1e9) if found else None


def _api_profile(facts: ServingFacts, rate: int, prices: Prices) -> ServingProfile:
    row = prices.api(facts.provider or "", facts.model or "")
    if row is None:
        return _unpriced(
            facts, rate, prices, f"no public price for {facts.model} on {facts.provider}"
        )
    tokens_in, tokens_out = facts.tokens_in or 0.0, facts.tokens_out or 0.0
    price_in, price_out = row.usd_per_1m_in or 0.0, row.usd_per_1m_out or 0.0
    per_request = (tokens_in * price_in + tokens_out * price_out) / 1e6
    month = per_request * rate * HOURS_PER_MONTH
    return ServingProfile(
        requests_per_hour=rate,
        chosen=HostCost(host=row, usd_per_month=month),
        usd_per_1k_requests=per_request * 1000,
        prices_as_of=prices.snapshot_date,
        basis=[*facts.basis, "rate limits are not modelled"],
    )


def _unpriced(facts: ServingFacts, rate: int, prices: Prices, why: str) -> ServingProfile:
    return ServingProfile(
        requests_per_hour=rate,
        prices_as_of=prices.snapshot_date,
        unpriced_because=why,
        basis=[line for line in facts.basis if line != why],
    )


def _machines_for(facts: ServingFacts, prices: Prices) -> list[Host]:
    kinds: tuple[Kind, ...]
    if facts.family == "tabular":
        kinds = ("cpu",)
    elif facts.family == "vision":
        kinds = ("cpu", "gpu")
    else:
        kinds = ("gpu",)
    return [host for kind in kinds for host in prices.machines(kind)]


def _memory_gb(facts: ServingFacts) -> float:
    if facts.family == "prompt":
        billions = (facts.parameters or 0) / 1e9
        return billions * LLM_GB_PER_BILLION + LLM_HEADROOM_GB
    if facts.family == "vision":
        return (facts.weights or 0) * FLOAT_BYTES / GB + RUNTIME_GB
    return 0.5


def _fits(host: Host, needed_gb: float) -> bool:
    room = host.vram_gb if host.kind == "gpu" else host.memory_gb
    return room is not None and room >= needed_gb


def _cost_on(host: Host, facts: ServingFacts, rate: int, ref: CpuReference) -> HostCost | None:
    assert host.usd_per_hour is not None
    seconds = _seconds_per_request(host, facts, ref)
    if seconds is None:
        return HostCost(host=host, usd_per_month=host.usd_per_hour * HOURS_PER_MONTH)
    capacity = max(1, int(3600 / seconds))
    instances = max(1, math.ceil(rate / capacity))
    return HostCost(
        host=host,
        instances=instances,
        capacity_per_hour=capacity,
        usd_per_month=instances * host.usd_per_hour * HOURS_PER_MONTH,
    )


def _seconds_per_request(host: Host, facts: ServingFacts, ref: CpuReference) -> float | None:
    """One request's time on this machine, or None when nothing measured covers it."""
    if facts.family == "prompt":
        return None
    if facts.family == "tabular":
        if host.kind != "cpu" or facts.estimator_family is None:
            return None
        ms = ref.tabular_ms.get(facts.estimator_family)
        return None if ms is None else ms * ref.tabular_margin / 1000
    macs = facts.multiply_adds or 0
    if host.kind == "gpu":
        if host.resnet50_224_ms is None:
            return None
        _, reference_macs = BACKBONE_SIZES[REFERENCE_BACKBONE]
        return host.resnet50_224_ms * macs / reference_macs / 1000
    ms = _cpu_ms(facts, ref)
    return None if ms is None else ms / 1000


def _cpu_ms(facts: ServingFacts, ref: CpuReference) -> float | None:
    """Measured for the pinned backbones at the sizes measured; scaled by multiply-adds
    from resnet18's measured rate for a stack the table has no row for."""
    backbone, image_size = facts.backbone or "", facts.image_size or 0
    macs = facts.multiply_adds or 0
    if backbone in ref.backbone_ms and backbone in BACKBONE_SIZES:
        return _interpolate(ref.backbone_ms[backbone], BACKBONE_SIZES[backbone][1], image_size)
    anchor = ref.backbone_ms.get("resnet18")
    if not anchor:
        return None
    nearest = min((int(s) for s in anchor), key=lambda s: abs(s - image_size))
    anchor_macs = BACKBONE_SIZES["resnet18"][1] * (nearest / REFERENCE_SIZE) ** 2
    return anchor[str(nearest)] * macs / anchor_macs


def _interpolate(table: Mapping[str, float], macs_224: int, size: int) -> float:
    """Linear in multiply-adds between the two measured sizes around the recipe's, and
    scaled outside the measured range from the nearest end."""
    points = sorted((int(measured), ms) for measured, ms in table.items())

    def macs_at(px: int) -> float:
        return macs_224 * (px / REFERENCE_SIZE) ** 2

    below = [(s, ms) for s, ms in points if s <= size]
    above = [(s, ms) for s, ms in points if s >= size]
    if below and above and below[-1][0] == above[0][0]:
        return below[-1][1]
    if below and above:
        (s0, ms0), (s1, ms1) = below[-1], above[0]
        return ms0 + (ms1 - ms0) * (macs_at(size) - macs_at(s0)) / (macs_at(s1) - macs_at(s0))
    edge_size, edge_ms = points[0] if above else points[-1]
    return edge_ms * macs_at(size) / macs_at(edge_size)


def _capacity_basis(chosen: HostCost, facts: ServingFacts, ref: CpuReference) -> list[str]:
    if chosen.capacity_per_hour is None:
        if facts.family == "prompt":
            return [
                "one GPU box whose memory fits the weights; requests an hour not estimated, "
                "because tokens a second on that GPU was not measured"
            ]
        return ["capacity on this machine not estimated: no benchmark rate for it"]
    ms = 3600 / chosen.capacity_per_hour * 1000
    if chosen.host.kind == "gpu":
        return [
            f"one request takes about {ms:.1f} ms, scaled by multiply-adds from the "
            f"published resnet50 rate for this GPU"
        ]
    if facts.family == "tabular":
        return [
            f"one row predicts in about {ms / ref.tabular_margin:.1f} ms, measured on the "
            f"corpus on {ref.measured_on} and taken as a 2-vCPU box, "
            f"with a {ref.tabular_margin:g}x margin"
        ]
    how = "measured" if facts.backbone in ref.backbone_ms else "scaled by multiply-adds"
    return [
        f"one request takes about {ms:.1f} ms, {how} on {ref.measured_on} and taken as a 2-vCPU box"
    ]


__all__ = [
    "BACKBONE_SIZES",
    "DEFAULT_REQUESTS_PER_HOUR",
    "ESTIMATOR_FAMILIES",
    "FEATURES",
    "HOURS_PER_MONTH",
    "estimator_family",
    "facts_from_code",
    "facts_from_prompt",
    "facts_from_recipe",
    "load_prices",
    "profile",
]
