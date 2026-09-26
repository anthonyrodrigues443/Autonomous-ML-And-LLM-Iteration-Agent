"""What a run's winner costs to serve, and where each number came from.

`ServingFacts` is what the harness knows about a winner, enough to price it: the
classes a table pipeline named, an image recipe's weights and multiply-adds, a prompt's
model and its tokens per record. `Prices` is the table the Pricer reads: the clouds' cached
lists where a refresh exists, the shipped snapshot where not, with a source per cloud.
`ServingProfile` is the answer: the cheapest machine or API that serves the request rate,
the monthly cost, and a basis line for every number, so an estimate is never mistaken
for a measurement. Extra fields are refused and the parts must agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

Family = Literal["tabular", "vision", "prompt"]
Kind = Literal["cpu", "gpu", "api"]


def money(usd: float) -> str:
    """A budget as the user typed it: whole dollars stay whole, cents are kept, so a
    $36.80 budget never prints as the $37 a price beside it rounds to."""
    return f"${usd:,.0f}" if float(usd).is_integer() else f"${usd:,.2f}"


class ServingFacts(BaseModel):
    """What the harness knows about a winner. One family's fields are filled."""

    model_config = ConfigDict(extra="forbid")

    family: Family
    components: list[str] = Field(default_factory=list)
    estimator_family: str | None = None
    n_features: int | None = None
    backbone: str | None = None
    image_size: int | None = None
    weights: int | None = Field(default=None, ge=0)
    multiply_adds: int | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None
    tokens_in: float | None = Field(default=None, ge=0.0)
    tokens_out: float | None = Field(default=None, ge=0.0)
    records_measured: int | None = Field(default=None, ge=0)
    parameters: int | None = Field(default=None, ge=0)
    # For a network sized from timm's table: its milliseconds per image at this size on
    # the reference box, already scaled from timm's measurement. None for the pinned
    # backbones, whose latency the reference table carries.
    reference_ms: float | None = Field(default=None, ge=0.0)
    basis: list[str] = Field(default_factory=list)
    unpriced_because: str | None = None

    @model_validator(mode="after")
    def _parts_agree(self) -> Self:
        if self.family == "vision" and self.backbone is None:
            raise ValueError("an image winner names its backbone")
        if self.family == "vision" and (self.weights is None) != (self.multiply_adds is None):
            raise ValueError("an image winner has both weights and multiply-adds, or neither")
        if self.family == "prompt" and (self.provider is None or self.model is None):
            raise ValueError("a prompt winner names its provider and model")
        if self.family == "prompt" and (self.tokens_in is None) != (self.tokens_out is None):
            raise ValueError("a prompt winner has tokens in and out, or neither")
        if self.family == "tabular" and (self.backbone or self.model):
            raise ValueError("a table winner has no backbone and no model under test")
        return self


class Source(BaseModel):
    """Where one cloud's rows came from: the shipped snapshot, or a refreshed list."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["shipped", "refreshed"]
    date: str
    url: str | None = None
    region: str | None = None

    @property
    def label(self) -> str:
        where = f" ({self.region})" if self.region else ""
        return f"{self.kind} {self.date}{where}"


class Host(BaseModel):
    """One row of the price table: a machine, or an API model."""

    model_config = ConfigDict(extra="forbid")

    cloud: str
    name: str
    kind: Kind
    region: str | None = None
    vcpu: int | None = Field(default=None, ge=1)
    memory_gb: float | None = Field(default=None, gt=0.0)
    vram_gb: float | None = Field(default=None, gt=0.0)
    usd_per_hour: float | None = Field(default=None, ge=0.0)
    usd_per_1m_in: float | None = Field(default=None, ge=0.0)
    usd_per_1m_out: float | None = Field(default=None, ge=0.0)
    # The published batch-1 latency of the reference network on this machine, when one
    # was found. A row without it prices one box and says the capacity is not estimated.
    resnet50_224_ms: float | None = Field(default=None, gt=0.0)
    source: str
    read_on: str
    note: str | None = None

    @model_validator(mode="after")
    def _priced_one_way(self) -> Self:
        if self.kind == "api":
            if self.usd_per_1m_in is None or self.usd_per_1m_out is None:
                raise ValueError("an API row has a price per million tokens in and out")
            if self.usd_per_hour is not None:
                raise ValueError("an API row has no hourly price")
            return self
        if self.usd_per_hour is None or self.memory_gb is None:
            raise ValueError("a machine row has an hourly price and its memory")
        if self.usd_per_1m_in is not None or self.usd_per_1m_out is not None:
            raise ValueError("a machine row has no token prices")
        if self.kind == "gpu" and self.vram_gb is None:
            raise ValueError("a GPU row has its VRAM")
        return self

    @property
    def label(self) -> str:
        return f"{self.cloud} {self.name}"


class CpuReference(BaseModel):
    """Batch-1 latencies measured on one machine and taken as a 2-vCPU box, because
    no cloud CPU was measured. Every profile that uses them says so."""

    model_config = ConfigDict(extra="forbid")

    measured_on: str
    note: str | None = None
    backbone_ms: dict[str, dict[str, float]]
    tabular_ms: dict[str, float]
    tabular_margin: float = Field(default=2.0, ge=1.0)

    @model_validator(mode="after")
    def _every_table_has_a_size(self) -> Self:
        for backbone, table in self.backbone_ms.items():
            if not table:
                raise ValueError(f"{backbone} has no measured size")
            if any(not size.isdigit() or ms <= 0 for size, ms in table.items()):
                raise ValueError(f"{backbone} needs whole-number sizes and positive ms")
        if any(ms <= 0 for ms in self.tabular_ms.values()):
            raise ValueError("a table family needs a positive ms")
        return self


class Prices(BaseModel):
    """The price table the Pricer reads: machines, API models, the reference latencies,
    and per cloud whether the machines came from a refreshed list or the shipped file."""

    model_config = ConfigDict(extra="forbid")

    snapshot_date: str
    hosts: list[Host]
    api_models: list[Host]
    cpu_reference: CpuReference
    # Per cloud, whether `hosts` came from a refreshed list or the shipped file. Empty
    # means everything is shipped, which is what the file itself says.
    sources: dict[str, Source] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rows_in_their_lists(self) -> Self:
        if any(h.kind == "api" for h in self.hosts):
            raise ValueError("hosts are machines; API rows go under api_models")
        if any(h.kind != "api" for h in self.api_models):
            raise ValueError("api_models are API rows")
        return self

    def machines(self, kind: Kind) -> list[Host]:
        return [h for h in self.hosts if h.kind == kind]

    def provenance(self, clouds: Iterable[str] | None = None) -> str:
        """What the line prints after the price: per cloud, refreshed or shipped, and
        the date, so an old number is never read as a current one."""
        if clouds is not None:
            names = list(clouds)
        elif self.sources:
            names = sorted(self.sources)
        else:
            names = sorted({h.cloud for h in self.hosts})
        parts = []
        for cloud in names:
            source = self.sources.get(cloud)
            parts.append(
                f"{cloud} {source.label}" if source else f"{cloud} shipped {self.snapshot_date}"
            )
        return ", ".join(parts) if parts else f"shipped {self.snapshot_date}"

    def api(self, provider: str, model: str) -> Host | None:
        wanted = (provider.lower(), model.lower())
        for row in self.api_models:
            if (row.cloud.lower(), row.name.lower()) == wanted:
                return row
        return None


class HostCost(BaseModel):
    """One machine or API priced for the request rate."""

    model_config = ConfigDict(extra="forbid")

    host: Host
    instances: int = Field(default=1, ge=1)
    capacity_per_hour: int | None = Field(default=None, ge=1)
    usd_per_month: float = Field(ge=0.0)


class ServingProfile(BaseModel):
    """What the winner costs to serve, and where each number came from."""

    model_config = ConfigDict(extra="forbid")

    requests_per_hour: int = Field(ge=1)
    chosen: HostCost | None = None
    by_cloud: list[HostCost] = Field(default_factory=list)
    usd_per_1k_requests: float | None = Field(default=None, ge=0.0)
    prices_as_of: str
    unpriced_because: str | None = None
    basis: list[str] = Field(default_factory=list)
    # The serving budget the run was given, and whether the chosen machine or API comes
    # in under it. Both None on a run with no budget, so its profile is the one it had.
    budget_usd_per_month: float | None = Field(default=None, gt=0.0)
    within_budget: bool | None = None

    @model_validator(mode="after")
    def _priced_or_says_why(self) -> Self:
        if (self.chosen is None) == (self.unpriced_because is None):
            raise ValueError("a profile is priced, or it says why it is not")
        if self.chosen is not None and self.usd_per_1k_requests is None:
            raise ValueError("a priced profile has a cost per thousand requests")
        if self.budget_usd_per_month is None and self.within_budget is not None:
            raise ValueError("within_budget needs a budget to be within")
        if (
            self.budget_usd_per_month is not None
            and self.chosen is not None
            and self.within_budget != (self.chosen.usd_per_month <= self.budget_usd_per_month)
        ):
            raise ValueError("within_budget disagrees with the price and the budget")
        return self

    def render(self) -> list[str]:
        """The terminal lines, plain words, one number each."""
        if self.chosen is None:
            return [f"serving: not priced: {self.unpriced_because}"]
        chosen = self.chosen
        where = chosen.host.label + (f" x{chosen.instances}" if chosen.instances > 1 else "")
        if chosen.capacity_per_hour is None and chosen.host.kind != "api":
            # One box, and whether it keeps up with the rate is not claimed.
            lines = [
                f"serving: about ${chosen.usd_per_month:,.2f} a month for one {where} "
                f"(whether it serves {self.requests_per_hour:,} requests an hour is not "
                f"estimated), prices: {self.prices_as_of}"
            ]
        else:
            lines = [
                f"serving: about ${chosen.usd_per_month:,.2f} a month at "
                f"{self.requests_per_hour:,} requests an hour on {where}, "
                f"{self.budget_clause()}prices: {self.prices_as_of}"
            ]
        lines.extend(f"  basis: {line}" for line in self.basis)
        others = [cost for cost in self.by_cloud if cost != chosen]
        if others:
            lines.append(
                "  also: "
                + ", ".join(f"{cost.host.label} ${cost.usd_per_month:,.2f}" for cost in others)
            )
        return lines

    def budget_clause(self) -> str:
        """ "within the $50 serving budget, " or "over the $50 serving budget, "; empty
        with no budget."""
        if self.budget_usd_per_month is None or self.within_budget is None:
            return ""
        side = "within" if self.within_budget else "over"
        return f"{side} the {money(self.budget_usd_per_month)} serving budget, "


__all__ = [
    "CpuReference",
    "Family",
    "Host",
    "HostCost",
    "Kind",
    "Prices",
    "ServingFacts",
    "ServingProfile",
    "Source",
    "money",
]
