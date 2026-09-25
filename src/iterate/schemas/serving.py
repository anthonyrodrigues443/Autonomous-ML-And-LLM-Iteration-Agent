"""What a run's winner costs to serve, and where each number came from.

`ServingFacts` is what the harness knows about a winner, enough to price it: the
classes a table pipeline named, an image recipe's weights and multiply-adds, a prompt's
model and its tokens per record. `Prices` is the dated snapshot shipped with the package.
`ServingProfile` is the answer: the cheapest machine or API that serves the request rate,
the monthly cost, and a basis line for every number, so an estimate is never mistaken
for a measurement. Extra fields are refused and the parts must agree.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Family = Literal["tabular", "vision", "prompt"]
Kind = Literal["cpu", "gpu", "api"]


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


class Host(BaseModel):
    """One row of the price snapshot: a machine, or an API model."""

    model_config = ConfigDict(extra="forbid")

    cloud: str
    name: str
    kind: Kind
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


class Prices(BaseModel):
    """The dated snapshot: machines, API models, and the reference latencies."""

    model_config = ConfigDict(extra="forbid")

    snapshot_date: str
    hosts: list[Host]
    api_models: list[Host]
    cpu_reference: CpuReference

    @model_validator(mode="after")
    def _rows_in_their_lists(self) -> Self:
        if any(h.kind == "api" for h in self.hosts):
            raise ValueError("hosts are machines; API rows go under api_models")
        if any(h.kind != "api" for h in self.api_models):
            raise ValueError("api_models are API rows")
        return self

    def machines(self, kind: Kind) -> list[Host]:
        return [h for h in self.hosts if h.kind == kind]

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

    @model_validator(mode="after")
    def _priced_or_says_why(self) -> Self:
        if (self.chosen is None) == (self.unpriced_because is None):
            raise ValueError("a profile is priced, or it says why it is not")
        if self.chosen is not None and self.usd_per_1k_requests is None:
            raise ValueError("a priced profile has a cost per thousand requests")
        return self

    def render(self) -> list[str]:
        """The terminal lines, plain words, one number each."""
        if self.chosen is None:
            return [f"serving: not priced: {self.unpriced_because}"]
        chosen = self.chosen
        where = chosen.host.label + (f" x{chosen.instances}" if chosen.instances > 1 else "")
        lines = [
            f"serving: about ${chosen.usd_per_month:,.2f} a month at "
            f"{self.requests_per_hour:,} requests an hour on {where}, "
            f"prices as of {self.prices_as_of}"
        ]
        lines.extend(f"  basis: {line}" for line in self.basis)
        others = [cost for cost in self.by_cloud if cost != chosen]
        if others:
            lines.append(
                "  also: "
                + ", ".join(f"{cost.host.label} ${cost.usd_per_month:,.2f}" for cost in others)
            )
        return lines


__all__ = [
    "CpuReference",
    "Family",
    "Host",
    "HostCost",
    "Kind",
    "Prices",
    "ServingFacts",
    "ServingProfile",
]
