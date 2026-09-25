"""The Pricer: facts in, a serving profile out, every number with its basis.

Runs on a fake snapshot so the arithmetic is checked to the cent, without torch and
without the shipped prices, which have their own test.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from iterate.core import serving
from iterate.schemas.serving import (
    CpuReference,
    Host,
    HostCost,
    Prices,
    ServingFacts,
    ServingProfile,
)
from iterate.targets import layers

pytestmark = pytest.mark.unit


def _host(**overrides: object) -> Host:
    base: dict[str, object] = {
        "cloud": "gcp",
        "name": "small",
        "kind": "cpu",
        "vcpu": 2,
        "memory_gb": 2,
        "usd_per_hour": 0.01,
        "source": "s",
        "read_on": "2026-09-25",
    }
    return Host(**{**base, **overrides})  # type: ignore[arg-type]


def _prices() -> Prices:
    return Prices(
        snapshot_date="2026-09-25",
        hosts=[
            _host(),
            _host(cloud="aws", usd_per_hour=0.02),
            _host(cloud="azure", memory_gb=4, usd_per_hour=0.04),
            _host(
                cloud="aws",
                name="t4",
                kind="gpu",
                vcpu=4,
                memory_gb=16,
                vram_gb=16,
                usd_per_hour=0.5,
                resnet50_224_ms=10.0,
            ),
            _host(
                cloud="gcp",
                name="l4",
                kind="gpu",
                vcpu=4,
                memory_gb=16,
                vram_gb=24,
                usd_per_hour=0.7,
            ),
        ],
        api_models=[
            Host(
                cloud="openai",
                name="mini",
                kind="api",
                usd_per_1m_in=0.15,
                usd_per_1m_out=0.6,
                source="s",
                read_on="2026-09-25",
            ),
        ],
        cpu_reference=CpuReference(
            measured_on="a test box",
            backbone_ms={
                "resnet18": {"64": 6.0, "128": 12.0, "224": 24.0},
                "resnet50": {"224": 50.0},
            },
            tabular_ms={"linear": 2.0, "boosting": 4.0, "tree_ensemble": 10.0},
            tabular_margin=2.0,
        ),
    )


# ─── the schema refuses parts that disagree ─────────────────────────────


def test_an_image_winner_names_its_backbone() -> None:
    with pytest.raises(ValidationError, match="backbone"):
        ServingFacts(family="vision")


def test_an_image_winner_has_weights_and_multiply_adds_together() -> None:
    with pytest.raises(ValidationError, match="both weights"):
        ServingFacts(family="vision", backbone="resnet18", weights=10)


def test_a_prompt_winner_names_provider_and_model() -> None:
    with pytest.raises(ValidationError, match="provider and model"):
        ServingFacts(family="prompt", model="m")


def test_a_table_winner_has_no_backbone() -> None:
    with pytest.raises(ValidationError, match="no backbone"):
        ServingFacts(family="tabular", backbone="resnet18")


def test_an_api_row_has_token_prices_and_no_hourly_price() -> None:
    with pytest.raises(ValidationError, match="per million"):
        Host(cloud="openai", name="m", kind="api", source="s", read_on="d")
    with pytest.raises(ValidationError, match="no hourly"):
        Host(
            cloud="openai",
            name="m",
            kind="api",
            usd_per_1m_in=1,
            usd_per_1m_out=1,
            usd_per_hour=1,
            source="s",
            read_on="d",
        )


def test_a_machine_row_has_an_hourly_price_and_memory_and_a_gpu_has_vram() -> None:
    with pytest.raises(ValidationError, match="hourly price and its memory"):
        Host(cloud="aws", name="m", kind="cpu", usd_per_hour=1, source="s", read_on="d")
    with pytest.raises(ValidationError, match="VRAM"):
        _host(kind="gpu")


def test_api_rows_do_not_hide_among_machines() -> None:
    api = Host(
        cloud="openai",
        name="m",
        kind="api",
        usd_per_1m_in=1,
        usd_per_1m_out=1,
        source="s",
        read_on="d",
    )
    with pytest.raises(ValidationError, match="api_models"):
        Prices(
            snapshot_date="d",
            hosts=[api],
            api_models=[],
            cpu_reference=CpuReference(measured_on="x", backbone_ms={}, tabular_ms={}),
        )


def test_a_profile_is_priced_or_says_why() -> None:
    with pytest.raises(ValidationError, match="says why"):
        ServingProfile(requests_per_hour=1, prices_as_of="d")
    with pytest.raises(ValidationError, match="says why"):
        ServingProfile(
            requests_per_hour=1,
            prices_as_of="d",
            unpriced_because="x",
            chosen=HostCost(host=_host(), usd_per_month=1),
        )


# ─── the arithmetic ──────────────────────────────────────────────────────


def test_an_api_model_costs_tokens_times_price() -> None:
    facts = ServingFacts(
        family="prompt",
        provider="openai",
        model="mini",
        tokens_in=180,
        tokens_out=4,
        records_measured=200,
    )
    profile = serving.profile(facts, 1000, _prices())

    per_request = (180 * 0.15 + 4 * 0.6) / 1e6
    assert profile.chosen is not None
    assert profile.chosen.host.name == "mini"
    assert profile.usd_per_1k_requests == pytest.approx(per_request * 1000)
    assert profile.chosen.usd_per_month == pytest.approx(per_request * 1000 * 730)
    assert "rate limits are not modelled" in profile.basis


def test_instances_round_up_to_cover_the_rate() -> None:
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 224})
    # 24 ms a request is 150,000 an hour on one box.
    profile = serving.profile(facts, 400_000, _prices())

    assert profile.chosen is not None
    assert profile.chosen.host.name == "small"
    assert profile.chosen.instances == 3
    assert profile.chosen.capacity_per_hour == 150_000
    assert profile.chosen.usd_per_month == pytest.approx(3 * 0.01 * 730)


def test_a_machine_with_no_rate_prices_one_box_and_says_so() -> None:
    facts = serving.facts_from_prompt(
        json.dumps(
            {"tokens_in_per_record": 100, "tokens_out_per_record": 4, "records_measured": 10}
        ),
        provider="ollama",
        model="gemma4:12b",
    )
    profile = serving.profile(facts, 1000, _prices())

    assert facts.parameters == 12_000_000_000
    assert profile.chosen is not None
    assert profile.chosen.host.name == "t4"
    assert profile.chosen.capacity_per_hour is None
    assert profile.chosen.usd_per_month == pytest.approx(0.5 * 730)
    assert any("not estimated" in line for line in profile.basis)


def test_a_model_too_big_for_every_machine_is_unpriced_with_the_reason() -> None:
    facts = serving.facts_from_prompt(None, provider="ollama", model="llama:70b")
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is None
    assert profile.unpriced_because is not None
    assert "fits 44.0 GB" in profile.unpriced_because


def test_the_cheapest_per_cloud_is_listed_and_the_cheapest_of_all_is_chosen() -> None:
    facts = serving.facts_from_code(None, "HistGradientBoostingClassifier()", n_features=5)
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is not None
    assert [cost.host.cloud for cost in profile.by_cloud] == ["gcp", "aws", "azure"]
    assert profile.by_cloud[0] is profile.chosen
    lines = profile.render()
    assert lines[0].startswith(
        "serving: about $7.30 a month at 1,000 requests an hour on gcp small"
    )
    assert lines[-1] == "  also: aws small $14.60, azure small $29.20"


def test_a_gpu_rate_scales_by_multiply_adds_from_the_reference_network() -> None:
    prices = _prices()
    t4 = next(host for host in prices.hosts if host.name == "t4")
    reference = serving.facts_from_recipe({"backbone": "resnet50", "image_size": 224})
    half = serving.facts_from_recipe({"backbone": "resnet50", "image_size": 112})

    # resnet50 at 224 px IS the reference network, so the T4 row's 10 ms applies as is;
    # a quarter of the multiply-adds takes a quarter of the time.
    on_reference = serving._cost_on(t4, reference, 1000, prices.cpu_reference)
    on_half = serving._cost_on(t4, half, 1000, prices.cpu_reference)
    assert on_reference is not None
    assert on_half is not None
    assert on_reference.capacity_per_hour == 360_000
    assert on_half.capacity_per_hour == pytest.approx(4 * 360_000, rel=0.01)


def test_cpu_latency_interpolates_between_measured_sizes_and_scales_past_them() -> None:
    ref = _prices().cpu_reference
    inside = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 96})
    past = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 320})

    between = serving._cpu_ms(inside, ref)
    beyond = serving._cpu_ms(past, ref)
    assert between is not None
    assert 6.0 < between < 12.0
    assert beyond == pytest.approx(24.0 * (320 / 224) ** 2)


def test_a_stack_the_table_has_no_row_for_scales_from_resnet18() -> None:
    facts = serving.facts_from_recipe({"backbone": "simple_cnn", "image_size": 64})
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is not None
    assert any("scaled by multiply-adds" in line for line in profile.basis)


# ─── facts from what the run knows ───────────────────────────────────────


def test_a_pretrained_backbone_is_sized_from_the_table_and_the_image_size() -> None:
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 128})

    weights, macs_224 = serving.BACKBONE_SIZES["resnet18"]
    # torchvision's count carries the 1,000-class ImageNet classifier; the served network
    # replaces it with the run's own final layer.
    assert facts.weights == weights - (512 + 1) * 1000
    assert facts.multiply_adds == int(macs_224 * (128 / 224) ** 2)
    assert facts.basis[0] == "resnet18 at 128 px"


def test_a_custom_head_adds_its_weights_to_the_backbone() -> None:
    plain = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 224})
    headed = serving.facts_from_recipe(
        {"backbone": "resnet18", "image_size": 224, "head": [("linear", 512), ("dropout", 0.5)]}
    )

    assert plain.weights is not None
    assert headed.weights is not None
    assert headed.weights - plain.weights == (512 + 1) * 512
    assert "with head linear(512) dropout(0.5)" in headed.basis[0]


def test_a_layer_stack_is_sized_by_the_grammar() -> None:
    spec = [("conv", 32), ("pool",), ("conv", 64), ("linear", 256)]
    facts = serving.facts_from_recipe({"backbone": "layers_net", "image_size": 64, "layers": spec})

    parsed = layers.parse(spec)
    assert parsed is not None
    assert facts.weights == layers.count_weights(parsed)
    assert facts.multiply_adds == layers.count_macs(parsed, 64)


@pytest.mark.parametrize(
    ("recipe", "why"),
    [
        ({"model": "hf_hub:timm/vit_base"}, "the agent's own network"),
        ({"backbone": "layers_net", "image_size": 64}, "carries none"),
        ({"backbone": "resnet101", "image_size": 64}, "not a backbone this version prices"),
    ],
)
def test_an_image_winner_the_pricer_cannot_size_says_why(
    recipe: dict[str, object], why: str
) -> None:
    facts = serving.facts_from_recipe(recipe)
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is None
    assert profile.unpriced_because is not None
    assert why in profile.unpriced_because


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("RandomForestClassifier", "tree_ensemble"),
        ("HistGradientBoostingRegressor", "boosting"),
        ("LGBMClassifier", "boosting"),
        ("KNeighborsRegressor", "nearest_neighbours"),
        ("SVC", "svm"),
        ("LinearSVC", "linear"),
        ("RidgeClassifier", "linear"),
        ("MLPClassifier", "mlp"),
        ("OneHotEncoder", None),
        ("StandardScaler", None),
    ],
)
def test_estimator_class_names_map_to_a_family(name: str, family: str | None) -> None:
    assert serving.estimator_family(name) == family


def test_the_last_working_cell_that_names_an_estimator_prices_a_table_winner() -> None:
    cells = [
        {"code": "m = RandomForestClassifier().fit(X_train, y_train)", "error": None},
        {"code": "m = HistGradientBoostingClassifier().fit(X_train, y_train)", "error": None},
        {"code": "m = KNeighborsClassifier().fit(X_train, y_train)", "error": "NameError"},
    ]
    facts = serving.facts_from_code(cells, "", n_features=12)

    assert facts.estimator_family == "boosting"
    assert facts.components == ["HistGradientBoostingClassifier"]
    assert facts.basis == ["boosting pipeline (HistGradientBoostingClassifier) on 12 features"]


def test_the_heaviest_family_on_the_winning_cell_prices_it() -> None:
    code = "p = Pipeline([('a', LogisticRegression()), ('b', RandomForestClassifier())])"
    facts = serving.facts_from_code(None, code, n_features=3)

    assert facts.estimator_family == "tree_ensemble"


def test_a_table_winner_with_no_known_estimator_is_unpriced() -> None:
    facts = serving.facts_from_code([{"code": "print(1)", "error": None}], "print(1)", n_features=1)
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is None
    assert profile.unpriced_because == "the winning code names no estimator this version prices"


def test_prompt_tokens_come_from_the_record_when_measured() -> None:
    record = json.dumps(
        {"tokens_in_per_record": 180.5, "tokens_out_per_record": 4, "records_measured": 200}
    )
    facts = serving.facts_from_prompt(record, provider="openai", model="mini")

    assert (facts.tokens_in, facts.tokens_out, facts.records_measured) == (180.5, 4.0, 200)
    assert facts.basis == ["180 tokens in and 4 out per record, measured on 200 holdout records"]


def test_prompt_tokens_are_estimated_from_the_text_when_every_answer_was_cached() -> None:
    record = json.dumps({"records_measured": 0})
    facts = serving.facts_from_prompt(record, provider="openai", model="mini", prompt_chars=400)

    assert facts.tokens_in == 100.0
    assert facts.records_measured == 0
    assert "estimated from the prompt text" in facts.basis[0]


def test_a_local_model_with_no_size_in_its_name_is_unpriced() -> None:
    facts = serving.facts_from_prompt(None, provider="ollama", model="mymodel")
    profile = serving.profile(facts, 1000, _prices())

    assert facts.parameters is None
    assert profile.chosen is None
    assert profile.unpriced_because is not None
    assert "size is not in its name" in profile.unpriced_because


def test_a_provider_with_no_public_price_is_unpriced() -> None:
    facts = serving.facts_from_prompt(None, provider="groq", model="llama-3.3-70b-versatile")
    profile = serving.profile(facts, 1000, _prices())

    assert profile.unpriced_because == "no public price for llama-3.3-70b-versatile on groq"
    assert profile.render() == [
        "serving: not priced: no public price for llama-3.3-70b-versatile on groq"
    ]


def test_the_profile_round_trips_through_json() -> None:
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 128})
    profile = serving.profile(facts, 1000, _prices())

    again = ServingProfile.model_validate_json(profile.model_dump_json())
    assert again == profile
    assert again.render() == profile.render()


# ─── what review found, kept out for good ────────────────────────────────


def test_a_machine_with_no_rate_is_offered_only_when_no_rated_machine_fits() -> None:
    """At a high rate the rated boxes need many instances, and an unrated box would
    look cheaper while promising nothing. It stays out until nothing rated fits."""
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 224})
    profile = serving.profile(facts, 2_000_000, _prices())

    assert profile.chosen is not None
    assert profile.chosen.capacity_per_hour is not None
    assert all(cost.capacity_per_hour is not None for cost in profile.by_cloud)


def test_zero_reported_tokens_are_not_a_measurement() -> None:
    """A backend that sends no usage leaves zeros, and zero tokens is not a price."""
    record = json.dumps(
        {
            "system": "s" * 400,
            "user_template": "{input}",
            "tokens_in_per_record": 0.0,
            "tokens_out_per_record": 0.0,
            "records_measured": 50,
        }
    )
    facts = serving.facts_from_prompt(record, provider="openai", model="mini")

    assert facts.records_measured == 0
    assert facts.tokens_in == pytest.approx(407 / 4)
    assert "the backend reported no token usage" in facts.basis[0]


def test_the_cached_answer_estimate_reads_the_winners_own_prompt() -> None:
    record = json.dumps({"system": "x" * 800, "user_template": "{input}", "records_measured": 0})
    facts = serving.facts_from_prompt(record, provider="openai", model="mini", prompt_chars=40)

    assert facts.tokens_in == pytest.approx(807 / 4)


def test_cells_after_the_one_that_wrote_the_predictions_do_not_price_the_winner() -> None:
    cells = [
        {
            "code": "m = HistGradientBoostingClassifier().fit(X_train, y_train)\n"
            "pd.Series(m.predict(X_holdout)).to_csv('predictions.csv', index=False)",
            "error": None,
        },
        {
            "code": "m2 = RandomForestClassifier().fit(X_train, y_train)\nprint(m2.score(X, y))",
            "error": None,
        },
    ]
    facts = serving.facts_from_code(cells, "", n_features=5)

    assert facts.estimator_family == "boosting"


@pytest.mark.parametrize("name", ["QuantileTransformer", "KNeighborsTransformer", "SGDOneClassSVM"])
def test_a_preprocessor_that_shares_a_prefix_with_an_estimator_is_not_one(name: str) -> None:
    assert serving.estimator_family(name) is None


def test_dropped_stages_narrow_the_head_and_are_said_out_loud() -> None:
    whole = serving.facts_from_recipe(
        {"backbone": "resnet18", "image_size": 224, "head": [("linear", 256)]}
    )
    cut = serving.facts_from_recipe(
        {"backbone": "resnet18", "image_size": 224, "drop_stages": 1, "head": [("linear", 256)]}
    )

    assert whole.weights is not None
    assert cut.weights is not None
    # The head starts from the kept stage's width, 256, instead of the last stage's 512.
    assert whole.weights - cut.weights == (512 + 1) * 256 - (256 + 1) * 256
    assert cut.basis[0] == "resnet18 with 1 stage dropped and head linear(256) at 224 px"
    assert "served network is smaller than this" in cut.basis[-1]


def test_a_stack_with_nothing_to_compute_prices_one_box_instead_of_dividing_by_zero() -> None:
    facts = serving.facts_from_recipe(
        {"backbone": "layers_net", "image_size": 64, "layers": [("pool",)]}
    )
    profile = serving.profile(facts, 1000, _prices())

    assert profile.chosen is not None
    assert profile.chosen.capacity_per_hour is None


def test_a_mixture_model_is_sized_by_all_its_experts() -> None:
    facts = serving.facts_from_prompt(None, provider="ollama", model="mixtral:8x7b")

    assert facts.parameters == 56_000_000_000


def test_a_recipe_with_layers_prices_the_stack_whatever_the_backbone_says() -> None:
    facts = serving.facts_from_recipe(
        {"backbone": "resnet18", "image_size": 64, "layers": [("conv", 16), ("linear", 8)]}
    )

    assert facts.backbone == "layers_net"
    assert facts.basis[0].startswith("your own stack conv(16) linear(8)")


def test_an_empty_recipe_is_unpriced_rather_than_priced_as_a_default() -> None:
    profile = serving.profile(serving.facts_from_recipe({}), 1000, _prices())

    assert profile.unpriced_because == "the winner left no recipe to price"


@pytest.mark.parametrize(
    ("backend", "base_url", "provider"),
    [
        ("openai", None, "openai"),
        ("openai-compatible", "https://api.openai.com/v1", "openai"),
        ("openai-compatible", "https://API.together.xyz/v1/", "together"),
        ("openai-compatible", "http://10.0.0.5:8000/v1", "openai-compatible"),
        ("vllm", None, "vllm"),
    ],
)
def test_the_provider_is_the_cloud_the_base_url_points_at(
    backend: str, base_url: str | None, provider: str
) -> None:
    assert serving.provider_for(backend, base_url) == provider


def test_the_reference_table_refuses_a_backbone_with_no_size() -> None:
    with pytest.raises(ValidationError, match="no measured size"):
        CpuReference(measured_on="x", backbone_ms={"resnet18": {}}, tabular_ms={})


def test_a_one_box_machine_says_the_rate_is_not_estimated_on_its_own_line() -> None:
    facts = serving.facts_from_prompt(None, provider="ollama", model="gemma4:12b")
    line = serving.profile(facts, 1000, _prices()).render()[0]

    assert line.startswith("serving: about $365.00 a month for one aws t4 (whether it serves 1,000")


def test_the_basis_says_interpolated_when_the_size_was_not_measured() -> None:
    facts = serving.facts_from_recipe({"backbone": "resnet18", "image_size": 96})
    profile = serving.profile(facts, 1000, _prices())

    assert any("interpolated between sizes measured" in line for line in profile.basis)
