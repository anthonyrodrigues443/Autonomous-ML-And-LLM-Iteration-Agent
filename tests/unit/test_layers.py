"""Tests for the layer grammar: it is pure Python, so CI covers every form a model can
type, every refusal by its text, and the weight, multiply-add and memory numbers that
the real `_simple_cnn` pins."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from evals.config import REPO_ROOT
from iterate.targets import layers
from iterate.targets.layers import RecipeError, Spec

pytestmark = pytest.mark.unit

CANONICAL: Spec = (
    ("conv", 32),
    ("pool",),
    ("conv", 64, 5),
    ("dropout", 0.3),
    ("linear", 256),
)
TYPED: list[Any] = [
    [("conv", 32), ("pool",), ("conv", 64, 5), ("dropout", 0.3), ("linear", 256)],
    (("conv", 32), ("pool",), ("conv", 64, 5), ("dropout", 0.3), ("linear", 256)),
    [["conv", 32], ["pool"], ["conv", 64, 5], ["dropout", 0.3], ["linear", 256]],
    ["conv:32", "pool", "conv:64:5", "dropout 0.3", "linear 256"],
    ["conv32", ("pool"), "cnn 64 5", "drop 0.3", "fc 256"],
    [{"conv": 32}, {"pool": []}, {"conv": [64, 5]}, {"dropout": 0.3}, {"linear": 256}],
    "conv(32) pool conv(64,5) dropout(0.3) linear(256)",
    "conv:32, pool, conv:64:5, dropout:0.3, dense:256",
    "cnn32 -> maxpool -> conv2d 64 5 -> drop0.3 -> fully_connected 256",
    '[("conv", 32), ("pool",), ("conv", 64, 5), ("dropout", 0.3), ("linear", 256)]',
]


# ─── the loose reader ────────────────────────────────────────────────────────


@pytest.mark.parametrize("typed", TYPED)
def test_every_form_a_model_types_reads_as_the_same_layers(typed: Any) -> None:
    assert layers.parse(typed) == CANONICAL


@pytest.mark.parametrize("typed", TYPED)
def test_the_canonical_text_reads_back_as_itself(typed: Any) -> None:
    spec = layers.parse(typed)
    assert spec is not None
    assert layers.parse(layers.text(spec)) == spec
    assert layers.parse(layers.code(spec)) == spec


def test_the_canonical_text_and_code_are_the_one_form_the_prompts_show() -> None:
    assert layers.text(CANONICAL) == "conv(32) pool conv(64,5) dropout(0.3) linear(256)"
    assert layers.code(CANONICAL) == (
        '[("conv", 32), ("pool",), ("conv", 64, 5), ("dropout", 0.3), ("linear", 256)]'
    )
    assert layers.text(layers.SIMPLE_CNN) == "conv(32) pool conv(64) pool conv(128) pool"
    assert layers.parse(layers.EXAMPLE) is not None
    assert layers.parse(layers.HEAD_EXAMPLE, "head") is not None


@pytest.mark.parametrize(
    ("typed", "spec"),
    [
        (None, None),
        ([], None),
        ("", None),
        (("conv", 32), (("conv", 32),)),
        ("pool", (("pool",),)),
        ([("conv", 32.0, 3.0, 1.0)], (("conv", 32),)),
        ([("conv", 32, 3, 1)], (("conv", 32),)),
        ([("conv", 32, 5, 1)], (("conv", 32, 5),)),
        ([("conv", 32, 3, 2)], (("conv", 32, 3, 2),)),
        ([("pool", "max")], (("pool",),)),
        ([("pool", 2)], (("pool",),)),
        ([("pool", "avg")], (("pool", "avg"),)),
        (["avgpool"], (("pool", "avg"),)),
        ([("dropout", 0.3)], (("dropout", 0.3),)),
    ],
)
def test_the_canonical_form_drops_what_the_builder_defaults(typed: Any, spec: Spec | None) -> None:
    assert layers.parse(typed) == spec


def test_a_parsed_stack_is_hashable_and_json_safe() -> None:
    spec = layers.parse(json.loads(json.dumps([["conv", 32], ["pool"], ["dropout", 0.3]])))
    assert spec == (("conv", 32), ("pool",), ("dropout", 0.3))
    assert len({spec, layers.parse("conv(32) pool dropout(0.3)")}) == 1
    assert json.loads(json.dumps(spec)) == [["conv", 32], ["pool"], ["dropout", 0.3]]


def test_no_spec_reads_and_writes_as_nothing() -> None:
    assert layers.text(None) == ""
    assert layers.code(None) == "None"
    assert layers.found_strict(None) is None
    layers.check((), size=64, batch=64, outputs=10)


def test_a_head_reads_the_same_way() -> None:
    assert layers.parse("linear(512) dropout(0.5)", "head") == (("linear", 512), ("dropout", 0.5))


@pytest.mark.parametrize(
    "clause",
    [
        "use conv(32) pool conv(64)",
        "layers: conv(32) pool conv(64)",
        "layers=conv(32) pool conv(64)",
        "change the layers to conv(32) pool conv(64)",
        "switch to conv(32) -> pool -> conv(64)",
    ],
)
def test_a_change_clause_reads_as_the_stack_it_names(clause: str) -> None:
    assert layers.parse(clause) == (("conv", 32), ("pool",), ("conv", 64))


@pytest.mark.parametrize(
    "prose",
    [
        "try a custom CNN, 20 epochs",
        "simple_cnn 64px",
        "a CNN (2015)",
        "swap to convnext_tiny",
        "20 epochs",
    ],
)
def test_prose_behind_a_leading_word_is_still_not_layers(prose: str) -> None:
    with pytest.raises(RecipeError, match="must be layers like"):
        layers.parse(prose)


@pytest.mark.parametrize(
    ("typed", "spec"),
    [
        ("conv-32, pool, conv-64", (("conv", 32), ("pool",), ("conv", 64))),
        ("conv-32-pool-conv-64", (("conv", 32), ("pool",), ("conv", 64))),
        ("linear-256", (("linear", 256),)),
        ("dropout-0.3", (("dropout", 0.3),)),
        (
            "conv32-conv64-pool-drop0.3-fc256",
            (("conv", 32), ("conv", 64), ("pool",), ("dropout", 0.3), ("linear", 256)),
        ),
    ],
)
def test_a_hyphen_before_a_number_separates_it_and_is_not_a_minus(typed: str, spec: Spec) -> None:
    assert layers.parse(typed) == spec


@pytest.mark.parametrize(
    "typed",
    [
        [("conv", "32"), ("pool",), ("linear", "256")],
        [["conv", "32"], ["pool"], ["linear", "256"]],
        [{"conv": "32"}, {"pool": []}, {"linear": "256"}],
        '[["conv", "32"], ["pool"], ["linear", "256"]]',
    ],
)
def test_a_number_quoted_as_a_string_reads_as_the_number(typed: Any) -> None:
    assert layers.parse(typed) == (("conv", 32), ("pool",), ("linear", 256))


def test_a_flat_layer_holding_a_quoted_number_is_one_layer() -> None:
    assert layers.parse(["conv", "32"]) == (("conv", 32),)
    assert layers.parse([("dropout", "0.3")]) == (("dropout", 0.3),)
    with pytest.raises(RecipeError, match="conv takes channels from 4 to 512"):
        layers.parse([("conv", "wide")])


def test_a_layer_that_takes_nothing_reads_as_null_or_as_an_empty_list() -> None:
    stack = (("conv", 32), ("pool",), ("linear", 256))
    assert layers.parse(json.loads('[{"conv": 32}, {"pool": null}, {"linear": 256}]')) == stack
    assert layers.parse('[["conv", 32], ["pool", null], ["linear", 256]]') == stack
    assert layers.parse([{"conv": 32}, {"pool": []}, {"linear": 256}]) == stack
    with pytest.raises(RecipeError, match="dropout takes one share"):
        layers.parse([("dropout", None)])


def test_a_json_string_reads_as_the_value_it_decodes_to() -> None:
    stack = (("conv", 32), ("pool",), ("conv", 64))
    assert layers.parse('[{"conv": 32}, {"pool": []}, {"conv": 64}]') == stack
    assert layers.parse('[["conv", 32], ["pool"], ["conv", 64]]') == stack
    assert layers.parse('{"conv": 32}') == (("conv", 32),)
    assert layers.parse("[conv(32) pool conv(64)]") == stack


def test_a_refusal_names_what_the_reader_has_to_change() -> None:
    with pytest.raises(RecipeError, match="remove '%'"):
        layers.parse(["conv(32)", "dropout 30%"])
    with pytest.raises(RecipeError, match=r"30 looks like a percentage, so write 0\.3"):
        layers.parse([("conv", 32), ("dropout", 30)])


@pytest.mark.parametrize(
    ("typed", "word"),
    [
        ("conv(32) then pool then conv(64)", "then"),
        ("conv(32) and pool and conv(64)", "and"),
        ("conv(32) followed by pool", "followed"),
    ],
)
def test_a_word_between_two_layers_is_named_in_the_refusal(typed: str, word: str) -> None:
    with pytest.raises(RecipeError, match=f"'{word}' is not a layer name"):
        layers.parse(typed)


# ─── the strict reader ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prose",
    [
        "try a custom CNN, 20 epochs",
        "simple_cnn 64px",
        "a CNN (2015)",
        "linear probe",
        "conv 3x3",
        "never build a network from scratch",
        "use resnet50 with a custom head and the last block unfrozen",
        "the paper trains a convnet on 128 px images for 20 epochs",
        "drop the linear layer and pool the features",
        "use a linear(512) head",
        "conv(2000) pool conv(64)",
        "",
    ],
)
def test_an_ask_of_prose_opens_nothing(prose: str) -> None:
    assert layers.found_strict(prose) is None


@pytest.mark.parametrize(
    ("ask", "spec"),
    [
        (
            "build conv(32) pool conv(64) pool dropout(0.3) linear(256) from scratch",
            (("conv", 32), ("pool",), ("conv", 64), ("pool",), ("dropout", 0.3), ("linear", 256)),
        ),
        ("head=linear(512) dropout(0.5)", (("linear", 512), ("dropout", 0.5))),
        ("conv(32), pool, conv(64)", (("conv", 32), ("pool",), ("conv", 64))),
        ("layers: conv(32) -> pool -> linear(128)", (("conv", 32), ("pool",), ("linear", 128))),
    ],
)
def test_an_ask_that_writes_the_stack_out_is_read(ask: str, spec: Spec) -> None:
    assert layers.found_strict(ask) == spec


@pytest.mark.parametrize(
    "prose",
    [
        "each stage doubles the width: conv(64), conv(128), conv(256), conv(512)",
        "the two branches use conv(64), conv(128) filters respectively",
    ],
)
def test_widths_counted_off_in_prose_open_nothing(prose: str) -> None:
    assert layers.found_strict(prose) is None


def test_a_stack_that_shrinks_the_map_is_still_read() -> None:
    assert layers.found_strict("conv(32,3,2) conv(64,3,2)") == (
        ("conv", 32, 3, 2),
        ("conv", 64, 3, 2),
    )


def test_a_written_stack_reads_however_the_sentence_judges_it() -> None:
    """A known boundary: the caller that owns research findings filters the comparative
    ones, because the layer grammar reads the stack either way."""
    assert layers.found_strict("resnet50 gets 0.94; a scratch conv(32) pool conv(64) net") == (
        ("conv", 32),
        ("pool",),
        ("conv", 64),
    )


def test_what_the_prompts_show_is_what_an_ask_can_copy_back() -> None:
    assert layers.found_strict(layers.text(CANONICAL)) == CANONICAL
    assert layers.found_strict(layers.EXAMPLE) == layers.parse(layers.EXAMPLE)


# ─── refusals ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("typed", "reason"),
    [
        (42, "layers must be layers like"),
        ("20 epochs", "layers must be layers like"),
        ({"type": "conv", "out_channels": 32}, "layers must be layers like"),
        ([42], "layers[0] is 42; write the name first"),
        ([("elu", 3)], "layers[0] names 'elu'; the layers are conv, pool, dropout and linear"),
        ([("relu",)], "conv already ends in batch norm and relu, so drop this layer"),
        (["batchnorm"], "conv already ends in batch norm and relu, so drop this layer"),
        (["batchnormalization"], "conv already ends in batch norm and relu, so drop this layer"),
        (["flatten"], "fit() pools the map to one vector for you, so drop this layer"),
        (["globalaveragepooling2d"], "fit() pools the map to one vector for you"),
        (["softmax"], "fit() adds the final layer and the loss does the softmax"),
        (
            [("conv", 3, 32, 3)],
            "conv takes the output channels first and works out the input channels itself",
        ),
        ([("conv", 32, "same")], "layers[0] is ('conv', 32, 'same'): conv takes channels"),
        ([("conv",)], "layers[0] is ('conv',): conv takes channels"),
        ([("conv", 32, 5, 2, 1)], "conv takes channels from 4 to 512"),
        ([("conv", 2)], "layers[0] conv channels=2 is outside 4 to 512"),
        ([("conv", 600)], "layers[0] conv channels=600 is outside 4 to 512"),
        ([("conv", -32)], "layers[0] conv channels=-32 is outside 4 to 512"),
        ([("conv", 32, 9)], "layers[0] conv kernel=9 must be one of 1, 3, 5, 7"),
        ([("conv", 32, 2)], "layers[0] conv kernel=2 must be one of 1, 3, 5, 7"),
        ([("conv", 32, 3, 3)], "layers[0] conv stride=3 must be 1 or 2"),
        ([("conv", 10**400)], "layers[0] holds a number that is not a size"),
        ([("conv", float("inf"))], "layers[0] holds a number that is not a size"),
        ([("linear", 10**400)], "layers[0] holds a number that is not a size"),
        ([("relu", 10**400)], "conv already ends in batch norm and relu"),
        ([("elu", 10**400)], "the layers are conv, pool, dropout and linear"),
        ([("pool", 3)], 'pool halves the image and takes nothing, or "avg"'),
        ([("dropout", 1.5)], "dropout takes one share between 0.05 and 0.9"),
        ([("dropout", 0.0)], "dropout takes one share between 0.05 and 0.9"),
        ([("dropout", True)], "dropout takes one share between 0.05 and 0.9"),
        ([("dropout",)], "dropout takes one share between 0.05 and 0.9"),
        ([("linear", 4096)], "linear takes one width between 8 and 2048"),
        ([("linear", 4)], "linear takes one width between 8 and 2048"),
        ([("linear", 256, 128)], "linear takes one width between 8 and 2048"),
    ],
)
def test_a_layer_the_builder_cannot_make_is_refused_by_name(typed: Any, reason: str) -> None:
    with pytest.raises(RecipeError) as raised:
        layers.parse(typed)
    assert reason in str(raised.value)


@pytest.mark.parametrize(
    ("typed", "name", "extra", "reason"),
    [
        (
            [("conv", 32)] * 13,
            "layers",
            {},
            "layers has 13 layers and the limit is 12; drop or merge some",
        ),
        (
            [("linear", 512)] * 5,
            "head",
            {},
            "head has 5 layers and the limit is 4; drop or merge some",
        ),
        (
            ["pool", ("conv", 32)],
            "layers",
            {},
            "layers[0] must be a conv layer; pool, dropout and linear come after one",
        ),
        (
            [("conv", 32), ("linear", 256), ("conv", 64)],
            "layers",
            {},
            "layers[2] is a conv after a linear layer; conv and pool layers come first",
        ),
        (
            [("conv", 32), ("linear", 256), ("pool",)],
            "layers",
            {},
            "layers[2] is a pool after a linear layer",
        ),
        (
            [("linear", 512), ("conv", 32)],
            "head",
            {},
            "head[1] names 'conv'; a head takes linear and dropout only, because the backbone "
            "already hands it one vector per image",
        ),
        (
            [("conv", 32), ("linear", 10)],
            "layers",
            {"outputs": 10},
            "layers ends in linear(10), the number of outputs; fit() adds the final layer "
            "itself, so remove it",
        ),
        (
            [("linear", 102)],
            "head",
            {"outputs": 102},
            "head ends in linear(102), the number of outputs",
        ),
        (
            [("conv", 512, 7)] * 4,
            "layers",
            {},
            "layers has about 38.6M weights and the limit is 30M (convnext_tiny has 28.6M); "
            "lower the widest conv or linear layers",
        ),
        (
            [("conv", 512), ("conv", 512)],
            "layers",
            {"size": 224, "batch": 64},
            "layers does about 119 billion multiply-adds per image at 224 px and the limit "
            "is 8 (resnet50 at 224 px does 4)",
        ),
        (
            [("conv", 512)],
            "layers",
            {"size": 224, "batch": 64},
            "layers keeps about 18.4 GB of activations at 224 px and batch 64, and the limit "
            "is 6 GB; put a pool after the first conv, lower the channels, or lower batch_size",
        ),
        (
            [("conv", 32), *[("pool",)] * 7],
            "layers",
            {"size": 64},
            "layers[7] puts a pool on a 1 px map: a 64 px image survives 6 halvings (a pool or "
            "a stride 2 conv); drop one, or raise image_size",
        ),
        (
            [("conv", 32), *[("pool",)] * 6, ("conv", 64)],
            "layers",
            {"size": 64},
            "layers[7] puts a conv on a 1 px map: a 64 px image survives 6 halvings",
        ),
        (
            [("conv", 32, 3, 2)] * 6 + [("conv", 64)],
            "layers",
            {"size": 32},
            "layers[5] puts a conv on a 1 px map: a 32 px image survives 5 halvings",
        ),
    ],
)
def test_a_stack_the_runner_cannot_train_is_refused_by_name(
    typed: Any, name: str, extra: dict[str, Any], reason: str
) -> None:
    spec = layers.parse(typed, name)
    assert spec is not None
    with pytest.raises(RecipeError) as raised:
        layers.check(spec, name, **extra)
    assert reason in str(raised.value)


def test_a_stack_that_only_the_session_size_refuses_passes_until_it_is_known() -> None:
    spec = layers.parse([("conv", 32), *[("pool",)] * 7])
    assert spec is not None
    layers.check(spec)
    layers.check(spec, size=384)
    with pytest.raises(RecipeError):
        layers.check(spec, size=64)


def test_the_caps_that_scale_with_the_batch_wait_for_the_batch() -> None:
    spec = layers.parse([("conv", 512)])
    assert spec is not None
    layers.check(spec, size=224)
    with pytest.raises(RecipeError, match="activations"):
        layers.check(spec, size=224, batch=64)


def test_a_smaller_batch_does_not_buy_a_more_expensive_network() -> None:
    spec = layers.parse([("conv", 256), ("conv", 256)])
    assert spec is not None
    for batch in (None, 8, 16, 64):
        with pytest.raises(RecipeError, match="multiply-adds per image"):
            layers.check(spec, size=224, batch=batch)


def test_the_baseline_stack_passes_every_check() -> None:
    layers.check(layers.SIMPLE_CNN, size=64, batch=64, outputs=10)
    layers.check(layers.parse(layers.EXAMPLE) or (), size=160, batch=64, outputs=102)
    layers.check(layers.parse(layers.HEAD_EXAMPLE, "head") or (), "head", outputs=102)


# ─── the numbers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outputs", [1, 10, 102])
def test_the_weight_count_is_the_one_the_real_simple_cnn_pins(outputs: int) -> None:
    assert layers.count_weights(layers.SIMPLE_CNN, outputs) == 93_696 + 129 * outputs


def test_the_weight_count_reads_a_head_from_the_backbone_width() -> None:
    head = layers.parse(layers.HEAD_EXAMPLE, "head")
    assert head is not None
    assert layers.count_weights(head, 10, features=512) == (512 + 1) * 512 + (512 + 1) * 10


def test_the_cost_of_a_head_reads_the_backbone_width_too() -> None:
    head = layers.parse(layers.HEAD_EXAMPLE, "head")
    assert head is not None
    assert layers.count_macs(head, 224, "head", features=2048) == 2048 * 512
    lead = layers.parse("dropout(0.5) linear(512)", "head")
    assert lead is not None
    assert layers.activation_bytes(lead, 224, 1, "head", features=2048) == 4 * (2048 + 2 * 512)


def test_the_multiply_add_count_is_the_measured_one_for_simple_cnn() -> None:
    assert layers.count_macs(layers.SIMPLE_CNN, 64) == 41_287_680


def test_the_activation_estimate_scales_with_the_batch() -> None:
    one = layers.activation_bytes(layers.SIMPLE_CNN, 64, 1)
    assert layers.activation_bytes(layers.SIMPLE_CNN, 64, 64) == 64 * one
    assert layers.activation_bytes(layers.parse([("conv", 512)]) or (), 224, 64) > 18 * 2**30


# ─── imports ─────────────────────────────────────────────────────────────────


def test_the_grammar_imports_with_no_torch_and_no_third_party() -> None:
    code = (
        "import sys\n"
        "import iterate.targets.layers\n"
        "print(sorted({'torch', 'numpy', 'pandas', 'sklearn'} & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "[]"
