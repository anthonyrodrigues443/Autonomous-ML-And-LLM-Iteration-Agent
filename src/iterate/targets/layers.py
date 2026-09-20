"""Architecture as data: the layer lists `fit()` takes, read and checked without torch.

`layers=[...]` is a whole network trained from zero; `head=[...]` replaces the single
linear layer on a pretrained backbone. Both are lists of (name, numbers), parsed into
tuples a frozen `Recipe` can hold and JSON can carry. There are four names. `conv` is a
whole block (convolution, batch norm, ReLU) taking channels and an optional kernel and
stride; `pool` halves the map; `dropout` takes a share; `linear` is a fully connected
layer plus ReLU. The harness puts global average pooling between the conv part and the
linear part, and always adds the final layer, one output per class or one number.

`parse` is the loose reader, for a `fit()` argument and a brief's change clause, and it
takes every form a model types. `found_strict` is the tight one, for a user ask and a
research finding, where prose must read as nothing at all. `text` is the one canonical
form shown to a model, `code` the Python it can paste back. Nothing here imports torch,
so CI checks every refusal; `build_scratch` and `build_head` take `torch` as an argument
the way the rest of the runner does.
"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from itertools import pairwise
from typing import Any

Layer = tuple[str | int | float, ...]
Spec = tuple[Layer, ...]

KINDS = ("conv", "pool", "dropout", "linear")
HEAD_KINDS = ("linear", "dropout")
MAX_LAYERS = 12
MAX_HEAD = 4
MAX_PARAMS = 30_000_000
# The compute cap is per image: an epoch costs the same multiply-adds whatever the batch
# size is, so a smaller batch must not buy a more expensive network.
MAX_MACS = 8_000_000_000
MAX_ACTIVATION_BYTES = 6 * 2**30
CHANNELS = (4, 512)
WIDTHS = (8, 2048)
KERNELS = (1, 3, 5, 7)
STRIDES = (1, 2)
DROPOUT = (0.05, 0.9)
EXAMPLE = "conv(32) pool conv(64) pool dropout(0.3) linear(256)"
HEAD_EXAMPLE = "linear(512) dropout(0.5)"
SIMPLE_CNN: Spec = (("conv", 32), ("pool",), ("conv", 64), ("pool",), ("conv", 128), ("pool",))

_FLOAT_BYTES = 4
# Sizes never reach this, and a number that does costs nothing to reject early.
_MAX_NUMBER = 10**12
_ALIASES = {
    "cnn": "conv",
    "conv2d": "conv",
    "convolution": "conv",
    "convolutional": "conv",
    "maxpool": "pool",
    "maxpool2d": "pool",
    "max_pool": "pool",
    "maxpooling2d": "pool",
    "max_pooling2d": "pool",
    "pooling": "pool",
    "avgpool": "pool",
    "avgpool2d": "pool",
    "avg_pool": "pool",
    "averagepooling2d": "pool",
    "average_pooling2d": "pool",
    "fc": "linear",
    "dense": "linear",
    "fully_connected": "linear",
    "fullyconnected": "linear",
    "drop": "dropout",
    "dropout2d": "dropout",
}
_AVG = frozenset({"avgpool", "avgpool2d", "avg_pool", "averagepooling2d", "average_pooling2d"})
_INSIDE_CONV = "conv already ends in batch norm and relu, so drop this layer"
_ADDED_FLAT = "fit() pools the map to one vector for you, so drop this layer"
_ADDED_FINAL = "fit() adds the final layer and the loss does the softmax, so drop this layer"
_REFUSED = {
    "relu": _INSIDE_CONV,
    "gelu": _INSIDE_CONV,
    "silu": _INSIDE_CONV,
    "activation": _INSIDE_CONV,
    "batchnorm": _INSIDE_CONV,
    "batchnorm2d": _INSIDE_CONV,
    "batchnormalization": _INSIDE_CONV,
    "layernorm": _INSIDE_CONV,
    "flatten": _ADDED_FLAT,
    "gap": _ADDED_FLAT,
    "globalaveragepooling2d": _ADDED_FLAT,
    "global_average_pooling2d": _ADDED_FLAT,
    "globalavgpool": _ADDED_FLAT,
    "adaptiveavgpool2d": _ADDED_FLAT,
    "softmax": _ADDED_FINAL,
    "logsoftmax": _ADDED_FINAL,
    "sigmoid": _ADDED_FINAL,
}
# Longest first, so `maxpool2d` wins over `pool`; the second key only pins the order.
_NAME_WORDS = sorted({*KINDS, *_ALIASES, *_REFUSED}, key=lambda word: (-len(word), word))
_WORDS = "|".join(_NAME_WORDS)
# The lookbehind keeps `cnn` out of `simple_cnn`; the lookahead still allows `conv32`.
_NAME_AT = re.compile(rf"(?<![a-z0-9_])(?:{_WORDS})(?![a-z_])")
_STRICT_AT = re.compile(rf"(?<![a-z0-9_])({_WORDS})(?![a-z_])\s*(\([^()]*\))?")
# No layer takes a negative number, so a leading "-" is a separator, never a sign.
_TOKEN = re.compile(r"[a-z_]+(?:2d)?|\d*\.\d+|\d+")
_FILLER = re.compile(r"[\s:=,()\[\]'\"|;+>-]+")
# What may sit between two layers of one ask: separators only, never a word.
_GAP = re.compile(r"[\s,;+|>\-\[\]'\"]*")


class RecipeError(ValueError):
    """A recipe the runner will not execute, with a reason the agent can act on."""


def parse(value: Any, name: str = "layers") -> Spec | None:
    """The canonical tuples for what a model typed: tuples, JSON lists, "conv:32"
    strings, one-key dicts or the canonical text. None and an empty list are no spec."""
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 1:
        value = [value]
    if isinstance(value, str):
        if not value.strip():
            return None
        if value.lstrip()[:1] in ("[", "{"):
            try:
                return parse(json.loads(value), name)
            except json.JSONDecodeError:
                pass
        value = _chunks(value, name)
    if not isinstance(value, list | tuple):
        raise _not_layers(value, name)
    if value and isinstance(value[0], str) and any(_is_number(_number(v)) for v in value[1:]):
        value = [value]
    return tuple(_layer(item, f"{name}[{i}]") for i, item in enumerate(value)) or None


def found_strict(value: Any, name: str = "layers") -> Spec | None:
    """The stack a user ask or a research finding states outright, or None. Only
    name(number) counts, a bare name other than pool ends the run, a run of fewer than
    two valued layers is prose, and so is a run of convs that never shrinks the map. A
    written-out stack reads whatever the sentence says about it, so a finding that
    measures one and calls it the loser still opens the lever; the caller that owns
    findings filters comparative sentences."""
    if not isinstance(value, str):
        return None
    low = value.lower()
    run: list[str] = []
    end = 0
    for match in _STRICT_AT.finditer(low):
        broken = bool(run) and not _GAP.fullmatch(low[end : match.start()])
        bare = match.group(2) is None and _ALIASES.get(match.group(1), match.group(1)) != "pool"
        if broken or bare:
            if (found := _run(run, name)) is not None:
                return found
            run = []
        if not bare:
            run.append(match.group(0))
        end = match.end()
    return _run(run, name)


def text(value: Any, name: str = "layers") -> str:
    """The one canonical text: what describe(), the FIT line and the ready line show,
    and what parse() reads back."""
    return " ".join(_written(layer) for layer in parse(value, name) or ())


def code(value: Any, name: str = "layers") -> str:
    """The Python a coder can paste into fit()."""
    spec = parse(value, name)
    if spec is None:
        return "None"
    return "[" + ", ".join(_pasted(layer) for layer in spec) + "]"


def check(
    spec: Spec,
    name: str = "layers",
    *,
    size: int | None = None,
    batch: int | None = None,
    outputs: int | None = None,
) -> None:
    """Every refusal a whole stack earns. `size` is the image size and `batch` the batch
    size, both known only once the session fills them in; with either missing the caps
    they scale are not checked."""
    if not spec:
        return
    limit = MAX_HEAD if name == "head" else MAX_LAYERS
    if len(spec) > limit:
        raise RecipeError(
            f"{name} has {len(spec)} layers and the limit is {limit}; drop or merge some"
        )
    kinds = [str(layer[0]) for layer in spec]
    if name == "head":
        for i, kind in enumerate(kinds):
            if kind not in HEAD_KINDS:
                raise RecipeError(
                    f"{name}[{i}] names {kind!r}; a head takes linear and dropout only, because "
                    "the backbone already hands it one vector per image"
                )
    else:
        if kinds[0] != "conv":
            raise RecipeError(
                f"{name}[0] must be a conv layer; pool, dropout and linear come after one"
            )
        dense = kinds.index("linear") if "linear" in kinds else len(kinds)
        for i, kind in enumerate(kinds):
            if i > dense and kind in ("conv", "pool"):
                raise RecipeError(
                    f"{name}[{i}] is a {kind} after a linear layer; conv and pool layers come "
                    "first, linear layers last"
                )
    last = spec[-1]
    if outputs is not None and last[0] == "linear" and int(last[1]) == outputs:
        raise RecipeError(
            f"{name} ends in linear({outputs}), the number of outputs; fit() adds the final "
            "layer itself, so remove it"
        )
    if name == "head":
        return
    if (weights := count_weights(spec)) > MAX_PARAMS:
        raise RecipeError(
            f"{name} has about {weights / 1e6:.1f}M weights and the limit is "
            f"{MAX_PARAMS / 1e6:.0f}M (convnext_tiny has 28.6M); lower the widest conv or "
            "linear layers"
        )
    if size is None:
        return
    macs, elements = _cost(spec, size, name)
    if macs > MAX_MACS:
        raise RecipeError(
            f"{name} does about {macs / 1e9:.0f} billion multiply-adds per image at {size} px "
            f"and the limit is {MAX_MACS / 1e9:.0f} (resnet50 at 224 px does 4); put a pool "
            "after the first conv, lower the channels, or lower image_size"
        )
    if batch is None:
        return
    if (used := elements * _FLOAT_BYTES * batch) > MAX_ACTIVATION_BYTES:
        raise RecipeError(
            f"{name} keeps about {used / 2**30:.1f} GB of activations at {size} px and batch "
            f"{batch}, and the limit is {MAX_ACTIVATION_BYTES / 2**30:.0f} GB; put a pool "
            "after the first conv, lower the channels, or lower batch_size"
        )


def count_weights(spec: Spec, outputs: int = 0, *, features: int = 3) -> int:
    """Weights and biases, batch norm included. `outputs` adds the final layer, which is
    the builder's and is not part of the spec."""
    total, width = 0, features
    for layer in spec:
        if layer[0] == "conv":
            channels, kernel, _ = _conv(layer)
            total += width * channels * kernel * kernel + 3 * channels
            width = channels
        elif layer[0] == "linear":
            total += (width + 1) * int(layer[1])
            width = int(layer[1])
    return total + ((width + 1) * outputs if outputs else 0)


def count_macs(spec: Spec, size: int, name: str = "layers", *, features: int = 3) -> int:
    """Multiply-adds for one image. A head starts from the backbone's `features` wide
    vector, not from the image."""
    return _cost(spec, size, name, features)[0]


def activation_bytes(
    spec: Spec, size: int, batch: int, name: str = "layers", *, features: int = 3
) -> int:
    """What the forward pass keeps for the backward pass: every layer's output, in
    float32, for the whole batch."""
    return _cost(spec, size, name, features)[1] * _FLOAT_BYTES * batch


def build_head(torch: Any, spec: Spec, features: int, outputs: int) -> Any:
    """The dense tail on `features` numbers per image, with the final layer added. With
    no tail this is the bare Linear a stock backbone carries, so a head-free recipe
    builds what it always built."""
    nn = torch.nn
    parts: list[Any] = []
    width = features
    for layer in spec:
        if layer[0] == "linear":
            parts += [nn.Linear(width, int(layer[1])), nn.ReLU()]
            width = int(layer[1])
        else:
            parts.append(nn.Dropout(float(layer[1])))
    final = nn.Linear(width, outputs)
    return nn.Sequential(*parts, final) if parts else final


def build_scratch(torch: Any, spec: Spec, outputs: int) -> Any:
    """A whole network from a stack, trained from zero. The two modules are named `body`
    and `head`, as simple_cnn's are, so one freezing rule and one set of saved weight
    names cover both."""
    nn = torch.nn
    end = _body_end(spec)
    body: list[Any] = []
    width = 3
    for layer in spec[:end]:
        if layer[0] == "conv":
            channels, kernel, stride = _conv(layer)
            body += [
                nn.Conv2d(width, channels, kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ]
            width = channels
        elif layer[0] == "pool":
            body.append(nn.AvgPool2d(2) if len(layer) > 1 else nn.MaxPool2d(2))
        else:
            body.append(nn.Dropout2d(float(layer[1])))
    trunk = nn.Sequential(*body, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    head = build_head(torch, spec[end:], width, outputs)
    return nn.Sequential(OrderedDict(body=trunk, head=head))


def _not_layers(value: Any, name: str) -> RecipeError:
    example, shown = (
        (HEAD_EXAMPLE, '[("linear", 512), ("dropout", 0.5)]')
        if name == "head"
        else (EXAMPLE, '[("conv", 32), ("pool",), ("linear", 256)]')
    )
    return RecipeError(f"{name} must be layers like {shown} or {example!r}, got {value!r}")


def _chunks(value: str, name: str) -> list[str]:
    """One text, one chunk per layer name in it."""
    low = value.strip().lower()
    marks = [m.start() for m in _NAME_AT.finditer(low)]
    if not marks:
        raise _not_layers(value, name)
    if _FILLER.sub("", low[: marks[0]]):
        # A change clause leads with a word, so loose mode must read what strict does.
        if (found := found_strict(low, name)) is not None:
            return [_written(layer) for layer in found]
        raise _not_layers(value, name)
    return [low[a:b] for a, b in pairwise([*marks, len(low)])]


def _run(run: list[str], name: str) -> Spec | None:
    if len(run) < 2:
        return None
    try:
        spec = parse(" ".join(run), name)
    except RecipeError:
        return None
    valued = [layer for layer in spec or () if layer[0] != "pool"]
    if len(valued) < 2:
        return None
    kinds = {layer[0] for layer in spec or ()}
    # Widths enumerated in prose read as convs with nothing between them to shrink the map.
    if kinds == {"conv"} and not any(_conv(layer)[2] == 2 for layer in spec or ()):
        return None
    return spec


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number(value: Any) -> Any:
    """A number a model quoted as a string, as the number; anything else untouched."""
    if isinstance(value, str) and re.fullmatch(r"\d*\.\d+|\d+", value.strip()):
        return float(value) if "." in value else int(value)
    return value


def _absurd(value: int | float) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return abs(value) > _MAX_NUMBER


def _whole(value: Any) -> bool:
    return _is_number(value) and not _absurd(value) and float(value).is_integer()


def _words(chunk: str, at: str) -> list[Any]:
    low = chunk.strip().lower()
    if not low:
        raise RecipeError(
            f'{at} is {chunk!r}; write the name first, like ("conv", 32) or "conv:32"'
        )
    if left := _FILLER.sub("", _TOKEN.sub("", low)):
        raise RecipeError(
            f"{at} is {chunk!r}: remove {left!r}; write only the layers, like {EXAMPLE!r}"
        )
    words: list[Any] = [
        (float(t) if "." in t else int(t)) if t[0].isdigit() or t[0] == "." else t
        for t in _TOKEN.findall(low)
    ]
    for word in words[1:]:
        # pool is the only layer whose argument is a word.
        if isinstance(word, str) and word not in ("max", "avg"):
            raise RecipeError(
                f"{at} is {chunk!r}: {word!r} is not a layer name; write only the layers, "
                f"like {EXAMPLE!r}"
            )
    return words


def _layer(item: Any, at: str) -> Layer:
    if isinstance(item, dict) and len(item) == 1:
        ((key, args),) = item.items()
        item = [key, *(args if isinstance(args, list | tuple) else [] if args is None else [args])]
    if isinstance(item, list | tuple) and len(item) == 1 and isinstance(item[0], str):
        item = item[0]
    if isinstance(item, str):
        item = _words(item, at)
    if not isinstance(item, list | tuple) or not item or not isinstance(item[0], str):
        raise RecipeError(f'{at} is {item!r}; write the name first, like ("conv", 32) or "conv:32"')
    # JSON writes a layer that takes nothing as null and its numbers as strings.
    item = [item[0], *(_number(a) for a in item[1:] if a is not None)]
    raw = re.sub(r"[\s-]+", "_", item[0].strip().lower())
    kind = _ALIASES.get(raw, raw)
    args, shown = list(item[1:]), tuple(item)
    if raw in _REFUSED:
        raise RecipeError(f"{at} names {raw!r}; {_REFUSED[raw]}")
    if kind in KINDS and any(_is_number(a) and _absurd(a) for a in args):
        raise RecipeError(f"{at} holds a number that is not a size; {_takes(kind)}")
    if kind == "conv":
        return _conv_layer(args, at, shown)
    if kind == "pool":
        how = [str(a).lower() for a in args if a != 2]
        if len(how) > 1 or any(h not in ("max", "avg") for h in how):
            raise RecipeError(f"{at} is {shown}: {_takes('pool')}")
        return ("pool", "avg") if how == ["avg"] or raw in _AVG else ("pool",)
    if kind == "dropout":
        if not (len(args) == 1 and _is_number(args[0]) and DROPOUT[0] <= args[0] <= DROPOUT[1]):
            percent = (
                f"; {args[0]} looks like a percentage, so write {args[0] / 100:g}"
                if len(args) == 1 and _whole(args[0]) and 1 <= args[0] <= 100
                else ""
            )
            raise RecipeError(f"{at} is {shown}: {_takes('dropout')}{percent}")
        return ("dropout", float(args[0]))
    if kind == "linear":
        if len(args) != 1 or not _whole(args[0]) or not WIDTHS[0] <= args[0] <= WIDTHS[1]:
            raise RecipeError(f"{at} is {shown}: {_takes('linear')}")
        return ("linear", int(args[0]))
    raise RecipeError(f"{at} names {raw!r}; the layers are conv, pool, dropout and linear")


def _conv_layer(args: list[Any], at: str, shown: tuple[Any, ...]) -> Layer:
    if len(args) == 3 and _whole(args[0]) and args[0] <= 4 and _whole(args[1]) and args[1] > 8:
        raise RecipeError(
            f"{at} is {shown}: conv takes the output channels first and works out the input "
            'channels itself, like ("conv", 32) or ("conv", 32, 5)'
        )
    if not 1 <= len(args) <= 3 or not all(_whole(a) for a in args):
        raise RecipeError(f"{at} is {shown}: {_takes('conv')}")
    channels, kernel, stride = (int(a) for a in [*args, *(3, 1)[len(args) - 1 :]])
    if not CHANNELS[0] <= channels <= CHANNELS[1]:
        raise RecipeError(
            f"{at} conv channels={channels} is outside {CHANNELS[0]} to {CHANNELS[1]}"
        )
    if kernel not in KERNELS:
        raise RecipeError(f"{at} conv kernel={kernel} must be one of 1, 3, 5, 7")
    if stride not in STRIDES:
        raise RecipeError(f"{at} conv stride={stride} must be 1 or 2")
    full: Layer = ("conv", channels, kernel, stride)
    return full if stride != 1 else full[:3] if kernel != 3 else full[:2]


def _takes(kind: str) -> str:
    return {
        "conv": "conv takes channels from 4 to 512, then an optional kernel (1, 3, 5 or 7) "
        'and stride (1 or 2), like ("conv", 32) or ("conv", 32, 5, 2)',
        "pool": 'pool halves the image and takes nothing, or "avg"',
        "dropout": 'dropout takes one share between 0.05 and 0.9, like ("dropout", 0.3)',
        "linear": 'linear takes one width between 8 and 2048, like ("linear", 256)',
    }[kind]


def _conv(layer: Layer) -> tuple[int, int, int]:
    channels, kernel, stride = (int(v) for v in [*layer[1:], *(3, 1)[len(layer) - 2 :]])
    return channels, kernel, stride


def _body_end(spec: Spec) -> int:
    conv = [i for i, layer in enumerate(spec) if layer[0] in ("conv", "pool")]
    return conv[-1] + 1 if conv else 0


def _cost(spec: Spec, size: int, name: str, features: int = 3) -> tuple[int, int]:
    """Multiply-adds and kept activation elements for one image, and the refusal when a
    conv or a pool meets a 1 px map. The final layer is the builder's, so neither number
    counts it."""
    macs = elements = 0
    side, width, halved = size, features, 0
    body = _body_end(spec)
    for i, layer in enumerate(spec):
        kind = str(layer[0])
        if side < 2 and kind in ("conv", "pool"):
            raise RecipeError(
                f"{name}[{i}] puts a {kind} on a 1 px map: a {size} px image survives {halved} "
                "halvings (a pool or a stride 2 conv); drop one, or raise image_size"
            )
        if kind == "conv":
            channels, kernel, stride = _conv(layer)
            if stride == 2:
                side, halved = (side + 1) // 2, halved + 1
            macs += side * side * width * channels * kernel * kernel
            elements += 3 * side * side * channels
            width = channels
        elif kind == "pool":
            side, halved = side // 2, halved + 1
            elements += side * side * width
        elif kind == "dropout":
            elements += side * side * width if i < body else width
        else:
            macs += width * int(layer[1])
            elements += 2 * int(layer[1])
            width = int(layer[1])
        if i + 1 == body:
            elements += width
            side = 1
    return macs, elements


def _written(layer: Layer) -> str:
    kind = str(layer[0])
    if kind == "conv":
        channels, kernel, stride = _conv(layer)
        inside = [str(channels)]
        if stride != 1:
            inside += [str(kernel), str(stride)]
        elif kernel != 3:
            inside.append(str(kernel))
        return f"conv({','.join(inside)})"
    if kind == "pool":
        return "pool(avg)" if len(layer) > 1 else "pool"
    return f"{kind}({layer[1]})"


def _pasted(layer: Layer) -> str:
    inside = ", ".join(f'"{v}"' if isinstance(v, str) else repr(v) for v in layer)
    return f"({inside},)" if len(layer) == 1 else f"({inside})"


__all__ = [
    "EXAMPLE",
    "HEAD_EXAMPLE",
    "MAX_ACTIVATION_BYTES",
    "MAX_HEAD",
    "MAX_LAYERS",
    "MAX_MACS",
    "MAX_PARAMS",
    "SIMPLE_CNN",
    "Layer",
    "RecipeError",
    "Spec",
    "activation_bytes",
    "build_head",
    "build_scratch",
    "check",
    "code",
    "count_macs",
    "count_weights",
    "found_strict",
    "parse",
    "text",
]
