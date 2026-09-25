"""The code-gen contract: how generated code is run and scored.

The agent writes modelling code, never file I/O; a fixed harness loads the data
and writes the predictions. The script receives holdout features and never the
labels. A code candidate is a `Candidate` whose ``changes`` carries ``"code"``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core.scoring import direction, requires_proba, score, task_for_metric
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from iterate.adapters.data.tabular import TabularDataset

# File names exchanged with the runner's working directory.
TRAIN_CSV = "train.csv"
HOLDOUT_CSV = "holdout.csv"
META_JSON = "meta.json"
PREDICTIONS_CSV = "predictions.csv"
# Sibling artifact, deliberately NOT a second column in predictions.csv: that file's
# validator rejects a two-column line as an index-column mistake (`to_csv` without
# index=False), a guard that already caught the single worst failure in 100+ live
# iterations. Widening it to sometimes mean probabilities would blunt it.
PROBABILITIES_CSV = "probabilities.csv"
# The prompt path's deliverable, written by `submit()` at the same moment as the
# predictions it produced. A run's real output is the prompt you can put into
# production, and a notebook cannot tell you which of its cells held the winner.
PROMPT_JSON = "prompt.json"
# The image path's twin of PROMPT_JSON: the recipe (or own-model line) that produced
# the predictions on disk, with their digest.
RECIPE_JSON = "recipe.json"
# The network behind a submitted `fit()`, beside the predictions it made; recipe.json
# carries its digest. Dotted and harness-only: `model.pt` and `best_model.pt` are the
# names an agent's own torch code writes.
NETWORK_PT = ".iterate-network.pt"
# Written by the host before the session starts: the recipe the carried best used, so
# `fit()` in a new session starts where the last one finished.
INCUMBENT_JSON = "incumbent.json"
# Where a session records a module an import could not find. A cell that caught the
# ImportError leaves no traceback, so this file is the only evidence.
MISSING_IMPORTS = ".missing-imports"

# The single function the agent must define; the harness calls it.
ENTRY_POINT = "train_and_predict"

# The harness around the LLM's train_and_predict. Loads inputs, calls the
# function, writes predictions. The LLM source is inserted between the two halves.
_PREAMBLE = f"""\
import json
import pandas as pd

with open({META_JSON!r}) as _f:
    _meta = json.load(_f)
_target = _meta["target"]

_train = pd.read_csv({TRAIN_CSV!r})
X_train = _train.drop(columns=[_target])
y_train = _train[_target]
X_holdout = pd.read_csv({HOLDOUT_CSV!r})  # FEATURES ONLY — labels are held back

# ─── agent-provided train_and_predict below ───
"""

_POSTAMBLE = f"""
# ─── harness: run the agent's function and write predictions ───
# A 2-tuple return is the opt-in probability contract: (predictions, probabilities).
# Returning predictions alone stays valid, so every pre-v0.4 function is unaffected.
_out = {ENTRY_POINT}(X_train, y_train, X_holdout)
_preds, _proba = _out if isinstance(_out, tuple) and len(_out) == 2 else (_out, None)
pd.Series(list(_preds)).to_csv({PREDICTIONS_CSV!r}, index=False, header=False)
if _proba is not None:
    pd.DataFrame(_proba).to_csv({PROBABILITIES_CSV!r}, index=False, header=False)
"""

# import-name -> pip distribution name, for the cases where they differ. PROVISIONAL:
# this hand-kept map is a stop-gap. The architecture for resolving + installing the
# agent's imports is TBD (tracked in DECISIONS.md / LIMITATIONS.md) and will be
# revisited; for now an unknown import falls back to its own name and a bad guess
# simply fails the install, surfacing as a captured experiment failure.
_IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    # On PyPI the bare import name is a different project, or nothing.
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "faiss": "faiss-cpu",
    "fitz": "pymupdf",
    "imblearn": "imbalanced-learn",
    "mpl_toolkits": "matplotlib",
    "open_clip": "open-clip-torch",
    "umap": "umap-learn",
}


def is_code_candidate(changes: dict[str, Any]) -> bool:
    """True if a candidate's ``changes`` carries generated code rather than a spec."""
    return isinstance(changes.get("code"), str) and bool(changes["code"].strip())


def assemble_script(code: str) -> str:
    """Wrap the agent's `train_and_predict` source in the I/O harness."""
    return _PREAMBLE + code.strip() + "\n" + _POSTAMBLE


# Plain text, not an import of iterate: an e2b kernel has no iterate package.
# Appended LAST on sys.meta_path, so it is asked only after every real finder failed,
# and it records a name only when the import came straight from a cell — a library
# probing its own optional dependencies records nothing.
IMPORT_WATCH = f"""\
def _iterate_watch(log):
    import linecache, os, sys
    for found in sys.meta_path:
        if hasattr(found, 'iterate_log'):
            found.iterate_log = log
            return
    machinery = os.path.dirname(__import__('importlib').__file__)
    class MissingImportWatch:
        iterate_log = log
        seen = set()
        def find_spec(self, name, path=None, target=None):
            if name in self.seen or name.partition('.')[0] in sys.stdlib_module_names:
                return None
            frame = sys._getframe(1)
            while frame is not None and (
                frame.f_code.co_filename.startswith(('<frozen', machinery))
            ):
                frame = frame.f_back
            entry = linecache.cache.get(frame.f_code.co_filename) if frame else None
            if entry is None or entry[1] is not None:
                return None
            parent = name.rpartition('.')[0]
            if parent and getattr(sys.modules.get(parent), '__file__', 1) is not None:
                return None
            for later in sys.meta_path[sys.meta_path.index(self) + 1:]:
                try:
                    if later.find_spec(name, path, target) is not None:
                        return None
                except Exception:
                    pass
            self.seen.add(name)
            try:
                with open(self.iterate_log, 'a') as out:
                    out.write(name + chr(10))
            except OSError:
                pass
            return None
    sys.meta_path.append(MissingImportWatch())
_iterate_watch(__import__('os').path.realpath({MISSING_IMPORTS!r}))
del _iterate_watch
"""


def session_preamble() -> str:
    """The trusted first cell of a session: loads ``X_train`` / ``y_train`` /
    ``X_holdout`` (features only) and prints the shapes.

    It also defines a ``finish()`` shim so a model that calls the tool as a
    function does not fail the cell, snapshots the inputs so the harness can
    restore them before every agent cell, and seeds the global RNGs so the
    rendered notebook re-executes to the same score. The thread caps come FIRST,
    before any import loads a BLAS runtime."""
    return (
        "import os\n"
        "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):\n"
        "    os.environ.setdefault(_v, '1')\n"
        "import json, random, pandas as pd, numpy as np\n"
        "random.seed(42); np.random.seed(42)\n"
        f"with open({META_JSON!r}) as _f:\n"
        "    _meta = json.load(_f)\n"
        "_target = _meta['target']\n"
        f"_train = pd.read_csv({TRAIN_CSV!r})\n"
        "X_train = _train.drop(columns=[_target])\n"
        "y_train = _train[_target]\n"
        f"X_holdout = pd.read_csv({HOLDOUT_CSV!r})  # FEATURES ONLY — labels held back\n"
        "_pristine_inputs = {'X_train': X_train.copy(), 'y_train': y_train.copy(), "
        "'X_holdout': X_holdout.copy()}\n"
        "def finish(*args, **kwargs):\n"
        "    print('finish is a tool call, not a Python function. This cell still ran; "
        "now invoke the finish tool to end the session.')\n"
        "print('loaded:', X_train.shape, 'train /', X_holdout.shape, 'holdout; target:', _target)\n"
        + IMPORT_WATCH
    )


def prompt_session_preamble() -> str:
    """The prompt path's opening cell: the tabular data contract plus three
    helpers. ``ask`` runs a prompt over records against the one model under test,
    ``evaluate`` scores with the RUN'S metric, and ``submit`` runs the prompt over
    the holdout and writes predictions and prompt together, so a prompt cannot be
    submitted apart from the predictions it produced.
    """
    return (
        "import os\n"
        "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):\n"
        "    os.environ.setdefault(_v, '1')\n"
        "import json, random, pandas as pd, numpy as np\n"
        "random.seed(42); np.random.seed(42)\n"
        "from iterate.core.prompting import Prompt\n"
        "from iterate.core.prompt_runtime import make_ask, AskStats, UNPARSEABLE\n"
        "from iterate.core.scoring import score as _score\n"
        f"with open({META_JSON!r}) as _f:\n"
        "    _meta = json.load(_f)\n"
        "_target = _meta['target']\n"
        "_labels = _meta.get('labels')\n"
        "_metric = _meta['metric']\n"
        "TASK = _meta['task']\n"
        f"_train = pd.read_csv({TRAIN_CSV!r})\n"
        "X_train = _train.drop(columns=[_target])\n"
        "y_train = _train[_target]\n"
        f"X_holdout = pd.read_csv({HOLDOUT_CSV!r})  # FEATURES ONLY — answers held back\n"
        "_pristine_inputs = {'X_train': X_train.copy(), 'y_train': y_train.copy(), "
        "'X_holdout': X_holdout.copy()}\n"
        "_columns = list(_meta['features'])\n"
        "_task_kind = _meta.get('task_kind', 'classification')\n"
        "_numeric = tuple(_meta['numeric_range']) if _meta.get('numeric_range') else None\n"
        "_median = _meta.get('median')\n"
        "_ask = make_ask(columns=_columns, labels=_labels, numeric_range=_numeric, "
        "backend=_meta['target_backend'], model=_meta['target_model'], "
        "base_url=_meta.get('target_base_url'), cache_path=_meta.get('cache_path'), "
        "max_workers=int(_meta.get('max_workers') or 8))\n"
        "BASELINE_PROMPT = Prompt(**_meta['baseline_prompt'])\n"
        # Three sessions across two live runs died a cell each to NameError on
        # BASEL_PROMPT / BASELINES_PROMPT. The name is long and a 12B fumbles it; a
        # short alias costs one line and removes a recurring wasted turn.
        "BASE = BASELINE_PROMPT\n"
        "def _ask_with_stats(prompt, rows):\n"
        "    frame = rows.to_dict(orient='records') if hasattr(rows, 'to_dict') else list(rows)\n"
        "    stats = AskStats()\n"
        "    return _ask(prompt, frame, stats=stats), stats\n"
        "def ask(prompt, rows):\n"
        "    out, stats = _ask_with_stats(prompt, rows)\n"
        "    print('ask:', stats.summary())\n"
        "    return out\n"
        "def evaluate(answers, truth):\n"
        "    raw = truth.tolist() if hasattr(truth, 'tolist') else list(truth)\n"
        "    if _task_kind == 'regression':\n"
        # An unusable answer becomes the training median here too. Scoring it any
        # other way in the session than the host does would let the agent chase a
        # number that is not the one it is judged on.
        "        preds = [_median if str(a) == UNPARSEABLE else float(a) for a in answers]\n"
        "        values = _score('regression', [float(t) for t in raw], preds, include=(_metric,))\n"
        "    else:\n"
        "        values = _score('classification', [str(t) for t in raw], "
        "[str(a) for a in answers], include=(_metric,), open_vocabulary=True)\n"
        "    return values[_metric]\n"
        "def submit(prompt):\n"
        "    answers, _stats = _ask_with_stats(prompt, X_holdout)\n"
        "    print('ask:', _stats.summary())\n"
        f"    pd.Series(answers).to_csv({PREDICTIONS_CSV!r}, index=False, header=False)\n"
        # Fingerprint of what the model answered, so the host can detect a later
        # cell overwriting predictions.csv while prompt.json still sits there.
        "    import hashlib as _hl\n"
        "    _digest = _hl.sha256(chr(10).join(str(a) for a in answers).encode()).hexdigest()\n"
        # The tokens the model under test spent per record, kept for the serving price.
        # Cached answers cost no tokens, so only the records that were really asked count.
        "    _n = _stats.calls\n"
        "    _tokens = {'tokens_in_per_record': _stats.prompt_tokens / _n if _n else None,\n"
        "               'tokens_out_per_record': _stats.completion_tokens / _n if _n else None,\n"
        "               'records_measured': _n}\n"
        f"    with open({PROMPT_JSON!r}, 'w') as _f:\n"
        "        json.dump({**prompt.as_dict(), 'answers_sha256': _digest, **_tokens}, _f)\n"
        "    print('submitted', len(answers), 'answers for the holdout')\n"
        "    return answers\n"
        "def finish(*args, **kwargs):\n"
        "    print('finish is a tool call, not a Python function. This cell still ran; "
        "now invoke the finish tool to end the session.')\n"
        "print('loaded:', X_train.shape, 'train /', X_holdout.shape, 'holdout; answers:', "
        "(_labels if _labels else 'free text'))\n"
        "print('task:', TASK)\n"
        # A worked example, PRINTED rather than only described in the instructions.
        # The first live prompt run called .split() on a Prompt and then handed one
        # to re.sub: the wording said what the objects were and never showed one
        # being used, and the session died without submitting.
        "print(chr(10).join(" + repr(_WORKED_EXAMPLE) + "))\n" + IMPORT_WATCH
    )


# Shown by the preamble as the session's first output, so the shape of a correct
# cell is on screen before the model writes one.
_WORKED_EXAMPLE = [
    "",
    "HOW TO WORK — copy this shape:",
    "  sample  = X_train.head(120)              # a SAMPLE; a full pass is minutes",
    "  truth   = y_train.head(120)",
    "  answers = ask(BASE, sample)              # BASE is short for BASELINE_PROMPT",
    "  print(evaluate(answers, truth))          # scores with THIS run's metric",
    "",
    "  wrong = [(sample.iloc[i].to_dict(), t, a)",
    "           for i, (t, a) in enumerate(zip(truth, answers)) if str(t) != str(a)]",
    "  for row, t, a in wrong[:10]: print(t, '!=', a, '|', row)",
    "",
    "  better = Prompt(system=BASE.system + chr(10) + 'your one change',",
    "                  user_template=BASE.user_template)",
    "  print(evaluate(ask(better, sample), truth))   # SAME sample, like for like",
    "  submit(better)                                # then call the finish tool",
    "",
    "A Prompt is an OBJECT, not a string: read .system and .user_template and build",
    "a new one with Prompt(...). String methods on it will raise.",
    "",
]


# Prepended to every agent cell: restores the canonical inputs from the pristine
# snapshot taken in the preamble, so in-place mutation of X_train/y_train/X_holdout
# in one cell cannot leak into the next. Guarded so a stray preamble failure (no
# snapshot) degrades to a no-op instead of a NameError on every cell.
RESET_INPUTS = (
    "if '_pristine_inputs' in dir():\n"
    "    X_train = _pristine_inputs['X_train'].copy(); "
    "y_train = _pristine_inputs['y_train'].copy(); "
    "X_holdout = _pristine_inputs['X_holdout'].copy()\n"
)


def fallback_baseline(task: str, *, with_proba: bool = False) -> str:
    """A host-authored floor submission, run when a session ends without a valid
    predictions file. A LINEAR model on purpose: the safety net must train in
    milliseconds under any thread weather. ``with_proba`` also writes
    `probabilities.csv`, or a probability metric would have an unscoreable net."""
    if task == "classification":
        import_line = "from sklearn.linear_model import LogisticRegression\n"
        model = "LogisticRegression(max_iter=1000, random_state=42)"
    else:
        import_line = "from sklearn.linear_model import Ridge\n"
        model = "Ridge(random_state=42)"
    # Probability metrics are classification-only, so a regression floor never needs it.
    proba_line = ""
    if with_proba and task == "classification":
        proba_line = (
            f"pd.DataFrame(_fb_model.predict_proba(_fb_Xh))"
            f".to_csv({PROBABILITIES_CSV!r}, index=False, header=False)\n"
        )
    return (
        "import pandas as pd\n"
        + import_line
        + "_fb_cats = X_train.select_dtypes(exclude='number').columns.tolist()\n"
        "_fb_Xt = pd.get_dummies(X_train, columns=_fb_cats)\n"
        "_fb_Xh = pd.get_dummies(X_holdout, columns=_fb_cats).reindex(columns=_fb_Xt.columns, fill_value=0)\n"
        "_fb_med = _fb_Xt.median()\n"
        "_fb_Xt = _fb_Xt.fillna(_fb_med)\n"
        "_fb_Xh = _fb_Xh.fillna(_fb_med)\n"
        f"_fb_model = {model}.fit(_fb_Xt, y_train)\n"
        f"pd.Series(_fb_model.predict(_fb_Xh)).to_csv({PREDICTIONS_CSV!r}, index=False, header=False)\n"
        + proba_line
        + "print('fallback baseline banked', len(_fb_Xh), 'predictions')\n"
    )


def submission_was_swapped(prompt_json: bytes | None, predictions: bytes | None) -> str | None:
    """Why the scored predictions are not the ones `submit()` produced, or None.

    `submit()` records the sha256 of the answers the model gave. If predictions.csv
    no longer matches, a later cell replaced it — a hardcoded rule, a lookup, a
    hand-edited file. That would score, and it would not be prompt engineering.

    Absent fingerprint means an older or hand-written submission, which is not
    evidence of anything and passes.
    """
    if not prompt_json or not predictions:
        return None
    try:
        recorded = json.loads(prompt_json).get("answers_sha256")
    except (ValueError, TypeError, AttributeError):
        return None
    if not recorded:
        return None
    import hashlib

    actual = hashlib.sha256(predictions.decode(errors="replace").strip().encode()).hexdigest()
    if actual == recorded:
        return None
    return (
        "predictions.csv does not match the answers submit() produced, so the "
        "submitted predictions did not come from the model under test"
    )


def prompt_fallback_baseline() -> str:
    """The prompt path's floor submission: the most common training answer, for
    every holdout row.

    Emphatically NOT "re-run the baseline prompt". The tabular floor learned this
    the expensive way — it trained a gradient-boosted tree until a session died to
    fit-cell timeouts and the floor timed out with it. A floor made of LLM calls has
    exactly that shape, and it would be slowest in precisely the situation that
    triggers it: a session that already spent its budget on calls. The majority
    answer needs no model, no network and no time.

    It writes prompt.json too, so a floored iteration still records what was being
    attempted rather than leaving a submission with no prompt behind it.
    """
    return (
        "import json, pandas as pd\n"
        "_fb_answer = str(y_train.astype(str).value_counts().index[0])\n"
        f"pd.Series([_fb_answer] * len(X_holdout)).to_csv({PREDICTIONS_CSV!r}, "
        "index=False, header=False)\n"
        f"with open({PROMPT_JSON!r}, 'w') as _f:\n"
        "    json.dump(BASELINE_PROMPT.as_dict(), _f)\n"
        "print('fallback banked the majority answer', repr(_fb_answer), 'for', "
        "len(X_holdout), 'rows')\n"
    )


def vision_session_preamble() -> str:
    """The image session's opening cell: `start()` decodes the images, carves the
    validation fold and returns every name the agent's cells work with.

    No thread caps, unlike the tabular preamble: torch sizes its own pools, and the
    device variables are set inside `start` before torch is first imported.
    """
    return (
        "import json, random, numpy as np, pandas as pd\n"
        "random.seed(42); np.random.seed(42)\n"
        "from iterate.core import vision_session as _vs\n"
        "globals().update(_vs.start('.'))\n"
        f"with open({META_JSON!r}) as _f:\n"
        "    _meta = json.load(_f)\n"
        f"_train = pd.read_csv({TRAIN_CSV!r})\n"
        "X_train = _train.drop(columns=[_meta['target']])\n"
        "y_train = _train[_meta['target']]\n"
        f"X_holdout = pd.read_csv({HOLDOUT_CSV!r})  # FILE NAMES ONLY — labels held back\n"
        "_pristine_inputs = {'X_train': X_train.copy(), 'y_train': y_train.copy(), "
        "'X_holdout': X_holdout.copy()}\n"
        "def finish(*args, **kwargs):\n"
        "    print('finish is a tool call, not a Python function. This cell still ran; "
        "now invoke the finish tool to end the session.')\n" + IMPORT_WATCH
    )


def vision_notebook_setup(started_from: Mapping[str, Any] | None) -> str:
    """The one cell the host adds ahead of a delivered image session, so Run All starts
    where the session did.

    It writes the recipe the session departed from and clears what an earlier Run All
    left in the folder: the session's own state file, which would otherwise carry the
    last re-run's fit, and the submission files, which the keep-best guard reads and
    would hold a second Run All's fit against a first one's. `best_model.pt`, the
    delivered network, is not among them and is never touched.
    """
    from iterate.core.vision_session import SESSION_JSON

    stale = (SESSION_JSON, PREDICTIONS_CSV, PROBABILITIES_CSV, RECIPE_JSON, NETWORK_PT)
    recipe = (
        f"with open({INCUMBENT_JSON!r}, 'w') as _f:\n"
        f"    _f.write({json.dumps(dict(started_from), default=str)!r})\n"
        if started_from
        else ""
    )
    return (
        "# Written by iterate, not by the session: the recipe it started from, and a\n"
        "# clean slate so this Run All behaves like the first one.\n"
        "import os\n"
        + recipe
        + "for _stale in (\n"
        + "".join(f"    {name!r},\n" for name in stale)
        + "):\n"
        "    if os.path.exists(_stale):\n"
        "        os.remove(_stale)\n"
    )


# First statement of every image cell: frees the last cell's device memory and starts
# this cell's fit clock, then restores the three input frames. `__import__` rather than
# a name, so a cell that deleted SESSION or re-bound the helpers still gets them back.
VISION_CELL_PREFIX = (
    "__import__('iterate.core.vision_session').core.vision_session.begin_cell(globals())\n"
    + RESET_INPUTS
)


def vision_worked_example(task: str) -> list[str]:
    """The shape of a correct image cell, printed by `start()` once the task is known.

    Task-aware because the two versions differ at line one: a number to predict has no
    CLASSES to size the head with, and a cross-entropy loss will not train it.
    """
    common = [
        "",
        "HOW TO WORK, one fit per cell:",
        "  f = fit(backbone='resnet18')      # every lever you do not pass stays as it is",
        "  submit(f)                         # the holdout predictions THIS fit made",
        "  g = fit(image_size=128)           # same recipe, new size, same fold",
        "  print(g.val, 'vs', f.val)         # like for like: both scored on VAL_IDX",
        # Never indented by two spaces: the contract test reads the indented blocks as the
        # runnable cells, and a third block would mean a third cell to keep working.
        "fit(layers=[('conv', 32), ('pool',), ('linear', 256)]) trains that network from",
        "zero; fit(head=[('linear', 512), ('dropout', 0.5)]) puts those layers where the",
        "backbone's final layer was. Four names: conv, pool, dropout, linear.",
        "",
        "YOUR OWN MODEL, from any library research names:",
        "  import time, timm, torch",
        "  import torch.nn.functional as F",
        "  NAME = 'efficientnet_b0'",
    ]
    if task == "regression":
        body = [
            "  model = timm.create_model(NAME, pretrained=True, num_classes=1).to(DEVICE)",
            "  size = IMAGE_SIZE",
            "  train_x, hold_x = pixels(size)",
            "  centre = float(labels[FIT_IDX].mean()); spread = float(labels[FIT_IDX].std())",
            "  opt = torch.optim.AdamW(model.parameters(), lr=3e-4)",
            "  t0 = time.perf_counter()",
            "  for epoch in range(3):",
            "      model.train()",
            "      for rows in batches(FIT_IDX, 64):",
            "          out = model(as_input(train_x[rows])).reshape(-1)",
            "          loss = F.mse_loss(out, (as_labels(rows) - centre) / spread)",
            "          opt.zero_grad(); loss.backward(); opt.step()",
            "      if seconds_left() < (time.perf_counter() - t0) / (epoch + 1):",
            "          break",
            "  val = predict(model, train_x[VAL_IDX]) * spread + centre",
            "  evaluate(val, model=NAME, image_size=size, epochs=epoch + 1)",
            "  submit_numbers(predict(model, hold_x) * spread + centre, model=NAME)",
            "",
            "Train on scaled labels, as fit() does, and give evaluate() and",
            "submit_numbers() numbers in the label's own units.",
        ]
    else:
        body = [
            "  model = timm.create_model(NAME, pretrained=True, "
            "num_classes=len(CLASSES)).to(DEVICE)",
            "  cfg = model.pretrained_cfg",
            "  size = cfg['input_size'][-1] if cfg.get('fixed_input_size') else IMAGE_SIZE",
            "  train_x, hold_x = pixels(size)",
            "  opt = torch.optim.AdamW(model.parameters(), lr=3e-4)",
            "  t0 = time.perf_counter()",
            "  for epoch in range(3):",
            "      model.train()",
            "      for rows in batches(FIT_IDX, 64):",
            "          loss = F.cross_entropy(model(as_input(train_x[rows])), as_labels(rows))",
            "          opt.zero_grad(); loss.backward(); opt.step()",
            "      if seconds_left() < (time.perf_counter() - t0) / (epoch + 1):",
            "          break",
            "  evaluate(predict(model, train_x[VAL_IDX]), model=NAME, image_size=size, "
            "epochs=epoch + 1)",
            "  submit_probabilities(predict(model, hold_x), model=NAME)",
        ]
    return [
        *common,
        *body,
        "",
        "model=NAME is REQUIRED on evaluate, submit_probabilities and submit_numbers:",
        "it is how this run records which model was tried.",
        "predict() reads .logits for you, so a Hugging Face model works as it is.",
        "",
    ]


def vision_fallback_baseline() -> str:
    """The image floor: the most common training class, or the training mean, for every
    holdout image, with class-share probabilities.

    It reads only the three files the host wrote, so it cannot fail on the device, the
    network or a spent fit budget — which is exactly the state a session is in when the
    floor fires.
    """
    return (
        "import json, numpy as np, pandas as pd\n"
        f"with open({META_JSON!r}) as _f:\n"
        "    _fb_meta = json.load(_f)\n"
        f"_fb_y = pd.read_csv({TRAIN_CSV!r})[_fb_meta['target']]\n"
        f"_fb_n = len(pd.read_csv({HOLDOUT_CSV!r}))\n"
        "if _fb_meta['task'] == 'regression':\n"
        "    _fb_pred = [float(_fb_y.astype(float).mean())] * _fb_n\n"
        "else:\n"
        "    _fb_classes = _fb_meta['classes']\n"
        "    _fb_share = _fb_y.astype(str).value_counts(normalize=True)\n"
        "    _fb_p = np.array([_fb_share.get(str(c), 0.0) for c in _fb_classes])\n"
        "    _fb_p = _fb_p / _fb_p.sum()\n"
        "    pd.DataFrame(np.tile(_fb_p, (_fb_n, 1)))"
        f".to_csv({PROBABILITIES_CSV!r}, index=False, header=False)\n"
        "    _fb_pred = [_fb_classes[int(_fb_p.argmax())]] * _fb_n\n"
        f"pd.Series(_fb_pred).to_csv({PREDICTIONS_CSV!r}, index=False, header=False)\n"
        "print('fallback banked', _fb_pred[0], 'for', _fb_n, 'holdout images')\n"
    )


def validate_train_and_predict(code: str) -> str | None:
    """Static check of a generated snippet; returns an error reason or ``None`` if OK.

    Catches the cheap-to-detect mistakes (won't parse, no `train_and_predict`,
    wrong arity) before we ever spend a run on it, so the proposer can re-prompt
    with a precise reason instead of burning an iteration on a doomed script.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"code did not parse: {exc}"
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY_POINT),
        None,
    )
    if func is None:
        return f"no top-level function named {ENTRY_POINT!r} was defined"
    n_positional = len(func.args.posonlyargs) + len(func.args.args)
    if n_positional < 3 and func.args.vararg is None:
        return f"{ENTRY_POINT} must accept (X_train, y_train, X_holdout)"
    return None


# Pure plumbing / containers — instantiated but not informative about the approach.
_NON_COMPONENT = frozenset(
    {"Pipeline", "ColumnTransformer", "FeatureUnion", "DataFrame", "Series", "ndarray", "array"}
)


def components_used(code: str) -> list[str]:
    """The class-like components a snippet instantiates — preprocessors, encoders,
    feature selectors, the estimator — in source order, deduped.

    A deterministic fingerprint of WHAT an attempt actually did (not just the model
    it named), so the proposer can see prior *preprocessing* and choose something
    genuinely different instead of repeating impute+one-hot every time. Catches
    CapWords classes (e.g. SimpleImputer, OneHotEncoder, StandardScaler,
    TargetEncoder, HistGradientBoostingClassifier); function-style helpers like
    ``pd.get_dummies`` are not captured (a known minor gap).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):  # walk is breadth-first, so re-sort by source position
        if isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name and name[0].isupper() and name not in _NON_COMPONENT:
                calls.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name))
    seen: dict[str, None] = {}  # source-ordered, deduped
    for _, _, name in sorted(calls):
        seen.setdefault(name, None)
    return list(seen)


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


_INSTALLER = re.compile(
    r"(?<![\w.-])(?:pip[\d.]*|pipx|pipenv|poetry|pdm|rye|pixi|hatch|uvx?|conda|mamba|micromamba"
    r"|ensurepip)(?![\w-])|-m\s*pip\b"
)
_INSTALL_VERB = re.compile(
    r"(?<![\w-])(?:install|add|sync|download|inject|upgrade|update|create|run|--with|-U)(?![\w-])"
)
_INSTALLER_MODULES = frozenset({"pip", "ensurepip"})
_SPAWN_MODULES = frozenset({"subprocess", "pty", "runpy"})
_OS_MODULES = frozenset({"os", "posix", "nt"})
_OS_SPAWNS = frozenset(
    {
        "system",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    }
)
_MAGICS = frozenset({"run_line_magic", "run_cell_magic"})
_SHELL_CALLS = _MAGICS | {"getoutput", "create_subprocess_exec", "create_subprocess_shell"}
_SPAWN_NAMES = _OS_SPAWNS | _SHELL_CALLS
# Magics that time, capture or configure Python code: their text is judged as a cell.
_CODE_MAGICS = frozenset(
    {"time", "timeit", "capture", "prun", "matplotlib", "load_ext", "autoreload", "config"}
)
_MAGIC_OPTIONS = re.compile(r"^\s*(?:-[A-Za-z]+\s*(?:\d+\s+)?)*")
_LOADERS = frozenset({"import_module", "__import__", "run_module", "find_spec", "load_module"})
_EVALUATORS = frozenset({"exec", "eval", "compile"})


@dataclass(frozen=True)
class InstallerCell:
    """Why a cell counts as an installer. ``query`` is set when it only asks what is
    installed (``!pip list``): still refused, with a pointer at importlib.metadata."""

    evidence: str
    query: bool


def _module_root(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def _const_str(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        return node.value.decode(errors="replace")
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_platform_system(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Name) and (node.value.id, node.attr) == ("platform", "system")


def _literal_strings(call: ast.Call) -> list[str] | None:
    """The strings a call's arguments hold when every argument is a plain literal that
    interpolates nothing (IPython expands ``{name}`` and ``$name``), else None."""
    out: list[str] = []
    for arg in (*call.args, *(k.value for k in call.keywords)):
        for node in ast.walk(arg):
            if (text := _text(node)) is not None:
                if "{" in text or "$" in text:
                    return None
                out.append(text)
            elif not isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.expr_context)):
                return None
    return out


def _parsed(code: str) -> ast.Module | None:
    from IPython.core.inputtransformer2 import TransformerManager

    try:
        return ast.parse(TransformerManager().transform_cell(code))  # type: ignore[no-untyped-call]
    except SyntaxError:
        return None


def runs_installer(code: str) -> InstallerCell | None:
    """Judged on the whole IPython-transformed cell: a process spawner plus an installer
    name in a string it can reach, or any load of pip. A spawner called with plain
    literals reaches only those; any other spawner reaches every string in the cell. A
    name check, not a boundary: an installer name built from pieces, an obfuscated call
    (``vars(os)['system']``, an aliased ``exec``, ctypes) and a script one cell writes
    and the next runs all pass. A cell that does not parse runs nothing, so it is not
    judged."""
    tree = _parsed(code)
    return None if tree is None else _installer_in(tree)


def _installer_in(tree: ast.Module) -> InstallerCell | None:
    strings: list[str] = []
    spawners: list[tuple[str, list[str] | None]] = []
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_root(alias.name) in _INSTALLER_MODULES:
                    return InstallerCell(f"import {alias.name}", query=False)
                if _module_root(alias.name) in _SPAWN_MODULES:
                    spawners.append((f"import {alias.name}", None))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            root = _module_root(node.module)
            if root in _INSTALLER_MODULES:
                return InstallerCell(f"from {node.module} import", query=False)
            names = {a.name for a in node.names}
            if (
                root in _SPAWN_MODULES
                or (root in _OS_MODULES and "*" in names)
                or (root != "platform" and names & _SPAWN_NAMES)
            ):
                spawners.append((f"from {node.module} import", None))
        elif isinstance(node, ast.Attribute):
            # Uncalled, only a module's own spawn function counts: X_train.system is a column.
            if (
                id(node) not in funcs
                and node.attr in _SPAWN_NAMES
                and isinstance(node.value, ast.Name)
                and node.value.id in _OS_MODULES
            ):
                spawners.append((node.attr, None))
        elif isinstance(node, ast.Name) and node.id in _SPAWN_MODULES:
            spawners.append((node.id, None))
        elif isinstance(node, ast.Call):
            called = _called_name(node.func)
            first = _const_str(node.args[0]) if node.args else None
            if (
                called in _LOADERS
                and first is not None
                and _module_root(first) in _INSTALLER_MODULES
            ):
                return InstallerCell(f"loads {first!r}", query=False)
            if called in _EVALUATORS and first is not None and (inner := runs_installer(first)):
                return InstallerCell(f"{called} of {inner.evidence}", query=inner.query)
            if called == "getattr" and len(node.args) > 1:
                attr = _const_str(node.args[1])
                if attr in _SPAWN_NAMES:
                    spawners.append((f"getattr {attr}", None))
            if called not in _SPAWN_NAMES or (
                isinstance(node.func, ast.Attribute) and _is_platform_system(node.func)
            ):
                continue
            if called in _MAGICS and first in _CODE_MAGICS:
                lines = [_const_str(a) for a in node.args[1:]]
                bodies = [None if t is None else _parsed(_MAGIC_OPTIONS.sub("", t)) for t in lines]
                if all(body is not None for body in bodies):
                    for body in bodies:
                        if body is not None and (inner := _installer_in(body)):
                            return InstallerCell(f"%{first} of {inner.evidence}", inner.query)
                    continue
            spawners.append((str(called), _literal_strings(node)))
        if (text := _text(node)) is not None:
            if _module_root(text) in _SPAWN_MODULES:
                spawners.append((text, None))
            strings.append(text)
    if not spawners:
        return None
    anywhere = next((evidence for evidence, reach in spawners if reach is None), None)
    reached: list[tuple[str, list[str] | None]] = (
        [(anywhere, strings)] if anywhere is not None else spawners
    )
    for evidence, texts in reached:
        for text in texts or ():
            if found := _INSTALLER.search(text):
                query = not any(_INSTALL_VERB.search(t) for t in texts or ())
                return InstallerCell(f"{evidence} with {found.group().strip()!r}", query=query)
    return None


def package_for_import(import_name: str) -> str:
    """The pip distribution name for a top-level import (e.g. 'sklearn'→'scikit-learn'),
    falling back to the import name. Used for install-on-demand of a missing module."""
    return _IMPORT_TO_PACKAGE.get(import_name.split(".", 1)[0], import_name.split(".", 1)[0])


def required_imports(code: str) -> list[str]:
    """The pip distributions a snippet needs: top-level imports, minus the stdlib.

    Deterministic (a plain AST walk, no LLM). Import names are mapped to their
    distribution name via `_IMPORT_TO_PACKAGE` where the two differ; unknown names
    fall back to themselves. Relative imports are ignored. See the note on
    `_IMPORT_TO_PACKAGE`: the resolve-and-install architecture is provisional.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".", 1)[0])
    return sorted(
        {_IMPORT_TO_PACKAGE.get(m, m) for m in modules if m not in sys.stdlib_module_names}
    )


def build_inputs(dataset: TabularDataset) -> dict[str, bytes]:
    """The files handed to the runner: train (with target), holdout FEATURES, meta.

    The holdout target is never written — it stays host-side for scoring, so the
    sealed holdout cannot leak through the sandbox boundary.
    """
    train = dataset.train_features.copy()
    train[dataset.target] = dataset.train_target.to_numpy()
    meta = {
        "target": dataset.target,
        "task": task_for_metric_safe(dataset),
        "features": list(dataset.features),
    }
    return {
        TRAIN_CSV: train.to_csv(index=False).encode(),
        HOLDOUT_CSV: dataset.test_features.to_csv(index=False).encode(),
        META_JSON: json.dumps(meta).encode(),
    }


def task_for_metric_safe(dataset: TabularDataset) -> str:
    """The task as the loader decided it, conveyed to the session through meta.json."""
    return dataset.task


def parse_probabilities(probabilities_csv: bytes | None, *, expected: int) -> list[list[float]]:
    """Parse `probabilities.csv` into rows of floats. Raises ValueError with a
    reason the agent can act on.

    One value per line for binary (the positive class), or one comma-separated
    value per class for multiclass. Shape agreement with the label set is checked
    downstream by `core.scoring`, which owns what each shape means.
    """
    # `is None` rather than falsy: a zero-byte file exists and "was not found" would
    # send the agent looking for a write bug it doesn't have.
    if probabilities_csv is None:
        raise ValueError(f"{PROBABILITIES_CSV} was not found in the working directory")
    raw = probabilities_csv.decode(errors="replace").strip()
    if not raw:
        raise ValueError(f"{PROBABILITIES_CSV} is empty")
    lines = raw.splitlines()
    if len(lines) != expected:
        raise ValueError(
            f"expected {expected} probability rows (one per holdout row), found {len(lines)}"
        )
    rows: list[list[float]] = []
    for n, line in enumerate(lines, start=1):
        try:
            rows.append([float(part) for part in line.split(",")])
        except ValueError:
            raise ValueError(
                f"{PROBABILITIES_CSV} line {n} is not numeric: {line[:40]!r}. Write raw "
                "probabilities, not labels, with index=False and header=False"
            ) from None
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{PROBABILITIES_CSV} rows have inconsistent widths: {sorted(widths)}")
    return rows


def score_predictions(
    dataset: TabularDataset,
    predictions_csv: bytes | None,
    *,
    metric: str,
    experiment_id: str,
    probabilities_csv: bytes | None = None,
    average: str | None = None,
    open_vocabulary: bool = False,
) -> ExperimentResult:
    """Score a script's predictions against the held-back holdout labels.

    `open_vocabulary` (the prompt path) makes an answer outside the target's values
    count as wrong instead of becoming a new class. A missing or malformed
    predictions file is a captured failure, never an exception. A bad probability
    file sinks the experiment only when the primary metric needs probabilities.
    """
    if not predictions_csv:
        return _failed(experiment_id, "no predictions file produced")
    raw = predictions_csv.decode(errors="replace").strip()
    if not raw:
        return _failed(experiment_id, "predictions file was empty")
    preds = raw.splitlines()
    expected = dataset.n_test
    if len(preds) != expected:
        return _failed(experiment_id, f"expected {expected} predictions, got {len(preds)}")

    needs_proba = requires_proba(metric)
    y_proba: list[list[float]] | list[float] | None = None
    if probabilities_csv is not None or needs_proba:
        try:
            rows = parse_probabilities(probabilities_csv, expected=expected)
            y_proba = [row[0] for row in rows] if len(rows[0]) == 1 else rows
        except ValueError as exc:
            if needs_proba:
                return _failed(experiment_id, f"{metric} needs probabilities: {exc}")
            y_proba = None

    # Coercing + scoring can fail on garbage predictions (wrong dtype, non-numeric
    # values for a numeric target, a label the metric can't handle). That's the
    # agent's code being wrong, not ours — capture it as a failed experiment (and
    # feed the reason back) rather than letting it crash the loop.
    try:
        task = task_for_metric(metric)
        y_pred = _coerce(preds, target=dataset.test_target, task=task)
        values = score(
            task,
            dataset.test_target.to_numpy(),
            y_pred,
            y_proba=y_proba,
            average=average,
            include=(metric,),
            open_vocabulary=open_vocabulary,
        )
    except Exception as exc:
        if y_proba is not None and not needs_proba:
            # A malformed bonus panel must not sink a valid label-metric run.
            try:
                values = score(
                    task_for_metric(metric),
                    dataset.test_target.to_numpy(),
                    y_pred,
                    average=average,
                    include=(metric,),
                    open_vocabulary=open_vocabulary,
                )
            except Exception:
                return _failed(
                    experiment_id, f"could not score predictions ({type(exc).__name__}: {exc})"
                )
        else:
            return _failed(
                experiment_id, f"could not score predictions ({type(exc).__name__}: {exc})"
            )
    metrics = Metrics(
        values=values,
        primary=metric,
        direction=direction(metric),
        n_samples=expected,
    )
    return ExperimentResult(experiment_id=experiment_id, metrics=metrics)


def _coerce(preds: list[str], *, target: object, task: str) -> list[int | float | str]:
    """Coerce string predictions to the holdout target's own type, so labels line
    up (e.g. int 0/1 vs string "0"/"1" would otherwise be a 'mixed types' error).
    A number to predict stays a float whatever the column's dtype."""
    if task == "regression":
        return [float(p) for p in preds]
    kind = getattr(getattr(target, "dtype", None), "kind", "O")
    if kind in "iu":  # integer labels (the common 0/1 classification target)
        return [int(float(p)) for p in preds]
    if kind == "b":
        return [bool(int(float(p))) for p in preds]
    if kind == "f":  # continuous (regression) or float labels
        return [float(p) for p in preds]
    return [p.strip() for p in preds]  # string / categorical labels


def _failed(experiment_id: str, reason: str) -> ExperimentResult:
    return ExperimentResult(experiment_id=experiment_id, error=f"code-gen contract: {reason}")


__all__ = [
    "ENTRY_POINT",
    "HOLDOUT_CSV",
    "IMPORT_WATCH",
    "INCUMBENT_JSON",
    "META_JSON",
    "MISSING_IMPORTS",
    "NETWORK_PT",
    "PREDICTIONS_CSV",
    "PROBABILITIES_CSV",
    "RECIPE_JSON",
    "TRAIN_CSV",
    "VISION_CELL_PREFIX",
    "InstallerCell",
    "assemble_script",
    "build_inputs",
    "components_used",
    "fallback_baseline",
    "is_code_candidate",
    "package_for_import",
    "parse_probabilities",
    "required_imports",
    "runs_installer",
    "score_predictions",
    "session_preamble",
    "validate_train_and_predict",
    "vision_fallback_baseline",
    "vision_notebook_setup",
    "vision_session_preamble",
    "vision_worked_example",
]
