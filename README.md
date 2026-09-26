# iterate

**Autonomous research-aware iteration agent for ML models and LLM prompts.**

[![PyPI](https://img.shields.io/pypi/v/iterate-ai)](https://pypi.org/project/iterate-ai/)
[![CI](https://github.com/anthonyrodrigues443/Autonomous-ML-And-LLM-Iteration-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/anthonyrodrigues443/Autonomous-ML-And-LLM-Iteration-Agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/iterate-ai)](https://pypi.org/project/iterate-ai/)
[![License](https://img.shields.io/github/license/anthonyrodrigues443/Autonomous-ML-And-LLM-Iteration-Agent)](LICENSE)

```bash
pip install iterate-ai

# your CSV, your target column, your metric. LLM runs on local Ollama ($0)
# or any OpenAI-compatible endpoint. Full setup: Quick start below.
iterate run --data examples/churn_tabular/data.clean.csv --target Churn
# (--metric is optional now: omit it and the agent picks one from your data, and says why)

# the same loop on an LLM prompt: a labelled eval set plus one line saying what the job is
# (python examples/toxicity_jigsaw/prepare.py builds that eval set first, no account needed)
iterate run --data examples/toxicity_jigsaw/data.csv --target label \
            --task "decide whether this Wikipedia comment is toxic"

# the same loop on images: a CSV of image paths and labels, or a folder laid out however it came
# (python examples/flowers102/prepare.py builds that CSV first, no account needed)
iterate run --data examples/flowers102/data.csv --target label --metric accuracy
iterate run --data path/to/your_image_folder      # no --target: the labels are found for you
```

`iterate` runs an autonomous experiment loop on your ML problem. The agent **writes and runs its own training code**, cell by cell, in a live Jupyter kernel: a Supervisor reads the run history and briefs one experiment, a coding agent executes it against real cell outputs and real tracebacks, a Summarizer distills every finished notebook so the next one inherits what worked and what failed. In v0.3 you **talk to it while it runs**: a terminal UI streams the session as a live transcript (syntax-highlighted cells, scores, briefs) over a pinned input box, and anything you type in plain English becomes a question answered from the run's notebooks, a steer for the current experiment, or a standing rule every later experiment respects. In v0.5 the same loop iterates an **LLM prompt**: give it a labelled eval set and one line saying what the job is, and the agent writes a prompt, measures it, reads what it got wrong, and rewrites it. In v0.6 it trains an **image model**: give it a folder of images or a CSV of image paths, and the run starts from a plain CNN trained from zero, then the agent fine-tunes pretrained backbones or writes its own torch code, for classes or for numbers. Every submission is scored on a sealed holdout, every attempt persists in memory, and the winner ships as a runnable notebook, or as `prompts.yaml` on a prompt run. 2,163 unit tests across 75 files; CI runs 2,115 of them on every push, and the other 48 need torch or a Mac.

| v0.6 today | On the roadmap |
|---|---|
| **It trains image models, on the same loop.** Point `--data` at a folder of images laid out however it came, or at a CSV of image paths, and the run starts from a plain CNN trained from zero; the agent then fine-tunes pretrained backbones or writes its own torch code. Classes and numbers, scored on the same sealed holdout a table gets, delivered as a runnable notebook | Cost-to-serve recommendations (v0.7) |
| **It iterates LLM prompts, not only models.** Pass `--task` with a labelled eval set and the agent writes a prompt, measures it on training rows, reads the misses, and rewrites it. Classification and regression, scored on the same sealed holdout a model gets, delivered as `prompts.yaml` | Inferred inputs + MCP auto-discovery (v0.9) |
| **You no longer pick the metric.** Omit `--metric` and the agent reads your data, searches the literature, and chooses one, then tells you why. An explicit choice always wins | One-sentence input (v1.0) |
| **A Critic reviews every experiment for leakage**, so a pipeline that fits on the holdout does not get to bank its score; **a Researcher grounds proposals in real papers** (OpenAlex + arXiv, no API key), citing work it actually retrieved | |
| A deterministic guard stack converts weak-model waste and outranks everything else: a user steer can shape a brief but never bypass a gate, and no agent can overturn a guard; talk to the run while it runs; winner ships as a runnable notebook | |

## Why I built this

I kept seeing the same failure mode on small AI teams. A model or a prompt ships, and under delivery pressure nobody iterates on it again, so it sits in production for months while baselines move on. Experiments get re-run because nobody wrote down why they failed the first time. Teams pay frontier-model prices because nobody checked whether a cheaper model with a better prompt would do the job. `iterate` is the institutional memory, research desk, and experiment runner those teams don't have time to build.

> **How this gets built:** [WORKFLOW.md](WORKFLOW.md) (the method) · [DECISIONS.md](DECISIONS.md) (every call I made against the AI's default) · [BUILD_LOG.md](BUILD_LOG.md) (the daily trail)

---

## The full pitch

> Every YC batch ships 200+ AI startups with 2-3 engineer teams. Under shipping pressure, two things break: nobody re-iterates models against new baselines, and LLM prompts sit in production for months untouched. Engineers re-run failed experiments because nobody logged why. Teams pay GPT-5 prices because nobody tested whether Haiku + better prompting would do the job at 1/50th the cost.
>
> AutoML brute-forces. Experiment trackers only log. Prompt evals only evaluate. AIDE iterates Kaggle problems once. `iterate` is being built as the system that runs an autonomous, literature-aware, memory-persistent improvement loop on **ML models, DL/vision models, AND LLM prompts** in production, optimizing for the best model you can actually **afford to serve**. That is the v1.0 vision; the releases below get there one dial at a time.

---

## Status

**v0.6 released: the same loop now trains image models.** v0.1 proved the autonomous loop, v0.2 made the agent write and run its own code, v0.3 put you in the loop without stopping it, v0.4 made `--metric` optional and added a Researcher and a Critic, v0.5 added LLM prompts as the second problem type, and v0.6 adds the third: images. Pass a folder of images or a CSV of image paths, and the run starts from a plain CNN trained from zero, then the agent fine-tunes pretrained backbones through a `fit()` helper or writes its own torch code, for classes and for numbers, scored on the same sealed holdout a table is. The honest floor: on a local 12B the agent mostly steers `fit()` (backbone, image size, learning rate, epochs, augmentation) and rarely writes torch of its own, and an image run is tens of minutes on a laptop GPU, not seconds. Behind every release there is an eval suite with a measured ceiling per dataset, so a flat result reads as a miss or as an exhausted problem instead of a guess.

**Agent-first:** the autonomous loop landed at v0.1, not as a late-stage add-on. Two dials turn release to release: the inputs you must give *shrink* (toward one-sentence input) and the problem types *grow* (tabular, then prompts, then DL/vision).

| Release | Phase | Status |
|---|---|---|
| v0.1 | **The agentic loop**: Proposer + Orchestrator + Terminator + Memory + CLI, first autonomous tabular run | shipped |
| v0.2 | **Sandboxed code-gen + the multi-agent cell-by-cell system** (Supervisor, coding agent, Summarizer) + notebook deliverable + the deterministic guard stack | shipped |
| v0.3 | **Interactive runs**: terminal UI (live transcript + input box), plain-English chat with queued delivery, pause / resume / stop, notebook Q&A, standing rules | shipped |
| v0.4 | **Researcher + Critic specialists**: literature-grounded proposals with real citations, leak review before a score banks; agent picks the metric + starting model; probability metrics | shipped |
| v0.5 | **`PromptTarget`: agentic prompt iteration** — you give a labelled eval set and a one-line task, the agent writes a prompt, reads what it got wrong, and rewrites it. Classification **and** regression, scored on a sealed holdout | shipped |
| v0.6 | **`DLModelTarget`: an image folder or a CSV of image paths runs the same loop a table does.** A plain CNN baseline, then briefed experiments that fine-tune pretrained backbones or write their own torch code, for classes and numbers, scored on a sealed holdout. **The winner is saved: `best_model.pt` plus a three-line loader**, and `best.ipynb` really runs again. **You can say what the network should be**: `layers=`, `head=` and `drop_stages=` on `fit()`, from a research finding or typed into a running run. GPU compatible: Apple (MPS) and NVIDIA (CUDA), with CPU as the fallback. Run live with a local 12B. | shipped |
| v0.7 | **Cost-constrained recommendation.** Give a serving budget and the run recommends the best-scoring model you can afford to run: the budget is a hard wall on what it costs to serve the winner, never a weight on the score; prices come live from the clouds' own lists; the Researcher names models with no cost in the question and the Pricer says which fit, back and forth until one does. Every winner leaves with a serving profile (cheapest machine or API, monthly cost, requests an hour, where each number came from), plus `iterate cost` and the cheaper prompt runs promised in v0.5 | in progress, release Sun 2026-10-11 |
| v0.9 | Infer features/target from the data + a description; **MCP discovery** of the data/code itself (absorbs the v0.8 milestone) | planned |
| v1.0 | One-sentence input + multi-backend benchmark + read-only dashboard + docs + launch (absorbs the v0.10 milestone) | planned |

---

## What it does

You give it a prepared CSV, the target column, and a metric. The agent does the rest: builds its own baseline, then runs one briefed experiment per iteration, cell by cell, against a sealed holdout it never sees. The split is yours if you want it to be: pass `--train` and `--holdout` instead of `--data` and the holdout is sealed exactly as you gave it, its rows shuffled once so their order cannot carry a label. Since v0.6 the data can be images too: a folder laid out however it came, a CSV of image paths, or your own `--train` and `--holdout` pair goes through the same loop, from a plain CNN baseline to fine-tuned pretrained backbones and the agent's own torch code ([What v0.6 adds](#what-v06-adds-images-same-loop)).

What a live run looks like:

```
coder[iter-04]: cell 6 ok (1.7s, 4/300s budget)
coder[iter-04]: cell 7 error: NameError: name 'Xb_cat' is not defined
coder[iter-04]: cell 8 ok (0.1s, 5/300s budget)
agent loop: iteration 4 'Model Swap - XGBoost' -> f1=0.6312

                        Run summary
 iter   model                              f1   delta vs baseline
 base   baseline                       0.5676                   -
    1   Baseline Model                 0.6251             +0.0575
    2   Class Weight Balancing         0.6251             +0.0575
    3   Hyperparameter Tuning          0.6279             +0.0603
    4   Model Swap - XGBoost  <- best  0.6312             +0.0636
    ...
best: Model Swap - XGBoost (f1=0.6312, +0.0636 vs baseline)
```

Each iteration is a real R&D session, not a script dump:

1. The **Supervisor** compresses everything tried so far into a two-line brief: the banked best (exact config, threshold, components), the known dead ends, and exactly ONE new move to try.
2. The **coding agent** rebuilds the carried best, applies the brief's one change, measures it like-for-like on a validation split, and submits only what it can defend. Errors are debugged from real tracebacks, cell by cell.
3. The **Summarizer** digests the session (what helped, what hurt, the takeaway), so run 7 knows what run 3 learned.
4. The harness scores the submission on the sealed holdout and saves the notebook immediately. Ctrl-C keeps everything already earned.

**The harness is the moat, not the model.** Every fact in a brief is machine-derived from the actual banked code, never LLM recall. A stack of deterministic guards catches the failure modes weak models actually produce: briefs that re-commission already-banked work, submissions byte-identical to earlier ones, briefed changes that never reached a line of code, sessions that die without submitting (a floor submission banks automatically). Each guard exists because a live forensic run demonstrated the failure it prevents; the stack was validated across 21 instrumented runs on a local 12B model, which ties its all-time best score inside the guarded loop.

**Deliverables.** The winner is exported as `best.ipynb`: a runnable, annotated notebook of the actual winning session (hypothesis, staged cells with their real outputs, dead ends labeled, findings). Open it from the run folder and Run All really runs: the three files the session read are written beside it, and a cell that errored in the session is tagged so Jupyter carries on past it. `--notebooks all` keeps one notebook per iteration: the full research journey, kept for reading rather than for running. An image run also saves the winning network as `best_model.pt`, the one that was scored, and three lines load it in your app (see "Use the model" below). Every experiment also persists in `.iterate/memory.db`, so the next run builds on this one.

---

## What v0.3 adds: you, in the loop

On a terminal, `iterate run` now opens an interactive session view: the run streams as a live transcript (each executed cell as a syntax-highlighted block with its status, seconds, and budget; briefs and scores as styled rows) above an input box that is always yours. Type anything, anytime, in plain English:

- **Questions** ("did we complete an iteration?", "why did iteration 3 fail?") get answered by the Supervisor from the run's actual recorded notebooks, not from model recall.
- **Instructions** ("try a smaller learning rate") reach the RUNNING session at its next cell, and the next brief sees them too. "next run, try catboost" waits for the next experiment.
- **Standing rules** ("from now on, never use lightgbm") are carried into every later experiment's planning, capped and lean.
- **`pause` / `resume`** park and continue the run at the next cell boundary, kernel kept alive (e2b leases included) and every clock suspended. **`/stop`** (or a double Ctrl-C) quits immediately and still prints the run summary of everything finished so far; a single Ctrl-C winds down gracefully first (the in-flight attempt banks its floor). Type **`/`** for the command palette: arrow keys move, Enter completes into the input box, a second Enter sends.

Messages queue while a cell or an LLM call is in flight; you get an instant ack saying when they will land. The message routing is decided by the Supervisor but EXECUTED by the harness, and the guard stack outranks chat: a steer can shape a brief, it can never re-commission banked work, unseal the holdout, or bypass a gate. `--plain` keeps the classic scrolling output (chat still works, line by line); piped, scripted, CI, and backgrounded runs behave exactly as before, non-interactive.

---

## What v0.5 adds: prompts, same loop

A prompt eval set is a CSV like any other: input columns plus one column holding the right answer. Pass `--task` and the run switches to prompt iteration. Nothing about the loop changes. The Supervisor still briefs one change per experiment, the coding agent still measures like for like, the Critic still reviews, the Summarizer still hands on what was learned. What changes is what a cell does: one model call per record instead of one fit.

- **The harness owns the model call.** Inside a session the agent writes the prompt and calls `ask(prompt, rows)`; it cannot change the model, the temperature or the endpoint between experiments, so two experiments differ by the prompt and nothing else. The allowed answers are a tool schema built from the label set, so an answer outside it cannot happen. `evaluate(answers, truth)` scores with the run's metric, and `submit(prompt)` runs it over the holdout and writes the predictions and the prompt together.
- **The holdout is sealed the same way.** Training rows are written with answers, holdout rows without, and holdout rows never enter the session. Few-shot examples can only come from training rows.
- **Classification and regression.** A closed set of labels is scored with f1, accuracy and the rest; a numeric answer (a rating, a score on a scale) with rmse, pearson, spearman or kendall. Free text is refused rather than scored, unless you pass `--allow-free-text` and accept exact-string matching.
- **Candidates are ranked on a fixed slice, the winner is re-scored on everything.** `--loop-holdout` (default 100) keeps the search cheap and paired; `best_score_on_full_holdout` in `prompts.yaml` is the number to quote.
- **`prompts.yaml` is the deliverable.** Every version, what changed, its score, and `best: true` on the one to put in production. Written by the harness, never by the agent.

```bash
iterate run --data eval.csv --target label --task "decide whether this ticket is urgent"
iterate run --data eval.csv --target label --task "..." --prompt-file current_prompt.txt   # start from the prompt you ship today
iterate run --data pairs.csv --target score --task "rate how similar the two sentences are, 0 to 5" --metric pearson
iterate run --data eval.csv --target label --task "..." --target-model gemma4:12b --target-backend ollama --model llama-3.3-70b-versatile --backend groq
```

The last form tunes a prompt for one model while a stronger model drives the run. The cost line is honest: a pass is one model call per record, so 100 records on a local 12B is minutes, not seconds. Every answer is cached, so re-measuring a prompt the run has already tried is free.

---

## What v0.6 adds: images, same loop

An image dataset is a table like any other: one column of image paths, one column holding the label. Nothing about the loop changes. The Supervisor still briefs one change per experiment, the coding agent still measures like for like, the Critic still reviews, the Summarizer still hands on what was learned, and the winner still ships as a runnable `best.ipynb`. What changes is what a cell does: it trains a network on your GPU instead of fitting a table model.

- **Three ways in.** A CSV of image paths with `--target`, your own split with `--train` and `--holdout`, or a folder laid out however it came with `--data <folder>` and no `--target`. For a folder, the run works out how the images link to their labels by rules (class folders, a table joined to the images on an exact key, one-hot columns, a split column, wrapper folders), shows what it found, and refuses what it cannot prove with the reason and the flags that would settle it. Where the rules can only say "one of these", the Linker, the sixth specialist, reads a listing of the folder and proposes which table, key and label column; the proposal is rebuilt and measured before you see it, and the pause takes yes, no, or a correction in plain English. A plan you accepted is remembered, so the same folder never goes through the model twice.
- **Seven checks before you say yes.** On a folder run, seven deterministic checks run on the rows it will write: coverage both ways, duplicate and conflicting labels, byte copies across the split, lookalikes, a second label source, class balance, and id-like columns that span the split. The report prints under the link block and lands as `monitor.json`. It reports and never relabels: a copy of a training image never stays in a holdout the run makes, and you can answer `drop` to take one out of a holdout you gave. The linked data lands in `.iterate/data/<name>/` with `raw_files/`, `train/`, `holdout/` and their CSVs.
- **The baseline is a plain CNN trained from zero**, the image twin of a table run's baseline: three conv blocks at 64 px, 20 epochs, no augmentation, the same model on every machine. Every pretrained model is a try the agent makes, so the gain over the baseline is the agent's.
- **`fit()` is the easy path, the agent's own torch code is the open one.** `fit()` trains one of three pinned pretrained backbones (resnet18, resnet50, convnext_tiny) or the plain CNN, plans its epochs against the cell's time budget, prints a line per epoch, and reports an out-of-memory error as a result instead of a crash. The agent moves the levers: backbone, image size, how much to unfreeze, learning rate and schedule, augmentation, epochs, label smoothing. When the research names a model `fit()` does not have, the agent writes its own torch code and the harness scores and submits it under that model's name. Either way the harness keeps the images, the holdout, the time limit and the submission.
- **Classes and numbers.** A label can be a class (a flower species, a land-use type) or a number (a storm's wind speed in knots). The run prints how it read the labels, and `--metric` settles it when whole-number labels could be either.
- **torch is installed for you, or up front.** An image run needs torch and torchvision. With install consent the harness installs them at the start of the run, before anything else trains; or install them yourself with `pip install 'iterate-ai[vision]'`. They never change mid-run.
- **It trains on this machine.** `--compute e2b` is refused for an image run: the sandbox has no GPU and none of your images. On a Mac every cell still runs inside the macOS sandbox.

```bash
iterate run --data examples/flowers102/data.csv --target label --metric accuracy     # a CSV of image paths, classes
iterate run --train examples/cyclone_wind/train.csv --holdout examples/cyclone_wind/holdout.csv \
            --target label --metric rmse                                             # your own split, a number label
iterate run --data path/to/image_folder                                              # a folder: the labels are found for you
iterate run --data path/to/image_folder --labels labels.csv --key image_id --target breed   # or name the label table yourself
```

The cost line is honest: an image run is tens of minutes, not seconds. Four live runs on a local gemma4:12b and an Apple M5 took 29 to 49 minutes for 2 or 3 iterations. EuroSAT, 27,000 satellite tiles, went from `f1_macro` 0.948 to 0.985 in a 3-iteration run of 43 minutes. Storm wind speed went from rmse 13.12 knots to 9.16 against a measured ceiling of 8.87, in 2 iterations and 48.5 minutes (main 6485be3, before the keep-best guard, machine under memory pressure). Flowers102, 102 flower species, went from accuracy 0.554 to 0.981 in 3 iterations and 39 minutes. That is past the 0.974 our own ceiling sweep had found, by 0.7 points where one standard error is 0.4: the agent fine-tuned convnext_tiny at 224 px, a pairing the sweep never ran. It is GPU compatible: Apple GPUs (MPS) and NVIDIA GPUs (CUDA), and it falls back to CPU. On a 12B the agent mostly steers `fit()`: across eleven live image sessions it wrote its own torch code in one, and that model scored below the `fit()` best, 0.9478 against 0.9848. Most of the gain is pretrained features, which is what transfer learning is. Labels are classes or numbers only: no detection, no segmentation.

---

## Quick start

**Local-first. $0. No API keys required.**

```bash
# 1. Install Ollama + a local model (one-time)
brew install ollama
ollama pull gemma4:12b         # the local model every release since v0.2 is validated on
ollama serve                   # background server at localhost:11434

# 2. Install iterate (pulls scikit-learn / XGBoost / LightGBM)
pip install iterate-ai         # "iterate" was taken on PyPI; the command is still `iterate`

# 3. Prepare a tabular CSV (your standard ML data cleaning) and run
iterate run --data train.clean.csv --target churn --metric f1

# 3b. Or iterate a prompt: a labelled eval set + one line saying what the job is
python examples/toxicity_jigsaw/prepare.py     # builds examples/toxicity_jigsaw/data.csv, no account needed
iterate run --data examples/toxicity_jigsaw/data.csv --target label \
            --task "decide whether this Wikipedia comment is toxic" --metric f1

# 3c. Or train an image model: a CSV of image paths (or a folder of images, no --target)
python examples/flowers102/prepare.py          # downloads Oxford Flowers102 (329 MiB), no account needed
iterate run --data examples/flowers102/data.csv --target label --metric accuracy
#     needs torch: the run installs it at the start with your consent, or: pip install 'iterate-ai[vision]'
```

The first run offers a one-time setup wizard (backend, model, compute, install consent); after that, flags override saved defaults per run.

```bash
# Run the generated code in an isolated cloud sandbox instead of locally:
iterate run --data train.clean.csv --target churn --metric f1 --compute e2b

# Use a cloud LLM backend (aliases: groq, together, deepseek, openai):
iterate run --data train.clean.csv --target churn --metric f1 \
            --backend groq --model llama-3.3-70b-versatile --api-key "$GROQ_API_KEY"

# Seed the baseline from an existing notebook/script (read as text, never executed):
iterate run --data train.clean.csv --target churn --metric f1 \
            --source baseline_notebook.ipynb --baseline 0.78

# Bound the whole run; keep every iteration's notebook:
iterate run --data train.clean.csv --target churn --metric f1 \
            --until 30m --notebooks all
```

Useful flags: `--train` + `--holdout` instead of `--data` (bring your own split: the holdout is sealed exactly as given; code path only, the `--spec` lane keeps `--data`), `--max-iterations`, `--patience`, `--until` (wall-clock bound), `--notebooks best|all|none`, `--compute local|e2b`, `--install/--no-install` (package-install consent), `--think` (reasoning mode for the coder, Ollama only), `--fresh` (archive memory, start a new chapter), `--plain` (classic output instead of the interactive UI), `--requests-per-hour` (how many predictions an hour the winner will serve, default 1,000; the end of the run prices the winner at that rate), `--spec` (the v0.1 allow-list path, kept as the fast lane). Prompt runs: `--task` (switches to prompt iteration), `--prompt-file` (your current prompt as the baseline), `--target-model` / `--target-backend` (the model whose prompt is tuned, separate from the one driving the run), `--loop-holdout` (records per candidate during the search). Image folders: `--labels` (the table that holds the labels, when the rules should not pick one), `--key` (the column in `--labels` that names each image), `--yes` (accept a link the rules could not fully prove, or one the Linker proposed, without the pause). Full reference: `iterate run --help`

**Where things land:** `.iterate/runs/<run_id>/best.ipynb` (the runnable winner), `meta.json` + `train.csv` + `holdout.csv` beside it (the bytes the session read; the holdout has no labels in it), `notebooks/` (with `--notebooks all`), `best.json` (config + score sidecar, and a `serving` block: what the winner costs a month to serve at your request rate, on the cheapest machine or API that can, with where each number came from; the same line prints after `best:` at the end of the run), `prompts.yaml` on a prompt run (every version with its score, the best marked, and the same `serving` block), `best_model.pt` on an image run (the winning network, read-only). `.iterate/` is git-ignored by a `.gitignore` written when the folder is first made, since a run folder holds a copy of your training rows. Code-path winners ship as notebooks by design; `--spec` winners also save `best_model.joblib`. `--output` moves the model file, and `best.json` goes beside it.

**Use the model.** An image run's `best_model.pt` holds the weights, the recipe, the class names, the image size and the label scale. Loading it downloads nothing, and it resizes your images the way training did.

```python
from iterate.vision import load

model = load(".iterate/runs/<run_id>/best_model.pt")
model.predict(["a.jpg", "b.jpg"])        # class names, or numbers in the label's own units
model.predict_proba(["a.jpg", "b.jpg"])  # one row per image, one column per entry of model.classes
```

Run All on an image `best.ipynb` trains the recipe again and writes its own network file. It never touches `best_model.pt`, which is read-only and is the network that was scored. `predict` takes a list of paths, PIL images or arrays. Your app needs `iterate-ai[vision]` installed, which brings iterate, torch and torchvision; `load` itself imports only numpy, PIL and torch, and it refuses a torch older than 2.6. The file is opened as plain weights (`weights_only=True`), so opening it runs no code, and a file that is not an iterate network is refused. If no try beat the baseline, or the winner is the agent's own torch code, no network is saved in v0.6.0 and the run says so at the end; `best.ipynb` holds the code that built it.

**Say what the network should be.** An image `fit()` takes two more shapes, so the architecture the research names can actually be trained. `fit(layers=[("conv", 32), ("pool",), ("conv", 64), ("pool",), ("dropout", 0.3), ("linear", 256)])` trains that whole network from zero. `fit(head=[("linear", 512), ("dropout", 0.5)])` puts those layers where a pretrained backbone's single final layer was, which is the most common thing people do with a pretrained model. There are four layer names, `conv` (a convolution, batch norm and ReLU together), `pool`, `dropout` and `linear`; the harness works out every input and output size and always adds the final layer itself.

You can ask for one mid-run. Type `try conv(32) pool conv(64) pool linear(256)` or `use a head of linear(512) dropout(0.5)` into a running image session and the harness reads the stack itself, tells you what it understood, and opens that lever for the next experiment. An ask buys one experiment; after that the run's own evidence decides again. A "no" word anywhere in the ask ("never build from scratch") opens nothing, because a ban is not a request, and prose that merely says the word ("try a custom CNN") opens nothing either: only `name(number)` counts.

**Safety.** A run reads only the data you name (`--data`, or `--train` and `--holdout`, plus a `--labels`, `--prompt-file` or `--source` file), and `--source` is read as text, never run. An image path or a link that leads outside what you named is refused before any file opens. A refusal names the link a path leaves through, and for a folder or a CSV of relative paths it prints a quoted copy command to paste; a CSV row that leaves by `..` or an absolute path is told to move the images under the CSV's folder or drop the rows, and a link that leads nowhere or goes round in a loop is told to remove the link. On a Mac every local cell runs in the macOS sandbox: it opens its own folder, the files the run hands it, its Python and model weights under `~/.cache/iterate/weights`, and nothing else, while the network stays open so weights and packages can download. Linux and WSL2 have no sandbox yet: the run says its cells are not confined, and `--compute e2b` runs them isolated. No API key is passed into a cell's environment except the one a prompt run's model needs and HF_TOKEN; where cells are not confined, a cell can still read saved keys such as `~/.config/iterate/config.toml` or a project `.env`. A cell never installs anything. On a local run, with your consent (`--install`), the harness installs a missing import one of four ways, picked by a dry run: install it, restart the kernel with it, save it for the next run, or refuse and say why; torch and torchvision never change mid-run, and every install carries a constraints file. On e2b the harness installs into its disposable sandbox. The holdout labels never enter the kernel.

> **Note on the one-line form.** The `iterate "improve our churn baseline"` experience,
> where the agent discovers the data, baseline, and metric itself, is the **v1.0 vision**,
> not v0.6. Today you pass `--data`/`--target` explicitly (`--task` for a prompt run, and no
> `--target` for a folder of images); the inputs shrink release by release (see the roadmap).
> Auto-discovery and cost-constrained serving are on the roadmap, not shipped yet.

---

## Three target families (the v1.0 shape)

| Target | What it iterates on | Status |
|---|---|---|
| `ModelTarget` | Trains a tabular model, scores it on a sealed holdout | **shipped (v0.1, code-gen in v0.2)** |
| `DLModelTarget` | Trains an image model for classes or numbers, from a plain CNN baseline to fine-tuned pretrained backbones and the agent's own torch code, and scores it on a sealed holdout. `iterate run` starts it from a folder, a CSV of image paths, or a `--train` and `--holdout` pair | **shipped (v0.6)** |
| `PromptTarget` | Runs an LLM prompt against a labelled eval set, one model call per record, scored on a sealed holdout | **shipped (v0.5)** |

All inherit from `BenchmarkTarget`. Same iteration loop, different execution path. (LLMs are **prompt-iteration only**; we don't fine-tune foundation models.)

---

## Pluggable data + tools via MCP (v0.9)

`iterate` will use **Model Context Protocol (MCP)** servers as its discovery layer: filesystem, Postgres, Notion and friends, so adding a data source is config, not code. The discovery workflow (agent introspects your tables, past experiments, and notebooks, then pauses for your gap-fill) lands at v0.9. Today the data interface is a prepared CSV, deliberately: the loop had to be proven before the input surface grows.

---

## Architecture

```
src/iterate/
├── core/                 # the reasoning engine
│   ├── agent_loop        # v0.2 loop: Supervisor briefs -> coder runs -> Summarizer digests
│   ├── supervisor        # strategist: grounded briefs + deterministic no-op guards
│   ├── coder             # cell-by-cell coding agent on a live stateful kernel
│   ├── summarizer        # per-experiment digest (cross-notebook knowledge transfer)
│   ├── researcher        # literature grounding with real citations (OpenAlex + arXiv)
│   ├── critic            # leak review before a score banks
│   ├── linker            # works out how a folder's images link to their labels
│   ├── prompt_runtime    # a prompt session's ask / evaluate / submit
│   ├── vision_session    # an image session's fit / evaluate / submit
│   ├── codegen           # code-gen contract, session preamble, floor submission
│   ├── orchestrator      # v0.1 spec-path loop (--spec)
│   ├── proposer          # spec-path proposer + dataset profiling
│   ├── reconstructor     # rebuild a baseline from --source (text only, never executed)
│   ├── memory            # persistent experiment store (sqlite)
│   ├── scoring           # sealed-holdout scoring, shared by both paths
│   ├── serving           # the Pricer: what the winner costs to serve, from a dated price snapshot
│   └── terminator        # deadline / patience / max-iterations gates
├── targets/              # BenchmarkTarget protocol + ModelTarget (tables), PromptTarget, DLModelTarget (images)
├── adapters/
│   ├── data/             # csv loading + profiling, image loading, the folder linker, the monitor, the workspace
│   ├── models/           # estimator registry (spec path)
│   └── compute/          # LocalKernel + E2BKernel (Jupyter), runners, the macOS cell sandbox, harness-side installs
├── deliver/              # runnable .ipynb rendering (sessions, leaderboards)
├── llm/                  # pluggable backends: native Ollama client + one
│                         #   OpenAI-compatible client (Groq/Together/Deepseek/OpenAI/vLLM)
├── prompts/              # every prompt in one yaml, versioned with the code
└── schemas/              # Pydantic types
```

**The LLM is plug-and-play; the harness does the lifting.** The same loop runs on a local 12B or a cloud 70B. The bet (the infra-over-model A/B is logged in [DECISIONS.md](DECISIONS.md) and [BUILD_LOG.md](BUILD_LOG.md)): a good enough harness makes weak local models perform like much bigger ones, and the guard stack is what closed that gap.

---

## Comparison with existing tools (the v1.0 target)

| Capability | AutoML (DataRobot/H2O) | W&B / MLflow | Braintrust / LangSmith | AIDE | **iterate** |
|---|---|---|---|---|---|
| Iterates ML models autonomously | ✓ | ✗ | ✗ | ✓ | **✓ shipped** |
| Agent writes its own training code | ✗ | ✗ | ✗ | ✓ | **✓ shipped** |
| Persistent memory across sessions | ✗ | log only | ✗ | ✗ | **✓ shipped** |
| Bounded autonomy (deadline / patience) | ✗ | ✗ | ✗ | partial | **✓ shipped** |
| Auditable reasoning trail (runnable notebooks) | ✗ | ✗ | ✗ | basic | **✓ shipped** |
| Iterates LLM prompts | ✗ | ✗ | eval only | ✗ | **✓ shipped** |
| Iterates DL / vision models | partial | ✗ | ✗ | partial | **✓ shipped** |
| Literature-aware proposals | ✗ | ✗ | ✗ | partial | ✓ |
| Cost-to-serve-aware optimization | ✗ | ✗ | ✗ | ✗ | planned v0.7 |
| Auto-discovers data + context (MCP) | ✗ | ✗ | ✗ | partial | planned v0.9 |
| Open-source | mostly ✗ | MLflow yes | ✗ | ✓ | ✓ |

Known limits are documented honestly in [LIMITATIONS.md](LIMITATIONS.md); the evaluation trail lives in [BUILD_LOG.md](BUILD_LOG.md), and version-over-version measurements in [evals/RESULTS.md](evals/RESULTS.md) — same datasets, same floor model, same budget, with a brute-force ceiling per dataset so a flat result can be read as either a miss or an exhausted problem. That harness is internal development tooling and is not part of the installed package.

---

## License

MIT. The framework is open-source. Adapters for proprietary data sources can be built on top.

---

## Author

Anthony Rodrigues: [GitHub](https://github.com/anthonyrodrigues443)
