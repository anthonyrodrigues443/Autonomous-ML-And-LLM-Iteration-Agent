# iterate

**Autonomous research-aware iteration agent for ML models and LLM prompts.**

> Every YC batch ships 200+ AI startups with 2-3 engineer teams. Under shipping pressure, two things break: nobody re-iterates models against new baselines, and LLM prompts sit in production for months untouched. Engineers re-run failed experiments because nobody logged why. Teams pay GPT-5 prices because nobody tested whether Haiku + better prompting would do the job at 1/50th the cost.
>
> AutoML brute-forces. Experiment trackers only log. Prompt evals only evaluate. AIDE iterates Kaggle problems once. `iterate` is being built as the system that runs an autonomous, literature-aware, memory-persistent improvement loop on **ML models, DL/vision models, AND LLM prompts** in production, optimizing for the best model you can actually **afford to serve**. That is the v1.0 vision; the releases below get there one dial at a time.

> **How this gets built:** [WORKFLOW.md](WORKFLOW.md) (the method) · [DECISIONS.md](DECISIONS.md) (every call I made against the AI's default) · [BUILD_LOG.md](BUILD_LOG.md) (the daily trail)

---

## Status

**v0.2 released: the agent writes and runs its own training code.** Install with `pip install iterate-ai`. In v0.1 the agent picked estimators from an allow-list; in v0.2 it works like an engineer in a notebook. A **Supervisor** reads the run history and briefs one experiment. A **coding agent** executes that brief cell by cell in a live Jupyter kernel: inspect, transform, fit, validate, submit, reading each cell's real output before writing the next. A **Summarizer** distills every finished experiment so the next one inherits what worked, what failed, and why. The winner ships as a runnable notebook.

**Agent-first:** the autonomous loop landed at v0.1 (Week 3), not as a late-stage add-on. Two dials turn release to release: the inputs you must give *shrink* (toward one-sentence input) and the problem types *grow* (tabular, then prompts, then DL/vision).

| Week | Phase | Status |
|---|---|---|
| 0 | Scaffolding + scope lock | done |
| 1 | Foundation: schemas + LLM client (tool-calling) + config + CLI | done |
| 2 | Tabular execution substrate: `BenchmarkTarget` + data adapter + `ModelTarget` + model factory + local executor | done |
| 3 | **The agentic loop**: Proposer + Orchestrator + Terminator + Memory + CLI, first autonomous tabular run (**v0.1**) | done |
| 4-5 | **Sandboxed code-gen + the multi-agent cell-by-cell system** (Supervisor, coding agent, Summarizer, pulled forward from the original v0.4 plan) + notebook deliverable + live progress + graceful Ctrl-C (**v0.2**) | done |
| 6 | **Full interactive CLI**: pause, mid-run chat, resume + token streaming (**v0.3**) | next |
| 7 | Researcher + Critic specialists; agent picks the metric + starting model (**v0.4**) | planned |
| 8 | `PromptTarget`: agentic prompt iteration (**v0.5**) | planned |
| 9 | `DLModelTarget`: vision transfer learning, validated on a local RTX 4050 (**v0.6**) | planned |
| 10 | **Cost-constrained recommendation** + serving profile + `iterate cost` (**v0.7**) | planned |
| 11 | Infer features/target from the data + a description (**v0.8**) | planned |
| 12 | **MCP discovery**: find the data/code itself (**v0.9**) | planned |
| 13 | Multi-backend benchmark + **Streamlit chat UI** + demos (**v0.10**) | planned |
| 14 | Full minimum-viable-input + polish + launch (**v1.0**) | planned |

---

## What v0.2 does

You give it a prepared CSV, the target column, and a metric. The agent does the rest: builds its own baseline, then runs one briefed experiment per iteration, cell by cell, against a sealed holdout it never sees.

```bash
iterate run --data churn.clean.csv --target Churn --metric f1
```

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

**Deliverables.** The winner is exported as `best.ipynb`: a runnable, annotated notebook of the actual winning session (hypothesis, staged cells with their real outputs, dead ends labeled, findings). `--notebooks all` keeps one notebook per iteration: the full research journey. Every experiment also persists in `.iterate/memory.db`, so the next run builds on this one.

---

## Quick start

**Local-first. $0. No API keys required.**

```bash
# 1. Install Ollama + a local model (one-time)
brew install ollama
ollama pull gemma4:12b         # the model v0.2 was validated on
ollama serve                   # background server at localhost:11434

# 2. Install iterate (pulls scikit-learn / XGBoost / LightGBM)
pip install iterate-ai         # "iterate" was taken on PyPI; the command is still `iterate`

# 3. Prepare a tabular CSV (your standard ML data cleaning) and run
iterate run --data train.clean.csv --target churn --metric f1
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

Useful flags: `--max-iterations`, `--patience`, `--until` (wall-clock bound), `--notebooks best|all|none`, `--compute local|e2b`, `--install/--no-install` (package-install consent), `--think` (reasoning mode for the coder, Ollama only), `--fresh` (archive memory, start a new chapter), `--spec` (the v0.1 allow-list path, kept as the fast lane). Full reference: `iterate run --help`

**Where things land:** `.iterate/runs/<run_id>/best.ipynb` (the runnable winner), `notebooks/` (with `--notebooks all`), `best.json` (config + score sidecar). Code-path winners ship as notebooks by design; `--spec` winners also save `best_model.joblib`.

**Safety boundaries:** your `--source` file is read as text, never executed. The generated code runs locally only with your consent (the setup wizard asks), or fully isolated with `--compute e2b`. The holdout labels never enter the kernel; scoring happens host-side.

> **Note on the one-line form.** The `iterate "improve our churn baseline"` experience,
> where the agent discovers the data, baseline, and metric itself, is the **v1.0 vision**,
> not v0.2. Today you pass `--data`/`--target`/`--metric` explicitly; the inputs shrink
> release by release (see the roadmap). Auto-discovery, prompt + vision targets, and
> cost-constrained serving are on the roadmap, not shipped yet.

---

## Three target families (the v1.0 shape)

| Target | What it iterates on | Status |
|---|---|---|
| `ModelTarget` | Trains a tabular model, scores it on a sealed holdout | **shipped (v0.1, code-gen in v0.2)** |
| `DLModelTarget` | Transfer-learns a vision model, scores it | planned (v0.6) |
| `PromptTarget` | Runs an LLM prompt against an eval set, scores outputs | planned (v0.5) |

All inherit from `BenchmarkTarget`. Same iteration loop, different execution path. (LLMs are **prompt-iteration only**; we don't fine-tune foundation models.)

---

## Pluggable data + tools via MCP (Week 12, v0.9)

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
│   ├── codegen           # code-gen contract, session preamble, floor submission
│   ├── orchestrator      # v0.1 spec-path loop (--spec)
│   ├── proposer          # spec-path proposer + dataset profiling
│   ├── reconstructor     # rebuild a baseline from --source (text only, never executed)
│   ├── memory            # persistent experiment store (sqlite)
│   ├── scoring           # sealed-holdout scoring, shared by both paths
│   └── terminator        # deadline / patience / max-iterations gates
├── targets/              # BenchmarkTarget protocol + tabular ModelTarget
├── adapters/
│   ├── data/             # csv loading + profiling
│   ├── models/           # estimator registry (spec path)
│   └── compute/          # LocalKernel + E2BKernel (Jupyter), runners, sandbox
├── deliver/              # runnable .ipynb rendering (sessions, leaderboards)
├── llm/                  # pluggable backends: native Ollama client + one
│                         #   OpenAI-compatible client (Groq/Together/Deepseek/OpenAI/vLLM)
├── prompts/              # every prompt in one yaml, versioned with the code
└── schemas/              # Pydantic types
```

**The LLM is plug-and-play; the harness does the lifting.** The same loop runs on a local 12B or a cloud 70B. The bet (see the [infra-over-model A/B in EVAL_LOG.md](EVAL_LOG.md)): a good enough harness makes weak local models perform like much bigger ones, and the guard stack is what closed that gap.

---

## Comparison with existing tools (the v1.0 target)

| Capability | AutoML (DataRobot/H2O) | W&B / MLflow | Braintrust / LangSmith | AIDE | **iterate** |
|---|---|---|---|---|---|
| Iterates ML models autonomously | ✓ | ✗ | ✗ | ✓ | **✓ shipped** |
| Agent writes its own training code | ✗ | ✗ | ✗ | ✓ | **✓ shipped** |
| Persistent memory across sessions | ✗ | log only | ✗ | ✗ | **✓ shipped** |
| Bounded autonomy (deadline / patience) | ✗ | ✗ | ✗ | partial | **✓ shipped** |
| Auditable reasoning trail (runnable notebooks) | ✗ | ✗ | ✗ | basic | **✓ shipped** |
| Iterates LLM prompts | ✗ | ✗ | eval only | ✗ | planned v0.5 |
| Iterates DL / vision models | partial | ✗ | ✗ | partial | planned v0.6 |
| Literature-aware proposals | ✗ | ✗ | ✗ | partial | planned v0.4 |
| Cost-to-serve-aware optimization | ✗ | ✗ | ✗ | ✗ | planned v0.7 |
| Auto-discovers data + context (MCP) | ✗ | ✗ | ✗ | partial | planned v0.9 |
| Open-source | mostly ✗ | MLflow yes | ✗ | ✓ | ✓ |

---

## Why this exists

Production AI teams forget what they've tried. So they keep retrying it. `iterate` is the institutional memory + research desk + experiment runner those teams don't have time to build.

Known limits are documented honestly in [LIMITATIONS.md](LIMITATIONS.md); the evaluation trail lives in [EVAL_LOG.md](EVAL_LOG.md).

---

## License

MIT. The framework is open-source. Adapters for proprietary data sources can be built on top.

---

## Author

Anthony Rodrigues: [GitHub](https://github.com/anthonyrodrigues443)
