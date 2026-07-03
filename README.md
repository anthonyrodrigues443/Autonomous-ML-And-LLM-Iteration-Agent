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
iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1
```

`iterate` runs an autonomous propose-run-score-remember loop on your ML problem. An LLM proposes a model and hyperparameters, the framework trains it leakage-safe, scores it on a sealed holdout, records the attempt in persistent memory, and keeps going until a deadline, patience, or plateau gate stops it. 304 unit tests across 30 files run in CI on every push.

| v0.1 today | On the roadmap |
|---|---|
| Autonomous model + hyperparameter iteration on tabular data (scikit-learn, XGBoost, LightGBM) | Sandboxed LLM code-gen for arbitrary models (v0.2) |
| Persistent memory in `.iterate/memory.db`; every run builds on the last | Interactive CLI: pause, mid-run chat, resume (v0.3) |
| Any OpenAI-compatible LLM backend; local Ollama by default, no API keys | LLM prompt iteration (v0.5), vision transfer learning (v0.6) |
| Best model saved as joblib; bounded loop with deadline / patience / plateau gates | Cost-to-serve recommendations (v0.7), MCP auto-discovery of data + context (v0.9) |

## Why I built this

I kept seeing the same failure mode on small AI teams. A model or a prompt ships, and under delivery pressure nobody iterates on it again, so it sits in production for months while baselines move on. Experiments get re-run because nobody wrote down why they failed the first time. Teams pay frontier-model prices because nobody checked whether a cheaper model with a better prompt would do the job. `iterate` is the institutional memory, research desk, and experiment runner those teams don't have time to build.

> **How this gets built:** [WORKFLOW.md](WORKFLOW.md) (the method) · [DECISIONS.md](DECISIONS.md) (every call I made against the AI's default) · [BUILD_LOG.md](BUILD_LOG.md) (the daily trail)

---

## The full pitch

> Every YC batch ships 200+ AI startups with 2-3 engineer teams. Under shipping pressure, two things break: nobody re-iterates models against new baselines, and LLM prompts sit in production for months untouched. Engineers re-run failed experiments because nobody logged why. Teams pay GPT-5 prices because nobody tested whether Haiku + better prompting would do the job at 1/50th the cost.
>
> AutoML brute-forces. Experiment trackers only log. Prompt evals only evaluate. AIDE iterates Kaggle problems once. `iterate` is the only system that runs an autonomous, literature-aware, memory-persistent improvement loop on **ML models, DL/vision models, AND LLM prompts** in production — optimizing for the best model you can actually **afford to serve** (cheapest cloud, cost/month, requests/hour) — pulling its own training data + context from your DBs, files, and docs (via MCP) — with human-approval safety gates and append-only reasoning logs to wherever your team reads.

---

## Status

**v0.1 released — the working agentic loop on tabular ML.** Install with `pip install iterate-ai`. Incremental releases follow through to the full **v1.0** (~early Sep 2026); the inputs you give shrink and the problem types grow release by release.

**Agent-first:** the autonomous loop is the **v0.1** milestone (Week 3), not a late-stage add-on. After that, two dials turn release to release — the inputs you must give *shrink* (toward one-sentence input) and the problem types *grow* (tabular → prompts → DL/vision). Full roadmap + daily trail in [BUILD_LOG.md](BUILD_LOG.md).

| Week | Phase | Status |
|---|---|---|
| 0 | Scaffolding + scope lock | done |
| 1 | Foundation — schemas + LLM client (tool-calling) + config + CLI | done |
| 2 | Tabular execution substrate — `BenchmarkTarget` + data adapter + `ModelTarget` + model factory + local executor | done |
| 3 | **The agentic loop** — Proposer + Orchestrator + Terminator + Memory + CLI → first autonomous tabular run (**v0.1**) | done |
| 4–5 | **Sandboxed code-gen** + cheap interactive wins (live progress, streaming, Ctrl-C) (**v0.2**) | — |
| 6 | **Full interactive CLI** — pause, mid-run chat, resume (**v0.3**) | — |
| 7 | Agent picks the metric + starting model (**v0.4**) | — |
| 8 | `PromptTarget` — agentic prompt iteration (**v0.5**) | — |
| 9 | `DLModelTarget` — vision transfer learning, validated on local RTX 4050 (**v0.6**) | — |
| 10 | **Cost-constrained recommendation** + serving profile + `iterate cost` (**v0.7**) | — |
| 11 | Infer features/target from the data + a description (**v0.8**) | — |
| 12 | **MCP discovery** — find the data/code itself (**v0.9**) | — |
| 13 | Multi-backend benchmark + **Streamlit chat UI** + demos (**v0.10**) | — |
| 14 | Full minimum-viable-input + polish + launch (**v1.0**) | — |

---

## What it does (the v1.0 vision)

**You give it one input. It figures out the rest.**

```
> iterate "improve our customer churn baseline"
```

Everything else is discovered:

| The agent autonomously finds | How |
|---|---|
| Which repo has the code | Filesystem + GitHub MCP — scan READMEs for keywords, fall back to recent commit activity |
| Training script + current baseline metric | Code parsing + MLflow / W&B MCP — extract from runs, comments, results JSON |
| Eval methodology + holdout split | Filesystem — find test/eval scripts |
| Relevant data tables | Postgres MCP — list, sample, infer relationships |
| Past experiment history (and why things failed) | Notion MCP — semantic search |
| Domain context | Synthesize from READMEs + commit messages |

It then **surfaces what it found and pauses for your gap-fill** before iterating:

```
agent> I found:
   Repo:       customer-platform/ml-models/churn (last commit: 3d ago)
   Training:   train_churn.py — CatBoost, F1=0.78 baseline
   Past tries: 4 attempts in Notion. Best: March, tenure features, F1=0.78.
   Tables:     users, subscriptions, support_tickets, events
   Missing:    No eval script located. Where does evaluation live?

> Eval is in customer-platform/eval/churn_eval.py. Also new plan_tier column.

agent> Got it. My top recommendation:
   → LightGBM + focal loss (Lin et al 2024) — addresses class imbalance 
     that broke March's attempt. Est +0.04 F1, 4 min runtime. Go?
```

Then the autonomous loop:

```
1. Research — arxiv + papers-with-code for relevant 2024-2026 work
2. Propose — LLM ranks candidate experiments by expected score gain
3. Memory check — has this been tried? did conditions change since the last failure?
4. Run — execute in a sandboxed environment
5. Score — compare against baseline
6. Log — write a reasoning-trail card to your logging target
7. Decide — continue or terminate (deadline / patience / plateau / idea-exhaustion)
```

Every decision cites either a paper or a past experiment. Every failure is logged with the **reason it failed** so the agent can revisit when conditions change.

---

## Three target families

| Target | What it iterates on | Example demo (ships with the framework) |
|---|---|---|
| `ModelTarget` | Trains a tabular model, scores it on a holdout | Tabular churn prediction (Kaggle) |
| `DLModelTarget` | Transfer-learns a vision model (fine-tunes a pretrained backbone), scores it | Image classification (validated on local RTX 4050) |
| `PromptTarget` | Runs an LLM prompt in production, scores outputs (LLM-as-judge or labeled set) | Jigsaw toxicity classification |

All inherit from `BenchmarkTarget`. Same iteration loop. Different execution path. (LLMs are **prompt-iteration only** — we don't fine-tune foundation models.)

---

## Pluggable tools + data sources (via MCP)

`iterate` uses **Model Context Protocol (MCP)** servers as its tool + data layer. Adding a new data source = config-only, no code changes.

Ships with:

| MCP server | What it enables |
|---|---|
| `filesystem` | Read local notebooks, past experiment logs, internal docs |
| `postgres` | DB introspection + read-only sampling (for data discovery) |
| `notion` | Search past experiment pages, write new experiment cards |

**The discovery workflow** (Week 11 feature — v0.8):

```
> iterate init --target churn_baseline --discover

[agent introspects via MCP]
  postgres.list_tables             → users, subs, tickets, ...
  postgres.describe_table("users") → schema
  notion.search("churn")           → 3 past experiment pages
  filesystem.search("churn|retention") → 2 local notebooks

Agent SUMMARY (paused for human review):
  Found 8 tables. Likely relevant: users, subscriptions, support_tickets.
  Past experiments in Notion: 3 attempts, best F1=0.78 (CatBoost, March).
  Inferred target: users.churned_30d. Inferred metric: F1.
  
  Any other artifacts I should know about?
  > [paste URLs, additional context, then 'go']
```

Add any other MCP server (Drive, GitHub, Slack, Sentry, custom) by editing one YAML file. The MCP-to-OpenAI-tool bridge layer means it works against Ollama, Groq, Together, Deepseek, OpenAI, and Anthropic alike.

---

## Quick start (v0.1)

**Local-first. $0. No API keys required.** v0.1 iterates **tabular** models — it
chooses the best model + hyperparameters from scikit-learn / XGBoost / LightGBM
for a prepared dataset.

```bash
# 1. Install Ollama + the tool-calling model (one-time)
brew install ollama
ollama pull qwen3:14b          # ~9.3 GB
ollama serve                   # background server at localhost:11434

# 2. Install iterate (heads-up: pulls scikit-learn / XGBoost / LightGBM)
pip install iterate-ai         # "iterate" was taken on PyPI; the command is still `iterate`

# 3. Prepare a tabular CSV (your standard ML data cleaning) and run
iterate run --data train.clean.csv --target churn --metric f1

# Seed the baseline from an existing notebook/script (read as text, never executed):
iterate run --data train.clean.csv --target churn --metric f1 \
            --source baseline_notebook.ipynb --baseline 0.78

# Use a cloud model instead of local Ollama:
iterate run --data train.clean.csv --target churn --metric f1 \
            --backend openai-compatible --base-url https://api.groq.com/openai/v1 \
            --model llama-3.3-70b --api-key "$GROQ_API_KEY"
```

The best model is saved to `.iterate/runs/<run_id>/best_model.joblib` (override with
`--output`) — load and use it directly: `joblib.load(path).predict(X)`. Every
experiment persists in `.iterate/memory.db`, so the next run builds on it.

Full CLI reference: `iterate run --help`

> **Note on the one-line form.** The `iterate "improve our churn baseline"` experience —
> where the agent discovers the data, baseline, and metric itself — is the **v1.0 vision**,
> not v0.1. Today you pass `--data`/`--target`/`--metric` explicitly; the inputs shrink
> release by release (see the roadmap). Auto-discovery, `iterate history` / `why-failed` /
> `best`, prompt + vision targets, and cost-constrained serving are all on the roadmap, not
> shipped in v0.1.

---

## Demo UI (Week 12 — v0.9)

A Streamlit-based chat interface that looks and feels like a desktop app — launches in your browser, runs entirely locally, screenshot-ready for demos:

```
Sidebar (live state):
  MCP status   filesystem, postgres, notion, github  (connected)
  Experiments  #001 win +0.04    #002 fail    #003 retry
  Memory       47 entries, 12 retried
  Cost         $0.03 today

Chat:
  > iterate "improve churn baseline"
    Scanning your repos... found 3 candidates.
    Reading customer-platform/... baseline CatBoost F1=0.78.
    Found 4 past experiments in Notion (best: March, tenure features).
    Anything else I should know about?
  > Eval lives in eval/churn_eval.py
    Got it. Top recommendation: ...
```

CLI is the canonical install. The Streamlit chat is the demo-ready interface.

---

## Architecture

```
iterate/
├── core/                # framework reasoning engine
│   ├── orchestrator     # the main loop
│   ├── researcher       # arxiv + papers-with-code retrieval
│   ├── proposer         # ranks candidates by score, within the serving budget
│   ├── serving_cost     # cheapest-cloud + cost/mo + req/hr estimator
│   ├── memory           # persistent store (sqlite)
│   ├── terminator       # deadline / patience / plateau gates
│   └── reporter         # PR-shaped report generator
├── targets/             # what gets iterated on
│   ├── base             # BenchmarkTarget protocol
│   ├── model            # ModelTarget (tabular)
│   ├── dl_model         # DLModelTarget (vision, transfer learning)
│   └── prompt           # PromptTarget (prompt-iteration)
├── adapters/            # pluggable I/O
│   ├── data/            # csv, kaggle, huggingface, postgres
│   ├── models/          # sklearn, xgboost, lightgbm, pytorch
│   ├── compute/         # local (MPS), gpu (RTX 4050), e2b, cloud (user's cloud / rented GPU)
│   └── logging/         # markdown, notion_mcp, slack
├── llm/                  # pluggable multi-backend LLM client
│   ├── base              # LLMClient protocol (provider-agnostic interface)
│   ├── openai_compatible # one client for ALL OpenAI-compatible backends
│   │                     #   (Ollama default · Groq · Together · Deepseek · OpenAI · vLLM)
│   └── anthropic_client  # Claude — the only non-OpenAI-compatible backend (optional, later)
└── schemas/             # Pydantic types
```

**The LLM is plug-and-play.** Claude, GPT, Llama 3.3, Deepseek — flip a config flag. The moat is the agentic harness (memory + research + tools + bounded loop), not the model — and the optimization target is the best model you can actually *afford to serve*: pure score inside a hard serving-cost budget, with a recommendation of the cheapest cloud to host it on, its monthly cost, and its requests/hour throughput.

---

## Multi-LLM backend (planned Week 12 benchmark)

| Backend | Est. cost per run | Notes |
|---|---|---|
| Claude Opus 4.7 | ~$4 | Best tool-use reliability |
| Claude Haiku 4.5 | ~$0.30 | Recommended default |
| Llama 3.3 70B (Together) | ~$0.20 | Free tier available via Groq |
| Deepseek V3 | ~$0.10 | Strong on code |

Week 12 will ship the head-to-head matrix on identical tasks — scored on quality **and** serving cost.

---

## Comparison with existing tools

| Capability | AutoML (DataRobot/H2O) | W&B / MLflow | Braintrust / LangSmith | AIDE | **iterate** |
|---|---|---|---|---|---|
| Iterates ML models | ✓ | — | — | ✓ | ✓ |
| Iterates DL / vision models (transfer learning) | partial | — | — | partial | ✓ |
| Iterates LLM prompts | — | — | eval only | — | ✓ |
| Literature-aware | ✗ | ✗ | ✗ | partial | ✓ |
| Persistent memory across sessions | ✗ | log only | ✗ | ✗ | ✓ |
| Revisits failures when conditions change | ✗ | ✗ | ✗ | ✗ | ✓ |
| Bounded autonomy (deadline / patience) | ✗ | ✗ | ✗ | partial | ✓ |
| Auditable reasoning trail | ✗ | — | ✗ | basic | ✓ |
| Human-approval gate | ✗ | n/a | ✗ | ✗ | ✓ |
| Logs to Notion / Drive / MD | ✗ | own dashboard | own dashboard | ✗ | ✓ |
| Multi-LLM backend | ✗ | n/a | partial | ✗ | ✓ |
| Cost-to-serve–aware optimization (cheapest cloud, $/mo, req/hr — best score you can afford to serve) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Auto-discovers training data + context (DB / MCP / Drive) | ✗ | ✗ | ✗ | partial | ✓ |
| Open-source | mostly ✗ | MLflow yes | ✗ | ✓ | ✓ |

---

## License

MIT. The framework is open-source. Adapters for proprietary data sources can be built on top.

---

## Author

Anthony Rodrigues — [GitHub](https://github.com/anthonyrodrigues443)
