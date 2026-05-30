# iterate

**Autonomous research-aware iteration agent for ML models and LLM prompts.**

> Every YC batch ships 200+ AI startups with 2-3 engineer teams. Under shipping pressure, two things break: nobody re-iterates models against new baselines, and LLM prompts sit in production for months untouched. Engineers re-run failed experiments because nobody logged why. Teams pay GPT-5 prices because nobody tested whether Haiku + better prompting would do the job at 1/50th the cost.
>
> AutoML brute-forces. Experiment trackers only log. Prompt evals only evaluate. AIDE iterates Kaggle problems once. `iterate` is the only system that runs an autonomous, literature-aware, memory-persistent improvement loop on **ML models, DL/vision models, AND LLM prompts** in production — optimizing for the best model you can actually **afford to serve** (cheapest cloud, cost/month, requests/hour) — pulling its own training data + context from your DBs, files, and docs (via MCP) — with human-approval safety gates and append-only reasoning logs to wherever your team reads.

> **How this gets built:** [WORKFLOW.md](WORKFLOW.md) (the method) · [DECISIONS.md](DECISIONS.md) (every call I made against the AI's default)

---

## Status

**Building — Week 3 in progress; agentic loop done (Days 1–5), v0.1 tags after Days 6–7.** First release **v0.1** (the working agentic loop) lands at the end of Week 3; incremental releases follow through to the full **v1.0** (~early Sep 2026).

**Agent-first:** the autonomous loop is the **v0.1** milestone (Week 3), not a late-stage add-on. After that, two dials turn release to release — the inputs you must give *shrink* (toward one-sentence input) and the problem types *grow* (tabular → prompts → DL/vision). Full roadmap + daily trail in [BUILD_LOG.md](BUILD_LOG.md).

| Week | Phase | Status |
|---|---|---|
| 0 | Scaffolding + scope lock | done |
| 1 | Foundation — schemas + LLM client (tool-calling) + config + CLI | done |
| 2 | Tabular execution substrate — `BenchmarkTarget` + data adapter + `ModelTarget` + model factory + local executor | done |
| 3 | **The agentic loop** — Proposer + Orchestrator + Terminator + Memory + CLI → first autonomous tabular run (**v0.1**) | in progress |
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

## What it does

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

## Quick start

**Local-first. $0. No API keys required.**

```bash
# Install Ollama (one-time setup)
brew install ollama
ollama pull qwen3:14b
ollama serve  # starts background server at localhost:11434

# Install iterate
pip install iterate

# Run it (the agent discovers everything else)
iterate "improve our customer churn baseline"

# Power-user: skip discovery, give explicit pointers
iterate init --data train.csv --target churn --baseline 0.78 --metric f1
iterate run --until 2026-06-01 --patience 15

# Inspect history
iterate history
iterate why-failed exp_042
iterate best
```

Full CLI reference: `iterate --help`

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

## Why this exists

Production AI teams forget what they've tried. So they keep retrying it. `iterate` is the institutional memory + research desk + experiment runner those teams don't have time to build.

---

## License

MIT (planned). The framework is open-source. Adapters for proprietary data sources can be built on top.

---

## Author

Anthony Rodrigues — [GitHub](https://github.com/anthonyrodrigues443)
