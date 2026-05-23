# iterate

**Autonomous research-aware iteration agent for ML models and LLM prompts.**

> Every YC batch ships 200+ AI startups with 2-3 engineer teams. Under shipping pressure, two things break: nobody re-iterates models against new baselines, and LLM prompts sit in production for months untouched. Engineers re-run failed experiments because nobody logged why. Teams pay GPT-5 prices because nobody tested whether Haiku + better prompting would do the job at 1/50th the cost.
>
> AutoML brute-forces. Experiment trackers only log. Prompt evals only evaluate. AIDE iterates Kaggle problems once. `iterate` is the only system that runs an autonomous, literature-aware, memory-persistent improvement loop on BOTH ML models AND LLM prompts in production — with human-approval safety gates and append-only reasoning logs to wherever your team reads.

---

## Status

🚧 **Building. Week 0 of 5.** First production-ready release planned for end of Week 5.

| Week | Phase | Status |
|---|---|---|
| 0 | Scaffolding + scope lock | ✅ |
| 1 | Framework skeleton + LLM client + first end-to-end smoke test | ⏳ |
| 2 | `ModelTarget` + sklearn/XGBoost adapters + first tabular iteration | — |
| 3 | `PromptTarget` + LLM-as-judge eval + first prompt iteration | — |
| 4 | Researcher + Proposer + Memory + Logging + **MCP layer** (filesystem / postgres / notion plug-ins for data discovery + experiment history search) | — |
| 5 | Termination logic + multi-LLM backend benchmark + **Streamlit chat UI** (demo-ready, looks like a desktop app) + demo video | — |

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
🤖 I found:
   Repo:       customer-platform/ml-models/churn (last commit: 3d ago)
   Training:   train_churn.py — CatBoost, F1=0.78 baseline
   Past tries: 4 attempts in Notion. Best: March, tenure features, F1=0.78.
   Tables:     users, subscriptions, support_tickets, events
   Missing:    No eval script located. Where does evaluation live?

> Eval is in customer-platform/eval/churn_eval.py. Also new plan_tier column.

🤖 Got it. My top recommendation:
   → LightGBM + focal loss (Lin et al 2024) — addresses class imbalance 
     that broke March's attempt. Est +0.04 F1, 4 min runtime. Go?
```

Then the autonomous loop:

```
1. Research — arxiv + papers-with-code for relevant 2024-2026 work
2. Propose — LLM ranks candidate experiments by expected impact ÷ cost
3. Memory check — has this been tried? did conditions change since the last failure?
4. Run — execute in a sandboxed environment
5. Score — compare against baseline
6. Log — write a reasoning-trail card to your logging target
7. Decide — continue or terminate (deadline / patience / plateau / idea-exhaustion)
```

Every decision cites either a paper or a past experiment. Every failure is logged with the **reason it failed** so the agent can revisit when conditions change.

---

## Two target families

| Target | What it iterates on | Example demo (ships with the framework) |
|---|---|---|
| `ModelTarget` | Trains a model, scores it on a holdout | Tabular churn prediction (Kaggle) |
| `PromptTarget` | Runs an LLM prompt, scores outputs (LLM-as-judge or labeled set) | Jigsaw toxicity classification |

Both inherit from `BenchmarkTarget`. Same iteration loop. Different sandbox execution path.

---

## Pluggable tools + data sources (via MCP)

`iterate` uses **Model Context Protocol (MCP)** servers as its tool + data layer. Adding a new data source = config-only, no code changes.

Ships with:

| MCP server | What it enables |
|---|---|
| `filesystem` | Read local notebooks, past experiment logs, internal docs |
| `postgres` | DB introspection + read-only sampling (for data discovery) |
| `notion` | Search past experiment pages, write new experiment cards |

**The discovery workflow** (Week 4 feature):

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
ollama pull qwen2.5-coder:14b
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

## Demo UI (Week 5)

A Streamlit-based chat interface that looks and feels like a desktop app — launches in your browser, runs entirely locally, screenshot-ready for demos:

```
┌──────────────────────────────────────────────────────────┐
│  iterate                                                  │
├─────────────────┬────────────────────────────────────────┤
│ MCP STATUS      │  > iterate "improve churn baseline"     │
│ ✓ filesystem    │                                          │
│ ✓ postgres      │  🤖 Scanning your repos...               │
│ ✓ notion        │     Found 3 candidate repos.             │
│ ✓ github        │                                          │
│                 │  🤖 Reading customer-platform/...        │
│ EXPERIMENTS     │     Baseline: CatBoost F1=0.78           │
│ #001 ✅ +0.04   │                                          │
│ #002 ❌         │  🤖 Found 4 past experiments in Notion.   │
│ #003 🔁         │     Best: March, tenure features.        │
│                 │                                          │
│ MEMORY          │  🤖 Anything else I should know about?    │
│ 47 entries      │                                          │
│ 12 retried      │  > Eval lives in eval/churn_eval.py      │
│                 │                                          │
│ COST            │  🤖 Got it. Top recommendation: ...       │
│ $0.03 today     │                                          │
└─────────────────┴────────────────────────────────────────┘
```

CLI is the canonical install. The Streamlit chat is the demo-ready interface.

---

## Architecture

```
iterate/
├── core/                # framework reasoning engine
│   ├── orchestrator     # the main loop
│   ├── researcher       # arxiv + papers-with-code retrieval
│   ├── proposer         # LLM ranks candidate experiments
│   ├── memory           # persistent store (sqlite)
│   ├── terminator       # deadline / patience / plateau gates
│   └── reporter         # PR-shaped report generator
├── targets/             # what gets iterated on
│   ├── base             # BenchmarkTarget protocol
│   ├── model            # ModelTarget
│   └── prompt           # PromptTarget
├── adapters/            # pluggable I/O
│   ├── data/            # csv, kaggle, huggingface, postgres
│   ├── models/          # sklearn, xgboost, lightgbm, pytorch
│   ├── compute/         # local, e2b sandbox
│   └── logging/         # markdown, notion_mcp, slack
├── llm/                 # multi-backend LLM client
│   ├── anthropic_client # Claude (Opus / Haiku)
│   ├── openai_client    # GPT
│   ├── together_client  # Llama via Together AI
│   └── deepseek_client  # Deepseek
└── schemas/             # Pydantic types
```

**The LLM is plug-and-play.** Claude, GPT, Llama 3.3, Deepseek — flip a config flag. The moat is the agentic harness (memory + research + tools + bounded loop), not the model.

---

## Multi-LLM backend (planned Week 5 benchmark)

| Backend | Est. cost per run | Notes |
|---|---|---|
| Claude Opus 4.7 | ~$4 | Best tool-use reliability |
| Claude Haiku 4.5 | ~$0.30 | Recommended default |
| Llama 3.3 70B (Together) | ~$0.20 | Free tier available via Groq |
| Deepseek V3 | ~$0.10 | Strong on code |

Week 5 will ship the head-to-head matrix on identical tasks.

---

## Comparison with existing tools

| Capability | AutoML (DataRobot/H2O) | W&B / MLflow | Braintrust / LangSmith | AIDE | **iterate** |
|---|---|---|---|---|---|
| Iterates ML models | ✅ | — | — | ✅ | ✅ |
| Iterates LLM prompts | — | — | eval only | — | ✅ |
| Literature-aware | ❌ | ❌ | ❌ | partial | ✅ |
| Persistent memory across sessions | ❌ | log only | ❌ | ❌ | ✅ |
| Revisits failures when conditions change | ❌ | ❌ | ❌ | ❌ | ✅ |
| Bounded autonomy (deadline / patience) | ❌ | ❌ | ❌ | partial | ✅ |
| Auditable reasoning trail | ❌ | — | ❌ | basic | ✅ |
| Human-approval gate | ❌ | n/a | ❌ | ❌ | ✅ |
| Logs to Notion / Drive / MD | ❌ | own dashboard | own dashboard | ❌ | ✅ |
| Multi-LLM backend | ❌ | n/a | partial | ❌ | ✅ |
| Open-source | mostly ❌ | MLflow yes | ❌ | ✅ | ✅ |

---

## Why this exists

Production AI teams forget what they've tried. So they keep retrying it. `iterate` is the institutional memory + research desk + experiment runner those teams don't have time to build.

---

## License

MIT (planned). The framework is open-source. Adapters for proprietary data sources can be built on top.

---

## Author

Anthony Rodrigues — [GitHub](https://github.com/anthonyrodrigues443)
