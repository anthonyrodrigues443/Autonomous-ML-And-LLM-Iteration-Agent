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
| 4 | Researcher + Proposer + Memory + Logging adapters (Notion / MD) | — |
| 5 | Termination logic + multi-LLM backend benchmark + Streamlit dashboard + demo | — |

---

## What it does

You give `iterate`:
- A **dataset** or **prompt + eval set** (a `BenchmarkTarget`)
- A **baseline metric** to beat
- **Constraints**: deadline, compute budget, patience threshold
- A **logging target**: Notion page, Markdown file, Google Drive

It runs an autonomous loop until it beats the baseline, runs out of ideas, or hits the deadline:

```
1. Research — query arxiv + papers-with-code for relevant 2024-2026 work
2. Propose — LLM ranks candidate experiments by expected impact ÷ cost
3. Memory check — has this been tried? did conditions change since the last failure?
4. Run — execute the experiment in a sandboxed environment
5. Score — compare against baseline
6. Log — write a reasoning-trail card to your logging target
7. Decide — continue or terminate (deadline, no-improvement, plateau, idea-exhaustion)
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

## Quick start

```bash
# Install
pip install iterate

# Initialize a project
iterate init --data train.csv --target churn --baseline 0.78 --metric f1

# Single iteration
iterate run

# Autonomous run until deadline
iterate run --until 2026-06-01 --patience 15

# Inspect history
iterate history
iterate why-failed exp_042
iterate best
```

Full CLI reference: `iterate --help`

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
