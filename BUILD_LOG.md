# Build Log

> Daily task tracking for `iterate`. Build sessions log here. Recruiters reading the repo see process, not just final code.
>
> Public file. Honest about what worked + what didn't.

---

## Scope & timeline (re-planned 2026-05-27 — agent-first)

**Agent-first.** The first real release (v0.1) is a *working agentic loop* on tabular ML with explicit inputs — the LLM reads the data, proposes a change, trains, scores, and iterates to the best model by a deadline. After that, two dials turn release to release: **(A) inputs required shrink** (toward one-sentence input) and **(B) problem types grow** (tabular → prompts → DL/vision). The agent is present from v0.1; everything later is capability expansion — not "turn the agent on at Week 7."

*(Supersedes the earlier breadth-first ordering, which wrongly deferred the agentic loop. ~14-week build, May 23 – ~early Sep.)*

**Updated 2026-05-28 — model flexibility has two tiers, and the second is bumped early:**
- **(b) Installed-library factory (v0.1):** a candidate names any estimator in an allow-listed installed library (scikit-learn / XGBoost / LightGBM) by import path + params — `{"model": "lightgbm.LGBMClassifier", "params": {…}}` — and we instantiate it. No hand-curated list.
- **(c) Sandboxed code-gen (v0.2, bumped early):** the Proposer *writes* the training code and runs it in an e2b sandbox → **any model at all**, not just installed libraries. This is the big capability and now lands right after the v0.1 loop.

**Updated 2026-05-30 — interactivity split into two milestones (Option B):**
- **v0.2 picks up the cheap interactive wins** alongside sandboxed code-gen: live progress display, streaming LLM responses, graceful Ctrl-C. About a day of extra work; fits inside the v0.2 window.
- **v0.3 is a new milestone for the real interactivity:** pause via Esc, mid-run chat with the LLM, resume. The hard engineering (async input, in-flight cancellation, conversational state) gets its own focused milestone.
- Everything that was v0.3+ shifts by one version. The Streamlit UI becomes v0.10 (its main "interactive interface" value is covered by v0.3's CLI; v0.10 carries demos, multi-backend benchmark, polish). Build is now ~14 weeks (was ~13).

**Updated 2026-06-01 — going multi-agent after v0.2 (at the Researcher milestone):**
- v0.1 and v0.2 stay **single-agent** (one Proposer LLM in a deterministic loop). The architecture moves to **multi-agent at the Researcher milestone (v0.4)** — the natural single-to-multi transition, where the second genuine LLM role appears.
- Shape: **specialist agents** (Researcher, Proposer, a Critic/Reviewer, later Discovery), each doing one focused job and handing **structured, typed output** to a **supervisor agent** that makes the decisions. Rationale: specialization raises per-agent tool-call reliability, and the supervisor reasons over digested high-quality context instead of raw everything. Our Pydantic schemas are the handoff contracts.
- Built on our own harness (no LangGraph — Week-1 decision stands). Executor + Memory stay deterministic; the supervisor takes the judgment calls. See DECISIONS.md.

- **Targets:** `ModelTarget` (tabular ML) · `PromptTarget` (production LLM prompts, prompt-iteration only) · `DLModelTarget` (vision, transfer learning — validated on local RTX 4050).
- **Moat — the specialized combination, not one feature:** a domain specialist for ML/DL/prompt iteration that does *together* what no single tool does — agentic iteration across **ML + DL models AND LLM prompts** · **persistent memory** (revisits past failures when conditions change) · **literature-grounded** proposals · **bounded autonomy** + human-approval gates · **auditable reasoning trail** · **cost-constrained optimization** (best score you can *afford to serve* — cheapest cloud, $/mo, req/hr) · **rich auto-discovered context** (DB / MCP / Drive). Cost-aware serving is the flagship for cost-sensitive startups; the moat is the *combination* + the specialization. (Full matrix: README comparison table.)
- **Compute:** pluggable backend — local MPS · RTX 4050 (GPU validation) · e2b · cloud-GPU adapter.

| Wk | Phase |
|---|---|
| 1 | Foundation — schemas + LLM client (tool-calling) + config + CLI · done |
| 2 | Tabular execution substrate — `BenchmarkTarget` + data adapter + `ModelTarget` + model factory (any installed sklearn/XGBoost/LightGBM estimator) + local executor |
| 3 | **The agentic loop** — Proposer + Orchestrator + Terminator + Memory → first autonomous tabular run (**v0.1**) |
| 4–5 | **Sandboxed code-gen** (Proposer writes training code, runs it in e2b → any model at all) **+ cheap interactive wins** (live progress display, streaming LLM responses, graceful Ctrl-C) → **v0.2** |
| 6 | **Full interactive CLI** — pause via Esc, mid-run chat with the LLM, resume → **v0.3** |
| 7 | **Multi-agent split** (Researcher + Proposer + Critic specialists → supervisor) + Dial A: agent picks the metric + starting model from research → **v0.4** |
| 8 | Dial B: `PromptTarget` — agentic prompt iteration → **v0.5** |
| 9 | Dial B: `DLModelTarget` — vision transfer learning (4050) → **v0.6** |
| 10 | Cost-constrained recommendation + serving profile + `iterate cost` → **v0.7** |
| 11 | Dial A: infer features/target from the data + a description → **v0.8** |
| 12 | Dial A: MCP discovery — find the data/code itself → **v0.9** |
| 13 | Multi-backend benchmark + Streamlit UI + demos → **v0.10** |
| 14 | Full minimum-viable-input + polish + launch → **v1.0** |

### Releases (incremental — ship a working slice, then iterate)

Semantic versioning: `0.x` = early/evolving, `1.0.0` = the full v1 vision. **The agentic loop is present from v0.1**; two dials then turn — inputs you must give *shrink*, problem types *grow*. Tag a GitHub release at each milestone; publish to PyPI from v0.1.0.

| Version | After | Problem types | Inputs you give (shrinking →) / New capability |
|---|---|---|---|
| v0.1.0 | Week 3 · **RELEASED 2026-05-31** | tabular | data + features + target + metric + baseline/notebook + deadline — **agentic loop on** (any installed-library model via the factory; best model saved as a joblib artifact) |
| v0.2.0 | Week 4–5 | tabular | *(same inputs)* — agent **writes & runs training code in a sandbox** → any model at all + **live progress / streaming / graceful Ctrl-C** |
| v0.3.0 | Week 6 | tabular | *(same inputs)* — **full interactive CLI**: pause the loop, chat with the LLM, resume |
| v0.4.0 | Week 7 | tabular | data + features + target + baseline + deadline  *(multi-agent: Researcher + Proposer + Critic specialists report to a supervisor; agent picks metric + starting model from research)* |
| v0.5.0 | Week 8 | + prompts | prompt + eval set + deadline |
| v0.6.0 | Week 9 | + DL / vision | data + target + deadline |
| v0.7.0 | Week 10 | all three | + serving budget / cloud  *(cost-constrained recommendation)* |
| v0.8.0 | Week 11 | all | data + a one-line description  *(infers features/target/metric)* |
| v0.9.0 | Week 12 | all | one sentence + a data source  *(MCP finds the data/code)* |
| v0.10.0 | Week 13 | all | + multi-backend benchmark + Streamlit UI |
| v1.0.0 | Week 14 | all | one sentence  *(full discovery)* |

---

## Format per session entry

```markdown
### YYYY-MM-DD | Phase N | Session summary

**Task:** [what you set out to do today]

**What shipped:**
- Files: src/iterate/foo.py, tests/unit/test_foo.py
- Commits: <sha>
- Behavior: [what now works that didn't before]

**What didn't:**
- [honest list of what got punted, broken, or harder than expected]

**Decisions:**
- [any architectural choice made + why — link to RESEARCH_LOG entry if applicable]

**Next session:**
- [what's queued for tomorrow]
```

---

## Week 1 Day-by-Day Plan

Realistic per-session scope (3 hours focused). One real commit per day.

| Day | Date | Focus | Lands |
|---|---|---|---|
| **1** | 2026-05-24 (Sun) | Pre-flight verification + Pydantic schemas | `src/iterate/schemas/experiment.py` (Experiment, ExperimentResult, Metrics, FailureCase, Candidate) + `tests/unit/test_schemas.py` |
| **2** | 2026-05-25 (Mon) | LLMClient protocol + OpenAICompatibleClient against Ollama | `src/iterate/llm/base.py` + `src/iterate/llm/openai_compatible.py` + smoke test that actually calls qwen2.5-coder:14b |
| **3** | 2026-05-26 (Tue) | CLI scaffold + config loader | `src/iterate/cli.py` (typer app, `iterate --help` works) + `src/iterate/config.py` (loads .env, validates) |
| **4** | 2026-05-27 (Wed) | First tool definition + tool dispatcher (just a stub — real ones land Week 2-4) | `src/iterate/tools/base.py` + a sandbox-stub tool to prove the loop |
| **5** | 2026-05-28 (Thu) | Anthropic adapter (the one non-OpenAI-compatible backend) — optional via `iterate[anthropic]` | `src/iterate/llm/anthropic_client.py` + parity tests |
| **6** | 2026-05-29 (Fri) | Memory store skeleton — sqlite + retrieval API (real population happens Week 4) | `src/iterate/core/memory.py` + `tests/unit/test_memory.py` |
| **7** | 2026-05-30 (Sat) | Polish + smoke test the full Week 1 stack: config loads → llm client connects → tool dispatcher routes → memory writes | Wk1 retrospective entry in BUILD_LOG |

**Slack day:** Sunday May 31 (rest, or catch up on anything that slipped).

**Note (2026-05-25):** Week 1's foundation — schemas + LLM client + config + CLI — shipped in **Days 1–3** (ahead of plan). The original Days 4–7 (tool dispatcher, Anthropic adapter, memory skeleton) were superseded by the expanded 11-week plan: memory + proposer + researcher → **Week 7**; tool dispatcher → **Week 7** (orchestrator); Anthropic adapter → optional/later. Week 1 is effectively complete; next is Week 2.

### Daily session shape

```
[20 min]  Read BUILD_LOG → pick today's task
[30 min]  Research via Claude chat — papers, libraries, alternatives
[10 min]  Log decision in RESEARCH_LOG (your words)
[90 min]  Write code (you decide what + why + critique/suggest/review; Claude writes)
[20 min]  Tests + verify
[10 min]  Commit (your own message) + push + PR + merge
[10 min]  Update BUILD_LOG: move task to Done with brief note
```

Total: ~3 hrs. If a session needs more, the task was too big — split it.

---

## Week 1 Backlog (P0 = blocking, ordered)

| # | Task | Files | Done? |
|---|------|-------|------|
| 1 | Project metadata: `pyproject.toml`, deps pinned, ruff + mypy config | `pyproject.toml` | done |
| 2 | `.env.example` with Ollama default + optional cloud backend keys (Groq/Together/Deepseek/Anthropic/OpenAI) + e2b + Kaggle | `.env.example` | done |
| 3 | Empty `src/iterate/` package skeleton (folders + `__init__.py`) | `src/iterate/**/` | done |
| 4 | Pydantic schemas — `Experiment`, `ExperimentResult`, `Metrics`, `FailureCase`, `Candidate` | `src/iterate/schemas/experiment.py` | done |
| 5 | `LLMClient` protocol — what every backend implements | `src/iterate/llm/base.py` | done |
| 6 | `OpenAICompatibleClient` — first real working LLM call (default: Ollama localhost:11434 + qwen2.5-coder:14b) | `src/iterate/llm/openai_compatible.py` | done |
| 7 | Smoke test — Ollama call end-to-end (plain chat + structured tool-calling, validated on qwen3:14b) | `tests/unit/test_openai_compatible.py` | done |
| 8 | CLI scaffold — working command skeleton (`iterate --help` · `version` · `config`); fixed typer single-command collapse | `src/iterate/cli.py` + `tests/unit/test_cli.py` | done |
| 9 | First commit message convention doc (semantic commits) | `BUILD_LOG.md` (this section) | done |
| 10 | Central config (pulled fwd from Day 3) — all defaults in one place, env/secret override | `src/iterate/config.py` | done |
| 11 | LLM contracts — `Message`/`ToolSpec`/`ToolCall`/`Usage`/`ChatResponse` | `src/iterate/schemas/llm.py` | done |

---

## Week 2 Day-by-Day Plan — Tabular execution substrate

**Week goal:** the machinery to run one tabular experiment — load data, apply a candidate's changes, train, score on a holdout, return an `ExperimentResult`. Proven with a *supplied* candidate (the agent that proposes candidates is Week 3).
**Target window:** ~Jun 1–7 (running ahead of plan — log by real date).

| Day | Focus | Lands | Done? |
|---|---|---|---|
| 1 | `BenchmarkTarget` protocol — the contract every target implements (`baseline()` + `run(candidate)` → `ExperimentResult`) | `src/iterate/targets/base.py` + tests | done |
| 2 | Tabular data adapter — load CSV, deterministic stratified split, content-hash | `src/iterate/adapters/data/tabular.py` + tests | done |
| 3 | `ModelTarget` (sklearn baseline) — wraps dataset + model + metric; `baseline()` train + score → `Metrics` | `src/iterate/targets/model.py` + tests | done |
| 4 | Model factory — build any allow-listed installed estimator (sklearn/XGBoost/LightGBM) from a `{"model","params"}` spec in `Candidate.changes` | `src/iterate/adapters/models/registry.py` + tests | done |
| 5 | Local executor — run one experiment (baseline or candidate), time it, and **capture failures** so a bad candidate can't crash the loop | `src/iterate/adapters/compute/local.py` + tests | done |
| 6 | Substrate end-to-end on churn — `baseline()` + `run(supplied candidate)` through the executor on real data (not yet agent-driven) | `examples/churn_tabular/` + integration test | done |
| 7 | Polish + Week 2 retro (BUILD_LOG) | wrap-up | done |

**Slack:** 1 day.

---

## Week 3 Day-by-Day Plan — The agentic loop (→ v0.1 agentic tabular)

**Week goal:** close the loop. The LLM autonomously proposes the next candidate, runs it on the Week-2 substrate, scores it, records it, and decides whether to continue — until a deadline / plateau. The first fully autonomous tabular run. Inputs still explicit (data, features, target, metric, baseline/notebook, deadline).
**Target window:** ~Jun 8–14 (log by real date).

**v0.1 contract (agreed 2026-05-27):**
- **Inputs:** `--data` + `--target` + `--metric` (required) · `--baseline` + `--source` notebook/md/txt (optional; `--baseline` requires `--source`; source is **read as text** by the LLM and rebuilt as a runnable spec we execute through our own eval — the user's actual code is **never executed**) · `--backend ollama|openai-compatible` + `--model` / `--api-key` / `--base-url` · `--max-iterations` / `--patience` / `--until` (deadline) · `--fresh` (archive existing memory, start a new chapter with factory-default baseline) · `--memory PATH` (override db path). Features auto-derive as all columns except target.
- **Baseline precedence** (first match wins): `--source` (reconstructed via LLM) → memory's prior best for this target (re-measured) → factory default. `--fresh` and any explicit baseline signal (`--source`, `--baseline + --source`) archive the existing memory db to `<name>.YYYYMMDD-HHMMSS.bak` before starting.
- **Working:** measure baseline via the precedence above → loop { propose → train → score on the sealed holdout → record → decide } until terminator (max_iterations / patience / plateau / deadline) fires. v0.1 candidate space = any installed allow-listed estimator (sklearn / XGBoost / LightGBM) via the model factory, named by `{"model","params"}`.
- **Output:** the **best model** (artifact + the winning config) + its score vs the **baseline we measured** + an **auditable report** of every experiment and why the winner won. Agent proposes; human reviews. All experiments + proposer failures persist in sqlite Memory across runs.
- **NOT in v0.1:** arbitrary/uninstalled models via sandboxed code-gen (v0.2) · live progress / streaming / Ctrl-C (v0.2) · interactive mid-run chat (v0.3) · the agent picking the metric (v0.4) · cost-constraint / serving profile (v0.7) · **executing user-provided source code — ever** (permanent security policy; the e2b sandbox at v0.2 runs the agent's OWN generated code, never the user's).

| Day | Focus | Lands | Done? |
|---|---|---|---|
| 1 | Proposer (+ native `OllamaClient` adapter for `think:false` + centralized `prompts.yaml`) — LLM proposes the next `Candidate` via a `propose_candidate` tool call from data summary + baseline + history | `core/proposer.py` + `llm/ollama_client.py` + `prompts/` + tests | done |
| 2 | Orchestrator — the loop: `baseline()` → propose → `run()` → score → record → decide → repeat (in-memory history; internal stop logic) | `src/iterate/core/orchestrator.py` + tests | done |
| 3 | Terminator — stop on deadline / patience / plateau via a delegated protocol; Orchestrator refactored to delegate | `src/iterate/core/terminator.py` + tests | done |
| 4 | Memory — record every experiment; feed **cross-run** history to the Proposer (sqlite + in-memory; structured proposer-failure records) | `src/iterate/core/memory.py` + tests | done |
| 5 | CLI `iterate run` (+ `--backend` factory, baseline precedence, `--fresh` archive) **and source-aware baseline reconstruction** — LLM reads `--source` md/txt/notebook as **text only** (never executes), rebuilds the approach as a spec, we run it through our eval → re-measured baseline | `src/iterate/cli.py` + `core/reconstructor.py` + `llm/factory.py` + tests | done |
| 6 | First autonomous tabular run on churn — reproducible committed demo (`prepare.py` + `iterate run`), verbosity suppression, proposer-yield polish, live agentic integration test | `examples/churn_tabular/` + integration tests | done |
| 7 | Polish + Week 3 retro + release **v0.1.0** (model persistence, dep trim, LICENSE, README reconcile, tag) | wrap-up | done |

**Slack:** 1 day.

---

## Week 4 Day-by-Day Plan — Sandboxed code-gen (→ v0.2)

**Week goal:** lift the model ceiling. v0.1 can only run allow-listed installed estimators via the `{"model","params"}` factory. v0.2 lets the agent **write its own training code** and run it in an **e2b sandbox**, so it can use any model the research points to (CatBoost, a custom net, a stacking pipeline, a library we never installed). It also ships a **notebook deliverable**: the winning experiment exported as a runnable, annotated `.ipynb` (works for a spec winner or a generated-code winner). Plus the cheap interactive wins (live progress, streaming, graceful Ctrl-C). The biggest single capability jump in the roadmap, hence a full day-by-day. Stays single-agent (multi-agent lands at v0.4).
**Target window:** ~Jun 1–10 (flows into early Week 5; log by real date). Days 1–2 done.

**Hard boundaries (locked):** we run **the agent's OWN generated code, in the sandbox, never the user's code** (the permanent security policy). The sealed-holdout principle holds: a generated script trains on train data only and is scored through **our** eval on the holdout it never sees.

**Design forks to settle on Day 1 (my recommendation in parens):**
- **e2b access + local option:** need an `E2B_API_KEY` (paid cloud sandbox, free tier exists; now in the `[sandbox]` extra). (Build behind the `ComputeBackend` protocol with a **local executor** that runs generated code on the user's machine. The local executor is both the keyless dev/test path AND a **user-facing backend** via `--compute local`. e2b is the **safe default** (isolated, contained blast radius for autonomously-generated code); local is an **explicit opt-in with a clear warning** (free, offline, uses the user's own GPU, but generated code runs with the user's permissions). Fits the existing "pluggable compute backend" vision.)
- **Code-gen vs spec coexistence:** (a new **code-candidate** type alongside the v0.1 `{"model","params"}` spec; the Proposer picks the spec path for installed models and the code path for anything beyond the three libraries. Keep the cheap reliable spec path; code-gen is the escape hatch, not a replacement.)

| Day | Focus | Lands | Done? |
|---|---|---|---|
| 1 | **`ComputeBackend` protocol** — extract the execution seam from `LocalExecutor` (it conforms; Orchestrator depends on the protocol); `SandboxExecutor` stub raising NotImplementedError. Settle the two design forks; RESEARCH_LOG entry on the code-gen contract + sandbox choice | `adapters/compute/base.py` + tests | done |
| 2 | **Code runner primitive** — `CodeRunner` protocol + `LocalCodeRunner` (subprocess) + `E2BCodeRunner` (e2b, lazy-imported, injectable sandbox); run a script with input files under a mandatory timeout, capture stdout/exit/outputs, teardown. The `ComputeBackend.execute` integration lands Day 5 (needs candidates) | `adapters/compute/runner.py` + tests | done |
| 3 | **Code-gen contract** — fill-in-a-function harness (LLM writes `train_and_predict`; we own the I/O); inputs = train + holdout FEATURES + meta (labels held back); script writes `predictions.csv`; we score through the shared `core.scoring`. Code-candidate = `{"code": ...}`. Proven end-to-end through `LocalCodeRunner` with a canned function (no LLM/e2b) | `core/codegen.py` + `core/scoring.py` + tests | done |
| 4 | **CodeProposer** — the LLM writes a training script to the contract (new prompt in `prompts.yaml` + tool). Coexists with the spec Proposer (option a). Conformance checks; failures captured, not crashed | `core/code_proposer.py` + tests | done |
| 5 | **Wire end-to-end + safety** — Orchestrator runs code-candidates through the sandbox executor; resource caps, timeout, no-network default, "own code only" enforced. First real sandboxed run on churn with a non-allow-listed model (e.g. CatBoost) | orchestrator wiring + integration test | done |
| 6 | **Notebook deliverable (B)** — export the winning experiment as a runnable, annotated `.ipynb` (a spec winner rebuilt as cells, or the generated-code winner wrapped with a markdown rationale); execute it to populate outputs (e2b's Jupyter kernel, or papermill/nbconvert on the local path); save next to `best_model.joblib`. The portfolio-worthy "here's exactly what the agent found, runnable" artifact | `deliver/notebook.py` (+ `nbformat`) + tests | done |
| 7 | **Cheap interactive wins** — live progress display (rich `Live`: iteration / model / score / best updating in place), streaming LLM responses (client stream path), graceful Ctrl-C (finish or abort current iteration, persist state, clean exit) | `llm/*` stream methods + CLI live view + tests | done (streaming re-scoped to v0.3, DECISIONS 2026-06-13) |
| 8 | Polish + Week 4 retro + release **v0.2.0** (tag + PyPI) | wrap-up | in progress (expanded: the release was gated on a quality bar, see the 2026-07 entries) |

**Slack:** 1 day (likely needed — sandbox infra + code-gen reliability are the riskiest work so far). v0.2 is now 8 days (added the notebook deliverable), so it runs into early Week 5.

---

## MCP + Discovery Backlog (preview — Week 9 under the agent-first plan)

> **Re-sequenced 2026-05-27 (agent-first):** Proposer (4.10) + Memory (4.11) moved **forward to Week 3** (the core agentic loop); Researcher (4.9) → Week 4 (Dial A). The MCP + discovery items below (4.1–4.8, 4.12–4.13) land at **Week 9** — they're Dial-A input-reduction (toward one-sentence input), *not* prerequisites for the agent.

This phase shifts the agent from "user provides every input" to **"user provides one input — `iterate 'improve our churn baseline'` — and the agent discovers the rest."**

### Autonomous discovery is the single biggest differentiator. It's the demo headline.

| # | Task | Files |
|---|------|-------|
| 4.1 | MCP client — connects to multiple servers via stdio/HTTP | `src/iterate/mcp/client.py` |
| 4.2 | MCP server registry — config-driven lifecycle (spawn/kill/health-check) | `src/iterate/mcp/registry.py` |
| 4.3 | MCP-to-OpenAI tool bridge — translate MCP tool defs to OpenAI tool schemas | `src/iterate/mcp/tool_bridge.py` |
| 4.4 | Wire filesystem MCP server (read local notebooks/docs/logs) | config + docs |
| 4.5 | Wire postgres MCP server (DB introspection + read-only sampling) | config + docs |
| 4.6 | Wire notion MCP server (search past experiment pages, write new ones) | config + docs |
| 4.7 | Wire github MCP server (scan repos for relevance) | config + docs |
| 4.8 | **Discovery agent** — given one-line goal, scans filesystem/GH/DB/Notion, infers baseline + metric + eval method + relevant tables, surfaces summary, pauses for human gap-fill | `src/iterate/core/discovery.py` |
| 4.9 | Researcher (arxiv + papers-with-code) | `src/iterate/core/researcher.py` |
| 4.10 | Proposer — uses memory + discovered context to rank candidates | `src/iterate/core/proposer.py` |
| 4.11 | Memory store integration — every experiment + tool call logged for audit | `src/iterate/core/memory.py` |
| 4.12 | Logging adapter via Notion MCP — write experiment cards to Notion | `src/iterate/adapters/logging/notion.py` |
| 4.13 | Logging adapter for plain markdown (fallback when no Notion) | `src/iterate/adapters/logging/markdown.py` |

### Discovery agent specifics (Task 4.8 — the differentiator)

The discovery agent is what makes the demo wow. It does:

1. Parse the one-line goal into search keywords
2. List candidate repos (filesystem + github MCP) — rank by README keyword match, fall back to recent commit activity
3. Read top 1-3 candidate repos: train scripts, notebooks, requirements.txt, model artifacts
4. Extract current baseline metric (from MLflow runs, W&B, code comments, results JSON)
5. Identify eval methodology (test split definitions, eval scripts)
6. Query Postgres MCP: list_tables, sample, infer relationships to the problem
7. Search Notion MCP: past pages mentioning the project + extract failure reasons
8. Synthesize "what I found" summary
9. Identify gaps ("I couldn't find X")
10. Pause for user input. Commit gap-fill into memory. Then iterate.

---

## UI + Benchmark Backlog (preview — Week 10 under the agent-first plan)

| # | Task | Files |
|---|------|-------|
| 5.1 | Terminator — patience / deadline / compute budget / plateau detection | `src/iterate/core/terminator.py` |
| 5.2 | Reporter — generates run summary + PR-shaped report | `src/iterate/core/reporter.py` |
| 5.3 | **Streamlit chat UI** — sidebar (MCP status + experiments + memory + cost), chat input, live agent reasoning stream | `src/iterate/ui/chat.py` |
| 5.4 | Second example target (intent_clinc150) to prove framework genericity | `examples/intent_clinc150/` |
| 5.5 | Multi-LLM backend benchmark — same task run on Ollama / Groq / Together / Deepseek / Anthropic | `examples/benchmark/` |
| 5.6 | Demo video walking through full discovery → iteration loop | `docs/demo.md` + recording |
| 5.7 | Final README polish, launch post assembly from LAUNCH_POST.md | `README.md`, `LAUNCH_POST.md` |

---

## Done

### 2026-07-07 | Week 4 Day 8 | Quality bar CERTIFIED; release prep (README rewrite, v0.2.0 bump, sdist trim)

**Task:** Close the quality-bar loop with a certified run, then execute the release mechanics.

**Certification (runs 20 + 21, gemma4:12b, churn/f1):** run 20 passed all 7 bar criteria: staged R&D (0 monoliths in 116 cells, first fully-clean run), pickup (digit-identical rebuilds of the carried best), progression (0.5620 -> 0.5997 -> 0.6333, ties the all-time record), failure-knowledge transfer (dead-ends lines rode every brief and were honored), exactly 1 process-failure duplicate (bar: <=2; honest measured nulls excluded), zero FAILED, and every residual flaw named with a reproduced root cause. Run 21 confirmed the three fast-follow fixes with no regression: 2 duplicates, zero FAILED, best 0.6312 via an XGBoost swap, honesty notes rendering in duplicate notebooks. The certified capability floor: 1-2 honestly-labeled duplicate/null iterations per run is the 12B being a 12B; the harness detects it (byte hashes, marker checks), labels it (stamps in memory + notebook headers), and converts it (guard rejections, fallback briefs, floor submissions).

**Release prep shipped:** the quality-bar workstream committed as three logical commits (supervisor grounding + brief guards; coder gates + submission floor + thrash guards; loop verdicts + honest deliverables; 387 unit tests green). README rewritten for v0.2 (multi-agent story, real run transcript, honest shipped-vs-planned tables, architecture diagram matching the actual tree); examples/ READMEs de-ghosted (PromptTarget placeholders labeled as v0.5); LIMITATIONS rows updated (multi-agent core shipped, seeding shipped, capability-floor row added). Version 0.2.0 in pyproject + __init__ + lockfile; PyPI description rewritten to what v0.2 actually does; sdist trimmed from the whole repo (~2MB) to package + docs (120K); make build/publish targets added so the release process stops living in memory.

**Next:** live e2b verification with a real key, the official demo run, merge call, tag + publish, launch post.

### 2026-07-05 | Week 4 Day 8 (cont) | The dedup guard stack: forensics-fix-rerun, runs 8-19

**Task:** Kill the remaining waste class: iterations whose submission is byte-identical to an earlier one.

**The loop that did it (each run's forensics named ONE dominant mechanism; each got a deterministic guard + regression test; then rerun):**
- runs 8-10: identical-submission gate now hashes against EVERY prior submission (sibling duplicates evaded a best-only check); duplicate iterations stamped in history so the scoreboard stops re-crediting orbited levers; baseline re-briefs rejected in code.
- runs 11-13: pre-issue novelty guard (a brief re-commissioning work the carried best already contains: class_weight set, grid searched, feature built, is rejected before dispatch); the "so far:" slot composed by code from the loop's carried best, killing hallucinated facts structurally; recurrence-ranked dead-ends line carries failure knowledge to the coder (blind re-probes of one pet idea: 8/10 notebooks -> ~0).
- runs 14-17: guard precision round: technique-level marker matching (the class NAME contains 'threshold'; only technique mentions count), threshold-retune guard (re-tuning an already-tuned banked threshold deterministically reproduces the incumbent), measured-lost guard (a technique that lost fairly this run cannot be silently re-briefed), move lint (fused lever tags, phantom scores), digest sanitization (a duplicate keeps no what-helped claims; fabricated wins about never-executed levers are machine-stripped).
- runs 18-19: when a guard violation persists through its one corrective retry, the harness now composes a deterministic fallback brief from the first untried lever class (novel by construction; both live firings converted to real measured experiments). Plus the one FAILED iteration in 121: predictions written with an index column (to_csv without index=False) passed the line-count check and died at scoring; the finish gate now catches the format in-session.

**Trajectory across the stack:** wasted iterations 4-6/run -> 1-2; supervisor-fault duplicates -> 0 in the certified run; zero FAILED in 120 of 121 iterations; two runs got their best score directly off a guard forcing a novel lever.

### 2026-07-04 | Week 4 Day 8 (cont) | Quality bar gates the release; supervisor grounding + coder no-op gates + submission floor

**Task:** The user's call (DECISIONS 2026-07-04): the v0.2 release is GATED on a quality bar defined as trajectory, not score: staged R&D notebooks, pickup of the carried best, progression across notebooks, failure-knowledge transfer, no silent process waste. Iterate on the current architecture until it holds; no new roadmap features.

**What shipped (first tranche):**
- **Grounded briefs**: the brief's "so far:" facts are composed by CODE from the recorded history (real best score, its components, the applied decision threshold), never LLM recall. Live forensics had caught the supervisor citing scores that never existed on the holdout.
- **Coder no-op gates at finish time**: the lever gate (briefed change absent from every NEW code line, diff-scoped against the carried code) and the identical gate (submission byte-identical to the carried best), each a one-shot corrective nudge. The lever gate converted ~5/5 live and twice produced the run best.
- **Submission guarantee**: a session that dies without a valid predictions file banks a floor (re-run the carried best, else a canned seeded baseline) as a labeled fallback cell; total-loss iterations (2/10 in the worst pre-fix run) went to zero.
- **Thrash guards**: 6-consecutive-errors breaker + a 30-minute session wall ceiling (kernel-time budgets deliberately do not charge LLM latency, which left a thrashing session unbounded in wall-clock); truncated cells (mid-token cutoffs) rejected unexecuted with a precise retry message.
- **Split-first hygiene in the coder's worked example**: fit-before-split leakage in the coder's own validation split went from 5/5 notebooks to 0 across every run since; the like-for-like rule (same decision threshold on both sides of any comparison) killed the false-kill pattern.

**Also:** the pre-release audit (4-agent workflow over plan/docs/packaging/logs) that scoped the release wrap-up, and the run-forensics workflows (10 parallel notebook readers + adversarial verify) that became the standing certification instrument.

### 2026-06-20 | Week 5 Day 4-5 | Pre-release hardening; the EDA-ledger regression + revert; coder forensics overturn a comfortable assumption

**Task:** Land the last pre-release hardening items, then chase the observed cross-notebook EDA repetition.

**Hardening shipped (commit b9b5943):** session RNG seeded in the preamble (the rendered notebook re-executes to the SAME score the run reported); e2b keepalive; graceful Ctrl-C (finalize memory, keep every earned notebook, no stack trace); per-cell progress line. Plus supervisor backend-error resilience (commit 5e43dcb): a groq tool_use_failed 400 (the model emitting stop as the STRING "false"; bool("false") is True) no longer crashes a run, and the coder prompt stopped re-deriving the host profile (nunique re-derivation 10/10 notebooks -> 0).

**The regression story (kept honest because it is the method):** a 3-part cross-notebook EDA-transfer feature was built, tested, and then REVERTED. Same-model before/after runs showed the additive context (a supervisor status line + an EDA ledger) regressed gemma4:12b: best 0.6325 -> ~0.61 with the supervisor collapsing onto one lever 5-6/10 iterations. More context a frontier model digests, a 12B chokes on; a post-revert run restored lever diversity, confirming causation. Kept: the robustness fix + the prompt de-dup. Lesson banked: deterministic guards over prompt nudges, lean context always.

**Forensics overturned "the coder is fine":** a 10-reader workflow over every notebook of a fresh run showed the CODER was the primary quality-bar blocker on the weak model (2/10 no-submission iterations, a hard-coded threshold carried for 6 iterations, leakage in its own derived split). That verdict re-sequenced everything that followed: coder reliability first, supervisor compounding second.


### 2026-06-10 | Week 5 Day 3 | Supervisor priority ladder + lever ledger + Hypothesis/Findings notebooks → first 10/10-above-0.60 run (documented 2026-06-11)

**Task:** Fix the last diagnosed bottleneck — the SUPERVISOR's strategy (run c7ddda92: 0/10 briefs touched imbalance despite the profile showing F1 + 73/27 from experiment 1; the coder found `class_weight` on its own only at iteration 10, val 0.548→0.619) — and make the cross-notebook knowledge handoff visible in the artifacts themselves.

**What shipped (all on the open PR #43 branch; prompts in their own revertable commits):**
- **Lever ledger** (`core/supervisor.py`): a deterministic "Levers tried: … | Levers NOT yet tried: …" line in the supervisor's context, scanned case-insensitively from every experiment's code across 7 technique lever classes (categorical-encoding, numeric-transform, imbalance-or-threshold, interactions-or-ratios, feature-selection, ensembling, hyperparameter-search). Full-history scan; explicit done/not-done coverage instead of hoping the strategist infers it.
- **Notebook R&D framing** (`deliver/notebook.py`): every session notebook now opens with a `## Hypothesis` markdown cell (the supervisor's brief verbatim — which carries the run's so-far knowledge) and closes with `## Findings` (the Summarizer's digest: what helped, what hurt, data insights, validation trail, takeaway). Each artifact reads hypothesis → staged work → findings, and notebook N's Findings visibly become notebook N+1's Hypothesis. Trophy-emoji best-title replaced with "best:" (house style).
- **Supervisor prompt rewrite** (16-agent research forge: AIDE/MLE-bench/DS-Agent strategist policies, planner mode-collapse literature, cross-episode memory formats, plus forensics on the actual run notebooks; writeup in RESEARCH_LOG 2026-06-09): a 5-rung **priority ladder** walked each turn — BASELINE (plain one-hot + median-impute + HistGradientBoosting, nothing else) → **METRIC LEVER** (imbalance-sensitive metric + minority class under ~40% + lever untried → brief `class_weight=balanced` NOW; threshold tuning is the rung's one allowed second firing) → REPAIR (once per idea) → UNTRIED CLASS (from the ledger) → REFINE BEST (model swaps only here). A **PIVOT rule outranks every rung** (two consecutive non-improving briefs on one lever class forbid a third). **COMPOUND**: every brief keeps the best configuration and adds exactly one named change. Two-slot brief format ("so far:" / "next:" naming technique + lever class + one profile-fact reason). Verified on the installed sklearn that HistGradientBoosting accepts `class_weight='balanced'`, so the experiment-2 brief is executable as written.
- **Input-reset made visible at recovery time** (after the validation run): session rules now say a column added to the canonical frames is GONE next cell while every variable the model creates persists — "engineer features freely, into your OWN derived frames" — and the same-error nudge gained the missing-column recovery recipe. Design principle stated: **open workspace, guarded boundary** — restrictions only at the validation boundary (sealed labels, verified submission, pristine canonical frames); everything inside the workspace is open.

**Validation run (5d56268c, gemma4:12b, churn/f1) — the best run on every tracked axis:**
- **10/10 experiments ≥ 0.60 (first time ever)**; best **0.6325**, mean 0.6215 — essentially the all-time score (0.6353, from the unreadable monolith era) with clean staged R&D notebooks (0/177 monolithic cells).
- The ladder worked as designed: experiment 1 = plain baseline (0.6118, strongest opener yet); **experiment 2 briefed class_weight** (vs iteration 10 by accident in the prior run); then numeric-transform → hyperparameter tuning → feature selection → ensemble, with a pivot to feature-selection finding the run best. Hypothesis/Findings cells rendered in all 10 notebooks with accurate, specific content.
- One repeated failure inside iteration 2 (16× the identical missing-column error): a hand-built feature baked into a fitted imputer, then the model's in-place patch on `X_holdout` silently undone by the input-protection reset each cell — the guard was right, its invisibility was the bug. Fixed same day (the recovery-recipe nudge above); the session still recovered and banked a score.

**Decisions (user, logged in DECISIONS.md):** open-workspace/guarded-boundary; the reset stays (fix visibility, not the guarantee).

**Next session:** docs (this entry), then the merge call on PR #43 — from the build side the branch is merge-ready: the stated quality bar (R&D notebooks + knowledge compounding between them) is demonstrably met.

### 2026-06-09 | Week 5 Day 2 | Summarizer agent + cross-notebook knowledge transfer; PR #43 opens; submission-first fix; supervisor diagnosed

**Task:** Build the knowledge-transfer layer the user scoped ("each notebook's summary passed to the next: what was tried, what the data showed, what worked, what didn't — without ever feeding whole notebooks as context"), land everything on a reviewable PR, and keep iterating on live-run evidence.

**What shipped:**
- **Summarizer** (`core/summarizer.py`) — the fourth LLM role, the v0.4 specialist pulled forward into v0.2 (user call): runs ONCE per finished experiment, reads that one session's cells + printed outputs, and produces an `ExperimentDigest` (new schema: techniques, data_insights, what_helped, what_hurt, score, val_trail, takeaway). Deterministic skeleton (components, score, validation trail) always filled by code; the LLM adds the insight fields; **never raises** — any backend failure degrades to the skeleton so a digest can't cost the run. Digests persist on `Experiment` through Memory (SQLite round-trip tested).
- **Supervisor consumption**: history now renders each experiment's digest (data / helped / hurt / next-idea) plus a deterministic **technique scoreboard** (best score whenever each technique appeared). The coder still sees only the brief + best code — digests stay out of the expensive per-cell loop, so context cannot bloat by iteration 5 (~150 tokens per digest). Flow: notebook → Summarizer (once) → digest in Memory → Supervisor reads all digests → brief → next coder.
- **PR #43 opened and deliberately NOT merged** (user call): `main` stays the known-good baseline for trivial comparison/revert; every prompt rewrite is isolated in its own commit so `git revert <sha>` drops a prompt alone and keeps the infrastructure.
- **Submission-first fix** (run 2a486f41 exposed it: 3/10 notebooks fit a model, printed a validation score, and stopped — zero errors, no predictions file): the staged MODEL→SUBMIT split let a weak model treat the validation score as the finish line. The coder prompt now makes the WRITTEN predictions file the first milestone ("a run that prints a score but never writes predictions has produced nothing and fails"), the finish-rejection nudge gained a concrete recovery recipe, and hand-built features must be computed identically on both frames via one shared function.
- **Supervisor diagnosed as the next bottleneck** (run c7ddda92, the first fully clean run: 10/10 submitted, staging held, scores 0.55–0.59): digests were accurate, the coder executed faithfully, but the strategist never briefed the metric-appropriate lever (imbalance) in 10 tries and orbited model swaps — including its own evidence (it10's digest: "class_weight: 0.548→0.619") arriving too late to compound. The fix became Day 3.

**Tests:** 300+ unit tests at each step (summarizer merge/fallback/no-raise, digest round-trip, scoreboard rendering, the validate-never-submit regression suite); ruff + mypy --strict clean throughout.

**Next session:** rewrite the supervisor (research-grounded), add explicit lever coverage, make the handoff visible in the notebooks.

### 2026-06-08 | Week 5 Day 1 | gemma4:12b + thinking experiments; research-grounded coder prompt; R&D staging locked as the bar

**Task:** Break the staging stalemate ("model-bound" per 2026-06-07) by changing the experiment variables — the floor model (user call: gemma4:12b deserved a chance) and thinking mode (user hypothesis: a planning scratchpad might buy staging discipline) — then attack the prompt with research instead of in-session tuning.

**What shipped + findings (each from a live run):**
- **Model change validated** (user's call): gemma4:12b with the unchanged harness ran 10/10 experiments in the 0.60–0.62 band (best 0.6200) — the first all-iterations-above-baseline run. Same prompt that qwen kept failing: the floor model is a choice, not a constant.
- **Thinking mode refuted as the staging lever**: a full-think run failed every supervisor turn (the thinking trace crowds out the single `plan_next` tool call — "no plan after 2 attempts" on every iteration), so `--think` became **coder-only** (supervisor + summarizer always no-think; two clients on the code path). The clean A/B then showed thinking made cells MORE monolithic (plans in-head, dumps the finished plan as one cell — up to 181 lines) and dropped the floor (5/10 < 0.60). Thinking stays available, off by default.
- **Thinking-trace capture**: `ChatResponse.thinking` → `Cell.thinking` → a "Model reasoning" markdown block above each code cell in the notebook — the verbatim record of what the prompt made the model think, kept as a prompt-debugging instrument.
- **Incremental notebook saves**: each finished iteration's notebook (and best-so-far `best.ipynb`) is written the moment it completes via an `on_experiment` hook, so a crash or Ctrl-C mid-run leaves every finished deliverable on disk.
- **Research-grounded coder prompt** (20-agent forge: web research on agentic/code-gen prompting, expert Kaggle R&D workflow, ReAct/CodeAct stepwise agents, small-output elicitation — plus our own failure data; writeup in RESEARCH_LOG 2026-06-08): the centerpiece is an 8-cell worked example (A–H) whose shape IS the unit of work — a weak model imitates one example over many rules. One-action-per-cell, dtype-based column selection, starting points are reference-only (rebuild, never paste), and no fabricated harness enforcement (the synthesizer caught two drafts bluffing "is rejected before it runs," read `coder.py`, and replaced the lie with real budget economics). The worked example was verified to run end-to-end on the actual churn data.
- **Run with the new prompt (c9bc0764): staging fixed decisively** — monolithic full-pipeline cells 31–35% → <1% (1/137), biggest cell 181 → 40 lines, every iteration submitted — at a score cost (best 0.59 vs the monolith era's 0.6353).
- **Decision (user, the week's pivotal call): R&D staging is LOCKED — the score delta is acceptable.** Proper research-style notebooks are the product bar, not just the number ("if the code being written is in proper R&D style that's better"). This overturned the 06-07 "staging is model-bound" conclusion: it was model-bound *for prose rules*; a worked example the model imitates beat it. The new bottleneck named the same day: cross-notebook knowledge transfer — winning techniques weren't compounding across experiments (TargetEncoder discovered in one notebook, dropped by the next).

**Next session:** the Summarizer + digest layer (knowledge transfer), on a PR.

### 2026-06-07 | Week 4 Day 7 | v0.2 multi-agent cell-by-cell system + reliability hardening (coder prompt still pending the quality bar)

**Task:** Build the v0.2 core decided on Day 6 — a two-agent, cell-by-cell system — and harden it against what live qwen3:14b runs surface, iterating on the real churn dataset (target `Churn`, metric `f1`) until a run works cleanly every time. Spanned several days of live iteration (2026-06-05 to 07); logged as one entry for one commit.

**Status — NOT a release.** The harness, architecture, and reliability work below are settled and tested. The **coder system prompt is still pending**: it is being authored separately to reach a quality bar (consistent f1 on the floor model), so the prompt wording in `prompts.yaml` `coder:` is **provisional** and will be replaced. v0.2 does not ship until that prompt clears the bar. See "What's pending" below.

**What shipped — the multi-agent cell-by-cell system (new):**
- `src/iterate/adapters/compute/kernel.py` — `StatefulKernel` protocol + two implementations. `LocalKernel` boots a real IPython kernel (`jupyter_client`), runs a cell, captures stream/execute_result/display_data/error as nbformat-ready output dicts, interrupts on timeout, and never raises on a failing cell (errors are feedback). `E2BKernel` reuses one e2b sandbox across cells for the same state-persistence. Both expose `start(inputs)`, `run_cell`, `install`, `namespace_summary`, `read_output`, `close`.
- `src/iterate/core/coder.py` — `CodingAgent`: drives ONE experiment as a live kernel session (write a cell → see its real output + the live variable list → write the next), ending on a VERIFIED finish tool that only accepts when valid predictions exist. Holdout labels never enter the kernel; predictions are scored host-side, so the sealed-holdout guarantee is unchanged.
- `src/iterate/core/supervisor.py` — `Supervisor`: the across-experiments strategist. Reads run history, compresses it, and hands the coder a brief; decides stop. One LLM via a `plan_next` tool (the tool boundary is where the v0.4 specialists graduate).
- `src/iterate/core/agent_loop.py` — `run_supervised`: the supervised loop (Supervisor briefs → Coder runs a session → scored result → Memory), returning the same `RunResult` so the CLI treats both paths uniformly. CLI `--code` now routes here (the one-shot path stays under `--spec`).
- Same-model-different-roles is legitimately multi-agent: roles, prompts, tools, and isolated contexts distinguish the agents; the backend model identity does not.

**What shipped — reliability hardening (each fix traced to a real failure on a live run):**
- **num_ctx fix (the big one).** `OllamaClient` never set `num_ctx`, so Ollama ran qwen at its 4096 default and silently FRONT-truncated the growing session — dropping the system prompt + tool schema mid-run (confirmed in the server log: `truncating input prompt limit=4096 keep=4`). Now pinned (default 16384, env-overridable) plus a prompt-side `context_budget` that elides the OLDEST observations first so the system prompt is never what truncates. The full-context design only actually reached the model after this.
- **Auto-install fixed for uv venvs.** `python -m pip` fails in uv venvs (no pip); install fell through silently and the agent looped on an import that could never resolve. Now falls back `pip` → `uv pip --python <kernel>` → `ensurepip`, and the outcome is made VISIBLE to the agent (installed-and-re-ran, or FAILED-so-switch-libraries) instead of a silent no-op.
- **Deadline charges KERNEL-execution seconds only**, not LLM latency — a slow local model gets the same working budget as a fast cloud one (`--until` now bounds the whole run via the terminator, not a single experiment).
- **Verified finish + improve nudge:** a session cannot end on a hallucinated "done"; a first valid finish with most of the budget unspent is met once with a nudge to make one more measured improvement.
- **Repeated-cell breaker** (refuses an identical re-submitted cell) and **same-error breaker** (escalates when one error signature recurs across cosmetically-different cells, naming the cause and forbidding cosmetic retries) — both kill the perseveration loops a 14B falls into.
- **`finish()` shim** in the trusted preamble: the conflated `finish()`-as-code call prints guidance instead of NameError-ing an otherwise-good cell.
- **Input protection:** the preamble snapshots `X_train`/`y_train`/`X_holdout` and the harness restores them before every agent cell, so in-place mutation in one cell cannot poison later attempts.
- **Crash containment:** one coder session raising (backend timeout, kernel death) is recorded as a failed iteration and the run survives, instead of taking down the whole loop. Ollama client timeout raised to 600s (local prefill is genuinely slow on a long session).
- Actual-run notebooks: the kernel's captured outputs are attached to the notebook cells (`build_session_notebook`), so the deliverable shows real execution results, not synthesized ones.

**What shipped — cross-experiment knowledge transfer (first leg, v0.2):**
- **Host-computed data profile** in `summarize_dataset` — cardinalities, missing counts, skew, class balance, and top numeric-target correlations, computed once from the training split and handed to BOTH the supervisor and every coder session. Established facts no session has to re-derive.
- **Within-session validation trail** in the supervisor's history view — `(val tries: 0.58 -> 0.61 -> 0.59)` per experiment, so attempts that LOST inside a session inform the next brief, not just the final score.

**Empirical findings (live churn / f1 runs, qwen3:14b):**
- Best clean run reached **f1 0.6353, 5/5 experiments succeeding** with the harness fixes + monolithic cells (a new local-qwen high; baseline 0.5676). The harness lifts the floor model on SCORE and RELIABILITY.
- **Staged-cells-vs-monolithic-script is MODEL-bound, not harness-bound** (RESEARCH_LOG 2026-06-07). A 14B defaults to writing a complete script and reverts to one big cell whenever handed a working blob to edit (every improve iteration); prompt wording reliably stages only the from-scratch iteration. Forcing staging on the floor model regressed reliability (0.5813, 2/5). Conclusion: lift the floor model on score/reliability via the harness; if staged R&D *notebooks* are wanted, do it at the deliverable layer, not by constraining a weak driver. The coder prompt's cell-structure target is therefore being settled out-of-band (see status).

**What's pending (before v0.2 release):**
- **The coder system prompt** — authored separately to reach the quality bar; the in-tree wording is provisional and will be replaced. This is the gating item.
- Seed the code-path RNG for run-to-run reproducibility (still carried from Day 6).
- The carry-forward (`_winning_code`) hands the next experiment a concatenated blob; if the finalized prompt assumes staged cells, revisit this.
- Live e2b verification of the cell-by-cell path with a real key; one clean demo run; version bump to 0.2.0; publish.

**Tests:** 282 unit tests; ruff + mypy --strict clean (43 src files). New suites: `test_kernel.py` (real-kernel state/error/timeout/outputs/namespace), `test_coder.py` (end-to-end through a real `LocalKernel` + real scoring with a scripted fake LLM; verified-finish, auto-install, breakers, input-reset, deadline accounting), `test_supervisor.py`, `test_agent_loop.py` (carry-forward, crash containment, history dedupe).

**Decisions (yours, logged in DECISIONS.md):** cells always on (no flag); supervisor + coder both land in v0.2 (coder-first); no per-cell cap (time/turns are the bound); full context to the coder; deadline charges kernel time not LLM latency; protect the canonical inputs in the harness; the coder prompt's writing-style target is model-bound and owned out-of-band.

**Next session:** integrate the finalized coder prompt when it arrives (preserving the placeholder contract + reliability guardrails), then the v0.2 release wrap-up (seed fix, live e2b, demo run, version bump, publish).

### 2026-06-04 | Week 4 Day 6 | Notebook deliverable + code-path hardening + prompt-vs-model

**Task:** Ship the human deliverable (a runnable notebook), then harden the code path against what live runs surfaced, and settle empirically what actually limits exploration depth. Run on the real churn dataset throughout, which is how the bugs + findings came out.

**What shipped:**
- `src/iterate/deliver/notebook.py` — `build_notebook(experiment, …)` renders one experiment to a schema-valid `.ipynb` (via `nbformat`): markdown header (approach, score, Δ vs baseline, rationale), a load-data cell, the experiment's actual code (`train_and_predict` for code candidates; a `ModelTarget` rebuild for spec candidates), and a score cell. Cells load + score through iterate's own `load_csv` / `core.scoring`, so the notebook reproduces the *exact* reported number, not a lookalike (faithfulness over self-containment, on purpose). `save_notebook` + `slug` helpers.
- CLI `--notebooks best|all|none` (default `best`): `best` writes `<run_dir>/best.ipynb`; `all` writes one notebook per experiment under `<run_dir>/notebooks/` plus the winner (the full journey); `none` skips. Code-gen winners now ship `best.ipynb` as the runnable artifact (a code-gen winner returns predictions, not a pickle — by design), dropping the bare `.py`; spec winners still pickle.
- `nbformat` added to core deps (the notebook is a headline v0.2 deliverable). One localized mypy override for the renderer (nbformat ships no stubs).
- Clarified split (yours): the **digest** (a compressed insight for the LLM's next iteration) and the **notebook** (full, human-facing) are different things and coexist. The backend already captures the whole experiment after every result (Memory); the digest is a v0.4 summarizer add, never the stored record.
- Tests: code + spec notebooks are schema-valid and contain the winning code/rebuild + the score; failed experiments note the failure; `best`/`all` emit the right files; slug is filesystem-safe. **Integration (run locally, green): a rendered notebook executes top to bottom through a real Jupyter kernel and prints the score** — proves it's genuinely runnable.
- 223 unit tests (+7); ruff + mypy --strict clean (39 src files).

**Hardening + improvements (from live runs on the churn dataset):**
- **Components-digest in proposer history (deterministic, no LLM).** `codegen.components_used` extracts the class-like components each past attempt actually instantiated (`SimpleImputer`, `OneHotEncoder`, `HistGradientBoosting…`), and the code proposer's history now shows `[used: …]` per attempt. Root-cause fix: before this the proposer only saw a one-line description + score, so it kept repeating the same impute+one-hot and only swapped the model. (The richer LLM-summary version is the v0.4 "A".)
- **Feature-engineering-first prompt.** Reframed the code-proposer prompt so feature engineering is the *main* lever (concrete menu: target/ordinal/frequency encoding, numeric transforms, interactions, aggregations, feature selection, class-imbalance handling), model-swapping demoted to secondary. Also passes the baseline model identity into context.
- **Two real bugs caught by running it:** (1) a code winner crashed writing `best.json` because the run dir wasn't created (the code path skips `save_model`'s mkdir); (2) bad predictions (type mismatch) let a `ValueError` escape `score_predictions` and crash the whole run instead of being a captured failure. Both fixed + regression-tested; `_coerce` now aligns prediction dtype to the holdout target.
- **Cloud aliases supply their own base URL** (`groq`→`api.groq.com/openai/v1`, + openai/together/deepseek), so `--backend groq` (or a saved config) needs only a model + key, no hand-typed `--base-url`. Surfaced by a saved-config run that 404'd.
- **Test isolation:** an autouse fixture points `XDG_CONFIG_HOME` at a temp dir so tests never read the developer's real `~/.config/iterate/config.toml`.
- 233 unit tests; ruff + mypy --strict clean (39 src files).

**Prompt-vs-model finding (A/B, churn / f1, logged in RESEARCH_LOG):** ran the same harness with local `qwen3:14b` vs Groq `llama-3.3-70b`. The 70B explored *models* far more (logistic regression won; it even built a stacking ensemble) but used the **identical preprocessing every iteration** — so the preprocessing monotony was **prompt-bound, not model-bound**. After the FE-first prompt, local qwen engineered a new feature (`TotalCharges_per_tenure`) and hit **f1 0.6166 (+0.049 vs baseline)** — the best result in any run, beating the un-prompted 70B. Conclusion: modeling depth scales with the model; feature-engineering depth was a prompt problem, now fixed. (Aggressive FE by a weak model also produced silent near-zero scores — strongest argument for cell-by-cell.)

**Decision (yours):** pull **cell-by-cell execution** (a stateful code-interpreter session) into **v0.2** rather than deferring to v0.3 (logged in DECISIONS.md). The catastrophic blind-FE failures are exactly what looking-at-the-data-as-you-build prevents.

**Known pending before v0.2 release:** seed the RNG on the code path (run-to-run variance still exceeds small deltas — reproducibility); a pre-run undefined-name lint (recurring "uses X, never imported" failures); the cell-by-cell session itself.

**Next session:** lay out + build the cell-by-cell (stateful code-interpreter) session for v0.2.

### 2026-06-03 | Week 4 Day 5 | Code path goes live (executor + install + defaults + config)

**Task:** Wire the code path end to end so the agent's generated `train_and_predict` actually runs, installs what it imports, and scores through the contract — and make the code path the default. Restructured per two product calls: code-gen is now the default mode, local is the default compute, and a setup wizard lets users save their own defaults.

**What shipped:**
- **Execution routing.** `SandboxExecutor(code_runner)` (`adapters/compute/sandbox.py`) routes code candidates to its `CodeRunner` and runs baselines + spec candidates in-process (shared `run_in_process` helper in `local.py`). New compute contracts in `compute/base.py`: `CodeJob` (script + inputs + outputs + packages) and the `SupportsCodeGen` target capability. `ModelTarget` implements it (`build_code_job` / `score_code_job`) — the target shapes the data and scores (it owns the sealed holdout); the executor owns the venue. Every failure (runner can't boot, crash, timeout, non-codegen target) is captured, never raised.
- **Install-on-demand.** `CodeRunner.run` gained `packages`; `required_imports` (Day 4) feeds it. `E2BCodeRunner` always installs into its disposable sandbox; `LocalCodeRunner(install=…)` installs missing imports into iterate's own env only with consent (`--install`), never silently — a missing import on local is a captured failure otherwise.
- **Output fed back into the loop.** A run's stdout (diagnostics the agent printed) lands on `ExperimentResult.logs`; failures carry the stderr traceback. The CodeProposer history now feeds the recent runs' output + errors back, so the agent learns the data and self-corrects. Bounded (~2k chars/iteration, last few iterations) and leakage-safe (holdout labels never enter the script).
- **Prompt rework** (nothing reads as library-limited): `code_proposer` is environment-aware (install vs ambient) and invites EDA/printing; the **Reconstructor now WRITES code** that reproduces the source faithfully (real CatBoost, custom nets — no "closest allow-listed equivalent"); the spec proposer is reframed as the fast curated fallback.
- **Defaults + config.** `iterate run` gains `--code/--spec` (default code), `--compute local|e2b` (default local), `--install/--no-install`. New `iterate setup` wizard + persisted `~/.config/iterate/config.toml` (`userconfig.py`); precedence is flag > saved config > built-in default; first run with no config offers the wizard (skipped in non-interactive shells). Code winners save their `train_and_predict` source (a code-gen winner returns predictions, not a pickle — by design); spec winners still pickle.
- Tests: full loop on the code path end to end (real `ModelTarget` + `SandboxExecutor(LocalCodeRunner)` + Orchestrator, no LLM); executor routing (code / spec / baseline / non-codegen target / runner-can't-boot); install passthrough + `_missing_packages`; Reconstructor-as-code; output-fed-back + env-note; setup wizard + config round-trip. Live e2b test of the whole code path (install-on-demand included), opt-in.
- 216 unit tests (+10 net); ruff + mypy --strict clean (37 src files).

**Design calls (yours, logged in DECISIONS.md):** no library boundary even by environment — install what the code imports; code + local as defaults (running generated code locally is a conscious setup choice); reconstructor emits code; feed run output back so the agent improves preprocessing.

**Not in Day 5:** the notebook deliverable (Day 6) turns a winning `train_and_predict` into a runnable `.ipynb`; a dedicated inspect/EDA step that doesn't cost a scored iteration is a v0.4 (supervisor) follow-up.

**Next session:** Week 4 Day 6 — notebook deliverable (B): export the winning approach as a clean, runnable notebook.

### 2026-06-02 | Week 4 Day 4 | CodeProposer (LLM writes the code)

**Task:** Add the third LLM caller (sibling of the spec `Proposer` and the `Reconstructor`): instead of naming an allow-listed estimator, it WRITES a `train_and_predict` function to the Day-3 contract. Built and proven in isolation with a fake LLM; wired into the loop on Day 5.

**What shipped:**
- `src/iterate/core/code_proposer.py` — `CodeProposer`: same `LLMClient` protocol + tool-call + retry machinery as the spec proposer, emits `changes = {"code": "<train_and_predict source>"}`. **No library allow-list on this path** — the prompt tells the agent to import whatever it needs; we install its imports before running (Day-5 executor). A cheap static guard (`validate_train_and_predict`) turns malformed snippets into a targeted re-prompt instead of a wasted run; a compact history formatter summarizes past attempts by description + score so whole function bodies are never echoed back into the prompt.
- `src/iterate/core/codegen.py` — two deterministic AST helpers (no LLM):
  - `validate_train_and_predict(code)` — parses, requires a top-level `train_and_predict` of the right arity; returns a precise reason or `None`.
  - `required_imports(code)` — top-level imports minus the stdlib, mapped to pip distribution names (`sklearn`→`scikit-learn`, `cv2`→`opencv-python`, …). Consumed by the Day-5 executor to install-on-demand.
- `code_proposer` prompt block in `prompts.yaml` (system / user / nudges / tool wording).
- Tests: build a code candidate from a tool call; **bridge test** runs a CodeProposer candidate through the real `LocalCodeRunner` + `score_predictions` (proves its output is directly contract-runnable, no LLM); non-parsing / wrong-name / no-tool-call retry then raise; recovery after one bad attempt; prompt carries the brief + metric; history summarized without raw code. Plus `required_imports` / `validate_*` unit tests (stdlib filtered, dotted + aliased names, relative imports ignored, arity + varargs).
- 196 unit tests (+17); ruff + mypy --strict clean (36 src files).

**Design call (yours):** no library allow-list even on the code path — the agent uses whatever it wants and we install its imports. Logged in DECISIONS.md. The import-name→package-name resolution is a provisional hand-kept map; the resolve-and-install **architecture is TBD** (you'll revisit it) — the soft-fail backstop (a bad install becomes a captured failure + retry) means the map only needs to cover the common stack to keep that rare.

**Not in Day 4:** executor routing on `is_code_candidate`, the install-then-run step in the sandbox, the first live e2b run, and the live qwen3 integration test — all Day 5.

**Next session:** Week 4 Day 5 — wire the code path end-to-end (executor routes code candidates, installs imports, runs in the sandbox, scores) + first real sandboxed run + safety.

### 2026-06-02 | Week 4 Day 3 | Code-gen contract

**Task:** Define the strict agreement between a generated training script and us, so the agent can write any modeling code and we still score it the same way on the same sealed holdout. Proven without an LLM or e2b.

**What shipped:**
- `src/iterate/core/scoring.py` — extracted `score` / `task_for_metric` / `direction` (+ the metric sets) out of `ModelTarget` so both the spec path and the code-gen path score identically (single ruler, no drift). `ModelTarget` imports from it; behavior unchanged.
- `src/iterate/core/codegen.py` — the contract:
  - **Fill-in-a-function harness:** the agent writes `train_and_predict(X_train, y_train, X_holdout) -> predictions`; `assemble_script` wraps it in a fixed preamble (loads `train.csv` / `holdout.csv` / `meta.json`) + postamble (writes `predictions.csv`). The LLM owns only the modeling; we own the I/O.
  - **Sealed holdout by construction:** `build_inputs` writes train (with target), holdout **features only**, and meta; the holdout labels never leave the host.
  - **Scoring:** `score_predictions` reads `predictions.csv`, checks length == n_holdout, scores via `core.scoring` → a `Metrics` panel. Missing / empty / wrong-length / unparseable → a captured failure, never a crash.
  - **Code-candidate = `{"code": ...}`** in `Candidate.changes`; `is_code_candidate` routes it (no new schema).
- Tests: end-to-end through the **real `LocalCodeRunner`** with a hand-written LogisticRegression `train_and_predict` (assemble → run → score → valid Metrics, no LLM/e2b); holdout labels absent from inputs; wrong-length / missing predictions captured as failures; a raising function captured by the runner. Plus the scoring extraction keeps `ModelTarget` green.
- 179 unit tests (+6); ruff + mypy --strict clean (35 src files).

**Next session:** Week 4 Day 4 — the CodeProposer (LLM writes `train_and_predict` to this contract; coexists with the spec Proposer).

### 2026-06-02 | Week 4 Day 2 | Code runner primitive (e2b + local)

**Task:** Build the low-level primitive that physically runs a Python script in a venue and returns its outputs. De-risks the riskiest unknown in v0.2 ("can we run code safely and get results back") before wiring it into the loop.

**What shipped:**
- `src/iterate/adapters/compute/runner.py`:
  - `CodeRunner` protocol — `run(script, *, inputs, outputs, timeout) -> RunResult`; must capture a failing script (nonzero exit / timeout), not raise.
  - `RunResult` — stdout / stderr / exit_code / outputs (name → bytes) / timed_out, with a `succeeded` property.
  - `LocalCodeRunner` — temp dir, write inputs + script, `subprocess.run` with mandatory timeout (kills on expiry), read named outputs back. The `--compute local` path; no isolation.
  - `E2BCodeRunner` — boot sandbox, upload inputs, run, download outputs, teardown in `finally`. `e2b_code_interpreter` lazy-imported (module loads without the `[sandbox]` extra); sandbox factory injectable for tests.
- Tests: `LocalCodeRunner` tested for real offline (round-trip, timeout, nonzero exit, missing-output); `E2BCodeRunner` tested with a fake sandbox (upload/run/read/teardown, execution-error mapping, teardown-on-raise); protocol conformance. A live e2b test in the integration suite skips without `[sandbox]` + `E2B_API_KEY`.
- 173 unit tests (+9); ruff + mypy --strict clean (33 src files).

**Honest scope flags (in the module docstring too):**
- **Not in Day 2:** the `ComputeBackend.execute(target, candidate)` integration — that needs code-candidates + the contract, so it completes Day 5. `SandboxExecutor` stays a stub until then. Day 2 is purely the runner primitive.
- **e2b not live-verified:** `E2BCodeRunner` is written to the documented e2b API and fake-tested, but not run against real e2b yet (no key in dev); the exact calls may need small fixes on first live run (Day 5 / when a key is added).
- **Network egress-deny is NOT yet enforced** for e2b (needs a custom sandbox template); flagged, not assumed. Local runner has no isolation by design.

**Next session:** Week 4 Day 3 — the code-gen contract (script I/O: gets train + holdout features, writes predictions; we score through our eval).

### 2026-06-01 | Week 4 Day 1 | `ComputeBackend` protocol (v0.2 foundation)

**Task:** Extract the execution venue into a swappable seam so the e2b sandbox (Day 2) drops in without touching the Orchestrator. Same "add the protocol when the second backend lands" call as the data source and terminator.

**What shipped:**
- `src/iterate/adapters/compute/base.py` — `ComputeBackend` protocol (`execute(target, candidate=None) -> ExperimentResult`, must capture failures not raise). `LocalExecutor` conforms unchanged.
- `src/iterate/adapters/compute/sandbox.py` — `SandboxExecutor` stub (raises NotImplementedError pointing at Day 2); conforms to the protocol so the seam is real.
- Orchestrator now depends on `ComputeBackend`, not the concrete `LocalExecutor`.
- RESEARCH_LOG entry settling the **execution venue** (e2b safe default for generated code + a local `--compute local` opt-in) and the **code-gen contract** (script gets train + holdout *features* only, writes predictions, we score through our eval — holdout labels never cross the sandbox boundary).
- 164 unit tests (+3); ruff + mypy --strict clean (32 src files).

**Decisions (forks settled, see Week 4 plan + DECISIONS direction):**
- e2b is the safe default for autonomously-generated code; **local execution is a supported opt-in** (`--compute local`) since the protocol makes it free to offer, with a warning that generated code runs with the user's permissions.
- Code-gen will be a **new candidate type alongside** the v0.1 `{"model","params"}` spec, not a replacement.

**Next session:** Week 4 Day 2 — the sandbox executor core (boot / upload / run / capture / timeout / teardown) + a local fallback executor.

### 2026-05-31 | v0.1.3 | Lazy CLI imports — instant `version`/`--help`

**Task:** `iterate version` (and `--help`/`config`) took ~2–3s because `cli.py` imported the full pandas + scikit-learn + orchestrator stack at module load, before any command ran.

**What shipped:**
- Moved the heavy imports (LocalExecutor, load_csv, SqliteMemory, Orchestrator, Proposer, Reconstructor, terminator, build_client, ModelTarget) out of the module top and **into `run()`** — the only command that needs them. `version`/`config`/`--help` now import only typer + rich + config.
- `import iterate.cli`: ~2–3s → **0.18s**; `iterate version`: **~0.2s**.
- Fixed the CLI tests' monkeypatching to target the source modules (lazy `from … import` inside `run()` bypasses a `cli`-module patch).
- 161 unit tests; ruff + mypy --strict clean. Version → 0.1.3.

### 2026-05-31 | v0.1.2 | Broaden the Proposer's model space (prompt fix)

**Task:** The Proposer kept re-proposing the 2–3 models named in the prompt examples (XGBoost / RandomForest / LightGBM) instead of exploring scikit-learn's full catalog — classic example-anchoring.

**What shipped (prompt-only, `prompts.yaml`):**
- System prompt now states the **full** estimator breadth explicitly (linear models, SVMs, k-NN, naive Bayes, discriminant analysis, single trees, the whole ensemble family, plus XGBoost/LightGBM) and instructs the LLM to **actively vary the model family** across iterations and match the task type.
- The `model` tool-field examples are now diverse (LogisticRegression, ExtraTrees, GradientBoosting, SVC, KNeighbors, XGB, LGBM) and explicitly labeled "examples, not a restricted list."
- Live check (5 iters, real qwen3): now proposes **4 distinct families** (XGBoost, LightGBM, RandomForest, GradientBoosting — the last never reached before) vs 2 before. Best f1 0.5676 → 0.5871.

**Honest limit:** local qwen3:14b still gravitates to tree ensembles (didn't reach linear/SVM/kNN) — defensible for tabular churn, and a cloud backend explores wider. Prompt did its job; further breadth is a stronger-model gain.

Version → 0.1.2 (bundles the v0.1.1 noise fix for a single PyPI publish).

### 2026-05-31 | v0.1.1 | Silence native training noise (demo polish)

**Task:** `verbose=-1` didn't fully muzzle LightGBM — its C++ core writes `[LightGBM] [Info] …` straight to the file descriptors, bypassing Python verbosity. Clean it up for a recordable demo.

**What shipped:**
- `_silence_native_stdio()` context manager wraps fit + predict: redirects fds 1/2 to devnull **and flushes libc stdio** (`ctypes` `fflush(None)`) before restoring, so buffered native output drains to devnull instead of leaking onto the terminal after the fds are restored.
- `_silence_lightgbm()` registers a null LightGBM logger once (its C++ logger bypasses C stdio buffering, so the fd redirect alone wasn't enough).
- Net effect: LightGBM info chatter, XGBoost per-round eval, and the benign sklearn feature-names warning are all gone; only the loop's own output shows.
- Regression test (`capfd`) asserts no `[LightGBM]` leaks. 161 unit tests; ruff + mypy --strict clean.
- Version → 0.1.1; published to PyPI.

### 2026-05-31 | Week 3 Day 7 | Release polish + Week 3 retro + v0.1.0

**Task:** Ship v0.1 honestly — fill the last contract gap (model persistence), trim the install, add release hygiene, reconcile the public docs with what v0.1 actually does, and tag.

**What shipped:**
- **Model persistence (the contract's "artifact"):** the executor used to train + score + discard the fitted model. Now `ModelTarget.save_model(spec, path)` refits the winner on train (same seed → exactly the scored model) and `joblib.dump`s the full pipeline; the CLI writes it to `.iterate/runs/<run_id>/best_model.joblib` (+ a `best.json` config sidecar) and prints the load line. `RunResult` gained `run_id`. Verified live: `joblib.load(path).predict(X)` works.
- **Dependency trim:** dropped `sqlalchemy` (unused — Memory is stdlib `sqlite3`); moved `e2b-code-interpreter` → `[sandbox]`, `kaggle`/`datasets` → `[datasets]` extras. `pip install iterate-ai` now pulls only what the v0.1 loop runs on.
- **PyPI dist name `iterate-ai`** (`iterate` was taken; import + command stay `iterate`).
- **`LICENSE`** (MIT).
- **README reconciliation:** Quick start rewritten to the real v0.1 flow (`pip install iterate-ai` → Ollama → `iterate run --data … --target … --metric f1`), the one-line discovery form clearly relabeled as the v1 vision, `iterate history`/`why-failed`/`best` marked roadmap. Status → "v0.1 released."
- 160 unit tests; ruff + mypy --strict clean (30 src files). Live CLI run on real Telco churn saves a working model.

**Decisions:**
- **Best model saved as a joblib artifact** — "we found the best model" is hollow if the user can't load it. Refit-on-train (matches the reported score) over refit-on-all-data (wouldn't match) for honesty in v0.1.
- **Lean core deps** — forward-looking libraries belong in extras, not forced on every install.

---

## Week 3 retro — v0.1 shipped (the agentic loop)

**The week in one line:** went from a tabular substrate to a working autonomous agent — `iterate run` reads a dataset, re-measures the baseline, and an LLM iterates model + hyperparameters to the best it can find, with persistent cross-run memory and a saved model artifact.

**Shipped (Days 1–7):** Proposer (+ native `OllamaClient` for `think:false`) · Orchestrator · Terminator (delegated protocol) · Memory (sqlite, cross-run) · CLI `iterate run` + source-aware baseline reconstruction · reproducible churn demo · model persistence + release.

**What worked:**
- **The four Protocol seams** (`LLMClient`, `BenchmarkTarget`, `Terminator`, `Memory`) — every component swappable without touching the loop. Adding the native Ollama client, the sqlite memory, and the terminator concretes were all adapter changes, not refactors.
- **Agent-first sequencing paid off** — the loop exists at v0.1, exactly the re-plan's bet.
- **Measure-don't-assume**, again — the 18-minute proposer hang ran down to qwen3 thinking-mode (only disablable on the native endpoint); the early-stopping failures ran down to a missing eval set.

**What didn't / punted:**
- **Local-model tool-calling yield** — qwen3 still occasionally replies without a tool call; mitigated (3 attempts + firm nudge), not solved. Cloud backend is the reliable path. Remaining levers (few-shot, lower temp, text-fallback parser) logged, deferred.
- **"Auditable report" = memory + summary**, not a generated document (Reporter is later).
- **No `iterate history`/`best`/`why-failed`** query commands yet (data's in memory; CLI surface is a natural early post-v0.1 add).
- LightGBM macOS-wheel slowness; richer structured failure replay.

**Decisions that shaped it (DECISIONS.md):** native Ollama as its own adapter · reconstruct-from-text, never execute user code · interactivity split (v0.2 cheap wins / v0.3 full chat) · `--baseline` requires `--source` · `--fresh` archives · best model saved as an artifact.

**Pace:** Weeks 1–3 (foundation → substrate → agentic loop + first release) done in ~9 days of sessions, ahead of the nominal cadence.

**Next: v0.2 — sandboxed code-gen** (the agent writes + runs its own training code → any model, not just the three libraries) + the cheap interactive wins (live progress, streaming, Ctrl-C).

### 2026-05-31 | Week 3 Day 6 | Reproducible churn demo + demo-clean polish

**Task:** Turn the ad-hoc CLI runs into a committed, reproducible v0.1 demo, and make the terminal output clean enough to record. (Tagging v0.1.0 is Day 7, after the retro.)

**What shipped:**
- `examples/churn_tabular/prepare.py` — Telco-specific cleaning (drop `customerID`, coerce `TotalCharges`, encode `Churn` Yes/No → 1/0) as a pure `clean()` fn + CLI entry; writes the committed `data.clean.csv`. **Data prep is not part of `iterate`** — standard ML glue, dataset-specific, kept out of the framework.
- Retired `examples/churn_tabular/run.py` (the Week-2 hand-fed-candidate demo — superseded by `iterate run`).
- `examples/churn_tabular/README.md` — rewritten for the v0.1 agentic flow (prep step + `iterate run` command + representative output + honest "prep is standard ML, the agent's job is the iteration" note).
- **Verbosity suppression** (`build_estimator`): inject quiet defaults (`verbosity=0` for XGBoost, `verbose=-1` for LightGBM) only when the candidate didn't set them and the class accepts them — the agent's explicit choice always wins. Kills the library training chatter that buried the loop's own output.
- **Proposer-yield polish:** default `max_retries` 1 → 2 (3 attempts) + a blunt retry nudge ("respond ONLY by calling the tool — no prose"). Reduces dropped iterations from local-model chatty replies.
- Tests: `test_prepare_churn.py` (cleaning is correct + idempotent), `tests/integration/test_agentic_loop_live.py` (real qwen3 + real ModelTarget end-to-end, opt-in), rewrote `test_churn_end_to_end.py` to use the committed clean CSV (no `run.py` dependency, no LLM).
- 158 unit tests pass; all 4 integration tests pass live; ruff + mypy --strict clean (30 src files).

**Finding (honest):** even with 3 proposer attempts, local qwen3:14b still occasionally replies without a tool call and an iteration is lost (recorded as a `ProposerFailure`, loop continues — graceful). This is the model tier's tool-calling ceiling, not a code bug. Remaining levers (few-shot example, lower temperature, a text-fallback parser, or a cloud backend) are deferred; the loop already survives misses correctly, and `--backend openai-compatible` is the reliable path for a flawless run.

**Next session:** Week 3 Day 7 — polish + Week 3 retro + tag **v0.1.0** (first release).

### 2026-05-30 | Week 3 Day 5 | CLI `iterate run` + source-aware baseline reconstruction + roadmap split

**Task:** Wire everything we've built into a single terminal command. Make `--baseline + --source` actually drive something — the LLM reads the source as text only and rebuilds the modeling approach as a runnable spec we execute through our own eval. Update the roadmap for the v0.2/v0.3 interactivity split.

**What shipped:**
- `src/iterate/cli.py` — `iterate run` command with all v0.1 flags + helpers (notebook walker, duration parser, db archiver, divergence check, baseline precedence, `rich.Table` summary, `RichHandler` live per-iteration log streaming).
- `src/iterate/llm/factory.py` — `build_client(name, …)` dispatches to `OllamaClient` or `OpenAICompatibleClient`.
- `src/iterate/core/reconstructor.py` — `Reconstructor` (sibling of `Proposer`; same LLM/tool-calling machinery; different prompt + tool `reconstruct_baseline`; lower temperature for fidelity).
- `src/iterate/prompts/prompts.yaml` — new `reconstructor` block (system + user template + tool description).
- `src/iterate/core/orchestrator.py` — optional `baseline_candidate: Candidate | None`; when given, the baseline is `executor.execute(target, baseline_candidate)` rather than the factory default.
- **Baseline precedence** inside the CLI (first match wins): `--source` (reconstructed) → memory's prior best for this target (re-measured; `--fresh` opts out) → factory default.
- **`--baseline` requires `--source`.** A number with no source describes nothing we can run; informational-only was the worst of both worlds — explicit CLI error.
- **`--fresh` archives, doesn't delete.** The existing memory db is renamed to a timestamped `.bak` rather than `rm`'d. Recoverable cheap safety net. Triggers on `--fresh`, `--source`, or `--baseline + --source` — explicit user input = "new chapter."
- **Cloud backends require an API key** (from `--api-key` or env); validated up front, not at first request.
- 153 unit tests pass (was 124; +29 net — factory 6, reconstructor 6, CLI 16, +1 orchestrator); ruff + mypy --strict clean (30 src files).

**Decisions (`DECISIONS.md`):**
- **`--baseline` requires `--source`** — informational-only baseline numbers are dead weight.
- **`--fresh` archives, doesn't delete** — non-destructive by default.
- **Roadmap split for interactivity (Option B):** v0.2 picks up the *cheap* interactive wins (live progress, streaming, Ctrl-C); v0.3 is a new milestone for *full mid-run chat* (pause/resume/conversational state). Everything that was v0.3+ shifts one version. Streamlit → v0.10. Build is now ~14 weeks (was ~13).

**Next session:** Week 3 Day 6 — first autonomous tabular run on real Telco churn data → tag **v0.1.0**.

### 2026-05-30 | Week 3 Day 4 | Memory (persistent history + cross-run continuity)

**Task:** Move history out of the Orchestrator's RAM and into a `Memory` protocol. Ship both an in-memory implementation (tests, ephemeral runs) and a sqlite-backed one (the real thing that survives `iterate run` exiting). Close the loop on Day 2's deferred "structured proposer-failure records."

**What shipped:**
- Files: `src/iterate/core/memory.py` (Protocol + 2 implementations + `ProposerFailure` dataclass), `tests/unit/test_memory.py` (14 tests, parameterized over both backends). Orchestrator refactored to delegate.
- **`Memory` Protocol** — `start_run` · `record` · `record_proposer_failure` · `history` · `proposer_failures` · `finish_run`. Same shape as `Terminator` — one Protocol, swappable backends.
- **`InMemoryMemory`** — dict-backed; ephemeral.
- **`SqliteMemory(db_path)`** — stdlib `sqlite3` (no ORM); auto-creates parent dir + schema on first use; persists across processes; one file on disk (default `.iterate/memory.db`, configurable via `ITERATE_MEMORY_DB`). `Experiment` and `ExperimentResult` go in as JSON blobs (pydantic round-trip); proposer failures live in a separate `proposer_failures` table.
- **Orchestrator refactor:** takes `memory: Memory` as a constructor arg. Calls `memory.start_run` at the top, records each `Experiment` through `memory.record`, structured `ProposerFailure` rows through `memory.record_proposer_failure`, queries `memory.history(target.name)` each iteration (so the Proposer sees **cross-run** history), `memory.finish_run` at the end. `RunResult.history` still returns just the current run's experiments.
- 124 unit tests pass (was 106; +18 net); ruff + mypy --strict clean (28 src files).

**Decisions:**
- **Stdlib `sqlite3`, not SQLAlchemy** — ~250 lines of straight SQL that reads top-to-bottom; the Memory protocol is the seam, so swapping in SQLAlchemy or Postgres later is an adapter change, not a refactor.
- **Cross-run history fed to the Proposer** by default — institutional memory is the value-prop. (A CLI `--fresh` flag at Day 5 can opt out.)
- **No programmatic dedupe** — give the LLM the full history and trust the prompt's "don't repeat." Add `has_been_tried(changes_hash)` only when failure modes show up in practice.
- **JSON-blob serialization** for `Experiment` / `ExperimentResult` — pydantic handles it cleanly; no schema changes when the models evolve; no ORM mapping to maintain.
- **Per-target scope** for history (no data-version hash yet) — that's a Week-9 concern when datasets evolve mid-project.

**Next session:** Week 3 Day 5 — CLI `iterate run` (+ `--backend` flag) and source-aware baseline reconstruction.

### 2026-05-30 | Week 3 Day 3 | Terminator (delegated stop logic)

**Task:** Extract the Orchestrator's internal stop logic into a clean `Terminator` protocol; add the missing stop conditions (deadline, plateau).

**What shipped:**
- Files: `src/iterate/core/terminator.py` (Protocol + 5 concretes + `LoopState` + factory), `tests/unit/test_terminator.py` (18 tests). Orchestrator refactored to delegate.
- `Terminator` protocol: one method, `update_and_check(state) -> str | None`. Stateful by design (Patience/Plateau track history); single method avoids notify-then-check ordering bugs.
- Concretes: `MaxIterations(n)`, `Patience(k)` (counts `proposer_error` too), `Deadline(seconds)`, `Plateau(window, epsilon)` (direction-agnostic spread), `Composite(*terminators)`.
- `Composite` calls **all** children every iteration (each maintains its own state correctly), then returns the first non-`None` reason.
- `default_terminator(...)` factory: sane Composite of `MaxIterations` + `Patience`, optional `Deadline`.
- **Orchestrator refactor:** dropped `max_iterations` / `patience` constructor args; takes `terminator: Terminator` instead. Tracks per-iteration outcome (`improved` / `no_improvement` / `proposer_error`) and elapsed wall-time, builds a `LoopState` each iteration, propagates whatever `stopped_because` reason the terminator returns.
- 106 unit tests pass (was 89; +17 net); ruff + mypy --strict clean (27 src files).

**Decisions:**
- **One method on Terminator** (`update_and_check`) rather than separate `notify` + `should_stop` — fewer places to call wrong, no ordering ambiguity.
- **`Composite` calls all children** every iteration (then returns the first reason) rather than short-circuiting — short-circuit would leave later terminators with stale state and they'd fire wrong on the next call.
- **`Plateau` shipped now** — small (~15 lines), and direction-agnostic via spread (max − min in the window) is more robust to noise than first-vs-last improvement.
- **Dropped the old Orchestrator constructor args cleanly** (no compatibility shim) — only the existing tests use the Orchestrator today; this was the cleanest moment to refactor.

**Next session:** Week 3 Day 4 — Memory (sqlite, persistent history, feed past attempts to the Proposer, recognise repeats across sessions).

### 2026-05-30 | Week 3 Day 2 | Orchestrator (closes the agentic loop)

**Task:** Wire the Week-2 substrate + Day-1 Proposer into the autonomous loop — `baseline → propose → execute → score → record → decide → repeat`.

**What shipped:**
- Files: `src/iterate/core/orchestrator.py` (`Orchestrator` class + frozen `RunResult` dataclass), `tests/unit/test_orchestrator.py` (9 tests)
- `RunResult` carries: the re-measured baseline, the full ordered `Experiment` history, the best successful experiment (or `None`), and `stopped_because` (`"max_iterations"` | `"patience"` | `"baseline_failed"`).
- `current_model` follows the best-so-far candidate — the Proposer's prompt always reflects what's currently in use.
- 89 unit tests pass (+9 from Day 2); ruff + mypy --strict clean (26 src files).

**Decisions (deliberately YAGNI for Day 2):**
- **In-memory history** — Memory (sqlite, Day 4) plugs in as a swap.
- **Internal stop logic** (`max_iterations`, `patience`) — Terminator (Day 3) takes over via a delegated protocol (same shape as the deferred `ComputeBackend` protocol).
- **`ProposerError` counts toward patience, no history entry** — the iteration was attempted; there's no `Candidate` to wrap as an `Experiment`. Day 4 Memory adds structured proposer-failure records.
- **No `run_agent.py` script or live integration test** — runnable end-to-end is the Day-5 CLI's job; Day 2 stays on deterministic fakes (no temporary code just for visual confirmation).

**Next session:** Week 3 Day 3 — the Terminator (deadline / patience / plateau as a delegated protocol).

### 2026-05-30 | Week 3 Day 1 | Proposer + native Ollama adapter + centralized prompts

**Task:** Start the agentic loop — the LLM proposes the next Candidate. Took two calendar days because the live path surfaced a hard backend constraint that had to be solved before Day 2 could land.

**What shipped:**
- **The Proposer** (`src/iterate/core/proposer.py`): turns an LLM call into a structured `Candidate` via a `propose_candidate` tool — REQUIRED `model` (current model or another, by import path), optional `params`, plus `description`/`rationale`/`expected_metric_delta`. Text-reply retry fallback (the LLMClient protocol exposes no `tool_choice`). `summarize_dataset(dataset)` helper for the data brief.
- **Native Ollama adapter** (`src/iterate/llm/ollama_client.py`): a NEW `LLMClient` implementation hitting Ollama's native `/api/chat` with `think:false`. Lives **alongside** `OpenAICompatibleClient` (unchanged) — Ollama gets its own adapter because its OpenAI `/v1` layer can't disable thinking. Added `ollama_host` to config.
- **Centralized prompts** (`src/iterate/prompts/prompts.yaml` + 12-line loader): every Proposer prompt — system, user template, history header, retry nudge, tool description + all 5 field descriptions — now lives in one YAML file. Wording can change without touching code.
- 80 unit tests + 1 integration (live qwen3:14b → valid Candidate in ~40s). ruff + mypy --strict clean (25 src files).

**The finding (measured, not assumed):**
- The first live Proposer call timed out at **18 minutes** (SDK retries × backend timeout). Diagnosed step by step: real-time streaming via `ollama run` showed qwen3 spending ~900 tokens on `<think>` reasoning before any answer. Tested all the documented thinking-off mechanisms — **`/v1/chat/completions` ignores them all** (`think:false` body param, `/no_think` soft prompt, `chat_template_kwargs:{enable_thinking:false}`). Only Ollama's **native `/api/chat`** honors `think:false`: **128s → 20s** for the same prompt, and the tool call is *richer* (with thinking off the model emitted explicit hyperparameters; with thinking on it sometimes returned no tool call at all). Recorded in memory so we never re-derive it.
- The fix is the new `OllamaClient`; the OpenAI client stays clean for cloud backends.

**Decisions (see DECISIONS.md):**
- **Baseline reproduction lands IN v0.1** (not Week 10) — `--baseline` and `--source` have to drive something or they're dead weight. Source is **read as text** by the LLM to reconstruct the approach and re-measure through our own eval. Slotted into Day 5 with the CLI.
- **Never execute user-provided source code, ever** (malware/RCE). The v0.2 sandbox runs the agent's OWN generated code, never the user's.
- **Native Ollama as its own adapter** (not bundled into the OpenAI client) — one backend's quirk shouldn't pollute the shared client.
- **Centralized prompts in YAML** — wording iterates more than code; one file = one place.

**Next session:** Week 3 Day 2 — the Orchestrator (baseline → propose → execute → score → record → decide → repeat).

### 2026-05-28 | Week 2 retro | Tabular execution substrate complete

**The week in one line:** went from an empty `targets/` package to a complete, tested substrate that runs one tabular experiment end-to-end — and re-planned the whole roadmap to agent-first while doing it.

**Shipped (Days 1–6):**
- `BenchmarkTarget` protocol — the contract every target obeys (`baseline()` + `run(candidate)`).
- Tabular data adapter — `load_csv` → deterministic stratified split → content-hashed `TabularDataset`, leakage-safe.
- `ModelTarget` — leakage-safe sklearn Pipeline, metric panel, deterministic.
- Model factory — any allow-listed installed estimator (sklearn/XGBoost/LightGBM) from a nested `{"model","params"}` spec.
- `LocalExecutor` — runs one experiment, times it, captures failures instead of crashing.
- End-to-end churn example on the real Telco dataset + an integration test.

**What worked:**
- The **contract cascade** — each piece shaped the next: the non-empty-`changes` validator from Week 1 forced `baseline()` to be its own method; the executor's failure capture exists *because* targets are allowed to raise.
- **Measure-don't-assume** earned its keep twice — the ~200x HistGB thread-oversubscription bug and the ~450x LightGBM macOS-wheel finding. Both would have crippled the loop; neither was the hardware.
- **Clean separation:** the target measures, the executor survives, the data adapter only loads + splits.

**What didn't / punted (tracked in the backlog):**
- Hard execution isolation (timeouts, resource caps) → v0.2 (e2b sandbox).
- Richer structured failure capture (vs a plain `error` string) → before v0.1 (Memory needs the "why").
- LightGBM macOS-ARM wheel is pathologically slow → documented; supported but out of the demo; fine on Linux.
- Hash-based splitting → later (a static per-run CSV doesn't need it yet).

**Decisions that shaped it (see DECISIONS.md):**
- **Agent-first re-plan** mid-week — the agentic loop became the v0.1 milestone instead of a Week-7 add-on.
- **Sandboxed code-gen (c) bumped to v0.2** — "run the model research recommends" shouldn't wait.
- **Nested candidate spec** over flat — clean model/params separation, the shape the Proposer will emit.

**Pace:** Week 2 done in 6 build sessions, on track.

**Next: Week 3 — the agentic loop → v0.1.** The Proposer generates the candidates we've been hand-supplying; the Orchestrator runs propose → execute → score → record; the Terminator stops on plateau/patience; Memory feeds history back. The first release where the agent drives.

### 2026-05-28 | Week 2 Day 6 | Substrate end-to-end on real churn data (+ a LightGBM macOS finding)

**Task:** Prove the whole tabular substrate works together on a real dataset — the last piece before the Week-3 agentic loop.

**What shipped:**
- Files: `examples/churn_tabular/run.py` + `README.md` + `data.csv` (public Telco Customer Churn, 7043 rows); `tests/integration/test_churn_end_to_end.py` (marked `integration`); a fast build-only factory test for XGBoost/LightGBM in `tests/unit/test_model_registry.py`
- End-to-end on real data: `load_csv` → `ModelTarget` → model factory → `LocalExecutor`. Re-measured baseline (HistGB) f1 **0.568** → best candidate (XGBoost) **0.576** (+0.008); a deliberately broken candidate is captured as a failure, not a crash.
- Dataset-specific cleaning (drop `customerID`, coerce `TotalCharges`, encode `Churn` Yes/No → 1/0) lives in the example, not the framework.
- 64 unit tests + 1 integration test green; ruff + mypy --strict clean (23 src files).

**The finding (measured, not assumed):**
- A LightGBM candidate took **~155s** vs XGBoost's 0.38s on identical data. Ran it down: not the thread wrapper (slow with *and* without `threadpool_limits`), not the hardware (XGB/HistGB sub-second), not a sklearn/LightGBM OpenMP conflict (slow even with LightGBM imported alone, no sklearn). Root cause: the **LightGBM 4.6 macOS-ARM pip wheel is pathologically slow to train** (~0.2s/tree, ~450x) — a known wheel/`libomp` issue, not our code, and absent on Linux / in the e2b sandbox.
- Resolution: LightGBM stays factory-supported (build-only unit test) but is omitted from the demo's candidate list; documented as a known issue. Not forcing a from-source build on all installs to fix a local-macOS-only problem. (Backlog + example README.)

**Next session:** Week 2 wrap / Day 7 polish, then **Week 3 — the agentic loop** (Proposer drives the candidates → v0.1). Substrate is complete: contract · data adapter · `ModelTarget` · model factory · executor · end-to-end example.

### 2026-05-28 | Week 2 Day 5 | Local executor (minimal failure capture)

**Task:** A compute venue that runs one experiment end-to-end and never lets a bad candidate crash the loop.

**What shipped:**
- Files: `src/iterate/adapters/compute/local.py` (`LocalExecutor`), `tests/unit/test_local_executor.py` (4 tests)
- `execute(target, candidate=None)` — `None` runs the baseline, otherwise the candidate; times the run and stamps `duration_seconds`.
- **Failure capture:** any exception from the target (broken params, a fit-time error, an off-list model) is caught and recorded on `ExperimentResult.error`; `metrics` stays `None` and nothing propagates, so the loop keeps going and Memory can read the reason.
- 63 tests pass; ruff + mypy --strict clean (22 src files).

**Decisions:**
- **No `ComputeBackend` Protocol yet** — `LocalExecutor` is the only backend; the Protocol gets extracted when e2b lands (v0.2), with cloud-GPU a third adapter on the same port. Same YAGNI call as the deferred `DataSource` protocol.
- **Crash = `error` string** for v0.1; a richer structured `FailureCase`/traceback for the Week-3 Memory store is tracked in the backlog (before v0.1).
- Hard isolation (timeouts, resource caps, killing runaway training) is the e2b sandbox's job → v0.2 (backlog).

**Next session:** Week 2 Day 6 — substrate end-to-end on a real churn dataset in `examples/`: `baseline()` + a supplied candidate through the executor, with an integration test.

### 2026-05-28 | Week 2 Day 4 | Model factory (any installed estimator) + bumped sandbox code-gen to v0.2

**Task:** Stop hard-coding the estimator. Build any allow-listed installed model from a candidate's spec — so the Proposer can switch model families, not just tune one.

**What shipped:**
- Files: `src/iterate/adapters/models/registry.py` (`build_estimator`), `tests/unit/test_model_registry.py` (8 tests); `ModelTarget` rewired to delegate to the factory (dropped its local `_make_estimator`)
- A candidate's `changes` is now a **nested spec** — `{"model": "<import.path>", "params": {…}}` — instead of flat hyperparameters. `model` is optional (defaults to `HistGradientBoosting` per task); `params` optional.
- Dynamic instantiation via `importlib`, **allow-listed** to `sklearn.*` / `xgboost.*` / `lightgbm.*` (anything else raises and points at the v0.2 code-gen path). `random_state` injected only when the estimator's signature accepts it (introspected) and not already set.
- 59 tests pass; ruff + mypy --strict clean (21 src files); suite still ~3.8s (threading cap holding).

**Decisions:**
- **Nested `{"model","params"}` spec** over flat hyperparameters — clean separation of *which model* from *its params*, no key collisions, and the exact shape the LLM will emit next ("this model, these params, from research"). (RESEARCH_LOG 2026-05-28.)
- **Two model-flexibility tiers, and (c) bumped early to v0.2:** (b) this factory = any *installed* allow-listed library, shipped now; (c) sandboxed code-gen = the agent *writes* training code and runs it in e2b → *any* model at all, moved to **v0.2** (right after the v0.1 loop). Scope/Releases tables re-sequenced above; later versions shift down one, build now ~12–13 weeks.
- Allow-list (not arbitrary import) is the safety boundary for (b); arbitrary/uninstalled models are exactly what the sandbox (c) is for.

**Next session:** Week 2 Day 5 — local executor (run one `Experiment`: build candidate → train → score → `ExperimentResult`, with failure capture).

### 2026-05-27 | Week 2 Day 3 | `ModelTarget` (tabular) + a ~200x perf fix

**Task:** First concrete target — train + score a tabular model (`baseline()` + minimal `run()`).

**What shipped:**
- Files: `src/iterate/targets/model.py` (`ModelTarget`), `tests/unit/test_model.py` (6 tests)
- `baseline()` + `run(candidate)` via a **leakage-safe** sklearn Pipeline (preprocess → estimator, fit on train only); `HistGradientBoosting` default; task + metric panel inferred from `--metric`; deterministic (seed)
- Demoed live: baseline f1 0.667 → best hand-supplied candidate 0.710 (+0.043). The substrate iterates (manually; the agent drives it Week 3).
- 50 tests; ruff + mypy --strict clean (20 src files)

**Finding + fix (the important one):**
- Model tests ran ~83s. Diagnosed to sklearn `HistGradientBoosting` **OpenMP thread oversubscription** on the 10-core M5 — **9.99s/fit on 120 rows vs 0.05s single-threaded (~200x)**. Not the hardware — tiny data + many threads = pure coordination overhead. Would have crippled the agentic loop (it runs many small experiments).
- Fix: cap threads during fit/predict via `threadpool_limits` (default 1, configurable `max_threads`). Full suite **183s → 4.2s**. Added `threadpoolctl` as a direct dep.

**Decisions:** estimator-family switching + richer candidate→model mapping = Day 4 (model adapters); robust error handling + execution venue = Day 5 (executor); `FailureCase` capture = Week 3.

**Next session:** Week 2 Day 4 — model adapters (sklearn + XGBoost; build a model from `Candidate.changes`).

### 2026-05-27 | Week 2 Day 2 | Tabular data adapter + agent-first re-plan

**Task:** Tabular data loading/splitting — and re-planned the whole roadmap to agent-first.

**What shipped:**
- Files: `src/iterate/adapters/data/tabular.py` (`load_csv` → `TabularDataset`), `tests/unit/test_tabular.py` (8 tests)
- Deterministic **stratified** split + dataset **content-hash** (data versioning); leakage-safe (split before preprocessing)
- `pandas` added to the mypy ignore list (treated like the other ML libs)
- 44 tests pass; ruff + mypy --strict clean (19 src files)

**Decisions:** (data-handling research → RESEARCH_LOG 2026-05-26)
- Stratified seed split + content-hash now; hash-based splitting deferred to Week 9 (evolving data); persist split snapshot → executor (Day 5).
- **Re-planned the roadmap to agent-first** (was breadth-first): the agentic loop is the **v0.1 milestone (~Week 3)**, not Week 7. Two dials thereafter — inputs shrink, problem types grow. Scope / Releases / Week 2-3 plans rewritten above; Proposer + Memory pulled forward to Week 3.
- Reframed the moat: specialization + the full differentiator combination, with cost-aware serving as the **flagship** (not the only moat).

**Next session:** Week 2 Day 3 — `ModelTarget` (sklearn baseline): `baseline()` train + score → `Metrics`.

### 2026-05-26 | Week 2 Day 1 | `BenchmarkTarget` protocol (v0.1.0 groundwork)

**Task:** Define the contract every target implements, so the orchestrator runs tabular / DL / prompt targets uniformly.

**What shipped:**
- Files: `src/iterate/targets/base.py` (the `BenchmarkTarget` Protocol), `tests/unit/test_targets_base.py` (4 tests)
- `Protocol` + `@runtime_checkable`, sync — `name`, `baseline() -> ExperimentResult`, `run(candidate) -> ExperimentResult`
- 36 tests pass; ruff + mypy --strict clean (18 src files)

**Decisions:** (see RESEARCH_LOG 2026-05-26)
- The target only **measures**; `baseline()` **always re-measures** the starting point through the target's own eval (never adopts a reported score) → every comparison is apples-to-apples.
- The target does not judge the winner — the orchestrator/terminator compares. Execution venue is the compute layer's concern, not the target's.
- A no-op Candidate is impossible (schema validator requires non-empty `changes`), which is *why* `baseline()` is its own method.

**Next session:** Week 2 Day 2 — tabular data adapter (`src/iterate/adapters/data/tabular.py`).

### 2026-05-25 | Week 1 Day 3 | CLI scaffold (working) + Week 2–3 plans

**Task:** Make the CLI scaffold real (Task #8) + log the missing Week 2 & 3 day-by-day plans.

**What shipped:**
- Files: `src/iterate/cli.py` (root callback + `version` + `config` commands), `tests/unit/test_cli.py` (4 tests)
- Fixed the typer **single-command collapse** bug — `iterate --help` now lists commands, `iterate version` works, `iterate config` prints resolved settings (api-key masked)
- BUILD_LOG: added Week 2 (ModelTarget / tabular) + Week 3 (PromptTarget / LLM-judge) day-by-day plans; reconciled the stale Week-1 Days 4–7
- 32 tests pass; ruff + mypy --strict clean

**What didn't:** nothing punted.

**Decisions:**
- Root `@app.callback()` to stop typer promoting a single command to the app root.
- Added a `config` command (debug aid + demonstrates the config layer wired to the CLI).

**Next session:** Week 2 Day 1 — `BenchmarkTarget` protocol (`src/iterate/targets/base.py`).

### 2026-05-24 | Week 1 Day 2 (same day as Day 1 — ahead of ETA) | LLM client layer — partial

**Task:** `LLMClient` protocol + `OpenAICompatibleClient` (Ollama) + smoke test. Pulled `config.py` forward from Day 3.

**What shipped:**
- Files: `schemas/llm.py` (Message/ToolSpec/ToolCall/Usage/ChatResponse), `llm/base.py` (`LLMClient` Protocol), `llm/openai_compatible.py` (sync client over the OpenAI SDK, Ollama default), `config.py` (central settings — all defaults one place, env/secret override), `tests/unit/test_openai_compatible.py`
- Deps: `pydantic-settings`; `.env.example` gains `ITERATE_BACKEND_TIMEOUT`; README `llm/` architecture corrected to the openai_compatible design; integration tests made opt-in
- 28 unit tests + a live smoke; ruff + mypy --strict clean (17 src files)
- Behavior: provider-agnostic LLM layer — swap backend by config alone; token usage surfaced for cost tracking

**What's tested:**
- Offline (deterministic, mocked SDK): translation both ways, tool-call parsing, usage defaulting — passing
- Live: plain chat end-to-end (`content='Ok'`, usage populated), error classification + retry, `test_live_ollama_smoke` — passing
- Live **structured tool-calling**: blocked at the time — see below

**What didn't (why Day 2 isn't fully done — the LLM):**
- `qwen2.5-coder:14b` returns tool calls as **plain text**, not structured `tool_calls` (verified even with `tool_choice="required"`); the `-coder` template lacks tool support. Our client is correct (parses structured calls — proven offline); the model is the gap.
- Lost ~1h to an Ollama version skew (desktop app 0.20.6 vs CLI 0.24.0) crashing the model runner — fixed by updating the app.

**Decisions:** (see RESEARCH_LOG 2026-05-24)
- Direct vendor SDKs, not LangChain. Sync client for v1. Tool-calling in the interface. LLM types in `schemas/llm.py`. Config centralized (defaults one place; secrets override). Next tool-driving model = **qwen3:14b** (validate qwen3:8b first; flip `config.iterate_model` once it tool-calls structurally).

**Update (later 2026-05-24 — carry-over RESOLVED):** `qwen3:14b` finished downloading and was validated through the client — `has_tool_calls=True`, args parsed to dict, `finish=tool_calls`. Flipped `config.iterate_model` default to `qwen3:14b` (+ `.env.example`). Day 2 now complete, including live agentic tool-calling. Noted: qwen3's thinking mode is on by default (spends tokens before the answer → needs generous budgets); bumped the live smoke to `max_tokens=512`.

**Next session (2026-05-25):**
- Day 3 proper: CLI scaffold (`iterate --help`, typer setup). Possibly handle qwen3 thinking-mode toggling when wiring prompts.

### 2026-05-24 | Week 1 Day 1 | Pre-flight verification + Pydantic schemas

**Task:** Verify the toolchain runs, then ship the 5 core domain schemas.

**What shipped:**
- Files: `src/iterate/schemas/experiment.py`, `tests/unit/test_schemas.py`, `.python-version` (3.12), `uv.lock`
- `Experiment`, `ExperimentResult`, `Metrics`, `FailureCase`, `Candidate` (Pydantic v2, `extra="forbid"`)
- Validators: finite/non-empty metrics, `primary` ∈ `values`, non-empty `changes`, success ⇒ metrics, completed ⇒ result
- Behavior: the loop's data contracts now exist + are validated; 20 unit tests green; ruff + mypy --strict clean

**What didn't:**
- Nothing punted. `mypy src` emits a benign "unused override section" note (only one file checked) — not an error.

**Decisions:** (see RESEARCH_LOG 2026-05-24)
- `Metrics` = flexible `values` dict + `primary` + `direction` (generic across ML/LLM; stable axis for plateau detection). LLM-designed eval plans deferred to a Week 4 *tool*, never a self-authored schema.
- Nested composition (not id references) — self-contained auditable snapshot; `id` kept on each model so the Week 4 Memory store can normalize/retrieve.

**Next session (2026-05-25):**
- Day 2: `LLMClient` protocol (`src/iterate/llm/base.py`) + `OpenAICompatibleClient` against Ollama + smoke test hitting qwen2.5-coder:14b.

### 2026-05-23 | Week 0 | Project scoped, repo scaffolded

**Task:** Lock in project scope + push initial folder structure.

**What shipped:**
- Folder structure (src/, tests/, examples/, etc.)
- `.gitignore` with project-specific entries (LAUNCH_POST, PRD, BIZ, GTM, BOTTLENECKS, EVAL_LOG, PROGRESS_NOTES, data/, models/, .iterate/)
- `README.md` (public hero)
- `BUILD_LOG.md` (this file)
- `RESEARCH_LOG.md` (citation trail template)
- `pyproject.toml`, `Makefile`, `Dockerfile` (placeholder), `.env.example`
- All `__init__.py` files for the `iterate` package skeleton

**What didn't:**
- No actual `iterate` code yet — pure scaffolding.

**Decisions:**
- Name: `iterate` (open-source, single-word brandable)
- Architecture: hexagonal — core + targets + adapters + llm separated cleanly
- v1 covers BOTH `ModelTarget` (sklearn/XGBoost first) AND `PromptTarget` (LLM-as-judge)
- LLM backends pluggable from day 1 (Claude default, Llama/Deepseek via adapters)
- Memory store will use sqlite (no external infra dependency)

**Next session (2026-05-24):**
- Task #4 (Pydantic schemas) → Task #5 (LLMClient protocol) → Task #6 (Anthropic client) → Task #7 (smoke test)

---

## Commit message convention

```
<type>(<scope>): <short summary>

[optional body explaining why, what changed, and any non-obvious choices]

[optional footer — refs to RESEARCH_LOG entries, closes BOTTLENECKS#N, etc.]
```

**Types:**
- `feat:` — new functionality
- `fix:` — bug fix
- `perf:` — performance work
- `refactor:` — no behavior change
- `test:` — tests only
- `docs:` — docs only
- `chore:` — tooling, config, deps
- `research:` — RESEARCH_LOG entry only (no code, locked-in research session)

**Examples:**
- `feat(llm): anthropic client with tool-use loop helper`
- `fix(memory): retrieve_relevant returned duplicates on partial match — added DISTINCT`
- `perf(researcher): cache arxiv API results to disk (eliminated re-fetch on retry)`
- `research(targets): chose Protocol over ABC for BenchmarkTarget — see RESEARCH_LOG 2026-05-24`

---

## Backlog (lower-priority, tracked)

Items not in this week's top P0 but worth keeping visible.

- **Hard execution isolation → v0.2 (sandbox).** The v0.1 local executor does *minimal* failure capture — catch the exception, record it on `ExperimentResult.error`, keep the loop alive. Real isolation belongs with the e2b sandbox path: per-experiment **timeouts**, **memory/CPU caps**, killing runaway training, and capturing stdout/stderr into `logs`. Deferred deliberately so v0.1 ships; revisit when building the (c) sandboxed code-gen path.
- **Richer failure capture → before v0.1.** The local executor records a crash as a plain `ExperimentResult.error` string. Before the first release, enrich it for the Week-3 Memory store: a structured `FailureCase` (error_type + the offending spec) and/or captured traceback, so the Proposer reliably avoids re-proposing a known-broken change. (User call 2026-05-28: string is fine now, improve before v0.1.)
- **Known issue — LightGBM slow on macOS ARM.** The LightGBM 4.6 prebuilt pip wheel for macOS ARM is pathologically slow to *train* (~0.2s/tree, ~450x XGBoost on identical data) — independent of thread settings (`threadpool_limits`, `OMP_NUM_THREADS`) and of whether sklearn is loaded. A known wheel/`libomp` issue, not framework code; does **not** reproduce on Linux or in the e2b sandbox (where v0.2 training runs). LightGBM stays factory-supported (build-only unit test) but is omitted from the churn demo's candidate list. Local-macOS fix: rebuild from source against brew `libomp` (`uv pip install --no-binary lightgbm lightgbm`); deliberately not forced on all installs. (Diagnosed 2026-05-28.)
