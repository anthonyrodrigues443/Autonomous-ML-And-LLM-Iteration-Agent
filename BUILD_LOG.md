# Build Log

> Daily task tracking for `iterate`. Build sessions log here. Recruiters reading the repo see process, not just final code.
>
> Public file. Honest about what worked + what didn't.

---

## Scope & timeline (re-planned 2026-05-27 — agent-first)

**Agent-first.** The first real release (v0.1) is a *working agentic loop* on tabular ML with explicit inputs — the LLM reads the data, proposes a change, trains, scores, and iterates to the best model by a deadline. After that, two dials turn release to release: **(A) inputs required shrink** (toward one-sentence input) and **(B) problem types grow** (tabular → prompts → DL/vision). The agent is present from v0.1; everything later is capability expansion — not "turn the agent on at Week 7."

*(Supersedes the earlier breadth-first ordering, which wrongly deferred the agentic loop. ~12–13-week build, May 23 – ~late Aug.)*

**Updated 2026-05-28 — model flexibility has two tiers, and the second is bumped early:**
- **(b) Installed-library factory (v0.1):** a candidate names any estimator in an allow-listed installed library (scikit-learn / XGBoost / LightGBM) by import path + params — `{"model": "lightgbm.LGBMClassifier", "params": {…}}` — and we instantiate it. No hand-curated list.
- **(c) Sandboxed code-gen (v0.2, bumped early):** the Proposer *writes* the training code and runs it in an e2b sandbox → **any model at all**, not just installed libraries. This is the big capability and now lands right after the v0.1 loop, displacing the old v0.2 (agent picks metric+model → now v0.3) and shifting later versions down by one.

- **Targets:** `ModelTarget` (tabular ML) · `PromptTarget` (production LLM prompts, prompt-iteration only) · `DLModelTarget` (vision, transfer learning — validated on local RTX 4050).
- **Moat — the specialized combination, not one feature:** a domain specialist for ML/DL/prompt iteration that does *together* what no single tool does — agentic iteration across **ML + DL models AND LLM prompts** · **persistent memory** (revisits past failures when conditions change) · **literature-grounded** proposals · **bounded autonomy** + human-approval gates · **auditable reasoning trail** · **cost-constrained optimization** (best score you can *afford to serve* — cheapest cloud, $/mo, req/hr) · **rich auto-discovered context** (DB / MCP / Drive). Cost-aware serving is the flagship for cost-sensitive startups; the moat is the *combination* + the specialization. (Full matrix: README comparison table.)
- **Compute:** pluggable backend — local MPS · RTX 4050 (GPU validation) · e2b · cloud-GPU adapter.

| Wk | Phase |
|---|---|
| 1 | Foundation — schemas + LLM client (tool-calling) + config + CLI · done |
| 2 | Tabular execution substrate — `BenchmarkTarget` + data adapter + `ModelTarget` + model factory (any installed sklearn/XGBoost/LightGBM estimator) + local executor |
| 3 | **The agentic loop** — Proposer + Orchestrator + Terminator + Memory → first autonomous tabular run (**v0.1**) |
| 4–5 | **Sandboxed code-gen** — the Proposer *writes* training code, runs it in an e2b sandbox → **any model at all**, not just installed libraries → **v0.2** |
| 6 | Dial A: agent picks the metric + starting model (basic research) → **v0.3** |
| 7 | Dial B: `PromptTarget` — agentic prompt iteration → **v0.4** |
| 8 | Dial B: `DLModelTarget` — vision transfer learning (4050) → **v0.5** |
| 9 | Cost-constrained recommendation + serving profile + `iterate cost` → **v0.6** |
| 10 | Dial A: infer features/target from the data + a description → **v0.7** |
| 11 | Dial A: MCP discovery — find the data/code itself → **v0.8** |
| 12 | Multi-backend benchmark + Streamlit UI + demos → **v0.9** |
| 13 | Full minimum-viable-input + polish + launch → **v1.0** |

### Releases (incremental — ship a working slice, then iterate)

Semantic versioning: `0.x` = early/evolving, `1.0.0` = the full v1 vision. **The agentic loop is present from v0.1**; two dials then turn — inputs you must give *shrink*, problem types *grow*. Tag a GitHub release at each milestone; publish to PyPI from v0.1.0.

| Version | After | Problem types | Inputs you give (shrinking →) |
|---|---|---|---|
| v0.1.0 | Week 3 | tabular | data + features + target + metric + baseline/notebook + deadline — **agentic loop on** (any installed-library model via the factory) |
| v0.2.0 | Week 4–5 | tabular | *(same inputs)* — agent **writes & runs training code in a sandbox** → any model at all, not just installed libs |
| v0.3.0 | Week 6 | tabular | data + features + target + baseline + deadline  *(agent picks metric + starting model)* |
| v0.4.0 | Week 7 | + prompts | prompt + eval set + deadline |
| v0.5.0 | Week 8 | + DL / vision | data + target + deadline |
| v0.6.0 | Week 9 | all three | + serving budget / cloud  *(cost-constrained recommendation)* |
| v0.7.0 | Week 10 | all | data + a one-line description  *(infers features/target/metric)* |
| v0.8.0 | Week 11 | all | one sentence + a data source  *(MCP finds the data/code)* |
| v0.9.0 | Week 12 | all | + multi-backend benchmark + Streamlit UI |
| v1.0.0 | Week 13 | all | one sentence  *(full discovery)* |

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
- **Inputs:** `--data` + `--target` (required) · `--metric` (required in v0.1 — agent *choosing* the metric is v0.3; may default by task type) · `--baseline` + `--source` notebook/md/txt (optional; source is **read as text** by the LLM to **reconstruct the baseline approach** — model + params + key preprocessing — which we run through our own eval for a comparable re-measured baseline; the user's actual code is **never executed**) · `--until` / `--patience` (optional; **patience** is the primary stop, deadline a backstop). Features auto-derive as all columns except target.
- **Working:** measure *our own* baseline (from the factory's default, or reconstructed from `--source` if given) → loop { propose → train → score on the sealed holdout → record → decide } until plateau or deadline. v0.1 candidate space = any installed allow-listed estimator (sklearn / XGBoost / LightGBM) via the model factory, named by `{"model","params"}`.
- **Output:** the **best model** (artifact + the winning config) + its score vs the **baseline we measured** + an **auditable report** of every experiment and why the winner won. Agent proposes; human reviews.
- **NOT in v0.1:** arbitrary/uninstalled models via sandboxed code-gen (v0.2) · the agent picking the metric (v0.3) · cost-constraint / serving profile (v0.6) · **executing user-provided source code — ever** (permanent security policy; the e2b sandbox at v0.2 runs the agent's OWN generated code, never the user's).

| Day | Focus | Lands | Done? |
|---|---|---|---|
| 1 | Proposer (+ native `OllamaClient` adapter for `think:false` + centralized `prompts.yaml`) — LLM proposes the next `Candidate` via a `propose_candidate` tool call from data summary + baseline + history | `core/proposer.py` + `llm/ollama_client.py` + `prompts/` + tests | done |
| 2 | Orchestrator — the loop: `baseline()` → propose → `run()` → score → record → decide → repeat (in-memory history; internal stop logic) | `src/iterate/core/orchestrator.py` + tests | done |
| 3 | Terminator — stop on deadline / patience / plateau via a delegated protocol; Orchestrator refactored to delegate | `src/iterate/core/terminator.py` + tests | done |
| 4 | Memory — record every experiment; feed **cross-run** history to the Proposer (sqlite + in-memory; structured proposer-failure records) | `src/iterate/core/memory.py` + tests | done |
| 5 | CLI `iterate run` (+ `--backend` factory) **and source-aware baseline reconstruction** — LLM reads `--source` md/txt/notebook as **text only** (never executes), rebuilds the approach as a spec, we run it through our eval → re-measured baseline | `src/iterate/cli.py` + reconstruction module + tests | todo |
| 6 | First autonomous tabular run on churn — LLM iterates to best by deadline → tag **v0.1.0** | `examples/churn_tabular/` + integration test | todo |
| 7 | Polish + Week 3 retro + release **v0.1.0** | wrap-up | todo |

**Slack:** 1 day.

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
