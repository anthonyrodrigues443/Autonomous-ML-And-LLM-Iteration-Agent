# Build Log

> Daily task tracking for `iterate`. Build sessions log here. Recruiters reading the repo see process, not just final code.
>
> Public file. Honest about what worked + what didn't.

---

## Scope & timeline (re-planned 2026-05-27 — agent-first)

**Agent-first.** The first real release (v0.1) is a *working agentic loop* on tabular ML with explicit inputs — the LLM reads the data, proposes a change, trains, scores, and iterates to the best model by a deadline. After that, two dials turn release to release: **(A) inputs required shrink** (toward one-sentence input) and **(B) problem types grow** (tabular → prompts → DL/vision). The agent is present from v0.1; everything later is capability expansion — not "turn the agent on at Week 7."

*(Supersedes the earlier breadth-first ordering, which wrongly deferred the agentic loop. ~11-week build, May 23 – ~Aug 8.)*

- **Targets:** `ModelTarget` (tabular ML) · `PromptTarget` (production LLM prompts, prompt-iteration only) · `DLModelTarget` (vision, transfer learning — validated on local RTX 4050).
- **Moat — the specialized combination, not one feature:** a domain specialist for ML/DL/prompt iteration that does *together* what no single tool does — agentic iteration across **ML + DL models AND LLM prompts** · **persistent memory** (revisits past failures when conditions change) · **literature-grounded** proposals · **bounded autonomy** + human-approval gates · **auditable reasoning trail** · **cost-constrained optimization** (best score you can *afford to serve* — cheapest cloud, $/mo, req/hr) · **rich auto-discovered context** (DB / MCP / Drive). Cost-aware serving is the flagship for cost-sensitive startups; the moat is the *combination* + the specialization. (Full matrix: README comparison table.)
- **Compute:** pluggable backend — local MPS · RTX 4050 (GPU validation) · e2b · cloud-GPU adapter.

| Wk | Phase |
|---|---|
| 1 | Foundation — schemas + LLM client (tool-calling) + config + CLI · done |
| 2 | Tabular execution substrate — `BenchmarkTarget` + data adapter + `ModelTarget` (sklearn/XGBoost) + local executor |
| 3 | **The agentic loop** — Proposer + Orchestrator + Terminator + Memory → first autonomous tabular run (**v0.1**) |
| 4 | Dial A: agent picks the metric + starting model (basic research) → **v0.2** |
| 5 | Dial B: `PromptTarget` — agentic prompt iteration → **v0.3** |
| 6 | Dial B: `DLModelTarget` — vision transfer learning (4050) → **v0.4** |
| 7 | Cost-constrained recommendation + serving profile + `iterate cost` → **v0.5** |
| 8 | Dial A: infer features/target from the data + a description → **v0.6** |
| 9 | Dial A: MCP discovery — find the data/code itself → **v0.7** |
| 10 | Multi-backend benchmark + Streamlit UI + demos → **v0.8 / v0.9** |
| 11 | Full minimum-viable-input + polish + launch → **v1.0** |

### Releases (incremental — ship a working slice, then iterate)

Semantic versioning: `0.x` = early/evolving, `1.0.0` = the full v1 vision. **The agentic loop is present from v0.1**; two dials then turn — inputs you must give *shrink*, problem types *grow*. Tag a GitHub release at each milestone; publish to PyPI from v0.1.0.

| Version | After | Problem types | Inputs you give (shrinking →) |
|---|---|---|---|
| v0.1.0 | Week 3 | tabular | data + features + target + metric + baseline/notebook + deadline — **agentic loop on** |
| v0.2.0 | Week 4 | tabular | data + features + target + baseline + deadline  *(agent picks metric + starting model)* |
| v0.3.0 | Week 5 | + prompts | prompt + eval set + deadline |
| v0.4.0 | Week 6 | + DL / vision | data + target + deadline |
| v0.5.0 | Week 7 | all three | + serving budget / cloud  *(cost-constrained recommendation)* |
| v0.6.0 | Week 8 | all | data + a one-line description  *(infers features/target/metric)* |
| v0.7.0 | Week 9 | all | one sentence + a data source  *(MCP finds the data/code)* |
| v0.8–0.9 | Week 10 | all | + multi-backend benchmark + Streamlit UI |
| v1.0.0 | Week 11 | all | one sentence  *(full discovery)* |

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
| 3 | `ModelTarget` (sklearn baseline) — wraps dataset + model + metric; `baseline()` train + score → `Metrics` | `src/iterate/targets/model.py` + tests | todo |
| 4 | Model adapters — sklearn + XGBoost; build a model from `Candidate.changes` (hyperparams / features) | `src/iterate/adapters/models/` + tests | todo |
| 5 | Local executor — run one `Experiment`: build candidate → train → score → `ExperimentResult` | `src/iterate/adapters/compute/local.py` + tests | todo |
| 6 | Substrate end-to-end on churn — `baseline()` + `run(supplied candidate)` → result (not yet agent-driven) | `examples/churn_tabular/` + integration test | todo |
| 7 | Polish + Week 2 retro (BUILD_LOG) | wrap-up | todo |

**Slack:** 1 day.

---

## Week 3 Day-by-Day Plan — The agentic loop (→ v0.1 agentic tabular)

**Week goal:** close the loop. The LLM autonomously proposes the next candidate, runs it on the Week-2 substrate, scores it, records it, and decides whether to continue — until a deadline / plateau. The first fully autonomous tabular run. Inputs still explicit (data, features, target, metric, baseline/notebook, deadline).
**Target window:** ~Jun 8–14 (log by real date).

| Day | Focus | Lands | Done? |
|---|---|---|---|
| 1 | Proposer — LLM proposes the next `Candidate` from the data summary + baseline + past results (+ optional source notebook) | `src/iterate/core/proposer.py` + tests | todo |
| 2 | Orchestrator — the loop: `baseline()` → propose → `run()` → score → record → decide → repeat | `src/iterate/core/orchestrator.py` + tests | todo |
| 3 | Terminator — stop on deadline / patience / plateau (minimal) | `src/iterate/core/terminator.py` + tests | todo |
| 4 | Memory — record every experiment; feed history to the Proposer; avoid repeats (sqlite, minimal) | `src/iterate/core/memory.py` + tests | todo |
| 5 | CLI `iterate run` — explicit inputs: `--data --features --target --metric --baseline/--notebook --until/--patience` | `src/iterate/cli.py` + tests | todo |
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

(empty)
