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
- Everything that was v0.3+ shifts by one version. The Streamlit UI becomes v0.10 (its main "interactive interface" value is covered by v0.3's CLI; v0.10 carries demos, multi-backend benchmark, polish). Build is now ~14 weeks (was ~13). *(Superseded 2026-07-25: 16 plan-weeks; see the update below.)*

**Updated 2026-06-01 — going multi-agent after v0.2 (at the Researcher milestone):**
- v0.1 and v0.2 stay **single-agent** (one Proposer LLM in a deterministic loop). The architecture moves to **multi-agent at the Researcher milestone (v0.4)** — the natural single-to-multi transition, where the second genuine LLM role appears.
- Shape: **specialist agents** (Researcher, Proposer, a Critic/Reviewer, later Discovery), each doing one focused job and handing **structured, typed output** to a **supervisor agent** that makes the decisions. Rationale: specialization raises per-agent tool-call reliability, and the supervisor reasons over digested high-quality context instead of raw everything. Our Pydantic schemas are the handoff contracts.
- Built on our own harness (no LangGraph — Week-1 decision stands). Executor + Memory stay deterministic; the supervisor takes the judgment calls. See DECISIONS.md.

**Updated 2026-07-25: post-v0.2 re-plan (v0.3 through v1.0 scoped day-by-day).**
- v0.2.0 shipped 2026-07-18 (PyPI + tag + GitHub release; v0.2.1 the same night; launch posts scheduled 2026-07-19). v0.2 grew far beyond its planned row: the multi-agent core (Supervisor + CodingAgent + Summarizer), cell-by-cell kernel sessions, the deterministic guard stack, and a release gated on a certified trajectory quality bar.
- Because the multi-agent core landed IN v0.2, v0.4 is no longer "go multi-agent". It graduates the remaining specialists (Researcher, Critic) at the existing `plan_next` tool boundary, hardens evaluation (probability metrics, CV option, leakage checks), and turns the first input dial (agent picks metric + starting model).
- Plan-weeks stopped tracking calendar weeks long ago (Weeks 1-3 took 8 calendar days; Weeks 4-5 took ~7 calendar weeks). The re-numbered tables below keep plan-weeks as scope units and add target calendar windows as ranges. Windows assume 1-2 sessions per calendar week during the semester and more during breaks; re-anchor at every release; public ETAs stay ranges, never hard dates.
- v0.4 and v0.9 get two plan-weeks each (the Week 4-5 lesson: a milestone carrying many tracked commitments overruns a single week).
- Day-by-day plans for Weeks 6 through 16 added below, plus a standing release checklist and a backlog disposition list (every deferred item re-homed or explicitly closed). Per-section Target window lines are retired in favor of the central releases table. Later plans firm up right before execution; expect reconciliation notes like Week 1's.

**Updated 2026-07-25 (cont): compressed to a 6-week Sunday-release sprint (Tony's call, same day).**
- One release every Sunday, project complete 2026-09-06: v0.3 Jul 26 · v0.4 Aug 2 · v0.5 Aug 9 · v0.6 Aug 16 · v0.7 Aug 23 · v0.9 Aug 30 (absorbs v0.8) · v1.0 Sep 6 (absorbs v0.10).
- Two merges make 9 milestones fit 7 Sundays: the v0.8 input-inference dial ships inside the v0.9 discovery release, and the v0.10 benchmark/dashboard/reporter ship inside v1.0. Version numbers v0.8.0 and v0.10.0 are skipped as standalone tags.
- The cadence is ~1 build day per calendar day, so each sprint day carries roughly two of the earlier plan-days. What makes Sundays real is the cut lists: every week names what ships and what moves to the post-v1.0 backlog. Anything unfinished on Saturday rolls forward; the Sunday release ships whatever passed the gate, and release notes state exactly what made it.
- The week sections below are rewritten to sprint calendar dates; the earlier scope-unit week numbering (6-16) is retired. v0.3 builds TODAY (Sat 2026-07-25) and releases tomorrow.

- **Targets:** `ModelTarget` (tabular ML) · `PromptTarget` (production LLM prompts, prompt-iteration only) · `DLModelTarget` (vision, transfer learning — validated on local RTX 4050).
- **Moat — the specialized combination, not one feature:** a domain specialist for ML/DL/prompt iteration that does *together* what no single tool does — agentic iteration across **ML + DL models AND LLM prompts** · **persistent memory** (revisits past failures when conditions change) · **literature-grounded** proposals · **bounded autonomy** + human-approval gates · **auditable reasoning trail** · **cost-constrained optimization** (best score you can *afford to serve* — cheapest cloud, $/mo, req/hr) · **rich auto-discovered context** (DB / MCP / Drive). Cost-aware serving is the flagship for cost-sensitive startups; the moat is the *combination* + the specialization. (Full matrix: README comparison table.)
- **Compute:** pluggable backend — local MPS · RTX 4050 (GPU validation) · e2b · cloud-GPU adapter.

| Wk | Phase |
|---|---|
| 1 | Foundation: schemas + LLM client (tool-calling) + config + CLI · done |
| 2 | Tabular execution substrate: `BenchmarkTarget` + data adapter + `ModelTarget` + model factory + local executor · done |
| 3 | **The agentic loop**: Proposer + Orchestrator + Terminator + Memory · **v0.1.0 shipped 2026-05-31** |
| 4-5 | **Sandboxed code-gen + the multi-agent core**: cell-by-cell kernel sessions (LocalKernel/E2BKernel), Supervisor + CodingAgent + Summarizer, deterministic guard stack, certified quality bar, notebook deliverable, per-cell progress + graceful Ctrl-C · **v0.2.0 shipped 2026-07-18** |
| Jul 25 | **Interactive CLI**: pause, mid-run chat, resume (streaming stretch) → **v0.3 · Sun 2026-07-26** |
| Jul 27 to Aug 1 | **Specialists + eval hardening**: Researcher + Critic, dossier + lean ledger, probability metrics, agent picks the metric + starting model → **v0.4 · Sun 2026-08-02** |
| Aug 3-8 | Dial B: `PromptTarget`, agentic prompt iteration (toxicity + intent examples) → **v0.5 · Sun 2026-08-09** |
| Aug 10-15 | Dial B: `DLModelTarget`, vision transfer learning (4050) → **v0.6 · Sun 2026-08-16** |
| Aug 17-22 | Cost-constrained recommendation + serving profile + `iterate cost` → **v0.7 · Sun 2026-08-23** |
| Aug 24-29 | Dial A, both steps: infer inputs from data + a description AND MCP discovery (filesystem/Postgres, gap-fill pause) → **v0.9 · Sun 2026-08-30** *(absorbs v0.8)* |
| Aug 31 to Sep 5 | One-sentence input + benchmark + dashboard + Reporter + docs + launch → **v1.0 · Sun 2026-09-06** *(absorbs v0.10)* |

### Releases (incremental — ship a working slice, then iterate)

Semantic versioning: `0.x` = early/evolving, `1.0.0` = the full v1 vision. **The agentic loop is present from v0.1**; two dials then turn — inputs you must give *shrink*, problem types *grow*. Tag a GitHub release at each milestone; publish to PyPI from v0.1.0.

| Version | Plan week | Target window | Problem types | Inputs you give (shrinking →) / New capability |
|---|---|---|---|---|
| v0.1.0 | 3 | **RELEASED 2026-05-31** | tabular | data + features + target + metric + baseline/notebook + deadline: **agentic loop on** (allow-listed installed models via the factory; joblib artifact) |
| v0.2.0 | 4-5 | **RELEASED 2026-07-18** | tabular | *(same inputs)*: agent **writes + runs its own training code cell-by-cell** (local kernel default, e2b sandbox flag); Supervisor + CodingAgent + Summarizer; guard stack + certified quality bar; notebook deliverable; per-cell progress + graceful Ctrl-C |
| v0.3.0 | sprint 1 | **Sun 2026-07-26** | tabular | *(same inputs)*: **full interactive CLI** (pause the loop, chat with the agent, resume); token streaming is the stretch item |
| v0.4.0 | sprint 2 | **Sun 2026-08-02** | tabular | data + features + target + baseline + deadline  *(Researcher + Critic specialists at the tool boundary; agent picks metric + starting model from research; probability metrics)* |
| v0.5.0 | sprint 3 | **Sun 2026-08-09** | + prompts | prompt + eval set + deadline |
| v0.6.0 | sprint 4 | **Sun 2026-08-16** | + DL / vision | data + target + deadline  *(validated on the RTX 4050)* |
| v0.7.0 | sprint 5 | **Sun 2026-08-23** | all three | + serving budget / cloud  *(cost-constrained recommendation + serving profile + `iterate cost`)* |
| v0.9.0 | sprint 6 | **Sun 2026-08-30** | all | data + a one-line description OR one sentence + a data source  *(absorbs v0.8: infers features/target/metric with a confirm pause; MCP discovery over filesystem + Postgres with the gap-fill pause)* |
| v1.0.0 | sprint 7 | **Sun 2026-09-06** | all | one sentence  *(absorbs v0.10: multi-backend benchmark + read-only dashboard + prose report; full discovery + docs + launch)* |

Sprint arithmetic, stated so it can be checked: 7 releases in 43 calendar days means ~1 build day per calendar day around college (9-5) and Keeper (6pm-2am); weekday build slots are late-night, weekends carry the heavy days. The compression is bought with the per-week cut lists below (cut items go to the post-v1.0 backlog, and LIMITATIONS.md states each cut honestly at release time). A slipped day rolls into that week's Saturday; a slipped WEEK does not move the Sunday: the release ships whatever passed the gate and the release notes say what made it. README's public status table gets synced to this calendar at the v0.3 release.

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

**Note (2026-05-25):** Week 1's foundation — schemas + LLM client + config + CLI — shipped in **Days 1–3** (ahead of plan). The original Days 4–7 (tool dispatcher, Anthropic adapter, memory skeleton) were superseded by the expanded 11-week plan: memory + proposer + researcher → **Week 7**; tool dispatcher → **Week 7** (orchestrator); Anthropic adapter → optional/later. Week 1 is effectively complete; next is Week 2. *(Those "Week 7" pointers are the old 11-week numbering; the items shipped in Weeks 3-4. Under the 2026-07-25 sprint calendar the Anthropic adapter lands in sprint 7, Tue 2026-09-01.)*

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

## Standing release checklist (every vX.Y from here on)

Learned from the v0.2 release arc (release mechanics alone took 11 calendar days when improvised):

1. **Release gate:** no new roadmap features ride along a release-gate iteration (locked 2026-07-04). If the release touched the loop, re-run the trajectory quality bar on the floor model before tagging.
2. **Build gate:** full unit suite + ruff + mypy --strict green; `make build`.
3. **Doc sync:** README (status table, test count, validated model, shipped-vs-planned rows), LIMITATIONS.md (retire fixed rows, add honest new ones), examples/ READMEs. LIMITATIONS.md explicitly pairs with this roadmap; keeping it current is release work, not optional polish.
4. **Version mechanics:** bump pyproject + `__init__` + lockfile, tag, `uv publish` (Tony runs it), GitHub release notes (feature-first, honest capability floor).
5. **Launch assets:** demo recorded from the published pip package; posts drafted in LAUNCH_POST.md as a vPREV-to-vX.Y diff (what it does, what changed, honest does-NOT-do list, what is next, repo link; no em dashes; max 275 chars per tweet; at most one process stat woven into a feature story). Tony schedules via native platform schedulers for US mornings.
6. **Post-launch listening window:** the v0.1 replies produced two roadmap decisions; budget one session to read and log reactions.

---

## Sprint 1: v0.3, the interactive CLI. Build Sat 2026-07-25, release Sun 2026-07-26

**Goal:** the public v0.3 promise from the launch posts, built today, shipped tomorrow: pause a live run, talk to the agent mid-loop, resume. Ordered must-ship-first; the day ends wherever it ends and Sunday ships what passed the gate.

**v0.3 contract (updated same day after Tony's design pass):**
- **Scope:** the default supervised code path only. The frozen `--spec` fast lane keeps its non-interactive behavior (documented, not built).
- **Chat is plain English, non-modal, queued:** the user types free sentences ANY time; while a cell or an LLM call is in flight they queue; delivery happens at the next boundary (after a cell finishes, or just before the next supervisor call). No command syntax: `pause`, `resume`, `stop` match as bare English words; everything else is interpreted. Line-based input means no raw terminal mode at all.
- **Intent is interpreted by the supervisor, routing is executed by the harness:** every Enter gets an INSTANT timing-only ack driven by a shared run-status field ("got it, cell 7 executing, delivering at the next safe point"), which is all the harness truthfully knows at that moment. At the boundary the supervisor runs one small structured `route_message` call (question | steer this notebook | steer later | standing rule) on the no-think client; the harness then moves the message: questions → the Q&A turn, standing rules → the capped rules list, current-notebook steers → injected into the live session (budget-nudge pattern, one capped user-role line) plus visibility at the next planning turn, later-steers → the next brief. On a failed classification after one retry the message defaults to the least-destructive route (current-notebook steer + next-brief visibility) and the console says so; interpretation degrades, never crashes.
- **Pause point is the boundary:** on `pause` the coder finishes the executing cell, then the loop parks in place (kernel alive with periodic e2b keepalive, Memory already durable). No mid-cell kernel interrupt (an interrupt is a kill, not a pause).
- **Clocks:** the run deadline and the 30-minute session wall ceiling suspend while paused; the kernel-time budget is naturally safe (it charges kernel seconds only).
- **Two kinds of guidance, both lean:** a plain message is a one-shot steer (current session note + next `plan_next` turn, one capped line each); `rule: <text>` is a standing instruction (e.g. "do not test lever X") rendered as a capped list (3 items max, dead-ends style) in EVERY subsequent planning turn, so it shapes all future notebooks, and stamped into `candidate.changes` for audit. Mid-session notes are permanent session context (elision trims tool outputs only), hence the hard caps; the EDA-ledger regression is the cautionary tale, and the validation run watches a steered session for quality collapse.
- **Questions get answered from the notebooks, safely:** queued questions trigger a SEPARATE supervisor Q&A turn at the boundary with a `read_notebook` tool. The tool is backed by Memory's stored cells + digests (sqlite), not by parsing .ipynb files, which honors the 2026-06-03 "the agent never re-parses notebooks" decision. The Q&A turn has a capped tool budget, degrades to "cannot answer" instead of raising, prints to the console, and its context NEVER enters `plan_next` (the supervisor's single structured tool call stays uncrowded).
- **Guards outrank guidance:** a guidance-induced duplicate still gets rejected and stamped; chat can never unseal the holdout or bypass a gate. Guard-veto reasons already print at INFO, so the user sees why a steer was overridden.
- **The default face is a terminal UI (scope added same day, built):** a scrollable log pane + a pinned always-yours input box (Textual), title bar with the run config; `--plain` keeps line-mode chat; non-tty and BACKGROUNDED runs stay non-interactive (foreground-tty gated + SIGTTIN-safe, so `iterate run ... &` cannot freeze). Ctrl-C in the TUI means graceful stop, never a hard quit that abandons a live kernel.
- **Adversarially reviewed before the release (4-lens workflow):** caught and fixed two blockers (a backgrounded run would have been SIGTTIN-suspended by the stdin listener; Q&A resolved iteration numbers against ALL runs in memory instead of the current run) plus stop-then-pause deadlock, markup-mangled replies, e2b keepalive starvation around Q&A drains, second-note truncation, stop-reason mislabeling, and guidance lost on a supervisor retry. 424 unit tests (was 391), ruff + mypy --strict clean.

**Today, Sat 2026-07-25 (ordered blocks, must-ship first):**

| # | Block | Lands | Done? |
|---|---|---|---|
| 0 | Ten minutes of small calls: flip saved default backend groq → ollama; decide `--compute`-twice (warn or leave) and dataset-switch (document only for now); confirm the v0.2 posts went out; drop a superseded-by-BUILD_LOG header note into PRD.md | config + PRD note | |
| 1 | Controller + pause/resume: `RunController` (message queue, standing rules, pause/abort flags, paused-seconds accounting, shared run-status field driving instant routing acks) + daemon stdin LINE reader (commands vs messages, prints the ack); checkpoint hook at the coder cell boundary and the loop iteration boundary; deadline + wall-ceiling suspension; `keepalive()` on the kernel protocol (e2b lease outlives a long pause) | `core/interactive.py` + kernel/coder/agent_loop wiring + tests | |
| 2 | Guidance delivery, both routes: current-session injection at the coder cell boundary (budget-nudge pattern, one capped user-role note) + one-shot steer line and standing-rules list into `decide()` via the typed seam; prompts.yaml guidance keys + one precedence sentence; stamp applied guidance into `candidate.changes` | coder + supervisor + prompts + tests | |
| 3 | Supervisor interpreter + Q&A turn: `route_message` structured tool (question / steer-now / steer-later / standing rule, one retry then safest-default) and `answer()` with the Memory-backed `read_notebook` tool (list experiments, fetch cells + digest of one, capped rendering), max 2 tool calls then forced answer, console output, zero bleed into `plan_next` | supervisor interpreter + Q&A + tests | |
| 4 | CLI wiring: construct controller + line reader in the supervised branch, thread into `make_coder` + `run_supervised`, print queued-delivery acks and Q&A answers via the shared console; integration test on deterministic fakes | CLI + tests | |
| 5 | Floor-model validation run on gemma4:12b exercising queue-while-busy, pause/resume, a steer, a standing rule honored in the NEXT notebook, a notebook question answered; check trajectory criteria hold | validation notes + fixes | |
| 6 | STRETCH, only if the day still has legs (first casualty): token streaming on the native `OllamaClient`; `--debug` flag; notebook-header absolute-score item | stream + small wins | |

**Tomorrow, Sun 2026-07-26 (release day, standing checklist compressed):** build gate (tests + ruff + mypy, `make build`) → doc sync (README status table to this sprint calendar, LIMITATIONS updated incl. honest cut notes, EVAL_LOG pointer fixed: publish a backfilled public copy or repoint) → bump + tag + `uv publish` + GitHub release (feature-first) → demo recorded from the pip package → posts drafted in LAUNCH_POST (vPREV-to-v0.3 diff, honest does-NOT-do incl. anything that slipped) and scheduled in the night window for US Monday morning.

**Cut from v0.3 (to the post-v1.0 backlog unless a later Saturday absorbs them):** `OpenAICompatibleClient` streaming, LocalKernel IPC transport, the remaining certification polish items, pre-run undefined-name lint, e2b egress-deny template + per-experiment caps, BOTTLENECKS/EVAL_LOG cadence decision (retire recommended, decide any Sunday), cross-run persistence of standing rules (v0.3 rules are run-scoped; surviving a process restart needs Memory schema and waits).

---

## Sprint 2: v0.4, specialists + eval hardening. Mon 2026-07-27 to Sat 2026-08-01, release Sun 2026-08-02

**Goal:** the two remaining specialists (Researcher, Critic) graduate at the supervisor's `plan_next` tool boundary, probability metrics land, and the first input dial turns (agent picks the metric + starting model from research). Public commitments riding on this release: "Literature-aware proposals" and "Researcher + Critic specialists; agent picks the metric + starting model" (README). This is the heaviest sprint; the cut list below is what makes it fit one week.

**Contract:** specialists are separate LLM roles with typed handoffs, graduating from supervisor tools without contract changes (the 2026-06-04 seam). Own harness, no LangGraph, ever. New supervisor context stays lean (watch the validation run for lever collapse). Eval changes respect the sealed holdout. Researcher citations must be genuine: `Candidate.citations`, `source="researcher"`, dedup against Memory so it neither re-reads papers nor re-runs known ideas.

**Carry-ins from the v0.3 live drive (taken same-day as v0.3.1 instead — see the release entry above):** a live run lost iteration 5 to a timeout spiral: a HistGB fit hit the 120s per-cell cap (the Week-2 thread-oversubscription class — agent-generated cells run in a raw kernel with no thread cap, unlike the v0.1 spec path), the coder retried the IDENTICAL fit, the canned floor trains the same family so the submission guarantee timed out too, and the failure recorded as "no predictions file produced", which teaches the next iteration nothing.
1. Kernel session preamble caps BLAS/OpenMP threads (the v0.1 `threadpool_limits` lesson applied to generated code) — kills the whole oversubscription timeout class.
2. Timed-out cells get their own deterministic nudge (names the operation, forbids an identical retry, demands a cheaper model or a subsample) and count toward the same-error breaker.
3. A session that dies records WHY (which cells timed out or errored, on what operation), so the failure feeds the supervisor's REPAIR rung, the dead-ends channel, and the digest — instead of a contract one-liner.
4. The canned floor submission switches to a fast estimator (logistic regression) so the safety net can never time out.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Jul 27 | Metric registry replaces the fixed 8-metric panel + probability capture through the predictions contract + ROC-AUC / log-loss / PR-AUC + configurable averaging; plumbed through CLI, briefs, notebook headers | scoring + contract + tests | done |
| Tue Jul 28 | Experiment dossier, deterministic (distill captured cell stdout into a structured per-experiment record feeding briefs) + lean tried/untried idea ledger on `components_used` (capped, out of the dense supervisor prompt; watch for the EDA-ledger regression pattern) | dossier + ledger + tests | done |
| Wed Jul 29 | Researcher: arxiv + papers-with-code clients (cached to disk) + the agent itself (goal + data profile + history → grounded technique suggestions with citations, consumed at the tool boundary) | `core/researcher.py` + adapters + tests | done |
| Thu Jul 30 | Critic: generated-code review for subtle leakage (fit-on-train-only, target leakage in FE) as a typed pre-execution check + eval-hardening verdicts; Summarizer graduates to author the dossier (Tue's deterministic distiller becomes its input and fallback) | `core/critic.py` + tests | done (Summarizer-authors-dossier deferred, see entry) |
| Fri Jul 31 | Dial A: agent picks the metric + starting model from research + the data profile; `--metric` optional; the RESEARCHER picks it (not the supervisor — the metric must exist before ModelTarget, which exists before the baseline, which is an argument to decide(), so at choosing time the supervisor has no baseline or history to reason from); validated against the registry for name and task, defaulting deterministically from the target dtype when research is unavailable; FIXED at run start and never changed mid-run, since one ruler is what makes history and cross-run baselines comparable; free unscored inspect/EDA step so exploration stops costing a scored iteration | proposer/supervisor + CLI + loop + tests | done (free inspect step cut, see entry) |
| Sat Aug 1 | Certification-style validation on gemma4:12b (bar criteria + citations genuine + Critic catches seeded leakage + no context regression) + fixes; buffer absorbs anything slipped Mon-Fri | validation + fixes | done |
| Sun Aug 2 | **Release v0.4.0** per the standing checklist; LIMITATIONS retires the metric-panel, proba-metrics, and averaging rows and states the CV cut | v0.4.0 out | done |

**Cut from v0.4 (post-v1.0 backlog):** CV/k-fold selection option, typed Session handoff (`_winning_code` blob stays), qwen3:14b re-run, `iterate history`/`best`/`why-failed` (any green Saturday can absorb these).

---

## Sprint 3: v0.5, PromptTarget. Mon 2026-08-03 to Sat 2026-08-08, release Sun 2026-08-09

**Goal:** the second problem type. Prompts as a `BenchmarkTarget`: same iteration loop, different execution path. Two public example commitments come due: `examples/toxicity_jigsaw/` and `examples/intent_clinc150/`. Prompt iteration only, never foundation-model fine-tuning (permanent scope lock).

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 3 | **Eval suite FIRST** (Tony's call after the v0.4 certification): the headroom table from EVAL_LOG becomes a runnable corpus, so a release is measured rather than argued about. Then the v0.5 work below | `evals/` + corpus + runner |
| Tue Aug 4 | Loop integration: prompt lever classes for the supervisor ladder, coder session writes prompt variants + scoring cells, guard stack audited for the new path (duplicate gates hash prompt text, dead-ends transfer) | wiring + tests | |
| Wed Aug 5 | `examples/toxicity_jigsaw/`: Jigsaw toxic-comment prompt iteration end-to-end | example + integration test | |
| Thu Aug 6 | `examples/intent_clinc150/`: CLINC150 intent classification; genericity fixes the second prompt target surfaces | example + tests | |
| Fri Aug 7 | Floor-model validation on the prompt path; demo-clean pass | validation | |
| Sat Aug 8 | Buffer + carried items from earlier cut lists if green | fixes | |
| Sun Aug 9 | **Release v0.5.0** per the standing checklist; examples/ README de-placeholdered; "CSV path until v0.5/v0.6" rows updated | v0.5.0 out | |

---

## Sprint 4: v0.6, DLModelTarget vision. Mon 2026-08-10 to Sat 2026-08-15, release Sun 2026-08-16

**Goal:** the third problem type. Vision transfer learning validated on the RTX 4050 (the public claim), with GPU-aware execution. Line up 4050 access for Wed-Fri NOW; MPS is the smoke path, not the claim.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 10 | Recipe research (torchvision/timm backbone, freeze + head; dataset that trains on a 6 GB card; MPS vs CUDA) + vision data adapter (image folders behind the existing data seam); RESEARCH_LOG entry | adapter + RESEARCH_LOG | |
| Tue Aug 11 | `DLModelTarget` baseline (frozen backbone + linear head, deterministic) | `targets/dl.py` + tests | |
| Wed Aug 12 | Candidate space via the code path: agent writes training cells (unfreeze depth, lr schedule, augmentation); GPU-aware kernel (device pick; OOM captured as a failure, never a crash) | wiring + tests | |
| Thu Aug 13 | RTX 4050 validation (the public claim) + MPS smoke test; kernel-time budget semantics under GPU training re-checked | validation notes | |
| Fri Aug 14 | Vision example + notebook deliverable (training curves in cells); if local floor-model latency is impractical for DL, document the honest cloud-backend recommendation | example + validation | |
| Sat Aug 15 | Buffer; if green, the cloud-GPU `ComputeBackend` interface (interface only, implementation stays trigger-based) | fixes | |
| Sun Aug 16 | **Release v0.6.0** per the standing checklist | v0.6.0 out | |

---

## Sprint 5: v0.7, cost-constrained serving. Mon 2026-08-17 to Sat 2026-08-22, release Sun 2026-08-23

**Goal:** the flagship differentiator. Semantics locked long ago: score is the pure objective inside a hard serving-cost wall, never score-per-dollar; zero reward for being cheaper than the budget. Deliverable is a serving profile: best model within budget, cheapest cloud to host it, estimated $/month, requests/hour.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 17 | Pricing research (per-cloud instance pricing snapshots, refresh policy; model-size/quantization → feasibility mapping) + serving-cost model + serving-profile schema; RESEARCH_LOG entry | `core/serving.py` + tests + RESEARCH_LOG | |
| Tue Aug 18 | Constraint wiring: `--serving-budget` makes infeasible candidates losers regardless of score (a feasible-region filter in the compare step, not a penalty term) | wiring + tests | |
| Wed Aug 19 | `iterate cost` command: per-experiment + cumulative agent operating cost by compute and LLM backend (`ExperimentResult.cost_usd` aggregation); operating cost reported, never a constraint (infra-over-model stance) | CLI + tests | |
| Thu Aug 20 | Quantization as a feasibility lever for DL winners (4050 if reachable; MPS/e2b fallback) | lever + tests | |
| Fri Aug 21 | End-to-end demo: the same task with and without a budget produces different recommendations; validation | example + validation | |
| Sat Aug 22 | Buffer + carried items if green | fixes | |
| Sun Aug 23 | **Release v0.7.0** per the standing checklist; comparison-table cost rows flip to shipped; the parked cost-win post angle becomes honest here | v0.7.0 out | |

---

## Sprint 6: v0.9, infer the inputs + MCP discovery (absorbs v0.8). Mon 2026-08-24 to Sat 2026-08-29, release Sun 2026-08-30

**Goal:** Dial A turns hard, both steps in one release. Half the week is the v0.8 scope (data + a one-line description in, proposed target/features/metric out, confirmed at a pause), half is trimmed MCP discovery (the agent finds the data/code itself over filesystem + Postgres, then pauses for gap-fill). The PRD discovery-heuristics table is the ready-made spec. Security scoping is not optional: isolated subprocesses, path-restricted filesystem, read-only Postgres, every MCP call logged to Memory for audit.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 24 | Inference module: host-computed data profile + description → typed proposal (target, features, metric + direction, task type) with stated reasons; low confidence asks, never guesses silently; confirm-pause UX; `--target`/`--metric` optional, `--yes` for scripts | `core/infer.py` + CLI + tests | |
| Tue Aug 25 | Data versioning: content-hash chapter scoping in Memory + hash-based splitting + dataset-switch warning; split snapshot persisted to `.iterate/runs/<id>/` | memory/split + tests | |
| Wed Aug 26 | MCP substrate: client (stdio first), config-driven registry, tool bridge (MCP tool defs → OpenAI schemas) | `mcp/` + tests | |
| Thu Aug 27 | Wire filesystem (path-restricted) + Postgres (read-only) servers + security scoping + audit logging to Memory | config + docs + tests | |
| Fri Aug 28 | Discovery agent: one-line goal → repo scan + baseline/eval extraction + DB table relevance → "what I found / what I could not find" gap-fill pause, grounding committed to Memory | `core/discovery.py` + tests | |
| Sat Aug 29 | End-to-end `iterate init --discover` on a seeded fixture repo + DB; validation across 4+ datasets for the inference path; buffer | integration test + validation | |
| Sun Aug 30 | **Release v0.9.0** per the standing checklist (release notes cover both dials; v0.8.0 is skipped as a standalone tag); LIMITATIONS states the trims honestly | v0.9.0 out | |

**Cut from v0.9 (post-v1.0 backlog):** Notion + github MCP servers, Notion/markdown logging adapters, `DataSource` protocol + Kaggle/HF loader (the `[datasets]` extra stays dormant; LIMITATIONS "beyond a local CSV" row updated to say so), HTTP MCP transport if stdio suffices, source-artifact ask on claimed prior scores.

---

## Sprint 7: v1.0, one-sentence input + the evidence release (absorbs v0.10). Mon 2026-08-31 to Sat 2026-09-05, release Sun 2026-09-06

**Goal:** make every public positioning claim literally true, then launch. `iterate "improve our churn baseline"` end to end: discovery → gap-fill pause → bounded iteration → serving profile + notebook + report, with human-approval gates throughout. The v0.10 evidence items (benchmark, dashboard, reporter) ship inside this release.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 31 | One-line form glue: goal sentence → discovery → gap-fill pause → run, sane defaults (budget, deadline), every power-user override still working; human-approval gates audit (propose-only everywhere, never auto-merge to a production path) | CLI + tests + audit | |
| Tue Sep 1 | Anthropic adapter behind the `LLMClient` protocol + benchmark harness (same task, N backends, score x agent cost x wall time, publishable table) | `llm/anthropic_client.py` + `examples/benchmark/` + tests | |
| Wed Sep 2 | Run the benchmark (churn + toxicity) and publish the table; the Ollama-floor leg doubles as the release quality gate (trajectory bar re-run: a new adapter touched the loop) | results | |
| Thu Sep 3 | Streamlit read-only dashboard (runs, experiments, memory, cost panel) + prose Reporter from Memory + digests, both minimal | `ui/dashboard.py` + `core/reporter.py` + tests | |
| Fri Sep 4 | `docs/` real user documentation (install, quickstart per target family, config, backends, compute, memory, MCP; empty scaffold since Week 0) + full doc reconcile (README pitch rows all true, LIMITATIONS honest v1 gaps, PRD rewritten or retired, BOTTLENECKS/EVAL_LOG final state) | docs/ + doc PRs | |
| Sat Sep 5 | Proof-points run (fill the LAUNCH_POST table: 30+ experiments overnight under $10, a past-failure retry win, papers cited, the benchmark table) + release-candidate regression (all three targets, both compute paths, 3+ backends) + launch assets (3-5 min demo video, posts) | proof runs + rc + assets | |
| Sun Sep 6 | **Release v1.0.0**: tag + PyPI + GitHub release (v0.10.0 skipped as a standalone tag); launch: demo video to YouTube + LinkedIn + X, post series over the following 2 weeks (launch, architecture deep-dive with the multi-agent staleness note inverted, retry win, cost win, pluggability), scheduled through the night-window convention | v1.0.0 out | |

**The v1.0 bar still holds:** it ships when the public claims are true. If the Saturday rc says a claim is not true yet, the Sunday release goes out as v0.10.0 (the evidence release) and v1.0.0 follows the next Sunday with the gap closed. That is the only sanctioned slip in this calendar.

---

## Backlog disposition (2026-07-25, sprint edition: every tracked deferral re-homed or closed)

"Sat buffer" means the item rides any green Saturday; "post-v1.0" is the honest backlog that survives the sprint and gets stated in LIMITATIONS at each release.

| Item | Was | Now |
|---|---|---|
| Token streaming (both adapters) | deferred out of v0.2 | v0.3 stretch block 5 (Ollama first); OpenAI-compatible → Sat buffer / post-v1.0 |
| LocalKernel IPC transport, `--debug` flag, 3 non-gating certification items | parked at v0.2 close | `--debug` + header item in v0.3 stretch; rest → Sat buffer / post-v1.0 |
| e2b egress-deny (custom template) + per-experiment memory/CPU caps | "v0.2.x" / old isolation backlog | Sat buffer; else post-v1.0 (security batch, first item) |
| Pre-run undefined-name lint on generated cells | promised before v0.2, silently dropped | Sat buffer; else post-v1.0 |
| Default backend flip (groq → ollama) + `--compute` twice + dataset-switch call | open small calls | v0.3 block 0 (today) |
| Metric panel lift + predict_proba metrics + averaging | v0.4 rows in LIMITATIONS | sprint 2, Mon Jul 27 |
| CV/k-fold selection option | v0.4 row | CUT → post-v1.0 |
| Experiment dossier (deterministic → Summarizer graduation) | knowledge-transfer ladder step 2 | sprint 2: deterministic Tue Jul 28, Summarizer authors it Thu Jul 30 |
| Lean tried/untried idea ledger (post-revert redesign) | knowledge-transfer ladder step 3 | sprint 2, Tue Jul 28 (watch validation for lever collapse) |
| Free inspect/EDA step (unscored) | named v0.4 follow-up | sprint 2, Fri Jul 31 |
| Typed Session handoff (replaces `_winning_code` blob) | "typed handoff at v0.4" | CUT → post-v1.0 |
| Researcher (citations, dedup) + Critic (leakage, eval verdicts) | v0.4 specialists | sprint 2, Wed-Thu Jul 29-30 |
| Agent picks metric + starting model | v0.4 dial | sprint 2, Fri Jul 31 |
| `iterate history` / `best` / `why-failed` | unscheduled small add | Sat buffer; else post-v1.0 |
| qwen3:14b re-run vs the worked-example prompt | monitor item | CUT → post-v1.0 |
| PromptTarget + toxicity_jigsaw + intent_clinc150 examples | v0.5 | sprint 3 (release Aug 9) |
| DLModelTarget + 4050 validation | v0.6 | sprint 4 (release Aug 16) |
| Cloud-GPU adapter | interface ~v0.6, implementation later | interface: sprint 4 Sat if green, else post-v1.0; implementation trigger-based |
| Cost-constrained recommendation + serving profile + `iterate cost` + quantization lever | v0.7 / v1 moat | sprint 5 (release Aug 23) |
| Infer features/target/metric + confirm pause | v0.8 | sprint 6, Mon Aug 24 (ships inside v0.9.0) |
| Hash-based splitting + data-version Memory scoping + split-snapshot persist | old Week 8/9 + IDEAS deferrals | sprint 6, Tue Aug 25 |
| Source-artifact ask on claimed prior scores | old Week 7-8 anchor | CUT → post-v1.0 |
| MCP client/registry/bridge + filesystem/Postgres servers + discovery agent + gap-fill pause | v0.9 / old Week 4 zombie | sprint 6, Wed-Sat Aug 26-29 |
| Notion + github MCP servers, Notion/markdown logging adapters | v0.9 substrate items | CUT → post-v1.0 |
| Kaggle/HF data sources + `DataSource` protocol (the `[datasets]` extra) | promised at v0.9 | CUT → post-v1.0; LIMITATIONS row updated to say so at the v0.9 release |
| Streamlit CHAT UI (old backlog 5.3: chat input + live reasoning stream) | Week-10-era preview | narrowed 2026-07-25 to the read-only dashboard (sprint 7, Thu Sep 3); interactive chat shipped in the v0.3 CLI instead |
| Multi-backend benchmark + Streamlit dashboard + prose Reporter + Anthropic adapter | v0.10 | sprint 7, Tue-Thu Sep 1-3 (ships inside v1.0.0) |
| Demo-asset checklist (charts, dashboard shots) | LAUNCH_POST backlog | sprint 7, Sat Sep 5 |
| One-line form + docs/ + proof points + PRD reconcile | v1.0 | sprint 7 (release Sep 6) |
| Semantic memory retrieval | "if wrong retrievals surface" | stays trigger-based, unscheduled |
| Import → package alias-map architecture revisit | provisional, TBD | stays open; revisit if resolve failures recur (soft-fail backstop holds) |
| Spec-path preprocessing flexibility; multi-target/multi-label | TBD | stay unscheduled, trigger-based |
| Proposer-yield levers (few-shot, temperature, text-fallback) | logged, not pulled | superseded in practice by gemma floor + cloud path; keep listed, unscheduled |
| PostgresMemory | conditional | only if multi-user/hosted materializes |
| Cross-notebook EDA-repetition retry | reverted feature | folded into the sprint 2 ledger work, guard-first, watch for regression |
| LightGBM macOS-ARM wheel slowness | known issue | documented, no action (fine on Linux/e2b) |
| Optuna / AutoML HPO | v2 ambition | stays out of v1 scope |

---

## MCP + Discovery Backlog (re-homed 2026-07-25: sprint 6, v0.9, release Sun 2026-08-30)

> **Re-sequenced 2026-05-27 (agent-first):** Proposer (4.10) + Memory (4.11) moved **forward to Week 3** (the core agentic loop); Researcher (4.9) → v0.4 (sprint 2, release 2026-08-02). Under the sprint calendar: 4.1-4.5 + 4.8 land in **sprint 6 (v0.9, release 2026-08-30)**; 4.6, 4.7, 4.12, 4.13 (Notion, github, logging adapters) are **cut to the post-v1.0 backlog**. They're Dial-A input-reduction (toward one-sentence input), *not* prerequisites for the agent.

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

## UI + Benchmark Backlog (re-homed 2026-07-25 per row; the core lands in sprint 7, v1.0)

| # | Task | Files |
|---|------|-------|
| 5.1 | Terminator — patience / deadline / compute budget / plateau detection · shipped in v0.1 | `src/iterate/core/terminator.py` |
| 5.2 | Reporter — generates run summary + PR-shaped report · sprint 7, Thu Sep 3 | `src/iterate/core/reporter.py` |
| 5.3 | **Streamlit chat UI** — sidebar (MCP status + experiments + memory + cost), chat input, live agent reasoning stream · narrowed 2026-07-25 to a READ-ONLY dashboard (sprint 7, Thu Sep 3); the chat value shipped in the v0.3 interactive CLI instead | `src/iterate/ui/dashboard.py` |
| 5.4 | Second example target (intent_clinc150) to prove framework genericity · sprint 3, Thu Aug 6 (v0.5) | `examples/intent_clinc150/` |
| 5.5 | Multi-LLM backend benchmark — same task run on Ollama / Groq / Together / Deepseek / Anthropic · sprint 7, Tue-Wed Sep 1-2 | `examples/benchmark/` |
| 5.6 | Demo video walking through full discovery → iteration loop · sprint 7, Sat-Sun Sep 5-6 (v1.0) | `docs/demo.md` + recording |
| 5.7 | Final README polish, launch post assembly from LAUNCH_POST.md · sprint 7, Fri-Sun Sep 4-6 (v1.0) | `README.md`, `LAUNCH_POST.md` |

---

## Done

### 2026-08-02 | Sprint 2 Day 7 | v0.4.0 released

**Task:** Release mechanics per the standing checklist.

**Release gate (step 1):** v0.4 touched the loop in three places, so the trajectory bar was re-run before tagging rather than after — 8 runs across 6 datasets on gemma4:12b. That gate found seven bugs and is written up in the Day 6 entry. Four of eight runs improved on their baseline; of the four that did not, two had zero measured headroom and reporting nothing was the correct answer.

**Build gate (step 2):** 583 unit tests, ruff + mypy --strict clean, CLI startup verified still free of sklearn. One thing caught here that would have failed the release: `make build` runs `ruff check .` over the WHOLE repo while CI runs `ruff check src tests`, so the new `docs/evidence/` reproduction script (33 lint errors, deliberately not production code) would have broken the build. Excluded in pyproject, which is what its own README already claimed.

**Doc sync (step 3):** README status paragraph, release table row to shipped, the roadmap cell, the comparison-table "Literature-aware proposals" row flipped to a tick, and the capability section heading which still read "What v0.2 does" two releases on. LIMITATIONS: five rows flipped to fixed, the metric-panel row rewritten as the closed-registry row, the probability and averaging rows retired, and the false integer-target backlog row replaced. First ever EVAL_LOG entry, carrying the headroom table and a dimension scorecard.

**Version mechanics (step 4):** 0.3.1 -> 0.4.0 in pyproject, `__init__`, and the lockfile; tagged v0.4.0. Tony runs `uv publish`.

**Launch assets (step 5):** X thread (7 tweets, all under 275) and LinkedIn post (355 words, inside the 323-458 range of v0.1-v0.3) drafted in LAUNCH_POST.md, feature-first per the v0.2 REV2 lesson that a score-led post reads wrong. Demo to be recorded from the published package on `laptop_price.csv --max-iterations 3`, which shows the metric dial, a real citation and the 412 -> 322 win inside about twenty minutes.

**What v0.4 shipped:** all five public promises. Researcher and Critic specialists, literature-grounded proposals with citations that cannot be fabricated, probability metrics (plus 42 more than promised, derived from scikit-learn), a selection-bias watch, and the first input dial to turn since v0.1 — `--metric` is optional.

**Honest state of the evidence:** the headline capability claim rests on one dataset. Laptop price captured 110% of measured headroom by retrieving a 2022 paper on high-cardinality features and beating a hand-parsed baseline. Five of the six other datasets were near-ceiling, which is the real shape of tabular ML rather than a weakness of the agent, but it does mean "captures large gains" is n=1. The posts claim a specific run rather than a general capability, which is the correct framing for what was measured.

### 2026-08-02 | Sprint 2 Day 6 | Certification: seven real bugs, none of which 583 unit tests could find

**Task:** Run the trajectory quality bar on gemma4:12b before tagging, per the standing release checklist ("if the release touched the loop, re-run the bar"). v0.4 touched it in three places, so this was mandatory rather than optional.

**Method change worth keeping: headroom-normalised scoring.** The first three runs all read "no candidate beat the baseline", and I could not tell a failing agent from a hard problem. So every dataset got a brute-force ceiling measured FIRST, by hand, no LLM. A run is then scored as the fraction of available gain captured, not as a raw delta. That single change turned every ambiguous result below into a clear one, and it is the seed of the v0.5 eval suite.

| dataset | task | headroom | agent captured |
|---|---|---|---|
| laptop price (1303x13) | regression | 82.34 rmse (20%) | **90.33 rmse (110%)** |
| diabetes (442x11) | regression | small | 2.52 rmse |
| churn (7043x20) | binary | 1.6% | 0 |
| heart, yours (9000x27) | binary | 0.6% | 0 |
| heart, UCI (303x14) | binary | **0.0%** | 0, correctly |
| wine (178x14) | multiclass | 0 (baseline 1.0000) | 0, correctly |

**The result that answers the capability question.** On laptop price the agent read the profile, retrieved "Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features" (2022), target-encoded the 618-unique Product column and friends, and reached 321.56 rmse against a 411.89 baseline. I had hand-parsed that dataset first, extracting 8GB and 1.37kg and screen pixels out of strings, and got 329.55. **It beat the hand-engineered ceiling**, its winning candidate is stamped `source="researcher"` with two citations, and the code fits every transform on the training fold only. Knowledge transfer is visible in the chain: iteration 1's digest takeaway was "replace one-hot with target encoding for high-cardinality", and iteration 2's brief was exactly that.

**Seven bugs, every one found by running on real data rather than by reasoning:**
1. **The agent kernel ran the wrong interpreter.** `start_new_kernel()` resolves the machine's registered `python3` kernelspec — on macOS the Command Line Tools build. Measured: kernel on python 3.9.6 / sklearn 1.6.1, harness on 3.12.12 / 1.8.0. Worse, `install()` targets `sys.executable`, so auto-installed packages never reached the kernel at all and the "no library boundary" design was silently broken on local compute. Pre-existing since v0.2. On a machine with no system sklearn, every run would have failed.
2. **Metric names were not importable.** The agent is told to optimize `average_precision`; `sklearn.metrics.average_precision` does not exist. 16 cells burned across the runs. The registry already held the real function name, so the coder is now told it. Two scorers wrapping private helpers are dropped from the vocabulary — a metric the agent cannot compute is not worth offering.
3. **Boolean columns aborted the run at the baseline.** `SimpleImputer` rejects bool dtype, and a bool+string frame makes it take the numeric path and die on the first string. Any yes/no column stored as a real boolean killed the run before iteration 1.
4. **Non-UTF-8 CSVs could not be loaded.** `pd.read_csv` assumes UTF-8; any European export with a euro sign is latin-1. First contact with the tool was a decoding traceback on a file that opens fine in a spreadsheet.
5. **Threshold levers on ranking metrics.** Three runs picked `average_precision` then spent iteration 2 on class weighting. Measured across five datasets: threshold tuning moves f1 by ~0.02 and moves average_precision and roc_auc by EXACTLY 0.0000, because a ranking metric is invariant to the threshold. Reproduction lives in `docs/evidence/`.
6. **The Critic reasoned about direction and got it wrong.** It called an RMSE holdout of 59.29 against validation 56.69 a "lucky split" — maximize logic on a minimize metric, 3 firings in 5 iterations. The harness now computes "the holdout is BETTER/worse than validation by X" with direction applied, and the model only judges whether the gap is suspicious. Re-run on the same dataset: 1 firing.

7. **Integer regression targets could not be loaded at all.** Found by the diamonds dataset: an integer target was always read as a class label, so 11,602 distinct prices became 11,602 classes and the stratified split raised before the run started. Prices, counts and years are all integers, so this is the common regression case rather than an edge one. Recorded on Day 5 as a backlog item and DEFERRED on the reasoning that a fix would change existing splits. That reasoning was wrong and one measurement showed it: every dataset that works has 20 or fewer distinct target values and is untouched, so the only targets affected are ones that already crashed. Deferring on an unchecked assumption cost a day.

**The lesson that repeated twice.** For bug 5 the prompt fix did nothing: the note reached the supervisor's system prompt (verified) and gemma4:12b briefed the dead lever anyway, three runs out of three. Converting it to a guard fixed it on the first run. That is the June EDA-ledger lesson landing again — guards beat prompt nudges on weak models — and it is why bug 6 was fixed by computing the fact host-side rather than explaining direction better.

**One "obvious" fix deliberately NOT made.** Two datasets carry an ID column (`patient_id`, `laptop_ID`) and flagging them looked like free value. Measured first: dropping `patient_id` is worth **-0.0031**. The agent independently reached the same conclusion on laptop, recording "Removing laptop_ID: 281.11 -> 290.51 (lost signal)". A no-op dressed as an improvement.

**Bar status:** staged R&D (10-28 cells per iteration, no monoliths), zero FAILED iterations, duplicates <=2, citations all resolving, Critic catching seeded leakage 3/3 with no false positive, the dial adapting (average_precision on imbalanced targets, f1 on balanced), and progression demonstrated where headroom existed. 581 unit tests, ruff + mypy --strict clean.

### 2026-08-01 | Sprint 2 Day 5 | Dial A: the first input to disappear since v0.1

**Task:** `--metric` becomes optional. This is the roadmap spine turning — v0.2 and v0.3 both read "(same inputs)" in the README release table and grew capability instead; v0.4 is the first release since v0.1 where the inputs you must give SHRINK.

**Design change from the plan, on Tony's call.** The plan had a dedicated `choose_setup()` pass to pick the metric. Wrong for two reasons he named: it would run research TWICE (once pre-loop to pick a metric, again at iteration 1 to pick techniques), and the metric — the thing the entire run optimizes — would have been decided by the THINNER of the two passes, grounded in a class balance rather than in literature. Revised to one retrieval feeding two judgements: the same fetched papers back both the setup choice and the technique suggestions, so the metric choice compounds off the full research. It is also fewer calls than before, since iteration 1 no longer researches separately.

Kept as a separate structured call rather than extra fields on `suggest_techniques`: a 12B asked for techniques, a metric AND a starting model in one emit starts dropping fields (DECISIONS 2026-06-01, one focused job per specialist call). Two calls over identical papers cost one local LLM call and zero extra API calls.

**The construction-order blocker, fixed first.** `ModelTarget` was built at cli.py:329 and the LLM client at 338 — the metric was needed nine lines before the thing that could choose it existed. `summarize_dataset` needs only the dataset and `build_client` needs neither, so both moved above. Mechanical, no behaviour change.

**What shipped:**
- `core/setup.py` — the single place a run's ruler is decided. Explicit wins; otherwise the proposal is validated against the registry AND against the target column; anything failing falls back to what v0.3 would have run (f1 / f1_macro / rmse). **The agent can only ever upgrade the default**, never produce a run that fails to start.
- The choice is printed with its reason before the first experiment. A tool that silently picks your evaluation metric is worse than one that asks.
- The starting model is treated as a WEAKER claim than the metric: a junk model name falls back on its own without costing the metric choice, because the agent rewrites the model every iteration anyway — a bad one costs one experiment, a bad ruler costs the run.
- 17 new tests; 569 unit tests, ruff + mypy --strict clean.

**A duplicate-source-of-truth bug caught by a failing test, the third of this sprint.** I wrote a fresh classification-vs-regression heuristic in `target_task`. `tabular.py` already had one (`_looks_like_classification`, used to decide whether to stratify), and they disagreed: on an integer target with many distinct values mine said regression, the loader said classification. The test failed inside `load_csv` — the dataset could not even be LOADED. `target_task` now delegates to the loader's heuristic. Two definitions of "is this classification" would eventually diverge, and a metric chosen for a task the data was never split for cannot score at all.

That also surfaced a pre-existing limitation, now recorded: an integer regression target (a count, a year) gets stratified and raises at load time. Fixing the heuristic changes every existing split and therefore every recorded score, so it needs its own change rather than being a side effect of this one.

**Verified end to end**, with live API calls on the agent path: explicit `--metric roc_auc` wins unchanged; offline gives `f1` exactly as v0.3 did; the agent on a 27/73 imbalanced target picks `average_precision` and explains that PR-AUC reflects the minority class where accuracy and ROC-AUC flatter the majority; a proposed `rmse` on a label target is rejected and falls back to `f1`.

**Cut: the free unscored inspect step.** It needs a fourth `AttemptOutcome`, must not burn patience or count toward `max_iterations`, and needs its own cap so an unproductive inspect cannot eat the budget. That is a loop change with real blast radius, and it is an efficiency improvement rather than part of the README promise ("agent picks the metric + starting model"). Not worth its risk the night before a release with the certification run still to go. Moves to the v0.5 week.

### 2026-08-01 | Sprint 2 Day 4 | The Critic: can this score be believed?

**Task:** The last specialist. Reviews every finished experiment before its score is allowed to bank as the run's best. This is the direct answer to the question a commenter asked in public — whether the agent can tell overfitting from genuine improvement (PROGRESS_NOTES:140) — and to the leak class that cannot be caught statically (PROGRESS_NOTES:162).

**Deviation from the plan row, argued and taken:** the row specifies leak review "as a typed PRE-EXECUTION check". Moved to post-session, pre-banking, for three reasons. There is no single "the code" before execution on the cell-by-cell path — a pre-execution check means an LLM call PER CELL, and sessions run 10-30 cells. The question is about the score, not the code: leakage matters because it inflates a holdout number, which is inherently a post-hoc judgement. And the mirage check needs the score anyway, so both jobs land in one call instead of two mechanisms.

**The design decision this day turns on — leak vetoes, mirage does not:**
- **leak** = a defect visible in the submitted code (a transform fitted on the holdout, the target used to build a feature). VETOES banking, because a number produced by cheating is not a score.
- **mirage** = a statistical suspicion about the gain, most often a holdout score far above the validation trail. FLAGS only, never vetoes.

A leak is checkable, so acting on it is safe. Whether a gain is "real" is probabilistic, and the sealed holdout is already this project's ruler — letting a 12B overrule it would put a model back into the control flow that direction, the guard stack and duplicate-hashing were all deliberately kept out of. So the Critic can subtract a win it can prove was cheated, and can raise a hand about one it merely doubts. Nothing it says can promote a losing experiment; it only ever takes away.

**What shipped:**
- `core/critic.py` + `critic:` prompts key. One structured call per experiment, reviewing the backward slice from the final predictions write (`submit_path_code`, promoted from private) so probes and abandoned branches are already excluded. It sees the holdout score, the previous best, and the validation trail the session printed — the val-vs-holdout gap IS the mirage evidence, so it is put in front of the model rather than left to be inferred. **Day 2's dossier supplies that trail**, which is the first place that day's work paid off.
- Verdicts stamped into `candidate.changes` as `critic_rejected` / `critic_flagged`, joining `duplicate_submission` and `lever_unmeasured`. The supervisor's technique table already excludes stamped experiments from crediting their techniques, so a vetoed experiment stops teaching a false lesson with **zero new prompt context**.
- The banking gate reads `_improves(...) and not was_rejected(...)`. Deterministic ruler first, veto second, and the veto can only subtract.
- **The history annotation is not optional colour.** Without it the supervisor reads a leaked 0.81 as the run's high-water mark and pushes that direction, so the scoreboard would be actively lying. A rejected score renders as `f1=0.8100 [REJECTED, <reason> — this number is not a result]`, the same pattern the floor and duplicate markers already use.
- Degrades like every other agent: a backend failure, a model that will not call the tool, a failed experiment, or a spec candidate with no code all yield a clean verdict. A flaky review must never cost a real result. String booleans are coerced, because the supervisor hit `bool("false") is True` live on groq and the same models drive the Critic.
- `--critique/--no-critique`, defaulting on. 19 new tests; 552 unit tests, ruff + mypy --strict clean.

**Verified end to end:** a pipeline fitting StandardScaler on `pd.concat([X_train, X_holdout])` scored 0.81 against a 0.60 baseline. It passes `_improves`. The Critic caught the concat, the experiment was stamped rejected, and it did not bank.

**Deferred, deliberately: the Summarizer graduating to author the dossier.** That is Day 2's carried deferral and the only part of this day that changes what the supervisor eventually reads, so it is the part carrying the June EDA-ledger risk. Shipping an unmeasured context change the day before a release, with Dial A and the certification run still ahead, is the trade that bites. It stays for the v0.5 week with the decision rule already written down: keep only if lever diversity holds.

### 2026-08-01 | Sprint 2 Day 3 | The Researcher: literature grounding with citations that cannot be invented

**Task:** The first specialist graduates at the tool boundary the architecture has described since 2026-06-01, and the first time this codebase talks to an external API.

**Plan correction, forced:** the row says "arxiv + papers-with-code clients". **papers-with-code no longer exists** — `paperswithcode.com` now 302-redirects to `huggingface.co/papers/trending`. Replaced with **OpenAlex**, which is a straight upgrade: ~320M works across journals and conferences versus arXiv's ~2.4M preprints, and it returns citation counts, which gives the Researcher a ranking signal it otherwise lacks. Both are keyless and free, so `pip install iterate-ai` still needs no account.

**Architecture correction, caught before building.** The plan (and my own first description) had the supervisor calling the Researcher as a tool. It does not, and should not. In this codebase the HARNESS orchestrates and agents never call each other: `agent_loop` calls `supervisor.decide()`, then `coder.run()`, then `summarizer.summarize()`. The one place a supervisor drives a tool loop is `answer()` (Q&A with `read_notebook`), and that is deliberately quarantined — its docstring says "none of this context ever enters `plan_next`", because `plan_next` must stay ONE structured call. That is the same constraint that turned thinking mode off for strict roles. So the Researcher is orchestrated exactly like the Summarizer, and the supervisor's ASK is a `want_research` boolean on the emit it already makes: one field, not an extra round trip.

**What shipped:**
- `adapters/research/` — OpenAlex + arXiv clients, disk-cached by query hash under `.iterate/research/` (survives across runs), arXiv rate-limited to its requested 1-per-3s, both hitting https directly since the http endpoint 301-redirects on every uncached query.
- `core/researcher.py` — two focused LLM calls with deterministic retrieval between them: `plan_queries` turns the host profile into 2-3 queries, the harness searches and dedupes, `suggest_techniques` picks from what came back. Split because a specialist with one narrow job tool-calls far more reliably on a weak model than one call juggling search, judgement and citation.
- **The citation guarantee is structural, not prompted.** The model picks a paper by its INDEX in the list it was shown; the harness resolves that index to the identifier the API returned. It is incapable of emitting a DOI that was never fetched. An out-of-range or non-numeric index DROPS the suggestion rather than keeping it with a blank citation. Prompting a model not to invent citations is a request; indexing makes it impossible, and a fabricated citation in a tool advertising "literature-aware proposals" is the worst bug this project could ship.
- **Crediting is conservative.** `Candidate.source` becomes "researcher" and citations are stamped ONLY when the brief actually took up a suggestion (technique phrase present, or two shared content words). A pass the supervisor read and ignored stamps nothing, because an unearned citation is no better than an invented one.
- Two deterministic guards around the supervisor's judgement: iteration 1 always researches (no history means nothing to base a `want_research` call on), and `max_research_calls` (3) caps the run so literature can never eat the experiment budget.
- `--research/--no-research`, defaulting on. Findings reach `decide()` as one capped block (420 chars, three lines) alongside guidance and standing rules.
- 17 new tests. 525 unit tests, ruff + mypy --strict clean.

**A real bug the tests caught:** `search_all` did not guard against a source raising. The built-in clients swallow their own network errors, but the AGGREGATOR did not, so any custom or later-added source broke the never-raises contract. Now guarded per source: one dead source degrades to the others' results, not to a dead run.

**A logging call that paid for itself immediately:** the Researcher's two broad `except` blocks logged at DEBUG. During testing one of them silently swallowed a pydantic ValidationError — a genuine programming error, invisible, presenting exactly like "the literature search found nothing". Raised to INFO with the exception text. Research silently producing nothing and research legitimately finding nothing look identical from outside, and only one of them is a problem.

**Verified end to end against the live APIs**, not only against fakes: a scripted model produced two queries, the clients returned 10 deduped papers, and both suggestions came back with real DOIs (`10.1038/s41586-024-08328-6`, `10.1186/s40537-020-00305-w`). Crediting attributed the brief that took up a suggestion and refused an unrelated one. Cold 3.1s, warm 0.00s from cache.


**Scope added same day (Tony's call): the metric vocabulary is derived, not hand-written.** Day 1 shipped 12 hand-picked metrics. Once the Researcher can propose techniques it can propose METRICS, and a hand-written set is the wrong shape for that — but letting a model supply the DIRECTION would reintroduce the silent-inversion failure the registry exists to prevent. sklearn resolves it: its scorer registry encodes direction structurally (all scorers are higher-is-better, loss metrics carry a -1 sign — verified to hold across all 58 with no exceptions). So the vocabulary is now derived from `get_scorer_names()`: 54 selectable metrics including Matthews correlation, balanced accuracy, jaccard and every f1/precision/recall averaging variant, with task derived from the scorer function's module and probability-requirement from its `_response_method`. Clustering scorers are excluded — they compare two label assignments rather than a prediction against a target, so offering them would invite the agent to pick something meaningless. The always-computed PANEL stays at 12 so history stays comparable and cheap; a selected metric outside it is computed on top via `include=`. Derivation reads private scorer attributes, so a canary test asserts it still works on the installed sklearn: an upgrade that moves them fails CI rather than silently shrinking a user's vocabulary back to 12.

**And it surfaced the string-label bug one level deeper.** Day 1 fixed our own panel inheriting sklearn's `pos_label=1`. The derived metrics hit the same wall for a different reason: most binary scorers (`jaccard`, `f1`, `precision`, `recall`) carry only `average="binary"` in their kwargs and no `pos_label` at all, so sklearn's own default applies. Inspecting kwargs would have missed every one of them; reading the function SIGNATURE is what catches it. Verified end to end on a Yes/No target.

### 2026-08-01 | Sprint 2 Day 2 | The deterministic record: experiment dossier + one definition of tried/untried

**Task:** Give the run two records that need no LLM — what a finished session can be observed to have done, and what the run has already spent itself on — and collapse the copies of that logic already scattered through the supervisor.

**What shipped:**
- `core/dossier.py`: distils a finished session into an observed record — techniques instantiated, data facts the cells printed, the validation trail, error signatures deduped on the same signature the coder's breaker uses, and the session shape (cells run / errored / timed out). The load-bearing rule is that it NEVER invents: every fact is a line the session printed, quoted, or a count of the cell records. That is exactly what makes it safe as the Summarizer's fallback, because a fallback that could hallucinate would be worse than no fallback. `build()` never raises — it is the path that runs when everything else has already degraded.
- `core/ledger.py`: tried/untried over two dimensions. Lever classes are marker-matched and can false-positive; components come from `codegen.components_used` via the AST and cannot, so `component_tried()` distinguishes `GradientBoostingClassifier` from `HistGradientBoostingClassifier` where substring matching cannot. The marker-neutralising pass is injectable, so the module owns no marker vocabulary of its own.
- The Summarizer's deterministic skeleton now reads the dossier instead of keeping a parallel copy; its dead `_val_trail` / `_FLOAT` helpers are gone.
- 24 new tests (`test_dossier.py` 13, `test_ledger.py` 11). 508 unit tests, ruff + mypy --strict clean.

**The consolidation was the real win.** The tried-set loop existed TWICE in `supervisor.py`, inline in `_fallback_move` and again in `_lever_ledger`. Both now read `run_ledger()`; zero duplicate loops remain. Same class of problem as Day 1's metric frozensets — a fact about the run with more than one definition.

**A trap caught doing it:** `_LEVER_MARKERS` and `_CANONICAL_MOVES` hold the SAME keys in DIFFERENT orders, deliberately — the first is display order for the prompt line, the second is fallback priority for `_fallback_move`. Naively sharing the sequence would have silently reordered text inside the supervisor's prompt, and list order is a salience signal to an LLM. So the shared piece is the tried SET only, never the order: share the derivation, never the presentation. A test pins it, since the invariant is otherwise invisible to whoever next reorders either dict.

**Correction to the sprint plan's premise.** The Day 2 row says to watch for the EDA-ledger regression pattern, which read as "a ledger in the prompt is the thing that regressed." It is not. Two different June events: the LEVER ledger shipped 2026-06-10 and HELPED (it is in the first 10/10-above-0.60 run entry, and it is live in the prompt today as `_lever_ledger`); the EDA ledger plus a supervisor status line was the thing built and reverted 2026-06-20. The component ledger still stays out of the prompt, but on the weaker ground that it is new context on an already-dense prompt, not that it repeats a measured failure.

**One visible behaviour change:** `val_trail` renders at a consistent 4dp instead of echoing whatever precision the session printed, and dedupes by value — so 0.55 followed by 0.5500 is one entry, not two.

**Deliberately not done, and why:** feeding the dossier into the Summarizer's PROMPT, and seeding the digest's insight fields from observation. Both change what eventually reaches the supervisor, so both are gated on a before/after rather than on being available. They belong to Day 4, where the Summarizer graduates to authoring the dossier, and Day 6 already carries "no context regression" as a bar criterion. The decision rule for that run: keep only if lever diversity holds — a score that rises while diversity collapses is the June pattern presenting itself as a win. Nothing in this PR changes a single character the supervisor reads.

### 2026-08-01 | Sprint 2 Day 1 | The metric layer: registry, probability metrics, configurable averaging

**Task:** Lift the fixed 8-metric panel — the first half of the sprint-2 Day 1 row. Probability metrics (ROC-AUC, PR-AUC, log-loss, Brier) become scorable, averaging becomes selectable, and the metric table becomes one source of truth instead of four.

**What shipped:**
- `core/scoring.py`: every metric is one row in `REGISTRY` carrying its task, direction, probability requirement and compute. `CLASSIFICATION_METRICS`, `REGRESSION_METRICS`, `PROBA_METRICS`, `LABEL_METRICS`, `task_for_metric()`, `direction()` and the new `requires_proba()` are all derived from it; `_MINIMIZE` deleted. Adding a metric is now one line.
- Probability panel: `score()` takes keyword-only `y_proba`; classification gains `roc_auc`, `average_precision`, `log_loss`, `brier` (binary only). Label metrics are still always computed, so a run's history stays comparable across iterations that did and didn't emit probabilities. Shape validation lives in `Inputs.positive_column()` / `class_matrix()`; malformed probabilities raise here rather than being swallowed, because whether that sinks an experiment depends on `requires_proba(primary)`, which only the caller knows.
- `resolve_average()`: binary/micro/macro/weighted for f1/precision/recall, with `None` preserving the pre-v0.4 behaviour exactly so no existing run changes score.
- 29 new tests in a new `tests/unit/test_scoring.py` (scoring had no direct test file — it was only covered through `test_model.py` and `test_codegen.py`). Two are structural: one asserts every exported set is derived from the registry, one asserts every registered metric actually scores, so a drifted set or a broken compute fails CI instead of a live run. 472 unit tests (443 at v0.3.1); ruff + mypy --strict clean.

**Two bugs found on the way, both pre-existing:**
- **`cli.py` kept its own copy of the metric names** and computed direction as "minimize if regression, else maximize". That held right up until `log_loss` became selectable — a classification metric that minimizes. Had it shipped, the Terminator's stop decision, the supervisor's `_best_holdout` and `_prior_best`'s cross-run baseline carry-over would all have been told lower loss is worse: the run banks its worst result as best and optimizes away from the goal, looking completely normal the whole time. Collapsing to the registry is the fix, and it is the reason the registry was worth doing rather than just widening the frozensets.
- **Binary string targets have never scored**, since v0.1. The label panel inherited sklearn's `pos_label=1` default, so a `Yes`/`No` target raised on f1/precision/recall. The churn example only avoids it because `prepare_churn` maps to 0/1, but `codegen._coerce` has a branch specifically for string labels, so any user pointing at their own CSV could hit it. The positive class is now named explicitly as the greater label — which had to happen regardless, so that the label panel and the probability panel agree on which class is positive.

**Portability call:** multiclass PR-AUC binarizes `y_true` explicitly rather than relying on sklearn's own multiclass support, which landed well after the `scikit-learn>=1.5` floor and raises below it. Same trap keeps Brier binary-only. Verified against the installed 1.8; the floor is what the pinned range has to survive.

**The probability contract, so the metrics are reachable from a real run:**
- `probabilities.csv` as a SIBLING artifact, never a second column in `predictions.csv`. That file's validator reads a two-field line as an index-column mistake (`to_csv` without `index=False`) — the guard that caught the single worst failure of the v0.2 live runs. Widening it to sometimes mean probabilities would have blunted it.
- `codegen.parse_probabilities` + `coder._validate_probabilities`: one value per line for binary, one comma-separated value per class for multiclass, with error strings the model can act on ("line 4 is not numeric… write raw probabilities, not labels").
- The policy split `core.scoring` deliberately refused to make: `score_predictions` sinks the experiment when `requires_proba(primary)` and the file is bad, and silently drops the bonus panel when it isn't. A malformed `probabilities.csv` must never cost an otherwise-valid f1 iteration.
- **Verified finish now checks probabilities too.** Caught at the finish gate rather than at scoring time, so a model that forgot the file is sent back into the session with turns left to write it instead of losing the whole iteration to the floor.
- **The floor writes probabilities when the metric needs them.** Without this, a probability run whose session died would bank a labels-only safety net that ROC-AUC cannot score — a total loss exactly when the net was supposed to catch it. LogisticRegression is already a probability model, so it cost one generated line.
- The one-shot harness takes an opt-in 2-tuple return `(predictions, probabilities)`; returning predictions alone stays valid, so every pre-v0.4 function is unaffected. The delivered notebook's scoring cell unpacks the same contract — it calls itself "the same ruler iterate used", and on a `roc_auc` run it would otherwise have printed a panel with no `roc_auc` in it.
- The spec path offers probabilities whenever the fitted pipeline has `predict_proba`, so a label-metric run gets the probability panel as a free bonus.
- `_proba_requirement` appends the extra submission rule to the coder prompt ONLY for a probability metric: an f1 run must not carry an instruction about a file it should never write, since every line competes for a weak model's attention.
- `--average` exposed on the CLI and threaded through `ModelTarget`, `CodingAgent` and `score_predictions`.

**Verified end to end** through the real `LocalCodeRunner`, not just in units: a `train_and_predict` returning `(labels, proba)` wrote both files, scored the full 8-metric classification panel, and the same run with the probabilities file removed failed with "roc_auc needs probabilities: probabilities.csv was not found" instead of silently scoring nothing.

**Briefs needed nothing:** `_format_history` renders `result.metrics.primary_value`, already metric-agnostic.

**Test count:** 483 unit tests (443 at v0.3.1), ruff + mypy --strict clean, CLI startup verified still free of sklearn.

### 2026-07-26 | v0.3.1 | The timeout-spiral patch (found live, fixed before publish)

**Task:** Tony's pre-publish test drive lost iteration 5 to a timeout spiral: a HistGB fit hit the 120s per-cell cap (the Week-2 thread-oversubscription class — generated code ran in a raw kernel with no thread cap), the coder retried the identical fit, the canned floor trains the same family so the safety net timed out too, and the failure recorded as a bare contract violation. Four deterministic fixes, patch-released as 0.3.1 before anything reached PyPI.

**What shipped:**
- Session preamble caps BLAS/OpenMP threads BEFORE any import (the v0.1 `threadpool_limits` lesson applied to generated code) — kills the oversubscription timeout class.
- Timed-out cells now count toward the consecutive-failure breaker and carry their own nudge (names the limit, forbids an identical retry, demands a cheaper family or a subsample).
- A dead session's failure record states WHY: "N cell(s) timed out; last killed: 'model.fit(...)'" — food for the REPAIR rung and the dead-ends channel instead of "no predictions file produced".
- The canned floor is now a linear model (LogisticRegression / Ridge, median-imputed): the safety net trains in milliseconds under any thread weather.
- `Cell` gains `timed_out`; 443 unit tests (440 at v0.3.0); ruff + mypy --strict clean.

**Deeper fix unchanged:** cross-iteration timeout knowledge (digest/dossier carrying "this model class stalls here") is sprint-2 work as planned.

**Launch complete (same day):** `iterate-ai 0.3.1` published to PyPI and verified from a fresh venv (`iterate version` → 0.3.1 from the pip install); feature-first launch posts (7-tweet X thread + LinkedIn, mirroring the v0.1/v0.2 structure) scheduled. Sprint 1 shipped on its Sunday: built Saturday, released Sunday, with the same-day patch for the live-found timeout class. Sprint 2 (v0.4) opens Monday.

### 2026-07-26 | Sprint 1 | v0.3.0: interactive runs (TUI + chat + pause/resume + hard stop)

**Task:** Build and ship v0.3 in one day per the sprint re-plan: talk to the run while it runs.

**What shipped:**
- Files: `core/interactive.py` (RunController: queued plain-English chat, control words, pause/abort, paused-clock accounting, live run snapshot), `ui/tui.py` (Textual interface: live transcript with two-tone syntax cell panels, command palette on "/", pinned input box), wiring through `coder.py` / `agent_loop.py` / `supervisor.py` / `kernel.py` / `cli.py`, new prompt keys for routing / Q&A / guidance / user notes.
- Chat: type anything, anytime; messages queue to the next safe boundary with a timing-only ack; the supervisor classifies intent (question / steer now / steer later / standing rule) and the HARNESS executes the routing. Questions are answered from the dataset profile + the LIVE session's cells + recorded notebooks (2-fetch cap); steers reach the running session at its next cell; standing rules ride every later planning turn (3 x 90 chars, lean by design).
- Controls: pause/resume at the cell boundary with all clocks suspended and the e2b lease kept alive; `/stop` or double Ctrl-C quits immediately and still prints the summary table from the loop's live snapshot; single Ctrl-C stays the graceful wind-down (floor banked, memory finalized).
- Guards outrank chat everywhere; applied guidance is stamped into `candidate.changes` for audit; non-tty, piped, and backgrounded runs are byte-identical to v0.2 (foreground-tty gate + SIGTTIN safety); `--plain` keeps line-mode chat.
- 440 unit tests (391 at v0.2.0), ruff + mypy --strict clean; `textual` added as a core dep; saved default backend flipped groq → ollama (rate-limit gotcha from the v0.2 launch).

**What didn't:**
- Token streaming (originally v0.3 scope) cut to the backlog; the transcript streams per cell and per event instead.
- Parked polish not taken (LocalKernel IPC transport, `--debug`, the 3 certification items, undefined-name lint, e2b egress template); all tracked in the disposition table.
- Q&A answers about the CURRENT run only; controls are exact words (`pause` / `resume` / `stop`, slash forms included) — natural-language stop routes as guidance.

**Found live during the test drive (and fixed same-day):**
- The sqlite Memory was created on the main thread but the TUI runs the loop on a worker thread → first write crashed; the Memory is now born on the thread that runs the loop, with a worker-thread regression test.
- Q&A was blind to the dataset profile and the in-flight session, so questions the screen had literally just answered ("how many categorical columns?", "what's the split?") came back empty; both are now in its context.
- Four-lens adversarial review before the test drive caught two more blockers pre-live: backgrounded runs would have been SIGTTIN-suspended by the stdin listener, and Q&A resolved iteration numbers against ALL runs in memory instead of the current one. Plus: stop-then-pause deadlock, markup-mangled replies, e2b keepalive starvation, second-note truncation.

**Decisions (user calls, this session):** hard-stop semantics (stop = quit now with the table; pause/resume are the waiting tools); plain-English chat over command prefixes (supervisor routes, harness executes, safest-default fallback); the TUI as the default face with `--plain` opt-out; transcript palette (two-tone code, role colors, full-width wrapped panels).

**Next session:** Sprint 2 (v0.4): Researcher + Critic specialists, probability metrics, dossier + lean ledger, agent picks metric + starting model. Release Sunday 2026-08-02.

### 2026-07-18 | Week 4 Day 8 (close) | v0.2.0 SHIPPED: PyPI + tag + GitHub release; both compute paths verified on the published package; new-dataset generalization run

**Task:** Execute the release and prove the published artifact does what the repo claims.

**Shipped:** PR #44 squash-merged to main (26 files, +3331/-424, first run of the new CI green), `iterate-ai 0.2.0` published to PyPI, tag + GitHub release out ("v0.2.0: the agent writes its own training code"). Release notes lead with the R&D-session story and the honest capability floor, not just numbers. A 0.2.1 patch followed the same day: third-party HTTP request logs (one line per LLM call, plus e2b keepalive/execute pairs around every cell) demoted to debug so a run's console shows only its own progress lines.

**Verification on the published package, not the repo checkout:** a 3-iteration e2b run on merged main saved every notebook live including two iterations with errored cells (the exact case that used to crash rendering: e2b's SDK v2 ships tracebacks as one string; fixed in #44 and now regression-proven on real sandbox traffic). Then a generalization run on a dataset the agent had never seen: UCI Adult income, 26.9k rows, 13 mixed features, 25.6% positive. Fresh chapter, f1 0.7187 baseline -> 0.7310 best, and the winning lever was class-weight balancing: the right move for an imbalanced target, picked without any churn-chapter memory to lean on. Digests came out dataset-specific (target-encode native_country, education x capital-gain interactions), and the run stopped itself on patience after three non-improving iterations. One labeled duplicate + one labeled unmeasured lever: inside the certified 1-2 floor band.

**Also caught in live use:** passing --compute twice (muscle memory) silently uses the last one, standard CLI behavior but worth knowing; and a fresh chapter is mandatory when switching datasets, because memory chapters key on the target family, not the file.

**Next:** demo video + launch posts (drafted, feature-first per the v0.1 post's structure), then v0.3 scoping (interactive CLI: pause, mid-run chat, resume).

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
