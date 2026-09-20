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

**Updated 2026-09-12: sprint 4 re-dated.** v0.5 shipped 2026-09-09, a month after its code was complete, because the release waited on the certification run. Sprint 4 (v0.6) now runs Sun 2026-09-13 to Sat 2026-09-19 with the release on Sun 2026-09-20; sprints 5 to 7 keep their scope and re-anchor their dates at the v0.6 release, per the standing rule that windows re-anchor at every release.

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
| Sep 13-20 | Dial B: `DLModelTarget`, images on the same loop: classes and numbers, a plain CNN baseline, fine-tuned pretrained backbones and the agent's own torch code · **v0.6.0 shipped 2026-09-20** *(re-dated 2026-09-12, confirmed 2026-09-16)* |
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
| v0.5.0 | sprint 3 | **RELEASED 2026-09-09** (planned Aug 9; slipped to Aug 11 for the regression half, then waited on the certification run) | + prompts | data + target + a task line (+ your current prompt, optional) + deadline: **the same loop on LLM prompts**, classification and regression, sealed holdout, `prompts.yaml` deliverable, eval suite with a measured ceiling per dataset |
| v0.6.0 | sprint 4 | **RELEASED 2026-09-20** (planned Aug 16; re-dated 2026-09-12 after v0.5 waited on its certification run, confirmed 2026-09-16) | + images | an image folder laid out however it came, or a CSV of image paths + target, or your own `--train` + `--holdout` split, + deadline: **the same loop on images**, classes and numbers, a plain CNN baseline, `fit()` over three pretrained backbones plus the agent's own torch code, sealed holdout; the data linker and the monitor for folders; a run reads only the data it was given, the macOS cell sandbox and harness-side installs for every local run. RTX 4050 over WSL2: SLOT_4050 |
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

## Sprint 3: v0.5, PromptTarget. Mon 2026-08-03 to Sat 2026-08-08, release Sun 2026-08-09 (released 2026-09-09)

**Carry-ins from the v0.4 certification (recorded 2026-08-02, ordered by what a user feels first):**
1. **Eval suite, FIRST in the week.** Day 6 spent an afternoon hand-computing brute-force ceilings per dataset to tell a failing agent from a hard problem, and found seven bugs no unit test caught. Make it a runnable corpus: datasets, measured headroom, a scoring harness. Two things it must fix about the v0.4 method — the ceilings were LOWER BOUNDS from a short sweep, not true maxima, so scoring against public leaderboard positions where they exist would replace guesswork with a number; and the corpus is the CI defence against dataset-shape bugs, since 583 unit tests missed all three (non-UTF-8, boolean columns, integer targets) because every fixture shared the same shape.
2. **The Researcher, Critic and Summarizer have no styled TUI events.** Only `brief`, `cell` and `score` are emitted; the two headline v0.4 specialists reach the transcript as dim ambient log lines via the logging handler, visually indistinguishable from routine chatter. They deserve the same weight the coder's cells get.
3. **Research cache files are opaque.** `.iterate/research/{source}-{hash}.json` stores a bare array of papers, so the QUERY that produced them is recorded nowhere. Debugging "why did it search for that" or auditing the trail means inferring the question from the answers. Store `{query, fetched_at, papers}` instead — a cache-format change, which is why it was not done on a tagged release.
4. **Threshold levers are not guarded on MULTICLASS.** `threshold_free` covers ranking metrics, but a multiclass problem has no single threshold regardless of metric, and the mobile-price run spent an iteration on "Threshold Refinement" against `accuracy`. Needs its own measurement before a guard, per the rule that no guard ships unmeasured.
5. **The Summarizer authoring the dossier** (Day 4 deferral) and **the free unscored inspect step** (Day 5 cut). The first changes what reaches the supervisor's planning context, so it is gated on a before/after with the decision rule already written down: keep only if lever diversity holds.
6. **`test_worker_thread_can_own_a_sqlite_memory` is flaky.** It waits a fixed `pilot.pause(0.3)` for sqlite work on a worker thread and fails intermittently on slower CI runners; it failed the v0.4 release PR once and passed on a re-run with no code change. Wait on a condition, not a duration.
7. **The agent misses thin margins.** Of 8 certification runs, 4 improved; two of the four that did not had small but real headroom available (churn 1.6%, mobile 2.1%). Worth understanding once the eval suite makes it measurable rather than anecdotal.

**Goal:** the second problem type. Prompts as a `BenchmarkTarget`: same iteration loop, different execution path. Two public example commitments come due: `examples/toxicity_jigsaw/` and `examples/intent_clinc150/`. Prompt iteration only, never foundation-model fine-tuning (permanent scope lock).

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Mon Aug 3 | **Eval suite FIRST** (Tony's call after the v0.4 certification): the headroom table from EVAL_LOG becomes a runnable corpus, so a release is measured rather than argued about. Then the v0.5 work below | `evals/` + corpus + runner | done (built Sat Aug 8) |
| Tue Aug 4 | Loop integration: prompt lever classes for the supervisor ladder, coder session writes prompt variants + scoring cells, guard stack audited for the new path (duplicate gates hash prompt text, dead-ends transfer) | wiring + tests | done (built Sat Aug 8) |
| Wed Aug 5 | `examples/toxicity_jigsaw/`: Jigsaw toxic-comment prompt iteration end-to-end | example + integration test | done (built Sun Aug 9) |
| Thu Aug 6 | `examples/intent_clinc150/`: CLINC150 intent classification; genericity fixes the second prompt target surfaces | example + tests | done (built Sun Aug 9) |
| Fri Aug 7 | Floor-model validation on the prompt path; demo-clean pass | validation | done (built Sun Aug 9: 5 live gemma4:12b runs) |
| Sat Aug 8 | Buffer + carried items from earlier cut lists if green | fixes | done (built Mon Aug 10: all 7 carry-ins resolved — 6 shipped, and carry-in 5's second half built, measured and reverted on the evidence, see entry) |
| Sun Aug 9 | **Release v0.5.0** — SLIPPED to Tue Aug 11 (Tony's call 2026-08-09): v0.5 ships classification AND regression together rather than half the promise. "Prompt iteration for ML tasks" is a claim worth a two-day slip | v0.5.0 out | done (released 2026-09-09: the code was complete Aug 11 and the release waited on the certification run, which ran Sep 8-9 and found two defects; see the release entry) |

---

## Sprint 4: v0.6, DLModelTarget vision. Sun 2026-09-13 to Sat 2026-09-19, release Sun 2026-09-20 (released 2026-09-20)

*(Re-planned 2026-09-12. The Aug 10-15 dates never ran: sprint 3's code was complete Aug 11 and its release waited on the certification run until Sep 9. Re-dated again 2026-09-14: Tony's call to put the data linker before the target added three build days, so the release moved one Sunday. Corrected 2026-09-16, Tony's two calls: a row is a calendar day and a day can carry several PRs, and the release is back on Sun Sep 20. The three linker PRs took two days, and by the end of Wed Sep 16 six PRs had shipped, so the rest of the image build fills Thu Sep 17 to Sat Sep 19, and Sun Sep 20 holds the RTX 4050 validation, the certification run and the release. The carry-ins below move to sprint 5, v0.7. The `Sprint 4 Day N` entry headings count calendar days from Sun Sep 13; merged PR titles and commits keep the numbers they shipped with.)*

**Carry-ins from the v0.5 certification and the sprint 3 evidence (recorded 2026-09-12, ordered by what a user feels first; all twelve moved to sprint 5, v0.7, on 2026-09-16 when Tony put the v0.6 release back on Sun Sep 20):**
1. **A prompt iteration scores about eight full 100-record variants, and most of them lose.** Ranking on 25 paired records is nearly as reliable as on 100 for choosing which of eight to keep, so exploratory variants should be ranked on a small slice and only the survivor promoted to the full slice before `submit`. Roughly four times fewer calls per iteration, on any backend. Tony's framing on 2026-09-08: "API calls taking time is no reason to argue time"; latency is the user's hardware, the multiplier is ours.
2. **The prompt floor scores exactly 0.0 under f1 when the positive class is not the majority.** The contended irony run banked "not ironic" for every row and scored 0.0. The baseline prompt's answers for the loop holdout are already in the answer cache from `baseline()`, so the floor can re-submit them at zero model calls and score the baseline number instead of zero.
3. **`prompts.yaml` does not label a fallback-banked or a duplicate version.** Both stamps exist on the experiment (`floor_banked`, `duplicate_submission`); the record does not read them. `kind: fallback` and `duplicate_of: vN`.
4. **The prompt path has no dead-lever guard.** The STS-B run spent iterations 2 to 4 refining one scale-definition lever, all three below iteration 1, and nothing stopped the repeat. The prompt family needs its own lever classes (the `_PROMPT_MOVES` ladder is the vocabulary) and a guard that fires when the same class has lost twice running. Replayed on the recorded STS-B run before it ships: it must fire at iteration 3.
5. **The free-text token cap is a bound, not a size.** 512 tokens for every free-text target; it should be sized from the training answers so a slow record costs seconds, not minutes.
6. **The eval runner never passes `--task`, so a version sweep would measure every prompt dataset as a tabular run** (`evals/runner.command_for`, found 2026-09-13). Still open; it moves to sprint 5 with the other carry-ins.
7. **Both tabular sweeps are lower bounds on churn, measurably.** A live run reached 0.6651 against the v2 ceiling of 0.6467 with bagging over tuned boosting, which neither sweep does. A third axis, hyperparameter search plus ensembling over boosting, is `v3`.
8. **Output-format discipline is a lever no sweep technique contains.** The STS-B winner was "answer with a decimal". Both prompt sweeps gain a seventh technique and STS-B and irony are re-measured; the cache makes the first six free.
9. **`tweet_irony` is the v0.5 certification dataset and has a demo video, but it is not in the repo.** Its folder, `tweet_emotion/`, and both `dataset.toml` files sit under a local git exclude (found 2026-09-13). Track them, with a README.
10. **The Summarizer-authors-dossier change was measured at n=1 per arm and reverted.** Three repeats per arm decide it. Compute, not build time: a nohup queue on an idle night.
11. **The version-over-version table is nearly empty.** Ceilings exist for all nine datasets; the 0.4.0 and 0.5.0 cells do not. Compute on an idle night.
12. **The harness deciding to inspect from the data profile** (since the floor model never sets `want_inspect`). Cut candidate: it needs its own before/after and the week already carries three measured items.

**Goal:** the third problem type. Image classification by transfer learning on a pretrained backbone, on the same loop, validated on the RTX 4050 (the public claim). MPS on the Mac is the smoke path, not the claim.

**Design calls, proposed 2026-09-12 (Tony cuts before Day 1):**
- **A vision dataset is a CSV of image paths plus a label column, or a folder with one subfolder per class.** `load_csv`, the sealed stratified split, scoring, memory, the notebook and the eval suite work unchanged; the adapter adds an image profile for the supervisor, a content hash over the image bytes, and opaque hard links so no path or slot order can carry the class.
- **The split can be the user's, for every family (Tony's call, 2026-09-13).** `--data` alone means the harness splits; `--train` plus `--holdout` means the holdout is sealed exactly as given. A class-folder tree with `train/` and `test/` inside is the same thing for images.
- **Images live in a flat folder with opaque filenames, labels only in the CSV**, so a holdout path can never encode its class. The Critic is told to look for path-parsing anyway.
- **The deterministic floor is a linear probe:** frozen pretrained backbone, embeddings cached once, logistic regression on top. Seconds on any device, and it doubles as the floor cell when a session dies, the same shape as the prompt path's majority answer.
- **The harness owns image loading, tensor caching, a `fit` helper with per-epoch prints and OOM capture, and `submit`.** The agent owns the levers: backbone, unfreeze depth, learning rate and schedule, augmentation, image size, epochs, label smoothing. Harness bounded, solutions open.
- **torch and torchvision live behind a `[vision]` extra.** e2b bundled into core because it is light; torch is not. A vision run without the extra gets one install line.
- **e2b is refused for vision runs** and recorded in LIMITATIONS: the sandbox is CPU and the images would need uploading file by file.
- **Two datasets, chosen by the Day 1 research against seven criteria.** Oxford Flowers102 is the headline example (8,189 photographs, 102 classes, one 329 MiB download; 1 of its images is in the ImageNet-1k training set and none of its test images is a near-duplicate of one) and EuroSAT is the certification dataset (27,000 satellite tiles, 10 classes, MIT licence, zero ImageNet overlap by construction: Sentinel-2 launched in 2015). Imagenette and Imagewoof, named in the 2026-09-12 planning note, are disqualified: every one of their images is an ILSVRC-2012 image, so a transfer result there measures a backbone remembering its own training set.
- **The 4050 laptop runs WSL2 with the CUDA torch wheel, reached over SSH on Tailscale.** The harness has POSIX-only tty checks, so native Windows is not the path.

| Date | Focus | Lands | Done? |
|---|---|---|---|
| Sun Sep 13 | Recipe research (datasets, backbone, MPS versus CUDA, the lever ladder) and the one measurement that sets the budgets: seconds per epoch on MPS at the chosen resolution with cached tensors. Image adapter: path-column detection, class-folder trees, path resolution, image content hash, opaque hard links, an image profile for the supervisor. The user's own split for every family (`--train` + `--holdout`, Tony's call). Two prepare scripts. The `[vision]` extra | RESEARCH_LOG entry, `adapters/data/images.py`, `load_split`, two examples, tests on tiny synthetic PNGs | done (PR #63) |
| Mon Sep 14 | **Two PRs.** **The data linker, PR 1 of 3: the rules ladder and the canonical data folder.** `iterate run --data <folder>` (or two folders, the user's split) works out how images link to labels by rules alone (wrapper folders collapsed, a table joined to the images on an exact key, one-hot columns, a split column, class folders with or without a train and test pair), refuses what the rules cannot prove with the reason and the flags that would settle it, and writes `.iterate/data/<name>/` with `raw_files/`, `train/`, `holdout/`, the two CSVs and the plan. **The Linker, PR 2 of 3.** The sixth specialist: when the ladder refuses a folder for want of a choice (no rule fits, two tables tie, two key columns disagree, a label table no key resolves), a capped listing of the folder with the join share the harness measured for every column goes to the model, and it answers one tool call: the table by number, the key and label columns by name, how the key names the file from a closed list. The harness rebuilds and measures the frame through the same function a rules plan goes through and reads the task from the labels; the pause becomes a conversation, yes, no, or a correction in plain English, five rounds at most, `--yes` in scripts; an accepted plan is remembered so a folder never goes through the Linker twice | `schemas/link.py`, `adapters/data/linking.py`, `adapters/data/workspace.py`, the folder lane in `run`, tests; `core/linker.py`, prompts.yaml `linker` keys, `plan_from_choice`, the pause loop, tests, a live pass on gemma4:12b | done (PRs #64 and #65) |
| Tue Sep 15 | **The monitor, PR 3 of 3.** Seven deterministic checks on the rows the folder will be written with, before the pause asks for a yes: coverage both ways, duplicate and conflicting labels, byte twins across the split, lookalikes at a coarse hash, a second label source, class balance, id-like columns that span the split; a typed report printed under the link block, saved as `monitor.json` beside the plan, and a counts-only brief with a seat in the supervisor's data summary (filled on Wed Sep 16). Reports, and never a relabel: a byte copy of a training image is left out of a holdout we make, and comes out of a holdout the person gave when they answer `drop` at the pause (Tony's call). The label audit and the vision spot check follow the target | `schemas/monitor.py`, `adapters/data/monitor.py`, `workspace.sides`, `linking.table_rows`, the pause, tests | done (PR #66) |
| Wed Sep 16 | **Two PRs.** **The image target.** `DLModelTarget`: probe baseline, a typed recipe runner for `run()` with a budget planned before the first step, code job and scoring, meta.json with family vision, and the `prepare_images` step that hands it byte-named copies. Vision ceiling sweep through the target's own `run()`, twelve recipes, no LLM, run overnight on MPS. **The image baseline and number labels, PR 1 of 3 for the image run (Tony's calls, 2026-09-16).** The baseline is a plain CNN trained from zero with basic prep, a classifier or a regressor: three blocks of 3 by 3 convolution, batch norm, ReLU and pooling at 32, 64 and 128 channels, a pooled linear head, 64 px, 20 epochs, AdamW with cosine decay, no augmentation, and no time cap, so it is the same model on every machine; the pretrained probe and every fine-tune become tries. Every mode predicts classes and numbers, so the image target learns numbers: a one-output head on labels scaled by the training mean and spread, mapped back before scoring, a ridge probe, the regression panel with Pearson and Spearman, and refusals that count the labels that are not numbers. A number-label example, NASA storm wind speed from satellite frames, split by storm, with its ceiling; the sweep's second version puts the baseline first, drops label smoothing for numbers, stores no error bar for them, and replaces an older image sweep's record while keeping the better ceiling; the eval corpus takes a `holdout` file. And the code path's scoring stops cutting a whole-number regression target's predictions to integers | `targets/dl.py`, `adapters/data/images.prepare_images`, `evals/vision_ceilings.py`, the family key in the corpus and the runner, two `dataset.toml`, headroom numbers; `targets/dl.py`, `core/codegen.py`, `evals/vision_ceilings.py`, `evals/corpus.py`, `examples/cyclone_wind/`, three ceilings re-measured | done (PRs #67 and #68): Flowers102 0.892 to 0.974 and EuroSAT 0.899 to 0.986 against the probe, then 0.554 to 0.974, 0.950 to 0.986 and storm wind 13.12 to 8.87 knots against the plain CNN |
| Thu Sep 17 | **What a run may read and what a cell may touch, for every local run, PR 2 of 3 for the image run (Tony's calls, 2026-09-16).** The harness reads only the data it was given, `--data` or `--train` and `--holdout`, and refuses an image path or a link that leads outside it before any file opens, with a quoted copy command to paste and each side of a user's split read against its own CSV's folder; a `--labels` table is read where the user names it. On a Mac every local kernel runs inside the macOS sandbox, opening its own folder, the files the run gave it, its Python and one folder for model weights, with none of the harness's API keys, and a one-line notice where no sandbox exists, in place of the cell guard by name planned at the target's gate. A cell never runs an installer, on any venue, and a missing import is installed by the harness through four routes planned by dry runs: install as is, restart the kernel with the new package (what the kernel held is lost and rebuilt), save it for the next run, or refuse. on `--compute local` torch and torchvision never change mid-run (Tony's call, 2026-09-13) and every install carries a constraints file, while e2b installs into its disposable sandbox as before; `ensure_torch`, which installs them at the start of an image run, is built for Fri Sep 18 to call | `adapters/data/inside.py`, `images.py`, `linking.py`, `workspace.py`, `adapters/compute/confine.py`, `adapters/compute/deps.py`, `kernel.py`, `codegen.runs_installer`, `coder.py`, prompts.yaml notes, `userconfig.cache_dir`, tests, live tabular and prompt runs sandboxed | done (built Thu Sep 17) |
| Fri Sep 18 | **The image run itself, PR 3 of 3.** The CLI dispatch for both input forms with the e2b refusal, torch and torchvision installed at the START of an image run when missing through `deps.ensure_torch`, under the one-time consent and with no time cap, the image line for a cell's BLOCKED note and the install rule in the image prompts, the vision preamble with the MPS pool cap and the fallback flag, the floor cell, GPU memory freed between cells, `fit()` for the pinned backbones and the plain CNN, its budget counted from the top of the cell, and helpers to submit predictions and probabilities from the agent's own torch code with any model research names (Tony's call); the metric step keeps the agent's choice with image wording, and an explicit `--metric` wins (Tony's call); prompts.yaml image keys for all four roles; the supervisor's image levers gated on this run's evidence rather than a fixed order; the copy of `monitor.json` into the run folder and the supervisor reading its brief; the eval runner's dev gate for image cells lifts with the CLI guard, tied by a test; torch never shares a process with a lightgbm fit, measured 2026-09-16. First live MPS run on gemma4:12b. | wiring, prompts, tests, first live run | done (built Fri Sep 18): EuroSAT, 3 iterations, no `--metric`, 43 min; baseline accuracy 0.9496 reproduced exactly and `f1_macro` 0.9477 to 0.9848; the harness installed torch and then timm for the agent's own code; `monitor.json` lands on the folder form only, and the CSV form has no report to copy |
| Sat Sep 19 | What only running finds: guards for the failure modes a 12B produces writing torch, a notebook check that training curves render, LIMITATIONS rows for MPS nondeterminism, e2b and floor-model torch reliability. Then the release gate per the standing checklist: README and LIMITATIONS sync, the version bump, launch drafts, the examples index and RESULTS regenerated | fixes, tests, docs, the release PR | slipped to Sun Sep 20. Saturday went to Day 6's review and merge (PR #71) and to the first live runs on merged main 6485be3, Flowers102, storm wind and a one-iteration table check; none of this row's build work started. See the Day 7 entry: the guards for a 12B's torch failures were not built because no such failure was seen, the MPS row says what was measured, which is repeatability, and RESULTS was left as it is |
| Sun Sep 20 | RTX 4050 validation, the public claim: install over WSL2, same example, CUDA versus MPS epoch time, VRAM peak, an OOM run to prove capture, and the example READMEs updated with the 4050 numbers. Tony's certification run on the vision example with measured headroom (sequential, Ollama restarted first), then **release v0.6.0** | validation notes, v0.6.0 out | Day 7's work ran today, with a cut line: PRs #72, #73 and #74 merged; the keep-best guard for image sessions: #76 merged, main 7e0c8b9; the re-certification run: laptop_price rmse 411.89 to 292.38 in 6 iterations, 73.3% of headroom against 56.2% at v0.5, no traceback; the prompt run: tweet_irony, 2 iterations, no traceback, cells confined, the researcher on the prompt wording, f1 0.7652 at baseline then 0.7544 and 0.7652, so no candidate beat it; Flowers102 on the keep-best code: accuracy 0.5544 to 0.9811, 3 iterations, 39 min, `KEPT` printed twice, both on ties; the RTX 4050 validation: SLOT_4050; Tony's certification run: SLOT_CERT; then the release, see the release entry |

**Background compute on idle nights, not build days:** carry-in 10 (the Summarizer A/B at three repeats per arm, about 4.5 hours) and carry-in 11 (the 0.4.0 and 0.5.0 cells), both now in sprint 5 with the other carry-ins. Both are nohup queues, run only when no live run is on Ollama.

**Cut list if the week runs long, in cut order:** the cloud-GPU `ComputeBackend` interface, the second corpus dataset beyond its ceiling row.

**Cut on 2026-09-20:** the cloud-GPU `ComputeBackend` interface, to post-v1.0. The week ran long by one day, Saturday was the release gate's day and not a green buffer, and LIMITATIONS now homes the row on the backlog. The second corpus dataset was not cut: EuroSAT, Flowers102 and the storm frames all have ceilings.

---

## Sprint 5: v0.7, cost-constrained serving. Mon 2026-08-17 to Sat 2026-08-22, release Sun 2026-08-23

**Carried in from sprint 4 (moved 2026-09-16):** the twelve carry-ins listed under sprint 4, from small-slice-then-promote prompt scoring to the harness deciding to inspect. Their dates re-anchor with this sprint's at the v0.6 release.

**Not re-dated yet (2026-09-20):** the windows in this heading, in the sprint 6 and sprint 7 headings and in the phase and release tables above are the ones set before v0.5 waited on its certification run, and all of them are past. The v0.6 release PR changes none of them. The new dates are Tony's call.

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
| Cloud-GPU adapter | interface ~v0.6, implementation later | interface: sprint 4 Sat if green, else post-v1.0 (CUT 2026-09-20 → post-v1.0); implementation trigger-based |
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

### 2026-09-20 | Sprint 4 release | v0.6.0 released

**Task:** release mechanics per the standing checklist, on the same day as Day 7's build, with a cut line.

**Release gate (step 1):** v0.6 touched the loop for every family (the stopped-cell restart, the import watch, the install routes, the macOS sandbox), and Day 7 changed `supervisor.py` on the path every regression run takes, so the gate as planned is live runs on the final code, one at a time, Ollama restarted first, from an environment holding scikit-learn 1.8.0. Each line below is what the plan asks for, and says nothing about the result until its token is filled. The re-certification as v0.4 and v0.5 ran it, laptop_price, rmse, gemma4:12b, 6 iterations, against their 55.4% and 56.2% of headroom, and the first live run with the lost-try guard awake on a lower-is-better metric: main 7e0c8b9 frozen by `git archive`, a clean venv on scikit-learn 1.8.0, 60 min, exit 0, no traceback, cells confined. 411.8904 → 427.0786 → 426.7399 → 318.3335 → 298.2250 → 298.2250 → **292.3833**, which is 73.3% of the 163.04 headroom. One run per release cannot separate that from a 12B's run-to-run spread, so it reads as a pass and not as an improvement. The lost-try guard fired on 2 of the 5 later decisions and read the direction correctly both times ("holdout 318.3335, did not beat 298.2250"). Both times the supervisor repeated itself and the run fell back to an untried lever: `imbalance-or-threshold`, a classifier move that cost iteration 5 as a labelled duplicate, then `hyperparameter-search`, which produced the best. What it refused is the finding: `gradientboosting` and `onehotencoder`, which every iteration's pipeline carried, the best included. The guard asks whether some experiment carrying the marker lost to the best, and not whether the best carries it too. It did not wreck the run, so the direction fix stays in v0.6 and the coarseness is a v0.7 row. The waste was labelled: 4 lever-gate fires, 2 identical-gate fires, 8 cell errors inside sessions that all finished, and one critic flag, holdout 298.23 against a best validation of 308.79. A 2-iteration prompt run with research on, the first live prompt run on the Day 6 kernel, passing on no traceback, confined cells and no "tabular-applicable" in the researcher's second call: passed. tweet_irony on f1, main 7e0c8b9 frozen, 68 min, exit 0, no traceback, and the prompt family's confined line printed. Ollama was reached through a recording pass-through so the request bodies could be read: 1,585 calls, 1,559 of them record answers, and the researcher's two calls, `plan_queries` and `suggest_techniques`, both carried the prompt family's wording with no "tabular" anywhere in them. The score is not what this run checks, and it did not move: baseline 0.7652, iteration 1 0.7544, iteration 2, few-shot examples, 0.7652, so no candidate beat the baseline in 2 iterations. One cell error, recovered inside its session. Flowers102 with `--metric accuracy`, on the keep-best branch at 3f6c8cb, the guard as merged, before #73 and #74 joined it: 0.5544 to 0.9811 in 3 iterations and 39 min 23 s, no traceback, against a stored ceiling of 0.9743. Then the two runs that carry the public claim, planned from the 0.6.0 wheel in a clean venv; each token records what its run was installed from. Tony's certification run on EuroSAT, `--data examples/eurosat/data.csv --target label --metric accuracy`, the CSV form so the stored 0.9496 baseline reproduces: SLOT_CERT. The RTX 4050 validation over WSL2, CUDA against MPS epoch time, VRAM peak and one out-of-memory run to prove capture, with the rule that a failure scopes the claim to MPS and changes no code after the gate: SLOT_4050.

**Build gate (step 2):** measured after #76 joined the branch: 1,703 unit tests collected with 1 deselected, `ruff check src tests evals` clean (CI itself runs `src tests`), mypy strict clean on 85 source files. CI runs 1,669 of the 1,703 and skips the 34 that need torch or the macOS sandbox. Wheel build and clean 3.12 venv check: SLOT_CERT

**Doc sync (step 3):** README (the top code block, the today table, the status paragraph, a "What v0.6 adds" section with the three input forms and the honest cost line, quick start 3c, the folder flags, where things land, the one-line-form note, the targets and comparison tables, the architecture tree, and a test count that says how many CI runs), LIMITATIONS (day names retired, the stale v0.6 homes moved to v0.7 or the backlog, the row about a later submit rewritten because it had become false, the lightgbm row corrected to what the code does, and new rows for scikit-learn 1.8.0, floor-model torch from one session, the token cap, native Windows, `best.json` for an image winner and the doubled score trail), the examples index and the three image example READMEs with their run commands, the release table and the sprint 4 table here. `evals/RESULTS.md` is left as it is: no live run writes the store, and regenerating it changes only the timestamp and adds a ceiling row for `tweet_irony`, a dataset the repo does not track until carry-in 9.

**Version mechanics (step 4):** 0.5.0 -> 0.6.0 in pyproject, `__init__` and the lockfile, with scikit-learn capped below 1.9 in the same single plain `uv lock`: two lines of `uv.lock` changed, 229 packages, every pin kept (scikit-learn 1.8.0, numpy 2.4.6, pandas 3.0.3, mypy 2.1.0). CI installs fresh, so that commit's run is the first full suite on scikit-learn 1.8.0 beside numpy 2.5.3 and pandas 3.0.6, and the cap ships only if it is green; the LIMITATIONS row ships either way. Tag, publish and the GitHub release follow the merge.

**Launch assets (step 5):** X thread (7 tweets, all at or under 275, counted by script) and LinkedIn post drafted in LAUNCH_POST.md as a v0.5-to-v0.6 diff, with the slipped promise in the does-NOT-do list. The demo is Tony's, recorded from the published package; the shortest live image run so far took 29 minutes.

**What v0.6 shipped:** `DLModelTarget` on the same loop, for classes and for numbers; a plain CNN trained from zero as the baseline, so every pretrained model is a try; `fit()` over three pinned pretrained backbones and the plain CNN, with its epochs planned against the budget and an out-of-memory error reported as a result; the agent's own torch code, scored and submitted under a model name, with the harness installing what it imports; image levers opened by this run's own evidence; three ways in, a CSV of image paths, the user's own `--train` and `--holdout` split for every family, and a folder laid out however it came, linked by rules, then by the Linker, the sixth specialist, with a pause that is a conversation; the monitor's seven checks before the yes; a run that reads only the data it was given, every local cell on a Mac inside the macOS sandbox with none of the harness's keys, and a cell that never installs while the harness installs through four routes; a cell that will not stop has its kernel restarted; three image examples with measured ceilings (Flowers102 0.554 to 0.974, EuroSAT 0.950 to 0.986, storm wind 13.12 to 8.87 knots); the code path no longer cuts a whole-number regression target's predictions to integers; and from Day 7, the lost-try guard and the scoreboard read the metric's direction, prompt runs get their own suggestion prompt, and notebooks settle progress bars. The keep-best guard for image sessions, #76: a fit that does not beat the one already submitted prints `KEPT` and writes nothing.

**Honest state of the evidence:** three live image runs on gemma4:12b and MPS carry the claim so far, all before the final code. EuroSAT on the Day 6 branch: `f1_macro` 0.9477 to 0.9848, 3 iterations, 43 min. Flowers102 on main 6485be3: accuracy 0.5544 to 0.9670 against a 0.9743 ceiling, 98.3% of headroom, about 29 min. Storm wind on main 6485be3, the machine under memory pressure: rmse 13.12 to 9.1617 against 8.87, about 93%, 48.5 min. After the keep-best guard, Flowers102 on branch commit 3f6c8cb: accuracy 0.5544 to 0.9811, 39 min, which is 101.6% of the stored headroom, because convnext_tiny at 224 px is a pairing the ceiling sweep never ran; the gap over the ceiling is 0.68 points, under two standard errors. On the final code: SLOT_CERT, SLOT_4050. Most of each gain is pretrained features. The 12B wrote torch of its own in one of eleven sessions, and that model lost to `fit()`, 0.9478 against 0.9848. Across releases, the trajectory bar on laptop_price: 55.4% of headroom at v0.4, 56.2% at v0.5, 73.3% at v0.6, one run each, so a pass and not a measured improvement. **What slipped:** v0.5's posts and release notes promised cheaper prompt sampling in v0.6, "a quarter of today's calls". It is carry-in 1, and it moved to v0.7 on 2026-09-16 with the other eleven when Tony put the release back on Sun Sep 20; the v0.6 posts and notes say so. The cloud-GPU interface was cut to post-v1.0. The eval suite's version-over-version table is still nearly empty (carry-in 11). **The calendar:** first planned for Aug 16, re-dated on Sep 12 after v0.5 waited four weeks on its certification run. The data linker added three build days and moved the release one Sunday on Sep 14, and on Sep 16 Tony put it back on Sun Sep 20 by moving the twelve carry-ins to v0.7. Day 7 did not start on Saturday, so its build, the gate runs and the release shared one day.

### 2026-09-20 | Sprint 4 Day 7 | What only running finds: the last submit wins, and two guards that took higher for better

**Task:** the Sat Sep 19 row, what only running finds and then the release gate. None of it started on Saturday, which went to Day 6's review and merge (#71) and to the first live runs on merged main, so Day 7 and release day are one day and the plan carries a cut line. The heading keeps the plan's name for the day; the date is the day it ran. The plan gate ran three scouts, the live runs, the release gate and the code left over from Day 6, and a skeptic over all three, and Tony approved every pick.

**Two live runs on main 6485be3, and a table check.** All on gemma4:12b and MPS, one after the other, on a Mac other work was also loading.

| run | finished | baseline | iterations | best against the ceiling |
|---|---|---|---|---|
| Flowers102, `--metric accuracy`, 3 iterations | about 29 min | 0.5544, as stored | 0.9364, 0.9670, 0.9438 | 0.9670 against 0.9743, 98.3% of headroom |
| storm wind, no `--metric`, 2 iterations | 48.5 min, longest cell 453 s of 750 s | 13.1203, stored 13.12; the metric step picked rmse by itself | 9.1617, 9.1981 | 9.1617 against 8.87, about 93% |
| mobile_price, 1 iteration, `--no-install` | under 4 min, 12 cells, one TypeError the model made | 0.9250 printed, 0.9450 recorded | 0.9400 | none quoted |

No harness traceback, no timeout and no `RecipeError` in any of them, and no cell error in either image run. The supervisor's guards fired in the storm run as they did on Friday ("rejected a banked-work re-brief", "lever not ready ... falling back") and the fallback worked. The epoch planner cut two storm fits to 9 and 8 of their 10 epochs. No cell reached its 750 s limit, so the stopped-cell fix is still covered only by its two kernel tests. The table check shows the loop behaves as it did before Day 6, and cannot show more: at one iteration nothing is carried, so it never reaches the lost-try guard or the scoreboard.

**The finding: the last submit wins.** The 12B calls `submit()` after every fit, never conditionally, and `Session._write` overwrites with no check. Over Friday's and Saturday's 8 image sessions, 7 trained more than once, all 7 submitted every fit, and in 3 of the 7 the last was not the best on validation: Flowers102 iteration 1 (val 0.9443, 0.9573, 0.9313), Flowers102 iteration 3 (0.9718, 0.9504) and storm iteration 1 (rmse 9.1694, 8.3456, 8.3266, 8.6088). Friday's EuroSAT run was monotonic by luck. In the storm run the model's next cell says, word for word, "Since I have already submitted the 8.3266 result, I will finish", which was false: its 8.6088 fit had overwritten it. The coder's prompt tells it the harness keeps the best, so the model believed something the code did not do. The knock-on is worse than the score: the losing recipe becomes the carried best. Flowers102 iterations 2 and 3 inherited lr 0.002 (val 0.9313) in place of lr 0.001 (val 0.9573). Storm iteration 2 inherited lr 0.0005, and because that losing fit had also been cut at 9 of 10 epochs, `vision_levers.ready()` returned only the cut-epochs repair, so the whole second iteration re-ran the losing try, asked for 10 epochs again and was cut to 8. LIMITATIONS said a second fit "costs time rather than a wrong answer". That was false and the row is rewritten.

**What the overwrite cost, replayed outside the harness.** These two numbers are evidence and not v0.6 scores. The overwritten Flowers102 try (val 0.9718) scores 0.9584 on the holdout against the recorded 0.9438, and the run's best stays 0.9670; a guess made before the replay, that it would beat 0.9670, was wrong, which is why it was replayed. The overwritten storm try (val 8.3266) scores 9.0190 against the recorded 9.1617, which would have been the run's best at 96.5% of headroom. The Flowers102 replay reproduced the live try exactly, val 0.950381679389313 and holdout 0.9437652811735942 both times; the storm replay matched on validation only, 8.326647102873675, because the recorded storm try was never replayed. And validation is not the holdout: the storm validation fold is random rows while the holdout is whole storms, so validation runs 0.6 to 0.7 knots optimistic, and iteration 2's better validation score, 8.5331, gave a worse holdout, 9.1981, than iteration 1's 8.6088 try gave. Picking by validation is the prompt's own rule, not a promise.

**What else running found.** A stale variable in the model's own print: storm cells wrote `f_high_res = fit(...)` and then `print(f.val)`, printing 9.1694 after fits that scored 8.3266 and 8.6088, and `dossier._val_trail` reads any line that says "val" or "score", so the stored trail carries the false value; every image trail is also doubled, 5 of 5 sessions, because the model echoes `f.val` unrounded beside the harness's own line. One coder reply ran to the 4,096-token cap, 5 min 24 s at 12.63 tokens a second, then a 26 s retry produced the cell: one of about 50 calls, silent in `run.log`, not charged to the kernel budget. Memory on the 24 GB Mac: swap peaked at 14.6 GB in the Flowers102 run and 18.7 GB in the storm run, macOS reported pressure in 31 of 58 and 59 of 98 half-minute samples, the Ollama server process grew from 12 GB to 17 GB, and when Ollama unloaded the model swap fell from 18.6 to 6.6 GB within two minutes; the cause was not found. An image winner's `best.json` says `"model": null` and `"params": {}`. Neither CSV form wrote a `monitor.json`, as LIMITATIONS already said. Both new notebooks were small, 13,131 and 20,495 bytes, only because resnet18's weights were cached on Friday. And the table check's baseline gap is not ours: the same source and the same split score 0.9450 on scikit-learn 1.8.0, what the lock pins, and 0.9250 on 1.9.1, what a fresh install resolved; churn f1 0.5776 against 0.5647, heart 0.8850 against 0.8933. A plain `HistGradientBoostingClassifier` script with no iterate code shows the same gap, and 1.8.0 beside the newer numpy 2.5.3 and pandas 3.0.6 gives 0.945, so scikit-learn 1.9 alone moves every stored baseline. It is not a v0.6 regression, since v0.5.0 on PyPI resolves 1.9.1 today. v0.6.0 caps scikit-learn below 1.9 and LIMITATIONS carries the row.

**MPS repeats, measured.** The Sat Sep 19 row planned a row for MPS nondeterminism. What was measured is the opposite: two `fit()` recipes re-run in a fresh process, resnet18 at 256 px with flip_crop for 5 epochs on Flowers102 and resnet18 at 224 px for 10 epochs on the storm frames, reproduced their validation score to 16 digits, and the Flowers102 one its holdout score exactly. So LIMITATIONS gets one sentence on the existing row, what was measured and what was not: the agent's own torch code and CUDA.

**No torch failure to guard.** The row also planned guards for the failure modes a 12B produces writing torch. All 15 agent cells in Saturday's two image runs, 8 storm and 7 Flowers102, were `fit()` calls with no cell error, so no shape, dtype, device, missing `model=`, ignored `seconds_left()` or guarded import appeared. The only session that has written its own torch is Friday's third iteration, a timm efficientnet_b0 that ran clean and scored 0.9478 against 0.9848 from `fit()`. No 12B torch failure has been seen, so no guard was built: a guard ships with the failure that proved it, and this one has none. LIMITATIONS says the evidence is one session.

**The four PRs, each small enough to drop alone.** A, #72: the trimmed-plan test read the real clock, had 0.04 s of headroom in the worst of three runs and failed for real at a load of 3; it now freezes the clock the way `cut_epoch` does 11 lines below it and asserts exactly 7.5 s left. First, because `make build` and `make publish` both run it. B, #73, one commit per fix, each test failing on main first: the lost-try guard reads the metric's direction (on main a 4.90 rmse loser was re-briefed and a 3.80 winner was refused against 4.10), the scoreboard reads it too (main printed `HGB 5.3000` as HGB's best when its best was 4.10, so v0.5's certification ran with an inverted scoreboard), and a prompt run gets its own suggestion prompt in place of one asking for "tabular-applicable" techniques, with table and image runs pinned byte for byte to what main sends. C, the keep-best guard for image sessions, gated on a live Flowers102 run that shows a `KEPT` line before the certification run, and otherwise moved to v0.7: merged as #76. The live run printed `KEPT` twice, both on a tie of 0.9840 with 0.9840, so the gate was met in letter only; keeping a better fit from a worse one rests on the replay of the three measured overwrites (0.9573 over 0.9313, 0.9718 over 0.9504, rmse 8.3266 over 8.6088), on a real sandboxed kernel check and on the tests, and LIMITATIONS says so. The same run showed the time budget planning convnext_tiny at 224 px down to 2 epochs whatever the fit asked for, a v0.7 row. D, #74: the notebook writer joins neighbouring stream outputs and applies carriage returns the way a terminal does, the rule nbconvert uses, and pretrained-weight loads pass `progress=False`; Friday's real `best.ipynb` goes from 1,247 stream outputs and 154,626 bytes to 8 and 13,727, with 0 of 55 printed lines lost. The "training curves render" check became a test that every epoch, FIT, MODEL and SUBMITTED line survives and no tick line does; a plot would have been a feature on release day.

**What review found.** On the plan, the skeptic found three holes and all three held: nobody had planned the re-certification run, which is checklist step 1 and a blocker, since Day 7 changes `supervisor.py` on exactly that run's path and a one-iteration accuracy run cannot reach the guard; nobody had planned a prompt run, though the prompt family had no live run on the Day 6 kernel; and the docs would have quoted numbers from code about to change. So the order became code, then runs on that code, then docs. It also kept the riders out of the two-bug PR: the scoreboard fix rides as its own commit on Tony's yes, and the two-line mypy fix waits for v0.7, since `make build` uses the locked mypy and CI never runs mypy. On #73, no blockers and five small limits in four LIMITATIONS rows, none safe to change on release day, all homed in v0.7: the fallback can offer classifier moves on a regression run, and in the same row the newly awake guard can fall back to a table move on a prompt run scored with a number; the researcher's goal line says "raise" even on rmse; the guard and the scoreboard read every earlier run's history whatever its metric; and the guard matches markers as substrings, so "iterations" can read as a re-try of `ratio`. On #74, joining outputs one add at a time took 4.4 s at 100,000 outputs and now joins once; the planned regex took 1.2 s on one 80,000-character line, so the rule is plain line-by-line code; and text printed before a bare carriage return on the same line is dropped by design, replayed over 2,015 real captured cells with no line lost.

**What is not fixed, by design.** Each has its LIMITATIONS line and a home. v0.7: the recipe in an image winner's `best.json`; the dossier trail rule, because it reverses a test merged in #71; a log line for a reply at the token cap, seen once; a task-aware fallback menu, which would change every regression run with no live check; a monitor report for a CSV given as a file, 50 to 70 source lines in every CSV image run; the two-line mypy fix for a fresh 3.12 dev install, the CI ruff scope and the pyproject description, none of which blocks the gate. Monitor: the stopped-cell fix live, and Ollama's memory growth. Left alone: `evals/RESULTS.md`.

**Tests:** 26 more than Day 6, 1,682 in the suite across 71 files; `ruff check src tests evals` clean; mypy strict clean on 85 source files. CI on main e843ab6 passed 1,648 and skipped 34, the torch-only and Mac-only tests. #76 then added 21: 1,703 in the suite, and CI passed 1,669 and skipped the same 34.

### 2026-09-18 | Sprint 4 Day 6 | Images train now: one loop, a `fit()` the agent steers, and its own torch where `fit()` stops

**Task:** PR 3 of 3 for the image run, from the ten calls Tony approved at the Day 6 plan gate. Before today a folder of images stopped once it was linked and a CSV of image paths stopped right after loading. Now both start the loop a table run takes: profile, metric step, the fixed plain-CNN baseline, research, supervisor briefs, coder cells against a live GPU session, a sealed holdout, the critic. Built in three lanes against one read-out and merged by hand, because all three touch `coder.py`, `prompts.yaml` and `researcher.py`.

**The CLI dispatch, and what an image run is measured by.** `cli.run` grew an image lane whose order is the point: every check that needs no data runs before anything is written or downloaded. For a folder that is the `--compute e2b`, `--task` and `--spec` refusals, `--baseline` without `--source`, the metric and average names and the API key, all before `ensure_torch` installs a gigabyte and before the linker copies a file; for a CSV or a `--train`/`--holdout` pair the same checks plus a class-count check on the loaded labels, before the copies. The `--fresh` memory archive moved below every refusal, so a refused run leaves `memory.db` alone. The linker's plan carries the task into `load_split`, so numbered class folders stay classes. `setup.offered_for_images` builds the metric list from the labels themselves: every name is scored on a small fixture, two classes or three depending on whether this run's labels are binary, `top_k_accuracy` is dropped at two classes, a holdout missing a training class drops the probability metrics, and a 2-class holdout under 3 or more training classes is refused outright rather than training a baseline that then fails at scoring. The whole list becomes the enum of the metric step's tool, so the model cannot name a metric that cannot score these labels, and the choice runs even when the paper search comes back empty. The task line says how the labels were read (`labels read as classes (10 text labels)`), and `--metric` is the override, which is Tony's call 4: no new flag.

**The kernel session.** `core/vision_session.py` is the image twin of the prompt runtime. Its `start()` sets the MPS pool cap before torch loads, decodes both sides once, holds back a fifth of every class with a floor of one row per class (Tony's call 7, the share tables use), and binds the 21 names a cell uses. `fit()` trains the four pinned backbones and prints one line per epoch plus a `FIT` payload that names the recipe it ran AND the recipe it started from; `evaluate(outputs, model=NAME)` prints `MODEL` and the submit helpers print `SUBMITTED`, with `model=` required (call 6) because without a name the gate cannot tell what was tried. Every fit starts from the recipe the run carried in, which the host writes into the session as `incumbent.json`, so a second session does not re-derive the best by hand; a switch between the plain CNN and a pretrained network starts from that kind's own reference instead. `begin_cell()` drops what IPython keeps of the last cell, frees the device and restamps the fit clock.

**The levers, read from what the session printed.** A tabular lever is a class name in the code; an image fit is one call that names every lever at once, so `core/vision_levers.py` reads the printed lines instead. The backbone is measured against the recipe the run carried in and every other setting against the recipe the fit printed under "from", so leaving the plain CNN reads as one backbone move and not as the epochs and depth its reference brought with it. `change_clause()` cuts a move's reason away before any guard reads it, so "resnet18 took 12s, so swap to convnext_tiny" proposes one value and not two. Model names come from `core/timm_models.txt`, 1,101 pretrained architectures captured from timm 1.0.29, plus Hugging Face `org/name` ids: the hand-written pattern it replaces missed 740 of them and matched the English words in a finding. `ready()` opens each lever from this run's own numbers with the fact that opened it, and the supervisor briefs one entry (call 5); a class that is not ready is nudged once and then replaced by the first ready move.

**Installs, and cells that will not stop.** A watch appended last on `sys.meta_path` in all three preambles records a module a cell failed to import even when the cell caught the error, and the coder plans it through the Day 5 routes, installs it and tells the agent to import it plainly (call 1). It never re-runs the cell and never restarts the kernel, whatever the cell did, because an image cell is a fit of minutes. A module under a shared prefix now resolves longest name first and is refused when no such distribution exists, so `google.cloud.storage` finds `google-cloud-storage` and `google.colab` refuses instead of installing PyPI's unrelated `google` (call 2; it reverses one Day 5 test). And for every family, a cell that runs past its limit is interrupted and then waited for, and a cell that swallows the interrupt has the kernel restarted under it (call 8). Without that the next cell was aborted unrun by ipykernel and came back as a success with no output.

**Measured, live on gemma4:12b, EuroSAT, 27,000 images, 3 iterations, no `--metric`, 43 min 18 s.** A throwaway venv with no vision extra, so the run installed torch and torchvision itself before anything else ran. The labels read as classes, 21,600 train and 5,400 holdout copied under names made from their bytes, the metric step picked `f1_macro` with its reason, and the confined line named this run's copy folder and the weights folder. The baseline reproduced the stored number exactly: accuracy 0.9496, `f1_macro` 0.9477, 20 of 20 epochs in 415 s under a loaded machine. Research returned 10 papers and 1 suggestion on image queries.

| iter | what the model wrote | val | holdout `f1_macro` | levers moved |
|---|---|---|---|---|
| base | plain CNN, 20 epochs, 64 px | | 0.9477 (accuracy 0.9496) | |
| 1 | `fit(backbone='resnet18')`, `fit(image_size=128)`, `fit(backbone='convnext_tiny', image_size=128)` | 0.9737, 0.9804, 0.9822 | **0.9848** (accuracy 0.9854) | backbone, image-size |
| 2 | one fit from the carried best, then one at 64 px | 0.9797, 0.9830 | 0.9846 | image-size, epochs |
| 3 | own torch code: timm `efficientnet_b0`, `predict`/`evaluate`/`submit_probabilities` | 0.9417 | 0.9478 | own-model, image-size |

Every guard that could fire, fired. The supervisor rejected iteration 2's first brief as banked work ("the carried best already trains with epochs=3"), then rejected `fine-tune-depth` as not ready and fell back to the ready `epochs` entry. Iteration 2's second cell asked for a fit the budget could not pay for and got a `RecipeError` naming the numbers and the two ways out ("one epoch needs about 359s and 311s of the fit budget are left; halve image_size or use resnet18"). Iteration 3 opened with `fit(backbone='efficientnet_b0')`, was told the four backbones `fit()` has, and wrote its own torch instead, whereupon the harness installed timm 1.0.29 for it in 1.4 s and the cell ran clean: this is call 1 working on a real run. The critic flagged iteration 3's holdout as a suspicious gain over its own validation score, by 0.0038, which is the margin the pre-fix validation trail gave; the shipped dossier reads the same cells as 0.9417 and a gap of 0.0061. The run stopped at `max_iterations` with `best.ipynb` and `best.json` in the run folder and every submission carrying its recipe and its moved levers.

**What the build and integration found.** The build moved off the read-out wherever running the code disagreed with it. The read-out's `deps` snippet passes `module=` into its own recursive `plan()` call, which makes `plan('google-cloud-storage', module='google.cloud.storage')` re-plan itself forever; the loop keeps main's guard and still resolves longest name first. Its `moved()` snippet reads the backbone against the recipe the fit printed under "from", which reads the first pretrained fine-tune as no backbone change at all, because the reference a kind switch starts from is itself a resnet18; the contract test caught it on a real session, and the backbone is now read against the recipe the run carried in. `refuse_labels` counts classes with `.dropna()` where the read-out's sketch used a bare `set()`, which is belt and braces rather than a fix: all three loaders refuse a missing label before it could reach the count, so the bare `set()` had nothing to miscount and the clause is not covered by a test. And the read-out's hint that a `--metric` disagreeing with the link should offer to flip the task is not built: following it skips the metric step and can crash, so the disagreement is printed and the run refuses instead. Each lane then reverted its own fixes one at a time to prove they were load-bearing: 11 of 11 in the kernel session and 16 of 16 in the levers failed the test that covers them, with every source file restored byte for byte afterwards. Before the live run the guard was re-checked as call 5 asks, and 37 of 37 entries `ready()` can emit, copied word for word and as the harness's fallback, pass the real `_vision_violation`; of the four recorded live briefs the two concrete ones pass, while the first draft that names two models and proposes no change is refused. It was not loosened for that brief, because the brief is genuinely ill-formed, and call 5's own contingency, the harness taking a ready lever, is the path the live run then used for real. Integration found two defects the live run exposed, each fixed with a test that fails without the fix. `dossier._val_trail` takes the last float on any line that says "val" or "score", and an image session's own `FIT` and `SUBMITTED` payloads say "val" inside their JSON, so the digest and the Findings cell of `best.ipynb` read `0.9737 -> 0.0000 -> 0.9804 -> 0.9804 -> 0.0000 -> 0.9822 -> 0.9822 -> 0.0000`. The supervisor's own trail already skipped those lines; the dossier now does too. And a float32 softmax leaves its rows about 2.5e-7 off 1, which is inside the tolerance the session accepts them under and outside the one scikit-learn scores with, so scoring iteration 3 printed "The y_prob values do not sum to one": the writer renormalises before the file is written. A `|` inside a code span had split a LIMITATIONS table row. The gates all pass: `ruff check src tests evals` and `ruff check .` clean, mypy strict clean on 85 source files, and `uv build` producing both artifacts with `timm_models.txt` inside the wheel. A CI-like run on a scratch copy, a fresh `.[dev]` venv with no torch and `sandbox_available` forced False before collection, passed 1,610 and skipped 33 of the same 1,643, which is 24 torch-only with "torch is not installed" and 9 Mac-only with "needs macOS sandbox-exec outside another sandbox", and nothing else changed. Three gaps are left as they are, and said plainly: the run folder has no `monitor.json` on the CSV form, because the monitor's report is written by the linker and a CSV given as a file has none; `best.ipynb` is 154 KB because torch's download progress writes 364 and 878 stream outputs into two cells; and no cell timed out, so the stopped-cell fix was exercised only by its two kernel tests. One wall-clock test on code this branch does not touch, `test_dl_torch.py::test_a_trimmed_plan_builds_its_schedule_over_the_epochs_it_runs`, fails inside the full suite on a loaded machine and passes 3 of 3 alone; CI skips it for want of torch, so it only bites `make build` on a busy Mac. The shared `choose_setup` was table-visible at first, since the Researcher stopped returning early when the paper search found nothing for every family; review caught that and the "choose anyway" is now a flag the CLI sets only for images, so a table or prompt run makes exactly the calls main made. And the read-out's own live command omits `--target label`, which every CSV form in the product requires, so the first attempt exited 2 on it.

**What review found, and what it moved.** Review raised 17 findings against the merged branch, every one reproduced against the real code before it was touched. Two were the same blocker from two directions: after an own-code win the host wrote that win's payload into the next session's `incumbent.json`, and a payload naming a `model` is not a fit recipe, so the next session's `Session.__init__` raised `RecipeError: the probe takes 0 epochs` out of the preamble cell and bound no helper at all. It was reproduced end to end through the real `run_supervised`, and it lands exactly where Friday's live run put its own code, on iteration 1. The fix is on both sides of the seam. The kernel drops a carry that names a model, fills a partial carry from THIS run's baseline instead of `Recipe`'s own defaults, and falls back to the baseline rather than raising when a carry does not validate, because no file the host writes may cost a whole iteration. The host now keeps the full payload for the prompt and the lever gate, where it is the honest record of what won, and writes only a fit-shaped recipe into `incumbent.json`, from that session's last `FIT` line. Stripping `model` out of the carried dict instead, which was the first proposal, was rejected on measurement: `moved()` reads `model_name(carried)`, so a re-try of the same model would then have read as a move. Two majors followed. A cell that raised AFTER its submit had its submission and its lever credit thrown away, although the helper writes `predictions.csv` before it prints `SUBMITTED` and the host scores that file whatever the cell did next; the run was scored on predictions it then refused to recognise, and the carried best reverted to the plain-CNN baseline. The readers now take the lines an errored cell printed before it died, with the out-of-memory marker appended last so the ledger stays in the order the cell ran. And the "choose the metric with no papers" gate was not scoped to images, so a table or prompt run whose search came back empty took an ungrounded metric where main took the default: proved at both levels, a unit probe against a main checkout and a 60-row regression CSV through the CLI with both paper clients stubbed to nothing. It is now an `allow_without_papers` flag the CLI sets only for images, so the run shape decides it at the call site rather than the role guessing from its family. Three minors changed code. The fine-tune reference a plain-CNN switch starts from carries no size, and `fit()` filled that hole from the size the fit ARRIVED at, so an explicit `image_size=` on the first pretrained fit read as an unmoved lever; the record now says the session size, which is what that reference would have run at, and the proposed fix of putting the size into the reference recipe was rejected because it was measured to change what trains, dropping a 160 px session to 64. `google` joined the probe list beside `google.colab`: measured in the throwaway venv Friday's run used, which has no top-level `google` at all, the parent import fails first, so the watch records the bare name, and the planner then chose PyPI's unrelated `google` 3.0.0 and told the agent to import it plainly. Dropping every recorded name to its top level would have fixed that and destroyed the rest, since `google.cloud.storage` would match a probe and be skipped. And the caught-import reader was gated on holding an installer, which made its own installs-off branch unreachable from the CLI and left a local run without `--install` silently ignoring a caught import, while the raised-import path told the agent about it; the gate is now on the venue, so e2b is still left to its own sandbox and the two paths agree. The test that covered the dead branch had been built on an (install off, installer present) pair the CLI cannot construct, and now uses the pair it does. One minor was refused on measurement: `refuse_labels` counting classes with `.dropna()` has no covering test because no supported input can deliver a missing label to it, so the BUILD_LOG sentence was corrected rather than a test written to pin a state the loaders already refuse. Five documentation findings were corrected with measured numbers: the critic's iteration-3 margin of 0.0038 is marked as what the pre-fix trail gave, against 0.0061 from the shipped dossier on the same stored cells; the quoted validation trail is the full eight entries the artifact holds; `start()` binds 21 names, not 22; the metric fixture is two classes or three, never the run's own count; and both example datasets USUALLY pick `f1_macro`, which is the project's own measurement, Flowers102 3 of 3 and EuroSAT 2 of 3. One more LIMITATIONS row now says what a folder run refused by a label check leaves behind: not only the torch install it already named, but the linked workspace under `.iterate/data/`, which the run prints on screen above the refusal and the corrected re-run reuses. Running the class-count checks before the copy was rejected: it reorders the approved lane, and the report the pause shows is built from the placed split. Thirteen tests went in and one was rewritten, each shown to fail with its fix reverted and nothing else changed: the whole own-code round trip from the printed worked example through `submitted()` to a session that opens on it, the errored-cell submission, the explicit size on a first fine-tune, the table run that finds no papers, the bare `google`, the local run with installs off, and the `MODEL` half of the validation-trail guard, whose live-run cause had no test at all. `ruff format` now runs on `agent_loop.py`, `critic.py`, `supervisor.py` and `test_dossier.py` too: the gate is on every changed file and those four were failing it, at the cost of about 90 reformatted lines in this diff, every hunk whitespace.

**Not in this PR, by design:** the two bugs on main outside images, a lost-try guard that ignores lower-is-better metrics and the table suggestion prompt on prompt runs, which go in a separate small PR on Saturday (call 10). `want_inspect` for images, and the per-class line `fit()` would print with it, which is v0.7 (call 3). A `--labels-as classes|numbers` flag, since tables have no such flag and a public flag is hard to take back (call 4). A refit on all the training images at submit, which would double every fit, 35 s for resnet18 on EuroSAT and 205 s for convnext_tiny on Flowers102 (call 7). Filtering torch's download progress out of `best.ipynb`, and running the monitor on a bare image CSV so the CSV form has a report to copy: both are design decisions, not fixes. The dead-ends tail is suppressed on the image path, because the Summarizer digests it is built from are not image-aware. The stopped-cell fix keeps only what a cell printed BEFORE the interrupt landed, dropping the line its own `KeyboardInterrupt` handler wrote and the interrupt traceback; main lost the same output, the agent is already told in words that the cell was stopped, and making the traceback the cell's `error` would drop a finished fit out of the lever ledger, so it is Saturday's, not this PR's. Saturday's runs: Flowers102 with `--metric accuracy` against the stored 0.9743 ceiling, and one storm iteration for numbers. Not measured: a cold torch or weights download, a multi-turn image session on anything but EuroSAT, the size of the fold handicap on full-size data, the image critic's three-call live check, and a live tabular or prompt run today to re-confirm the families the kernel and install changes also touch.

**Tests:** 207 more than main, 1656 in the suite; `ruff check src tests evals` clean, `ruff format --check` clean on every changed and untracked `.py`; mypy strict clean on 85 source files. Every image test runs on a fake runner and real PNGs, so the 283 tests across the files this PR and its review touched pass with torch, torchvision and timm blocked from importing, which is CI's shape.

### 2026-09-17 | Sprint 4 Day 5 | Your files stay yours, cells stay in their folder, installs find a way

**Task:** PR 2 of 3 for the image run, from Tony's four calls at the image-run plan gate on 2026-09-16 and the nine picks he approved at the Day 5 gate. The first plan guarded the workspace with a word check on image cells only, let the kernel install nothing while every install pinned every installed version, and had the image run's first cell ban the agent's own torch code. Tony's calls: "do not touch the files" holds for every local run, and a run given `--data` or `--train` and `--holdout` touches nothing outside them; a cell never runs an installer; installs must not block what the research needs, so the harness installs through four routes, and a restart that loses the cached images is fine; and the agent may use any model the research names, so the sandbox keeps weight downloads working. All four are in DECISIONS. This PR changes every local run, not only images. The CLI still stops before an image run trains; that is Fri Sep 18.

**The harness reads only the data it was given.** A new `adapters/data/inside.py` answers one question, whether a file is really inside a folder. It follows every link and resolves `..` after the link, and when the text check fails it compares each parent folder's disk id, because a Mac spells one folder `DATA` or `data`. A CSV of image paths is checked row by row against its own folder before any file opens, and each side of a user's split is read beside its own CSV; on main a holdout CSV in another folder read its images from the training CSV's folder. The frame keeps the checked path, so what is hashed and copied is what was checked, and a row swapped for a link out after the check is refused at the write. A folder goes through a new walk, `confine`, before the inventory and again before the copy: a link out, a dangling link out and a link loop are refused, where main crashed on a loop, a Hugging Face cache snapshot gets its own advice, and a copy that fails is a message, not a traceback. A refusal that leaves through a link names the link and, for a folder or a CSV of relative paths, prints a quoted copy command: `cp -RLc` on macOS, a clone that shares disk blocks, and `cp -RL --reflink=auto` elsewhere. A row that leaves by `..` or an absolute path is told to move the images under the folder or drop the rows, and a link loop is told to remove the link. A `monitor.json` that is a link is not read. The CLI checks an image CSV on both sides before any client, role or kernel exists, and a row outside exits 2. A table key written from the root with a slash, `/shard0/0.png`, links 30 of 30 again, after the first design dropped it to 0. `LINK_VERSION` goes to 2, so each remembered folder relinks once. The first design's refusal of a `--labels` table outside `--data` is gone: its keys only ever reach images inside `--data`, so it protected nothing, and it broke the usual Kaggle layout of `train/` beside `labels.csv`. The eval corpus reads each side beside its own CSV too.

**Cells stay in their folder.** A new `adapters/compute/confine.py` writes the sandbox rules for one kernel, and `LocalKernel` starts inside `sandbox-exec` when a probe says it can. A cell reads system and library folders and where Python imports from, asked of a child interpreter, minus any entry that holds the home folder or a protected path. It reads and writes its own folder, a short scratch folder under `/tmp`, the weights folder and a prompt run's answer cache. Installer programs cannot run by name, and neither can anything a cell wrote; the interpreter's own pip still runs, so the guarantee is that a cell cannot write to any environment, not that it cannot install into its own folder. The protected paths are `.iterate`, the memory db, `--data`, `--train`, `--holdout` and `--labels`. The probe carries a deny rule, because a bare probe passes inside another sandbox and the kernel then dies. Every kernel, sandboxed or not, gets none of the harness's keys, matched ignoring case, with a test that every `Settings` field ending in `_key` is on the list; a prompt kernel gets only its target's key, resolved on the host, so a key kept only in the project `.env` now reaches it. HF_TOKEN stays. Only while sandboxed, TORCH_HOME and HF_HOME point into `~/.cache/iterate/weights` and inherited Hugging Face cache overrides are dropped; without a sandbox they stay as the user set them, since moving them there buys nothing and forces downloads. A refused path reaches the agent as a BLOCKED note naming it, with a line saying where its data already is, picked by family, and a refused program gets its own note. The kernel gains `restart()` and `loaded_modules()`. A local run prints one dim line, `cells are confined: they open their own folder, and model weights under ~/.cache/iterate/weights, nothing else`, or, where no sandbox exists, that cells are not confined.

**Cells never install; the harness does, four ways.** `codegen.runs_installer` reads the whole cell, after IPython's own rewriting of `!` and `%`, before it runs. It refuses an installer run through `!pip`, `%pip`, a subprocess or `os.system` call, `exec` of such a string, spawn functions imported from any module, `posix` and IPython's included, `getattr(ip, 'system')`, and pipenv, poetry and uvx; a cell that only asks what is installed is pointed at `importlib.metadata`. The refusal runs before the repeated-cell check and counts toward the six-error breaker, on every venue. On a missing import a new `adapters/compute/deps.py` plans with uv dry runs, which work out what an install would change and change nothing, with pip's report as the fallback. It then installs and re-runs the cell; or installs, restarts the kernel, re-runs the session setup and the cell, and tells the agent its variables are gone; or saves the package for the next run when it would move something iterate itself has loaded; or refuses with the reason. torch and torchvision are refused by name before any process starts, and a package that clashes with what iterate requires is refused, not saved. After a restart the repeated-cell memory and both error counters start over. Every rung of the one install ladder, pip, then uv, then ensurepip, carries `-c` with a constraints file: a route install pins what the host and the kernel loaded plus torch, torchvision and iterate's requirements, and `kernel.install` and the code-job install pin every installed version, the code job capped at 600 s. Saved installs live in `~/.cache/iterate/installs/<venv id>.json`, keyed by the environment's real path, and run as the first statement of `iterate run` with consent, before pandas loads, one dim line each. Coder rule 3 now reads "Never install packages in a cell", and each route and refusal has its note; without consent the agent is told installs are off. `ensure_torch` is built for Fri Sep 18. e2b keeps installing through its kernel, as on main.

**Measured on this Mac, the sandbox.** Median of 5, interleaved, at a load average of 3.29:

| | Sandbox on | Sandbox off |
|---|---|---|
| Kernel start | 0.513 s | 0.472 s |
| Tabular preamble | 0.184 s | 0.185 s |
| Close | 0.114 s | 0.114 s |

The start times overlap, 0.393 to 0.536 s against 0.384 to 0.523 s, so this says no visible cost, not an exact one. Building the rules takes 0.013 s a start; the probe, 8.6 ms, and the import-path child, 17.0 ms, run once per process. A sandboxed tabular session on a heart-disease copy scored logistic regression and a random forest with cross-validation on 2 workers and fit lightgbm 4.6.0, and the host read 1,800 of 1,800 predictions; a planted OPENAI_API_KEY was not visible. A sandboxed prompt session read 3 answers the host had cached in 0.002 s, saw only its target key while GROQ_API_KEY was planted on the host, and was refused the sibling `memory.db` and a new file beside the answer cache. Through the coding agent, a cell that read an outside CSV got the BLOCKED note naming it, and a cell that ran `uv pip list` was refused unrun with the `importlib.metadata` note; the program note is covered at kernel level, and at coder level with a faked refusal.

**Measured, the installs,** in a throwaway venv pinned to the repo's versions, on a sandboxed kernel driven by a scripted model:

| Route | Package | Plan | Install | What happened |
|---|---|---|---|---|
| install | tabulate | 0.52 s | 0.16 s, 58 pins | the cell re-ran and printed its table |
| restart | plotly 7.1.0 over a loaded narwhals 1.14.0 | 0.59 s | 1.13 s, 55 pins | the kernel restarted in 0.975 s, the same rebuild cell ran again after it, and the kernel stayed sandboxed |
| next run | sktime 1.1.0, moving pandas 3.0.3 to 2.3.3 and scikit-learn 1.8.0 to 1.7.2 | 0.5 s | 6.38 s at the next start | a later `iterate run` from another folder, through `python3` rather than `python`, found it and installed it before pandas loaded, and a new sandboxed kernel imported it on pandas 2.3.3 |
| refuse | a name not on the index, pycaret, torch | | | not found; pycaret needs pandas below 2.2 and iterate needs 2.2 or above; torch never changes mid-run |

A `!pip install catboost` cell was refused unrun, a `!pip list` cell got the importlib note, and the session then scored 1.0 without a floor. The installer check, on 836 honest cells from past runs: none flagged, at 0.50 ms a cell. It caught 44 of 46 adversarial installer cells, both misses building "pip" from pieces, flagged 0 of 23 near misses, and caught 9 of the 13 shapes the skeptic found; the 4 it misses are a script one cell writes and the next runs, which no check of a single cell can see. 17 mutants each undid one fix or pick, and the tests caught all 17.

**Measured, live on gemma4:12b.** mobile_price, 2 iterations, `--no-install`, the same flags on main 2ce2fab and on the branch: accuracy at baseline 0.9450 on both, iteration 1 0.9375 on both, iteration 2 0.9425 on the branch and 0.9375 on main. The branch printed the confined line and ran 33 cells over 2 sessions; no notebook holds a permission error or a BLOCKED note, and its 4 cell errors were TypeError and NameError. The run folders, notebook sections and memory tables match main's. The branch run took 13 minutes and main's 8.5; the model's time was not separated from the kernel's. A prompt run on tweet_irony, 1 iteration on a 20-record loop holdout, printed the confined line with the answer cache, wrote 595 answers to the cache from the sandboxed kernel, and took f1 from 0.5000 to 0.6364; its `prompts.yaml` has the same keys as a main-era run. It took 41 minutes, with `OLLAMA_NUM_PARALLEL` unset. EuroSAT, Flowers102 and the storm frames went through the branch CLI to its "stops here" with the data hashes and counts main gives, and all 14 eval corpus hashes are unchanged, so every stored ceiling still matches.

**What the build and integration found.** Integration found a bug the sandbox made. matplotlib and Homebrew's fontconfig keep their caches under the home folder, which the sandbox closes, so the first `import matplotlib.pyplot` cell of every session took 9.41 s, against 0.50 s unsandboxed, and printed 110,386 characters of stderr, 385 Fontconfig errors among them; the 20,000-character tail the agent reads held none of the cell's own output. It showed because a catboost install pulled matplotlib into the throwaway venv. While sandboxed, MPLCONFIGDIR and XDG_CACHE_HOME now point into the weights folder too: the first session on a machine takes 9.58 s with one Fontconfig line and keeps its output, and later sessions take 0.52 s with nothing on stderr. A Mac-only test fails on the old code. Integration also found that a live model never reached the install routes. In two `--install` runs, each with a steer naming CatBoost typed into the chat, in the first the coder wrapped `from catboost import CatBoostClassifier` in `try/except ImportError`, so the cell never failed and nothing was planned, and in the second it wrote "Since CatBoost is not available, I will stick with HistGradientBoosting" and never imported it. 8 of the 836 honest cells guard a third-party import the same way, 7 of them xgboost. A scripted session with a plain import planned in 1.3 s, installed catboost 1.2.10 and 7 dependencies, re-ran the cell and scored 0.94 in 7.8 s; with the guarded import it planned nothing. Whether the harness should catch a guarded import is Tony's call, and LIMITATIONS says so. The build moved off the read-out where running said to. The sandbox probe runs on first use, not when the kernel is made, because two install tests patch `subprocess.run` and caught the probe when run alone. The sandbox refuses `uv` by its bare name, so `blocked()` looks a bare name up on PATH; the read-out's version gave no note. The saved-installs file is keyed by the environment's real path, and its entries are no longer filtered by interpreter, because the `iterate` script runs `.venv/bin/python3` while `python -m` reports `.venv/bin/python`; an entry stays when the package index is unreachable, and a constraint drops extras, which pip refuses. A refusal names the first link that leads outside, skipping links that stay inside and stopping once a row climbs above the folder, and it names a link whenever any refused row leaves through one, not only the first row. A CI-like run with the platform set to Linux and GNU cp first on the path passed 1,402 tests and skipped 9 of the same 1,411, the 9 Mac-only sandbox tests; none of its 37 kernel starts was sandboxed, and the two tests that paste the copy command ran `cp -RL --reflink=auto` and passed. The eval runner still never passes `--task` for a prompt dataset, on main too; that is carry-in 6.

**What the review forced.** The adversarial pass before the PR, four lenses with a skeptic on each lens's findings, reported 19 findings and confirmed all 19. In the data rules: a user split whose holdout CSV sits in a subfolder, with rows written relative to the training CSV, loaded on main, and on the branch its whole holdout silently pointed at files that do not exist, because only the training rows were checked on disk; the holdout rows are sampled too now, and a missing holdout is refused with a count and the fix. The copy command reached the user inside the error panel, wrapped with borders at the terminal's width, so it did not paste; it now prints on its own line first. In the sandbox: the list of installer programs held exact names only, so a versioned pip, uvx or another interpreter's `-m pip` still ran; the list now scans every folder on the path and the environment's own bin, and the docs say what actually holds, that a cell cannot write to any environment while the interpreter's own pip can still install into the cell's folder. In the installs: the planner turned down every install as unresolvable when uv was not on the path and the environment had no pip, like the repo's own, and at the next start it deleted every saved install; it now finds uv beside the interpreter and in the usual folders, runs ensurepip once, and keeps a saved install it cannot plan. Smaller ones, each fixed with a test that failed first: a pending-installs file of the wrong shape crashed every run before the flags were read; a route install's pinned dry run could add torch or break a package in iterate's own tree; uv's real messages for a torch clash and for a package with no usable versions gave the agent the wrong reason; a namespace package such as `google.generativeai` was refused as part of protobuf; one-cell installer shapes such as `platform.os.system(...)` passed the check; an honest cell with a timing magic and a column named `uv` was refused; table keys written as absolute paths through a link resolved 0 of 10; a file link that loops on itself, or a dangling link inside the folder, still ended in a traceback; the docs promised a copy command for every refusal, a constraints file and a fixed torch on e2b, and hidden API keys on machines with no sandbox; one measurement described the build before the installer rule landed; and the link-naming rules had no test. And `packaging`, which the planner imports, is now a declared dependency.

**Not in this PR, by design:** the image run on Fri Sep 18, which calls `ensure_torch` at the start, adds the image line for a BLOCKED note (the coder picks that line by family, so an image cell refused a path would raise without it), puts the install rule in the image system prompt, and rebuilds the vision preamble's state from disk after a restart. The prompt family's system prompt has no install rule yet; its agent learns it from the first refusal note. A Linux sandbox: the 4050 runs unconfined for now. A catch for a guarded import. Not measured: a cold 1 GB torch download, CUDA version pins, the install ladder on Linux, and whether a 12B obeys "do not retry this import". The Critic's brief naming the workspace, which the Thu Sep 17 row carried before the plan gate, is not in the approved plan. `tests/unit/test_coder.py` stays 81 lines off `ruff format`, as on main.

**Tests:** 240 more than main, 1449 in the suite; `ruff check src tests evals` clean; mypy strict clean. The Mac-only sandbox tests skip on Linux, and the torch checks still run in child processes.

### 2026-09-16 | Sprint 4 Day 4 | A plain CNN sets the bar, and images predict numbers too

**Task:** PR 1 of 3 for the image run, from Tony's calls at the image-run plan gate. The first plan made the pretrained probe the image baseline and left number labels for a later release. Tony's calls: the baseline is a plain CNN trained from zero with basic prep, a classifier or a regressor, "like we do for tabular", with every pretrained model a try research picks; every mode predicts classes and numbers; the baseline has no time cap; and the image run ships as three PRs, this one, then what a run may read and a cell may touch, then the run itself. "99% similar to tabular: the data is different, the models are different, the way of working is the same." All four calls are in DECISIONS. The CLI still stops before an image run trains; that is Fri Sep 18.

**The baseline.** `simple_cnn` sits beside the three pinned backbones: three blocks of 3 by 3 convolution, batch norm, ReLU and 2 by 2 pooling at 32, 64 and 128 channels, a global average pool and one linear head, 93,696 weights plus 129 per output. The baseline recipe trains it from zero at 64 px, or the images' own size when that is smaller, for 20 epochs with AdamW at 1e-3, cosine decay, batch 64 and no augmentation; the prep is the resize, centre crop and fixed pixel scaling every recipe gets. It runs as a fixed job: the epoch plan never trims it and no deadline cuts it, so it is the same model on a laptop CPU as on a GPU. A try may train `simple_cnn` for up to 30 epochs; it has no pretrained weights, so the probe, head-only training and a probe head are refused for it with the reason.

**Measured before choosing, on an Apple M5 with MPS.** With a constant learning rate the CNN swung between 0.823 and 0.937 on EuroSAT across its last three epochs, too loose for a floor. Cosine decay ended at 0.950 and 0.945 on two seeds, its last five epochs inside half a point. Flowers102 at 64 px scored 0.554 in 55 s; at 128 px, with the constant rate, 0.545 at four times the time. On storm frames the CNN at 128 px was worse than at 64, 16.5 knots against 13.1. 64 px it is.

**Numbers.** The target reads the task from the labels and refuses a metric of the other kind, naming both. A number label is checked on both sides of the split, and the refusal counts what is not a number, for example 2 training and 0 holdout labels, before anything is decoded; a constant label and labels too large to scale are refused too. Training labels are scaled to mean 0 and spread 1 with the training mean and population spread, the head has one output trained on squared error, and predictions are mapped back before scoring. On synthetic discs the scaling took r2 from 0.53 to 0.82 for the radius and from -1.74 to 0.78 for the area, whose raw spread the head never reached. The probe for a number is a ridge fit on the frozen features; its weights copied into resnet18 give the probe's predictions to within 3e-7. Scoring is the regression panel plus Pearson and Spearman, a non-finite prediction is a failed result that counts them, the epoch line prints the training r2, and `meta.json` names the task, the label spread and the baseline recipe. Label smoothing is refused for a number.

**A scoring bug on the table path.** The code path coerced every prediction to the holdout target's type, so a regression target stored as whole numbers had every fractional prediction truncated before scoring: rmse 1.53 in-process against 1.63 through the file, on a synthetic set. It now keeps floats for any regression metric. diamonds' price is such a column; its stored treatments ceiling, 527.48, was scored with the cut and is not re-measured. The best treatment's predictions score 527.4767 with the cut and 527.4854 as floats, so the stored ceiling is 0.009 better than the fixed path gives: the model over-predicts by 6.5 on average, and the cut pulled its predictions toward the truth.

**The number-label example: storm wind speed.** The design pass first picked KonIQ-10k, a photo-quality score, and its skeptic sank it: the signal is blur, noise and compression, which a 64 px resize removes, so the baseline would have sat near the mean. A research pass checked a dozen more. NASA's Tropical Cyclone Wind Estimation set won: infrared frames of 494 storms, wind speed in knots, CC-BY-4.0, every file served with no account. It needs a split by storm, since frames 30 minutes apart are near copies; the official test set continues 227 of its 371 storms from training. `examples/cyclone_wind/prepare.py` keeps every 6th frame of every storm, 11,908 frames and 248 MB, puts 99 of 494 storms in the holdout with a fixed seed, asserts no storm is on both sides, and pins sha256 values for both label files and for a manifest of every frame's size and hash. Two surprises on the way: the server returns 403 to Python's default user agent, so the script names itself, and 1,025 frames from 60 storms are 1093 px files with three identical channels, on both sides of the split; in training their wind spread matches the rest, while the holdout's 153 come from 14 weaker storms, mean 38 knots. One more training frame is a 366 px colour file. The harness converts and resizes every image, so nothing needs handling. The gate before any code depended on it, error bars from resampling whole holdout storms:

| | RMSE, knots | error bar |
|---|---|---|
| guess the training mean | 26.6 | 1.7 |
| the plain CNN, 64 px, two seeds | 13.1 and 13.5 | 0.7 |
| resnet18 probe, 160 px | 13.2 | 0.6 |
| resnet18 fine-tune, 3 epochs, 128 px | 9.3 | 0.4 |

The CNN beats the mean by 13.4 knots, 95% interval 10.9 to 15.8, and the fine-tune beats the CNN by 3.8, interval 2.8 to 4.8. A per-frame formula for the error bar came out two to five times smaller than the storm-level one, depending on the row and the formula, so a number-label sweep stores no error bar: it cannot see the storms. The example's README carries the storm-level one.

**The sweep, version 2.** Row 0 is the baseline, run through `baseline()` exactly as a run measures it, then the twelve version 1 recipes unchanged; for a number the label-smoothing rung drops out, and each row records Pearson and Spearman. A dataset may name a `holdout` file, loaded as the user's own split by every sweep; a dataset without one keeps its content key, checked on all 13 corpus datasets. The store keeps the better of two ceilings, and the table sweeps rely on that across their two methods, so a stored ceiling from an older image sweep is replaced instead, with its number carried when it was better and the method saying where it came from; nothing else changes that rule.

**Ceilings, re-measured on an Apple M5 with MPS:**

| | storm wind | Flowers102 | EuroSAT |
|---|---|---|---|
| Holdout | 2,276 frames, 99 storms | 1,636 images, 2 left out as byte copies | 5,400 tiles |
| The plain CNN, the baseline | 13.12 knots | 0.5544 | 0.9496 |
| Ceiling | 8.87 knots | 0.9743 | 0.9857 |
| Best recipe | resnet18 fine-tune, 3 epochs, 224 px, 250 s | convnext_tiny fine-tune, 3 epochs, 160 px, 283 s | convnext_tiny fine-tune, 3 epochs, 64 px, 134 s |
| Headroom over the baseline | 4.25 knots | 42.0 points | 3.6 points, about 22 standard errors |

**What the rows say.** The new baseline moves the floor, not the ceilings. Every pretrained row on Flowers102 and EuroSAT repeated the first version of the sweep to the fourth decimal, so the renames and the number path changed nothing for classes, and both ceilings stand where the target's own PR measured them. From the plain CNN, Flowers102 holds 42 points, 80% of which the resnet18 probe reaches on frozen features alone. EuroSAT holds 3.6: the plain CNN beats every frozen backbone on tiles, 0.950 against 0.899 for the resnet18 probe and 0.944 for convnext_tiny's, so there a pretrained model earns its place only by fine-tuning. The storms sit between the two: the resnet18 probe ties the CNN at 13.17 knots, convnext_tiny's frozen features reach 11.62, and the fine-tunes of resnet18 at 160 and 224 px, resnet50 and convnext_tiny land between 8.87 and 9.13, a tie inside the storm-level error bar. Five epochs scored 9.36 against three epochs' 9.11, and flip and crop 9.74. All three ceilings are ties, all 38 recipes ran the epochs they planned, and the three sweeps took 71 minutes.

**What the build's audits forced.** Four read-only audits of the built code, each with its own lens, the number math, classification unchanged, the eval store, and the tests against their bugs, proved eight findings and the fix pass confirmed all eight. A text label in training reached pandas' raw conversion error in the image profile before the target's counted refusal; the profile now reads the spread from the numbers it can parse. Labels large enough to overflow gave an infinite scale and a 20-epoch fit on NaN; they are refused. The eval runner sent `--train` and `--holdout` to released versions that have no such flags, where the cell would have failed on every sweep; those versions now skip a holdout dataset with the reason. A ceiling carried twice named the wrong source. And four guards had no test that failed without them. Each fix has a test, mutation-checked.

**What the review forced.** The adversarial pass before the PR, six lenses with a skeptic on each lens's findings, reported 25 findings, confirmed 20, six of them the same finding seen by two or three lenses, and refuted 5. The largest were in the docs, not the code. The cut's effect on the stored diamonds ceiling was an estimate that kept only the variance term; re-scoring the same predictions both ways moved it by 0.009, 29 times the estimate, and in the flattering direction, because that model over-predicts price by 6.5 on average. The storm README called the ResNet-18 paper's 10.5 knots a single-frame result on a storm split of its own; the paper feeds the current frame and the two before it and uses the competition's test set, which shares storms with training, so neither published number is measured the way this example is. The CPU time for the baseline, 16 minutes, came from a step timed while other jobs ran; timed alone the step takes 0.072 s at 4 threads, about 9 minutes, and the DECISIONS row carries the corrected number. The per-frame error bar is two to five times smaller than the storm-level one depending on the row and the formula, not three to four; one of the 11,908 frames is a third kind of file; the holdout's large frames come from 14 weaker storms; convnext_tiny's frozen features pass the plain CNN on the storms by 1.5 knots, so the claim that only fine-tuning does was cut back to EuroSAT; the carry rule keeps the better ceiling, not the higher; and the new sprint row had moved one of Tony's attributions onto the wrong clause. In the code: a pretrained recipe asking for more than 30 epochs was told 0 to 30 and then refused at 30; the prepare script never re-checked a frame already on disk, wrote frames in place, and did not retry a response cut short; and no test caught a sweep going back to the single-file loader, or a wrong train_r2 value. Each code fix has a test that failed without it.

**Not in this PR, by design:** everything the run needs, which the Thu Sep 17 and Fri Sep 18 rows list. SKIPP'D as a second number-label example. A loss other than squared error for a number, and test-time augmentation, which the agent's own code can do. A per-storm error bar inside the sweep.

**Tests:** 78 more than main, 1209 in the suite; `ruff check src tests evals` clean; mypy strict clean. The torch checks run in child processes and pass here; where torch is absent they skip.

### 2026-09-16 | Sprint 4 Day 4 | DLModelTarget, and the ladder that measures its ceiling

**Task:** the plan's original Day 2, after the three linker PRs. The third target on the same loop, and the no-LLM sweep that says how much headroom each vision dataset holds. The CLI does not start it yet; that is Fri Sep 18. The sweep ran the night the code was built, so its numbers are in this PR.

**What the target takes.** What the other two take: a CSV of image paths and labels, loaded through `load_csv` or `load_split`. A new `prepare_images` step runs the Day 1 pieces in order: find the path column, make the paths absolute, hash every file, leave a byte copy of a training image out of a holdout the harness split, profile, and copy every image under its own sha256 into `~/.cache/iterate/images/<version>/`. The twin rule is the linker's, so a sweep and a linked run seal the same holdout: Flowers102's 1,638 holdout rows become 1,636 either way. The cache sits outside every project on purpose, because under `.iterate` one hop up from a holdout copy would reach a linked workspace's `holdout.csv`. The step folds the monitor's brief into the dataset's facts when one sits beside the CSV, and picks the size to train at from the median short side: 64 for EuroSAT's tiles, 160 for photographs. The target refuses a path column that is not those copies, and a numeric label.

**The baseline and the recipe.** `baseline()` is the linear probe: a frozen resnet18 embeds every image once and a logistic regression learns on top. `run()` takes a typed `Recipe`: a backbone from a pinned list of three, unfreeze none, head or all, optimiser, schedule, augmentation, image size 32 to 384, epochs 0 to 12, batch size, learning rate, label smoothing, a head started from the probe, seed. Anything outside the lists and ranges is a failed result naming the field, never a raise. Every fit rebuilds the backbone from its pinned weights, so no fine-tune can move a later probe.

**The budget.** 540 seconds per evaluation, counted from the top of the call, so decoding and the probe count too. The runner times ten full training steps, optimiser included, at a zero learning rate so the timing moves nothing, adds a tenth for what train mode costs on top, keeps back the time it will need to predict, plans the whole epochs that fit, and builds the learning-rate schedule over exactly those. A trimmed run therefore finishes its schedule; the first draft stopped resnet50 at 224 px after 4 of 5 epochs with the rate still at 19% of peak. If not even one epoch fits, the recipe is refused with the advice. A clock check on every step is the safety net, and a cut epoch still predicts. Pixels are decoded once per size, and a size that would take more than a quarter of RAM is refused before any decoding.

**Three things the measuring found.** On MPS an out-of-memory error is a plain `RuntimeError`, not `torch.OutOfMemoryError`, so the runner reads the message; and by default the MPS pool pages to disk instead of raising at all. Capping it at the recommended working set was not enough: convnext_tiny at 384 px and batch 64 raised there only after swap grew by 9.7 GB on this 24 GB Mac. Five large recipes set the new cap: resnet18 at 224 px peaks at 4.1 GiB, resnet50 and convnext_tiny at 160 px at 5.3, resnet50 at 224 px at 8.4, convnext_tiny at 224 px at 10.6, none of them growing swap. The runner now starts torch with the pool at 0.7 of the working set, 12.4 GiB here; the 384 px recipe raises at 12.0 GiB with 1.8 GB of swap growth, and convnext_tiny at 224 px still fits and scores 0.976. The cap cuts the paging by four fifths; it does not remove it. One hot epoch at a learning rate of 1e-2 took EuroSAT from its 0.899 probe down to 0.772, so the default stays the Day 1 bench, AdamW at 1e-3. The rate moves in the sweep only inside the head-only and probe-head rungs, so no row measures it alone. And the first draft sorted integer labels as text: with twelve classes the same probabilities scored a roc_auc of 1.0 through `run()` and 0.60 through the code job. Classes now come from the training labels in their own type, the order the code-path scorer uses, and `meta.json` lists them.

**torch and lightgbm cannot share a process on this Mac.** Found by the first full suite run with the torch tests in it: it died with a segmentation fault inside a lightgbm fit two hundred tests later. Measured on its own, torch then a lightgbm fit crashes, and a lightgbm fit then torch hangs, stuck four minutes on a fit that takes milliseconds. Each ships its own OpenMP runtime. So the torch tests run in child processes and one of them asserts torch never loads in the test process, and a ceilings run starts every vision sweep in a child process of its own, since a run over the whole corpus fits lightgbm for its tabular rows. The Friday row carries the rule into the live run.

**Measured before the sweep, through the real loaders, Apple M5, MPS, resnet18:**

| | Flowers102 | EuroSAT |
|---|---|---|
| Images | 8,189 at 160 px | 27,000 at 64 px |
| Linear probe | 0.892 in 19 s | 0.899 in 15 s |
| One fine-tune epoch | 19 s | 19 s |
| Peak GPU memory | 2.4 GB | 1.2 GB |

**The sweep.** `evals/vision_ceilings.py` runs twelve recipes through the target's own `run()` with no LLM: three probes, resnet18 and convnext_tiny at the base size and resnet18 at the larger one, and nine fine-tunes, the Day 1 bench for 3 and 5 epochs, head only, the probe head with SGD, label smoothing, flip and crop, resnet50, convnext_tiny, and the bench at the larger size. The larger size is twice the base up to 224: 128 for tiles, 224 for photographs. The probe is the baseline, a failing recipe is recorded rather than fatal, and every row keeps its seconds and its epochs planned and run. A dataset says `family = "vision"` in its `dataset.toml` to get this sweep; no version runs a vision dataset as an agent cell yet, and a test runs the real command so the eval runner's dev gate lifts on the same day as the CLI's guard.

**Headroom.** Measured on an Apple M5 with MPS, each recipe once, 28 minutes for both datasets, then measured again under the final code after the review: every row but the head-only one matched the first run to the fourth decimal. All 12 recipes scored on both, and none was trimmed by the budget: every fine-tune ran the epochs it planned.

| | Flowers102 | EuroSAT |
|---|---|---|
| Holdout | 1,636 images, 2 left out as byte copies | 5,400 images |
| Probe, the baseline | 0.8918 | 0.8989 |
| Ceiling | 0.9743 | 0.9857 |
| Best recipe | convnext_tiny fine-tune, 3 epochs, 160 px, 259 s | convnext_tiny fine-tune, 3 epochs, 64 px, 137 s |
| Headroom over the probe | 8.3 points | 8.7 points |
| One standard error at the ceiling | 0.39 points | 0.16 points |

**What the rows say.** The ceiling is a tie, not a winner: on Flowers102 resnet18 at 224 px reached 0.9719, inside one standard error of convnext_tiny, and on EuroSAT resnet50 and resnet18 at 128 px both reached 0.9850. A stronger frozen backbone is most of the gain on its own: the convnext_tiny probe scored 0.9566 on Flowers102 in 25 seconds, against resnet18's 0.8918. Head-only training at a learning rate of 1e-2 scores above the probe it starts from, 0.8985 and 0.9196, and far below a full fine-tune; in the first run, before the review froze the backbone's batch-norm statistics for it, it scored below the probe on both, 0.8833 and 0.8596. The probe head with SGD scored below the plain bench on both, 0.9224 against 0.9554 and 0.9685 against 0.9785, the same trap the one-epoch measurement showed. Label smoothing and flip-and-crop each added 0.6 points on Flowers102 and nothing or less on EuroSAT, whose tiles have no upright to learn. Five epochs beat three by about a third of a point on both, inside one standard error on Flowers102. That is the order the supervisor's lever ladder should try on Fri Sep 18: backbone, then image size, then epochs, with augmentation dataset by dataset. The store keys each ceiling by the CSV and every image's bytes, `fc33b843` for Flowers102 and `cdbad85d` for EuroSAT; re-preparing the data from the prepare scripts before Sunday must reproduce those keys for these ceilings to carry over. RESULTS gains the two rows. The results store also holds the tweet_irony prompt ceiling from 2026-08-11, which joins RESULTS with carry-in 9 once that dataset is tracked.

**What the review forced.** The adversarial pass before the PR, three lenses with a refuter on every finding, confirmed seven findings and refuted three; most of the rest of its list was tests that were missing. The largest: the epoch plan timed a forward and backward pass only, so it planned more epochs than fit and built the schedule over steps the deadline then cut. Timing now covers the full training step, optimiser included, at a zero learning rate so the timing moves nothing, with a tenth added for train mode. A 16-bit image decoded almost all white and a 0 to 1 float image all black, because `convert` clips; both are scaled now. A vision dataset's key in the results store covered the CSV only, so an image swapped behind an unchanged CSV reused a stale ceiling; the key now covers the image bytes, and both ceilings were re-measured under it. The test that ties dev's vision gate to the CLI guard ran only the folder form, while a dev cell passes a CSV; it runs both now. The sweep test gave every rung the same score, so it could not see which row became the ceiling; its fake now scores rungs differently. From the rest of the list: a CPU out-of-memory message went unrecognised; head-only training put the frozen backbone's batch-norm layers in train mode, so their statistics drifted away from the features the probe head was fitted on, and with them frozen that rung went from below the probe to above it on both; NaN probabilities raised out of `run()`; one training class and fractional labels were not refused; a relative `XDG_CACHE_HOME` put the copies under the working directory; a missing file named on both sides of the split shared one opaque name; the default size used the smaller of two medians rather than the median short side; and a child sweep whose output could not be read was left running. Each is fixed with a test, and so are the guards the review found untested. Measuring the cap on build day, the check the plan promised, found the rest, in the paragraph above.

**Not in this PR, by design:** everything that needs the run the CLI starts, which the Thu Sep 17 and Fri Sep 18 rows now list: the dispatch, the preamble and floor cell, the prompts, the lever ladder, the workspace guard, the run-folder copy of the report and the supervisor reading the brief, and the installer seam. A vision winner's weights file, since a code winner's artifact is its notebook. efficientnet_b0 stays off the backbone list until it passes an MPS smoke test.

**Tests:** 92 new, 1131 in the suite; ruff clean at CI scope; mypy strict clean. The torch tests run in child processes and pass here; where torch is absent they skip.

### 2026-09-15 | Sprint 4 Day 3 | The data linker, PR 3 of 3: the checks a person reads before saying yes

**Task:** the last linker piece. Before the pause asks for a yes, seven deterministic checks run on exactly the rows the folder will be written with, and their report prints under the link block. No model is involved anywhere in it.

**What the split needed first.** The 80/20 split used to happen inside the workspace writer, after the pause, so nothing before the pause could see both sides. `workspace.sides` now makes it, the pause checks it, and the writer lays out exactly what was checked. The class floor moved with it and learned the split's arithmetic, at least as many images on each side as there are classes, so nine images in three classes are refused with the way out before any copy, where before this PR it was a traceback after `raw_files/` had been copied. A class the stratified split still leaves on one side is not a refusal but a warn from the checks. The workspace name now hashes the two sides, so a holdout with twins dropped and one without land in different folders.

**Seven checks, one report.** Coverage both ways: images no row claims, rows that name no image, files that cannot be read or decoded. Labels: the same bytes under two labels on the training side, a table key listed twice under two labels, and the same inside the holdout, told to the person and kept out of the count the model may see. Twins: the same bytes on both sides of the split. Lookalikes: a 64-bit difference hash one bit off, both thumbnails with detail, which finds re-encoded and resized copies and not crops or mirrors; a note, never a warn, because the hash cannot tell a copy from a plain scene. A second label source: folder names beside a table plan, or a table column beside a class-folder plan, when it agrees on at least half the images. Class balance: a class on one side of the split only is a warn, and a class more than 20 times smaller than the largest is a note. Groups: an id-like column, by name or by shape, whose values sit on both sides of the split. The report is typed, printed under the block with three examples per finding and the way out, saved as `monitor.json` beside `link.json` with up to a thousand details per finding, and its brief, built from counts and shares only, has a seat on the dataset and the image profile that the target fills on Day 4, so no file name and no label can reach a model through it.

**Tony's two calls at the gate, recorded in DECISIONS.** Warn, never stop, and the copies come out of the holdout: when the split is ours, no holdout given, no train and test pair, no split column, a holdout image whose bytes sit in training is left out before the checks run and the report says how many; when the split is theirs the twins warn lists the pairs and the pause takes `drop` or `remove` to take them out of the holdout, listed in `monitor.json`, counted in the linked block, and remembered beside the plan so a re-run drops them again without asking. And no hard-coded class count: the ratio note at 20 to 1 and the absent-class warn are the whole class check, because the metric chosen from the profile is what answers an imbalance. A warn makes even a rules-proven plan wait for a yes on a terminal; `--yes` accepts it in a script.

**Measured on the three archives, after the link.** Flowers102 (8,189 images, a CSV of paths, our split): 6.5 s; two holdout images left out as byte copies of a training image, and two more that look like one at the coarse hash, both re-encodes under the same label. EuroSAT (27,000 tiles, class folders, our split): 4.6 s; no twins, two lookalike tiles, one across two classes. Imagewoof (12,954 images, train and val folders): 4.1 s; two val images byte-identical to a training image under a different breed, one conflicting pair inside val itself, and its `noisy_labels_0` column read as a second source that agrees with the folders on every image. The thumbnail decode is the cost, 0.16 ms per 64 px tile to 4 ms per 1024 px JPEG, so the pass is skipped over 2 GB of images and the byte checks run alone.

**What the review forced.** The adversarial pass before the PR, three lenses with two independent refuters on every finding, thirty-one agents, confirmed one blocker and thirteen smaller holes and refuted none. The blocker: the group-column check walked every column of the table, the label column included, and matched its name list as substrings, so a target called `class_id` or a column called `width` was reported as a leak on every holdout row, a warn that would have made a clean dataset wait and, once the target read the brief, told the model it leaked; the plan's own columns are skipped now and the names match on whole words. The rest, all in the pause and the split: an answer made only of punctuation crashed the loop; any correction that began with drop or remove was read as the twin command and never reached the Linker, and with twins present it dropped holdout images nobody asked to drop, so the command is now the whole answer; a plan remembered from an earlier yes skipped the pause even when the checks warned, and the drop answered on the first run was not remembered, so a re-run wrote a second workspace with the twins back in; a drop that emptied the holdout went through and wrote a holdout with no rows; a copy filed under two labels whose halves fell on opposite sides of our split was reported only as left out, never as a conflict, so the left-out rows now count with the training copy they matched; the floor claimed one image per class on each side and only checked the arithmetic, so a class the stratified split leaves on one side is now a warn and the claim is corrected; a rules-proven plan whose split was refused still asked a question no answer could help; the split line undercounted after a drop; the time read counted the hashing twice for a given split; the writer hashed a given holdout it never used; and one dead line in a test. Every one is fixed and has a test, and the review's untested guards, the one-bit lookalike neighbourhood, a byte copy across a given split, a group found by shape alone, the coverage line when the thumbnail pass is skipped, have theirs. What it could not break: `table_rows` against the old `apply` on 240 random tables, `sides` against the old split inside `write` on every shape, idempotence with the name on the sides, the brief carrying no file name and no label, nothing written before the pause, and every warn and note on the three archives true when the named files were opened.

**Not in this PR, by design:** the copy of the report into the run dir and the fold into the supervisor summary, which need the vision dataset that lands with the target on Day 4; the `facts` seat is there for it. The LLM label audit and the vision spot check, which follow the target. A report for a CSV of image paths given as a file, which does not go through the linker yet. A script flag to drop twins: `--yes` keeps them.

**Tests:** 53 new, 1039 in the suite; ruff clean at CI scope; mypy strict clean.

### 2026-09-14 | Sprint 4 Day 2 | The data linker, PR 2 of 3: the Linker, and a pause that is a conversation

**Task:** the second of the three linker pieces. When the rules ladder refuses a folder for want of a choice rather than for a fault in the data, a model gets one look and proposes; a person says yes, no, or what to change; whatever is accepted is remembered. The three rules from the plan held all the way through: the Linker runs only where the ladder could not settle it, the task comes from the label values and never from the model, and nothing it says is shown before a frame was built from it and measured.

**Where it may speak, and where it may not.** A refusal now says which kind it is. No rule links the images and a table is present, two tables tie, two key columns point at different files, a label-looking table no key resolves, a class folder with folders inside beside a table: those are choices, and the Linker is asked. A folder of parquet files, no images, a split column with a value nobody named, a table that never reaches the holdout folder, a class with one image: those are faults, and the run stops with the reason exactly as before. A rules-only folder never builds a client, and `--labels` with `--key` and `--target` beats both the Linker and any remembered plan.

**What it is shown, and what it may say.** A listing capped at ten thousand characters, about 2,500 tokens: the layout to depth four with image and table counts, the train and holdout folders when the root has them, every table numbered with its row and column counts, up to twelve columns each with their kind and distinct count, the share of rows the harness could match to an image when the column is read as a path, a file name, a stem or a bare number, five sample rows (two, then none, when the budget is tight), and any README, marked as something that may be wrong. It answers with one `propose_link` call: the table by number, checked against the list it was shown; the key and label columns by name, checked against the table's header; how the key names the file from a closed list; and a split column only when the table has one. There is no `task` field and no free text that reaches a plan. The harness turns the picks into a plan through `plan_from_choice`, and the ladder was refactored so that a plan from the rules, from flags and from the model are all measured by the one function: a column the header does not have, a key that resolves under 90%, a split column without train and test values, a table number that was never listed, a JSON blob in prose instead of the call, all end as no plan and a reason. A split column the model leaves out is still found by name, as the rules find it, so the user's split cannot be lost to an omission.

**The pause.** The block reads as before, with one line under it, `the Linker: <its sentence>`. On a terminal the prompt takes yes, no, or anything else, which becomes a correction the Linker reads on its next look together with its previous answer; an empty line is asked again; five corrections at most, then the run stops and names the flags; a correction it cannot turn into a plan says so and asks again, and it still counts as one of the five, since it was a model call. A plan the rules or the flags proved takes only yes or no: a change of mind there is a job for the flags, not for the model. Without a terminal the block prints and the run stops asking for `--yes`, which accepts a measured plan whoever built it. A plan a person said yes to is written under `.iterate/data/plans/`, keyed by the folders' file paths and sizes and every table's bytes, so a re-run on the same data prints the block with `remembered from an earlier yes` and never calls the model, while an edited label file is a different key. A remembered plan is measured again on the folder as it is now before it is trusted, and the key folds in the ladder's version, so a new rule sends remembered folders back through it. Only plans the model had a hand in are remembered; a rules plan is recomputed in under a second and has nothing to save, which also means a yes to a rules plan between 90% and 98% is asked for again on the next run.

**Two folders.** With `--train` and `--holdout`, each folder goes through the rules on its own. When the rules refuse the second and settled the first, the first folder's choice is tried on the second by table name and columns, measured there like any plan and noted as taken from the other folder; only when it does not hold does the second folder get its own proposal. A folder the rules refused always pauses, whichever way its plan came. Corrections reach only the folders the rules refused: the first such folder gets the note, and a second one takes that choice when it holds there or gets its own proposal with the same note.

**Live on gemma4:12b, folders the rules refuse:**
- A table with a key that resolves and three unnamed columns (`region`, `species_code`, `photographer`): `file` read as the file name, target `species_code`, 100% coverage, 15.3 s. Its sentence: "The 'file' column matches 100% of the images as basenames, and 'species_code' contains the categorical labels." The correction "the label is the region, not the species" came back as target `region` in 5.7 s.
- The Kaggle tie, `train.csv` against `sample_submission.csv`: `train.csv`, `id` read as the file number, target `label`, 6.7 s. Its sentence: "Table 2 contains multiple distinct labels, whereas table 1 only contains one distinct label." The right call for the right reason, made from the listing's distinct counts.
- A third fixture, a label table no key resolves beside a metadata table that does, turned out to be a rules case: the metadata table had a column named `rating`, so the ladder linked it and the Linker was never asked. Kept in the log because it is what most such folders look like.

**What the review forced.** The adversarial pass before the PR, three reviewers on correctness, the trust boundary and plan fidelity, confirmed one blocker and six smaller holes, all in the conversation loop and none in the specialist or the ladder, which a differential run of 800 random folders found identical to main. The blocker: a correction whose plan measured fine but left no holdout rows was refused at the frames, yet the refused plan had already replaced the one on offer, so the next yes wrote one plan to the folder and remembered another, and every later run recalled the wrong one and died on the same refusal with no way back but deleting a file. The plan on offer now changes only after its frames were built. The rest: any answer but yes or no at a rules or flags pause went to the model, even with `--labels`, and "Yes." with a full stop counted as a correction; a correction in the pair form re-planned the folder the rules had proved; the plan was remembered before the folder was written, so a class with one image left a memory that blocked every later run; a plan carried over from the first folder to a second the rules refused inherited the rules' name and skipped the pause; the split column found by name could be the label column; the API key check had moved ahead of the folder lane, so a rules-only folder on a cloud backend was refused for a key it never needed; and the non-terminal refusal had lost its coverage number. The trust-boundary pass found the boundary holding (every model field checked against what was shown, nothing written before the pause, a raising client or a Ctrl-C at the prompt leaving no file behind) and three containment gaps: a table number spelled with a superscript digit crashed the run, because `isdigit` accepts what `int` refuses; the model's sentence and a CSV header reached the terminal as live rich markup and raw escape bytes; and the plan key walked a symlink loop and read an unreadable file before the rules ran. Now `isdecimal` with a guarded `int`, printable characters only from the model, every printed block escaped, and a key that folds an unreadable entry in rather than stopping, follows a symlinked folder once and stops at the inventory's depth. The same pass named a pre-existing drop in the pair form, a table inside one of the two folders carrying its own split column lost the rows marked test; that is refused now with the way out. Every one is fixed and has a test, and the reviewers' six untested guards (a remembered plan that no longer holds, the floor at a middle value, a transferred choice that does not hold, `--labels` without `--key` on disagreeing columns, a split column the header lacks, a one-row table and a decimal key) have theirs.

**Not in this PR, by design:** the monitor, tomorrow. Labels in file names, JSON annotations, headerless lists, multi-label columns and nested class folders with no table beside them stay refused with the flags: the Linker can only pick from a table's header, so each of those needs a rung in the ladder, not a smarter model. The run still stops once the folder is written, until the image target and the run switch land.

**Tests:** 75 new, 986 in the suite; ruff clean at CI scope; mypy strict clean.

### 2026-09-14 | Sprint 4 Day 2 | The data linker, PR 1 of 3: rules, and the folder a person can browse

**Task:** Tony's ideal, stated during the Day 1 review: point `--data` at a folder with anything scattered inside, or `--train` and `--holdout` at two folders, and have iterate work out the whole dataset itself, produce the CSV of image path and target, and lay the data out in a canonical folder. The design pass costed it at four pieces: a rules ladder, a canonical folder, the Linker with a confirm pause, and a monitor. This PR is the first two, and no model is involved anywhere in it.

**What a folder run does.** Tony's correction during the review, recorded in DECISIONS: one command, not two, so the linking is a step of `iterate run` rather than an `iterate link` a user has to remember. An inventory walks the folder to depth six, collapsing wrapper folders (`archive/archive/`) and skipping `__MACOSX` and dot-entries. Then the ladder tries shapes in a fixed order and the first plan that builds a frame with enough coverage wins: flags beat rules, a table joined to the images on an exact key beats class folders. The key can be a path, a file name, a stem or a bare integer id, tried in that order; the target is chosen by column name, then by a one-hot block of 0/1 columns that sum to one, then by being the only other column; a split column or a train and test folder pair is the user's split. A plan that resolves 98% of rows or more prints and continues; between 90 and 98 it prints, names what is dropped, and waits for a yes (`--yes` in scripts); under 90 it is not a plan. The block a person reads, verbatim from the one-hot fixture:

```
data: .../onehot
labels: train.csv, column 'image_id' is the file name without its extension, target 'healthy, rust, scab (one-hot)'
task: classification
split: none given, 24 images split here 80/20
coverage: 100.0% of the table's rows found their image
```

**The canonical folder** lands under `.iterate/data/<folder>-<hash>/`: `raw_files/` is a real copy of everything the user gave, so their originals are never touched; `train/` and `holdout/` are hard links into it, class subfolders for classification and a flat folder for regression, so the layout costs no extra disk; `train.csv` and `holdout.csv` carry paths relative to the folder; `link.json` records the plan and the sources. The name is a hash of the source files and the plan, so linking the same folder twice lands on the same workspace and does nothing. The CSVs load through the Day 1 seam unchanged, and the kernel still gets the byte-named copies the image adapter makes: the class folders are for people.

**Measured on the shapes it must handle.** Every real archive on disk and five synthetic layouts, all under a second: EuroSAT's class folders (27,000 images), Imagewoof's train and val trees, Flowers102's CSV of paths, bare integer ids joined on the stem, a doubled wrapper folder with a split inside, one-hot label columns, a path column with a regression score, a split column, and a parquet-only folder refused by name.

**What the tests and the review forced.** The first workspace writer created `raw_files/` before checking whether the copy was needed, skipped the copy, and every hard link then pointed at nothing; the smoke on the real archives found it before any test did. A table with a label column that no key could resolve let the class-folder rule quietly win by reading `images/` and `copies/` as two classes; a label-looking table now blocks that fallback. Then the adversarial review run before the PR confirmed thirteen more, three of them blockers: a blank cell in any text column crashed the ladder, because pandas 3 keeps NaN through `astype(str)` and every column is tried as a key; two folders with the same basename overwrote each other inside `raw_files/`, so the sealed holdout's bytes became the training set; and the common Kaggle layout, `train/`, `test/`, `train.csv` and `sample_submission.csv`, tie-broke to the submission file and wrote a workspace with zero training rows. The rest were of the same family: duplicate keys counted as hits, two key columns that disagreed picked the first, a split column merged `val` into the holdout, a same-length label edit reused a stale workspace, a class label with a path separator made folders outside the split, hidden files inside a class folder became rows, a class with one image crashed the stratified split after the copy. Every one is now a refusal with the reason, a note on the plan, or a value that cannot occur, and every one has a test. The one thing the review got wrong it said itself: `apply` trusting a plan's coverage is only a hole once a plan can come from somewhere other than `plan()`, which is PR 2's problem to close.

**Refusals, all by name:** a folder of parquet or hdf5 files (unpack it to images first), a folder with no images, images with no rule that links them, a label table no key resolves, two tables that link equally well, two key columns that point at different files, a split column with a value nobody named or with two holdout names, a table that never reaches the holdout folder, a class folder with folders inside, a class with a single image, `--key` without `--labels`, `--labels` with two folders, an output folder inside the data, a flag naming a column the table does not have.

**Not in this PR, by design:** JSON annotations, headerless text lists, labels in file names, multi-label columns, nested class folders, and any layout where two plans tie. Those are the Linker's territory, in the next PR the same day. The run itself waits for the image target and the run switch: until then a folder run links, confirms, writes the folder and stops with a line saying so, and the same guard stops a plain CSV of image paths, which before this PR went into the tabular loop on file names.

**Tests:** 56 new, 911 in the suite; ruff clean at CI scope; mypy strict clean.

### 2026-09-13 | Sprint 4 Day 1 | The image seam, and the two datasets that can carry the claim

**Task:** the first day of the third target family. Not the target, and not the wiring: the seam a vision dataset enters through, the datasets that can honestly carry a transfer-learning claim, the research entry behind every choice, and one measurement that sets the budgets. Plus one call from Tony that reshaped the input contract for every family.

**The research, and what it overturned.** Six research angles ran in parallel (datasets, backbones, MPS, CUDA on a 6 GB card, transfer-learning levers, codebase fit), a synthesis merged them, and the eight claims the design rests on were each handed to a skeptic told to refute them. All eight held. The one that mattered most disqualified the dataset the sprint plan had named: Imagewoof and Imagenette carry 1,350 images per class, which is exactly ILSVRC-2012's 1,300 training images plus its 50 validation images per class. Every image is an ImageNet image, so a probe on ImageNet features there is a backbone remembering its own training set. Flowers102 has one overlapping image in 8,189; EuroSAT is satellite imagery no ImageNet backbone has seen. The full entry is in RESEARCH_LOG.

**The measurement (M5, MPS, resnet18, batch 64, images pre-decoded to one uint8 tensor, 3 epochs, no DataLoader):**

| dataset | px | train / holdout | probe | fine-tune, 3 epochs | s per epoch | peak MPS memory |
|---|---|---|---|---|---|---|
| imagewoof | 160 | 9,025 / 3,929 | 0.853 | 0.870 | 28 | 3.5 GB |
| flowers102 | 160 | 6,551 / 1,638 | 0.892 | 0.958 | 21 | 2.4 GB |
| flowers102 | 224 | 6,551 / 1,638 | 0.932 | 0.972 | 39 | 4.4 GB |
| eurosat | 64 | 21,600 / 5,400 | 0.897 | 0.974 | 14 | 1.2 GB |
| eurosat | 128 | 21,600 / 5,400 | 0.919 | 0.981 | 53 | 2.4 GB |

Overlap shows up as a number: Imagewoof leaves under two points to fine-tuning, the other two leave six to eight. Every fine-tune fits the existing 600 s cell and the 1,800 s session holds eight or more, so the tabular budgets carry over unchanged. Decoding 27,000 JPEGs into the cached tensor costs 4 to 6 s; hashing every image's bytes costs under 2 s.

**Tony's call, recorded in DECISIONS: the split can be the user's, for every family.** `--data` alone means the harness splits, 80/20 and stratified as before. `--train` plus `--holdout` means the split is theirs and the holdout is sealed exactly as given, nothing reshuffled. For images a folder with `train/` and `test/` inside is the same thing. His reason was trust: a team that does not want a tool reshuffling its data keeps control of the one thing every score depends on. The three wrong combinations fail before the setup wizard and before any file is read.

**The leak the folder form exposed, and what the review found in the first fix.** A user's class-folder tree names the class in every path, and a class-sorted tree names it in the ORDER too. The first version hard-linked every image under a seeded shuffle of slot numbers. The adversarial review run before the PR refuted it three ways: the rows the kernel receives were still in class order, because only the file NAMES had been shuffled; the seed is public, so the slot permutation was invertible; and a hard link shares the source's inode, so the file's timestamp still sorted by class on a tree written class by class. The shipped rule is structural on every channel: every image is COPIED under a run-scoped image cache with its own sha256 as its name, no suffix (a format could be a label), one fixed mtime, written to a temporary name and renamed so an interrupted copy never sits in a slot; a missing file maps to an opaque path that does not exist; and both loaders shuffle row order once with the fixed seed, membership and labels untouched, which also applies to a user's CSV split. Copying 27,000 tiles costs a few seconds and the disk of the dataset once. The prepare scripts keep a flat layout too, since the examples are public.

**What shipped.**
- `adapters/data/tabular.py`: `split_frame` and `dataset_from_frames` behind `load_csv` and the new `load_split`; a `user_split` flag on the dataset; rows shuffled once with the fixed seed (the review showed a label-sorted holdout would otherwise have handed `--loop-holdout` a single-class slice); empty labels refused; the holdout index kept disjoint from train; a warning when the holdout carries a class train never saw.
- `cli.py` and `deliver/notebook.py`: the delivered notebook of a `--train`/`--holdout` run loads the user's split with `load_split`, where the first version re-split the training file (review finding); `--train` and `--holdout` naming the same file is refused.
- `adapters/data/images.py`, new: image-column detection (strict: one feature column, every value an image suffix, a sample on disk), class-folder trees with or without a split inside, absolute paths, per-file sha256, a data version covering the bytes, byte-named copies, and a header-only profile (sizes, modes, formats, unreadable files, byte-identical images across the split). On its first run the profile found byte-identical images in both Flowers102 and Imagewoof.
- `cli.py`: `--train` and `--holdout`, the exclusivity rule, the same-file refusal, and a refusal on the `--spec` lane.
- `examples/flowers102/` and `examples/eurosat/`: prepare scripts that download, verify checksums, flatten into `images/NNNNN.jpg` with the label only in the CSV, and assert no filename carries a class token. Both ran on the real archives.
- `pyproject.toml`: `pillow` in core, a `[vision]` extra for torch and torchvision.
- Tests: 37 new, 844 in the suite. Synthetic PNGs cover portrait, grayscale, corrupt, missing and cross-split duplicate images; the materialise test asserts that a class-sorted source does not become a class-sorted slot range.

**How the task is decided, settled during the review.** Tony proposed replacing the hard-coded twenty-distinct-values rule with "regression when distinct values exceed 10% of the rows". Measured before deciding, across the nine corpus datasets and six synthetic targets:

| target | rows | distinct | ratio | rule: 20 | rule: 10% | truth |
|---|---|---|---|---|---|---|
| sts_benchmark | 800 | 48 | 6.0% | regression | classification | regression |
| age, integer | 10,000 | 72 | 0.7% | regression | classification | regression |
| score on a 0.2 grid | 1,000 | 26 | 2.6% | regression | classification | regression |
| 40 integer class ids | 300 | 40 | 13.3% | regression | regression | classification |
| 15-row float regression | 15 | 15 | 100% | classification | regression | regression |

The ratio rule breaks STS-B in our own corpus and every score on a fixed scale, because real regression targets repeat values far more than 10% of the rows. The count rule breaks tiny regression sets. The column's kind carries most of the signal, so the shipped rule reads kind before count: text and booleans are classes; a number with a fractional part is regression, always; an integer-valued column is the one ambiguous case and reads as classes while it stays under twenty distinct values; an explicit `--metric` names the task and overrides the guess, where before a regression metric on a low-cardinality integer target was rejected as a contradiction. The run prints what it decided and why when nothing was named. Tony kept this rule over his own on the evidence. The rule now lives in one place, `looks_like_classification`, with the decision carried on the dataset as `task`; the three private copies in codegen, the proposer's profile and the image profile are gone.

**Not here, by design:** the target, the preamble, the folder form of `--data`, and the prompts. The folder form came on Day 2, the target on Day 4, and the preamble and prompts are planned for Fri Sep 18.

**Two things found in passing, both filed on the sprint table.** `evals/runner.command_for` never passes `--task` for a prompt dataset, so a version sweep would measure the prompt corpus as tabular runs; it is carry-in 6, still open. And `examples/tweet_irony/`, `examples/tweet_emotion/` and their `dataset.toml` files are not tracked by git at all, only excluded locally, so carry-in 9 means adding them to the repo, not writing a README.

### 2026-09-09 | Sprint 3 release | v0.5.0 released

**Task:** release mechanics per the standing checklist.

**Release gate (step 1):** v0.5 built a whole new execution path, so the bar was a live certification run by Tony on a dataset chosen for headroom rather than one the sweep had already read as flat: `tweet_irony` (SemEval-2018, 1,200 balanced tweets, the label hashtags verified stripped). It found two defects. The first, one record call generating 8,111 tokens and eating a 600s cell, is fixed in #59 and re-verified on a clean full run: 2,304 Ollama calls, every one HTTP 200, stop on patience, the winner re-scored on the full 240-record holdout. The second, `prompts.yaml` not labelling a fallback or a duplicate, is logged for v0.6. The comment cleanup (#60) rode between the fix and this PR with an AST-identity proof.

**Build gate (step 2):** 807 unit tests, ruff clean at CI scope (`src tests evals`), wheel built and installed into a clean venv, `iterate 0.5.0` verified from the wheel.

**Doc sync (step 3):** README (the today table, the status paragraph, a "What v0.5 adds" section with the four run forms, quick start 3b, the flags line, the targets and comparison tables, the one-line-form note), LIMITATIONS (the two input rows flipped), the release table and the sprint 3 table here.

**Version mechanics (step 4):** 0.4.0 -> 0.5.0 in pyproject, `__init__` and the lockfile. Tag, publish and the GitHub release follow the merge.

**Launch assets (step 5):** X thread (7 tweets, all under 275) and LinkedIn post drafted in LAUNCH_POST.md, feature-first. Demo recorded by Tony from a clean folder on `tweet_irony.csv`.

**What v0.5 shipped:** `PromptTarget` on the same loop, classification and regression; the harness-owned `ask` / `evaluate` / `submit` triple; the sealed-holdout split reused unchanged; `prompts.yaml`; ranking on a fixed slice with the winner re-scored on the full holdout; the technique-sweep ceilings for prompt datasets and the eval suite with a ceiling for all nine corpus datasets; four prompt examples (toxicity, CLINC150 intents, Davidson hate speech, STS-B); all seven v0.4 carry-ins closed.

**Honest state of the evidence:** three live results carry the claim. Toxicity: f1 0.8611 to 0.882, reproduced three times. STS-B: pearson 0.8690 to 0.9326 on the full holdout, where the winning edit (answer with a decimal) is a lever none of the six sweep techniques contains. Irony: 0.7652 to 0.7667 on the ranking slice and 0.756 on the full holdout, against a measured ceiling of 0.7647; on a dataset with 0.02 of headroom the agent reached the sweep's ceiling and did not pass it. The eval suite's version-over-version table is still nearly empty: ceilings exist for every dataset, the per-version cells for 0.4.0 and 0.5.0 do not, and filling them is a day of compute not yet spent. **The calendar:** planned for Aug 9, slipped to Aug 11 for the regression half, then waited four weeks for the certification run. The slip from Aug 11 to Sep 9 was not build time.

### 2026-09-09 | Sprint 3 release gate | Comments say the constraint, the log says why

**Task:** a ten-line comment block above two constants in `prompt_runtime.py` was the visible case; a scan of `src/iterate` found the same pattern 59 times: 26 comment blocks of six lines or more (four of ten or more) and 33 docstrings of fifteen lines or more, concentrated in supervisor, coder, cli, scoring and kernel. Each carried one of three things that belong here rather than in code: a measurement, an anecdote from a live run, or an argument for a design already made.

**The rule applied.** A comment stays when it states a constraint the code cannot show: an ordering, a failure case a regex guards against, a thread-safety rule. Everything else was cut. Module docstrings are one paragraph, what the module is and the one rule a caller must know. Function docstrings are the return value and the constraint.

| | before | after |
|---|---|---|
| comment blocks of 6+ lines | 26 | 0 |
| docstrings of 15+ lines | 33 | 0 |
| comment lines | 885 | 748 |
| source lines | 13,560 | 12,970 |

**Proof of no behaviour change.** The AST of every file, with docstrings stripped, is byte-identical before and after. ruff clean, 807 unit tests unchanged.

**Measurements that lived only in code, now recorded here so nothing is lost.**
- Multiclass threshold levers (2026-08-10): across 57 class-prior reweightings on a 4-class target the best achievable move was +0.0000 on both f1_macro and accuracy, against +0.0036 and +0.0022 for the same experiment run as binary. An argmax has no single threshold.
- The identical-submission gate exists because live runs produced six byte-identical submissions in a row, and a later run wasted 4 of 10 iterations on sibling duplicates a best-only check could not see.
- The prompt path reads `submit()`'s outputs on the live path, not through `score_code_job`; before that was wired, a run improved f1 three times and delivered none of the three prompts.
- Boolean columns: SimpleImputer rejects bool dtype, and a frame mixing bool with string columns takes the numeric path and dies on the first string. The baseline aborted before iteration 1 on any dataset with a yes/no column stored as a real boolean; the fix is casting to object before imputing.
- Free text is not scoreable by this target: exact-string matching rates three correct summaries at 0.0000, which is why `target_kind` needs both the cardinality signal and the one-answer-per-row signal before it calls a column free text.
- The e2b sandbox default lifetime is 300s, shorter than a cell-by-cell session; the kernel renews a 900s sliding lease per cell, under the 3600s Hobby-plan cap, so a crash orphans at most one lease.

### 2026-09-08 | Sprint 3 release gate | One record ate a cell

**Task:** the v0.5 certification run on `tweet_irony` (Tony's, sequential, nothing else on Ollama) died the way the contended one had: two 600s cells killed, the session out of budget at 1200/1800s, the majority-answer fallback banked. This time there was no contention to blame, so the cause had to be in the harness.

**What Ollama's log said.** Cell 2 finished 27 record calls at about 2.3s each. Cells 3 and 4 finished almost none:

| window | record calls finished | one call's prompt | tokens it generated |
|---|---|---|---|
| cell 2, no few-shot | 27 | ~330 | short |
| cell 3, 150 rows | 3 | 386 | 8,111 |
| cell 4, 60 rows | 0 | 386 | 6,562 |

One record per cell. The coder's cell 3 introduced a few-shot block written as one `Tweet ... Result: ironic` line per example, and on one tweet gemma4:12b answered and then kept extending the pattern for ten minutes at 13 tokens a second. Ollama runs with `OLLAMA_NUM_PARALLEL=1` by default, so the other seven workers queued behind that one call until the cell was killed at 600s. Twice.

**The defect.** `_one` in `prompt_runtime.py` sent the record call with no token cap. The plumbing was already there: the Ollama client maps `max_tokens` to `num_predict`, the OpenAI-compatible client passes it through, and the coder has used it since v0.2. The record path was the only caller leaving it unset.

**The fix.** Closed-set and numeric answers are capped at 64 tokens (a label in a tool call is under 20), free text at 512. A cut-off reply is one wrong row, which is the rule this module already states: a prompt that provokes unusable output is a worse prompt. Four tests in `test_prompt_runtime.py`; the suite is 807.

**Why it never showed before.** Three prompt runs and four technique sweeps had passed through the same code and none of them used that few-shot layout. Rare, and total when it hits, which is the profile of a defect a certification run exists to find.

**Secondary, machine-side.** The Ollama runner was holding 18 GB on a 24 GB Mac with the system swapping (context checkpoints at 119 MiB each, up to 32 of them). That slows every call and is not the harness's to fix; restart Ollama before a long run.

### 2026-08-11 | Sprint 3 Day 7 | The regression half, actually run

**Task:** v0.5 promises classification AND regression on the prompt path, and the two-day slip was taken for exactly that. But the regression half had never been run live end to end, and `sts_benchmark` was the only one of the nine corpus datasets with no measured ceiling — so the claim in the release notes had no number behind it. Not a release gate (Monday's tabular re-certification satisfied that), but "we ship regression" with zero live regression runs is the kind of thing that comes back.

**The ceiling first, because a prompt result without one is unreadable.** That was the lesson CLINC taught on Day 5 and it was only ever fixed for classification. The rating task needs its own six techniques, because the moves that shift a number are not the moves that shift a label — a scale has no answers to sit between.

| technique | pearson |
|---|---|
| minimal | 0.8917 |
| describe-the-scale | 0.8467 |
| anchored-examples | 0.9478 |
| use-the-whole-range | 0.8787 |
| reasoning | 0.8969 |
| **scale-plus-examples** | **0.9510** |

Ceiling 0.9510 over a 0.8917 baseline, so 0.0593 of headroom — between toxicity's 0.028 and Davidson's 0.096.

**Describing the boundary ALONE is now harmful three datasets out of three.** `define-the-labels` scored below minimal on both classification sets, and `describe-the-scale` does the same here. It was a finding about label sets; it is a finding about rating scales too. Paired with examples the same description wins or nearly does, which is the shape toxicity showed. Davidson stays the counter-case where the definition costs 0.09, so the best technique still differs by dataset.

**The live run: baseline 0.8690, best 0.9326 on the full 160 holdout, +0.0636.** Four iterations, stopped on patience.

| iter | brief | pearson (ranking slice) |
|---|---|---|
| base | minimal prompt | 0.8690 |
| **1** | **Baseline Measurement** | **0.9222** |
| 2 | Describe Scale (Rung 2) | 0.9201 |
| 3 | Describe Scale (Rung 2) | 0.9165 |
| 4 | Scale Definition for Objects | 0.9211 |

**The winning edit was output discipline, and no technique in the sweep contains it.** "You must provide a decimal score between 0 and 5." A minimal prompt makes gemma4:12b answer in whole numbers, which throws away most of the resolution Pearson needs on a continuous target. That single line is most of the +0.0636. Six standard techniques, none of them mentions decimals — so this is a lever the mechanical sweep structurally cannot find, which is the cleanest argument yet for an agent iterating per dataset rather than a fixed technique list.

Against the sweep: the agent gained +0.0636 on 160 records, the best of six techniques gained +0.0593 on 200. Different slices, so not like for like — the same caveat already on the record for toxicity — but the agent is at least matching a competent sweep and found a move outside its vocabulary.

**A first reading I had to correct, because it was wrong and it changed the conclusion.** Watching the run live I called iterations 2 and 3 a reproduction of the sweep's "describing the scale is harmful" result. Reading the actual prompts afterwards killed that: ALL FOUR versions describe the scale and anchor it with examples, including the winner. None of them was the bare technique. What separates them is density — v1 carries one extra rule, v3 carries three CRITICAL RULES and is both the densest and the worst. The ranking is monotone in how much instruction got piled on, which is the **same mechanism as the Summarizer revert measured the day before**: denser context degrades a floor model. Two measurements on unrelated surfaces pointing the same way is worth more than either alone, and I would have missed it by trusting the lever names in the log instead of reading what was written.

**The gap it exposed: the prompt path has no dead-lever guard, and this run is the first evidence of what that costs.** Iterations 2, 3 and 4 are three refinements of one lever, all three lost to iteration 1, and nothing stopped the repeat. The duplicate gate hashes prompt TEXT, so two wordings of the same move sail through it — the two rejections it did fire were baseline re-briefs. The tabular lever ledger and dead-lever guard were deliberately switched off for this family on Day 3, recorded then as "switched off rather than mistranslated", which was the right call with no evidence. There is evidence now. Re-homed to v0.6, not improvised the day of a release, for the same reason the deterministic inspect step was.

**Not a feature.** Nothing under `src/iterate` changed today. This is evidence for a claim v0.5 makes, plus one honest limitation found by making it.

### 2026-08-10 | Sprint 3 Day 6 | The carry-in list, closed

**Task:** Tony's bar for the release — "1 to 6 actually do it all then only we will do the v0.5 release since it was promised that way". All seven v0.4 certification carry-ins, not the cheap ones.

**Carry-in 7 turned out to be a question about the eval suite, not about the agent.** It read "the agent misses thin margins (churn 1.6%, mobile 2.1%)". The Aug 9 sweep could not confirm it: `brute_force_sweep_v1` measured churn, heart and mobile at EXACTLY zero headroom, three of five tabular datasets unreadable. v1 varies the estimator with the preprocessing fixed, so a margin living in how the columns are encoded is invisible to it however many models it tries.

So the fix was a second axis. `evals/treatments.py` sweeps eight feature treatments through the agent's OWN code path — `build_code_job` writes the same sealed holdout, the same runner executes it, `score_code_job` applies the same ruler — which means a treatment that wins is a thing the agent could actually have written, not a number from a privileged script.

| dataset | baseline | v1 (models) | v2 (treatments) | headroom | best treatment |
|---|---|---|---|---|---|
| churn | 0.6449 | 0.6449 | 0.6467 | +0.28% | frequency-encoding |
| heart_risk | 0.8967 | 0.8967 | 0.9000 | +0.37% | calibrated |
| mobile_price | 0.9450 | 0.9450 | 0.9550 | +1.06% | numeric-interactions |

Sweeping the other three tabular datasets settled a second question: **neither sweep dominates.** v2 also raised diamonds (537.14 to 527.48), while v1 still holds adult_income (0.7259 against v2's 0.7218) and laptop_price (248.85 against 323.73). Four ceilings from treatments, two from model families. A dataset's headroom lives on one axis or the other and there is no way to know which without looking at both, which is why v2 is an addition rather than a replacement.

**The answer to carry-in 7, corrected the same day by a live run.** The first reading of the table above was "the margins are real but 2-5x smaller than the hand estimate, and each one is a feature treatment rather than a model swap". A live v0.5 churn run then reached **0.6651**, against a baseline of 0.6449 and a v2 ceiling of 0.6467 — **eleven times the headroom the sweep could find**, and more than the 1.6% the v0.4 hand pass estimated. Its winning move was a `BaggingClassifier` over tuned `HistGradientBoosting` with a power transform, which is neither a model family in v1 nor a treatment in v2.

So the corrected answer is sharper and less flattering to my own sweep. **The margins are real and LARGER than any brute-force sweep here has found; both sweeps are weak lower bounds on churn, because neither does hyperparameter search or ensembling-over-boosting.** And carry-in 7's premise does not reproduce under v0.5: on churn the agent does not miss the thin margin, it finds an order of magnitude more of it than a competent sweep does. The code was read for leakage precisely because a 1122% capture is when to be suspicious — transformers are fit on the train sub-split only, the holdout is transformed and never fitted, and its labels never leave the host.

What survives from the first reading is the part the sweeps CAN speak to: of the headroom a fixed sweep can find, all of it on these three datasets is in feature treatments rather than model choice. The corpus no longer has an unreadable dataset, and the honest label on every ceiling is still "lower bound", now with a measured example of how loose that bound can be. A hyperparameter-and-ensembling axis is `v3`.

**A real bug fell out of building it: no probability metric could be scored on the sandbox code path.** `build_code_job` listed only `predictions.csv` as an output, so the `probabilities.csv` the script correctly wrote was never collected and `score_code_job` never passed it. Every `average_precision` / `roc_auc` / `log_loss` candidate under `--sandbox e2b` scored as a hard failure — two of the eight certification datasets are scored that way. Measured before and after on churn: `code-gen contract: average_precision needs probabilities` became 0.6362. It survived two releases because nothing tested `ModelTarget`'s two `SupportsCodeGen` methods at all; there are now tests that do.

**A second real bug, found by a test I expected to pass.** The dossier's data-fact extractor skipped any line containing the substring `"val"`, meaning to skip validation scores. It also skipped `missing values: 11`, `unique values in PaymentMethod: 4`, `value_counts: ...` and `interval columns: 3` — the most common EDA output there is. Every one of those facts was printed by a session, dropped by the extractor, and never reached the supervisor. Now word-bounded (`_` counts as a boundary, so `val_f1: 0.55` is still a score), and the word list gained the six things the new inspect step is told to print.

**The free inspect step shipped, and the three things that cut it from v0.4 turned out to be consequences of one wrong assumption.** The cut said it needed a fourth `AttemptOutcome`, must not burn patience, must not count toward `max_iterations`. All three follow from modelling an inspection as an EXPERIMENT. It is not one, and the schema says so: `ExperimentResult` rejects a result with neither metrics nor an error, which is the model refusing to represent an unscored run as an outcome.

Modelled instead as what it actually resembles — a Researcher pass — it needs none of them. The supervisor asks with `want_inspect` (one more field on the emit it already makes, exactly like `want_research`), the harness runs an unscored session that only prints, and the facts fold into the next `decide()` as text. No experiment, no memory record, no terminator interaction, no fourth outcome. The only budget it can spend is wall-clock, capped at `max_inspect_calls=2`.

The inspect session gets its own system prompt rather than the coder's with a sentence added. The coding prompt is dense with instructions to fit a model and submit predictions, and on a floor model those get followed — here, training anything is the failure mode.

**The inspect step is inert on the floor model, and so is a v0.4 feature nobody had checked.** A feature the model never asks for is dead code however well it is wired, so I probed it: gemma4:12b set `want_inspect` **zero** times across 18 live iterations in three runs, and zero times across three targeted probes including one built to invite it (34 unlabelled numeric columns, three iterations without improvement).

Probing `want_research` in the same call is what made it a finding rather than a disappointment. **It never fires either.** It shipped in v0.4 described as reaching the run "when the supervisor asks for it", and on the floor model the supervisor has never once asked — every research pass in every run on record is the automatic iteration-1 pass. So the limitation is the PATTERN, not my new field: a floor model emitting one structured call leaves optional booleans at their default.

Both ship. They are correct, tested, free when unset, and work on any model that sets them. What changes is the documentation, which now says plainly that on gemma4:12b neither fires. The deterministic version — the harness deciding to inspect from the data profile instead of asking — is the shape this codebase normally reaches for ("deterministic guards over prompt nudges", banked in June), and it is re-homed to v0.6 rather than improvised on release eve.

**The Summarizer now authors the dossier**, the other half of carry-in 5. Two changes, both of which alter what reaches the supervisor: the harness's verified observations go into the Summarizer's prompt above the raw cells, and empty insight fields are seeded from them. Seeding is ADDITIVE to an empty field only — observation is a floor under the digest, never a correction of it, because the machine can see what happened and only the model can see why. The deterministic fallback stays pure: it runs precisely when the LLM could not be trusted to have run at all.

This is the one change in the list carrying the June EDA-ledger risk, so it shipped behind a before/after with the decision rule written down in advance: keep only if lever diversity holds.

**It was measured and reverted the same day.** Churn, gemma4:12b, 6 iterations per arm, one variable:

| | arm 0 (withheld) | arm 1 (observations on) |
|---|---|---|
| best average_precision | **0.6651** | 0.6507 |
| gain vs baseline | +0.0202 | +0.0058 |
| distinct lever classes | 5 | 5 |
| most-repeated lever | 2 of 6 | 2 of 6 |
| data_insights across digests | 13 | **16** (+23%) |
| what_hurt items across digests | 9 | **13** (+44%) |

**Diversity held, and I reverted anyway.** The rule as written would have passed it, and the rule as written was incomplete: it names the June SYMPTOM (lever collapse) and not the June MECHANISM (a denser planning prompt degrading a floor model). The mechanism is right there in the last two rows — the change measurably grew what reaches the supervisor, and the score moved the wrong way by 0.0144. A change with no measured benefit, a regression in the predicted direction, on the highest-risk surface in the codebase, the night before a release, ships on hope rather than evidence.

What it does NOT establish is that the change is bad: n=1 per arm cannot separate 0.0144 from a 12B's run-to-run variance. So it is re-homed to v0.6 with 3 repeats per arm, not closed. The seeding never fired for `what_hurt` in either arm (the model always filled it), so the +44% there is the model writing more BECAUSE it read the observed block — which is the density effect, not the seeding.

**Correction to the decision rule, banked:** "keep only if lever diversity holds" is a necessary condition stated as a sufficient one. A context change now also has to show a benefit to earn its place. Absence of the known failure mode is not evidence of value.

**Tabular re-certification: the loop changes cost nothing.** The standing release checklist says re-run the trajectory quality bar on the floor model if the release touched the loop, and v0.5 touched it in five places (the coder's four family parameters, the supervisor's three ladders, the multiclass guard, the Summarizer's inputs, the inspect step). Same dataset and same model as the v0.4 flagship run, laptop price on gemma4:12b, 6 iterations.

| | rmse | gain | capture |
|---|---|---|---|
| baseline | 411.8904 | | |
| ceiling (`brute_force_sweep_v1`, 9 models) | 248.8509 | headroom 163.0395 | |
| v0.4 certification run | 321.5600 | 90.3304 | 55.4% |
| **v0.5 re-certification** | **320.1858** | **91.7046** | **56.2%** |

Marginally better, which is the answer the bar was asked for: nothing in five days of loop changes degraded the agent. The trajectory holds too — 411.89 → 427.08 → 334.46 → 334.32 → 325.62 → **320.19** → 328.58, with monotone improvement across iterations 2 to 5, and five distinct lever classes in six iterations (target encoding, numeric transformation, feature selection, model swap, imbalance-or-threshold). No collapse onto one lever.

**The waste was labelled, not silent, which is the other half of the bar.** The identical gate fired twice, the lever gate three times, and six cells errored inside sessions that still finished. Every one of those is a guard doing its job on a 12B floor model rather than a run quietly banking a re-run as a result.

**Also closed:** carry-in 2 (the Researcher, Critic and Summarizer had no styled TUI events and reached the transcript as dim ambient lines, indistinguishable from routine chatter), carry-in 3 (research cache files record the query that produced them, with backward-compatible reads of the old bare-list format), carry-in 4 (multiclass threshold guard, measured first: a threshold move is worth +0.0036 on binary f1_macro and exactly +0.0000 on multiclass, which is what makes it a dead lever worth naming), carry-in 6 (the flaky TUI test now waits on a condition rather than a fixed 0.3s).

---

### 2026-08-09 | Sprint 3 Day 5 | Regression: the second thing prompts are actually used for

**Task:** Tony's scope call — v0.5 ships classification AND regression, because the LLM tasks people actually put in production are the two that produce something structured enough to act on. "If a photo task is there, companies want the colours he is wearing from an enum and a rating, because only then can they do something." Free-form generation was the gap I had been aiming at and it was the wrong one.

**Most of it was already there.** 16 regression metrics with correct directions, and the registry has always accepted hand-curated non-sklearn entries — `brier`, `mae`, `mse` and `rmse` are exactly that. The blocker was one hardcoded word: `PromptTarget` called `score("classification", ...)`. `ModelTarget` had derived the task from the metric since v0.1.

**The decision I refused to guess: what is an unusable answer worth in a numeric target?** For labels this mattered enormously — mapping the sentinel to the positive class scored 0.857 against an honest 0.800, so garbage would have paid. Measured the same way here, on a 300-row rating task with 10% refusals:

| handling | rmse | pearson | |
|---|---|---|---|
| model answered every row (honest) | 0.701 | 0.890 | the bar |
| **drop the unparseable rows** | **0.691** | **0.894** | **flatters — refusing wins** |
| substitute the training mean | 0.802 | 0.848 | penalises |
| substitute the training median | 0.800 | 0.849 | penalises |
| substitute the worst case | 1.387 | 0.607 | penalises |

**Dropping is the trap, and it is the option that looks most reasonable.** A prompt that refuses on the rows it finds hard scores BETTER than one that answers them honestly, on both metric families. I had expected the mean to be the danger, since the mean is the best constant guess under rmse; it is not, because an honest answer beats the mean. Chose the **median**: it penalises on both families and, unlike worst-case, a handful of refusals cannot swamp the score (30 refusals move rmse from 0.70 to 1.39 and drown out everything the prompt did).

**Kendall, Pearson and Spearman — and they are not in sklearn.** Tony asked whether the metric vocabulary was sklearn-only. It never was, but the three correlations are genuinely absent from sklearn's scorer catalogue, so they are curated on scipy, which is already a hard dependency. Two design points fell out:

*They are selectable but NOT in the always-computed panel.* Adding them to PANEL changed the recorded metric set of every regression run that has ever happened, which a test caught immediately. History has to stay comparable, so they live in a new `CURATED_EXTRAS` that joins the REGISTRY without joining the panel.

*A test asserted every metric names an importable SKLEARN function.* That was true until today. The constraint it was protecting is real — the agent is told the symbol so it can compute the metric itself, and v0.4 burned 16 cells on a name that did not exist — but sklearn was incidental to it. Metrics now carry a `module`, and the test checks importability wherever the function lives.

**Why correlations matter for a rating task, in one measurement:** a prompt that ranks every item correctly but rates everyone three points high scores **pearson 1.000 and rmse 3.000**. For a rating that first verdict is usually the useful one, because a constant offset can be calibrated away and the ordering cannot. And the likeliest failure — answering the same number to everything — makes scipy call the correlation undefined, so it is guarded to 0.0 rather than nan, which would either crash the run or compare falsely against every real score.

**A numeric answer tool, the counterpart of the enum.** `{"type": "number", "minimum": …, "maximum": …}` built from the TRAINING answers, so a rating outside the observed range is not something the model can express. Bounds from training only: a range widened by the holdout would tell the model which values the answer key uses. Number parsing refuses more than it accepts — "GPT-3 rates it 4" and "between 3 and 4" both come back unusable, because naming two numbers invents a precision the model did not express.

**The supervisor needed a third ladder, not a translated second one.** "Define the hard case" names an edge between two answers, and a number has no answers to sit between. The scoring ladder's rungs are what actually moves a rating prompt: describe what the endpoints MEAN, anchor with examples spanning the range, fix a consistent lean, fix predictions bunching in the middle. Selected by `task_for_metric`, so the metric picks the ladder.

**`examples/sts_benchmark/`:** 800 sentence pairs scored 0 to 5. Chosen because it is not a contrived scoring example — it IS the standard one, and Pearson and Spearman are its canonical metrics, so the correlations get used on the dataset the field already measures them with. First example with two input columns, so multi-column rendering finally runs.

**Gates:** 782 unit tests (was 762), ruff clean repo-wide, mypy --strict clean.

### 2026-08-09 | Sprint 3 Day 4 | The second example, and the boundary it forced

**Task:** CLINC150, whose real job was never "another example" — it was to break what the binary task hid. Twenty answers instead of two, hierarchical label names, and a metric that cannot be won on the easy classes.

**It broke three things before a single line was written for it.** Checking the real labels first found each one.

**1. Coercion, fixed twice.** `time` is a substring of `timer`, so the clear reply "the intent is timer" matched two labels and was discarded as ambiguous. Two-label sets never collide, which is why this could not appear until the second example. My first fix — prefer the longest match — then broke the most common binary pair there is: "could be toxic or not toxic" resolved confidently to "not toxic", because comparing labels made "toxic" look like a mere substring. **A confident wrong answer is worse than an unparseable one, because it is invisible.** My own earlier test caught it. Coercion now scans by POSITION with longest-alternative-first, which gets both right, and nine cases are pinned.

**2. Free text scored a confident zero.** Three genuinely correct summaries, scored the way `PromptTarget` scores: **0.0000**, because the only thing measured was exact string equality. Nothing crashed; it just handed back a meaningless number. Now refused before the run with a message saying why, `--allow-free-text` to override. The boundary belongs in the code, not only in the README, because the boundary IS the product decision.

**3. A 1-to-10 rating was being read as ten classes.** Which scores predicting 9 against a truth of 10 exactly as badly as predicting 1. Fixed by NOT deciding it here: `target_kind` answers "what is this column" (closed set / numeric / free text) and the METRIC answers "classification or regression", as it does everywhere else. A rating is ten ordered classes under f1 and a score under rmse, and both are legitimate — deciding from the dtype would overrule the user.

**Task detection delegates rather than duplicates.** `looks_like_classification` already makes the discrete-versus-continuous call for the tabular split. Writing a second copy is the bug this project hit three times in one sprint, so the prompt path calls it and adds only the third case the tabular path never needed.

**A bug I introduced and caught the same hour.** Conflating "too many answers for an enum" with "not a closed set" refused full CLINC150 — 150 intents across 2400 rows, ratio 0.06 — as free text. The RATIO decides the kind; the CAP only decides whether an enum is feasible. Separated, and the enum being dropped is now said out loud, because it removes the guarantee that an answer is in the set. Full 150-intent CLINC is therefore supported honestly: it works, with a stated cost.

**The Critic was never broken — I misread silence.** It ran on all three experiments of every run and returned clean verdicts, which produce no log line. What was wrong is that it spent a call per experiment asking about scalers fitted on combined frames and target encoding over all rows, none of which can happen here. It now asks the questions that apply.

**And checking it exposed a real hole.** Nothing stopped a cell overwriting predictions.csv AFTER `submit()` with a hardcoded keyword rule, while prompt.json sat there looking legitimate. That scores well and is not prompt engineering at all. `submit()` now records a sha256 of what the model answered and the host compares it to the predictions it reads. Verifiable, so it is a hard rejection rather than a Critic opinion — a leak vetoes, a suspicion only flags.

**The Researcher stops searching for the wrong literature.** Its queries said "TABULAR dataset" and steered toward encodings and boosting. The prompt version names fine-tuning and architecture as NON-levers so they are not searched for, and points at annotation-guideline work — the same edge cases that make human annotators disagree are the ones the model gets wrong, and those papers state the rules explicitly.

**Metric vocabulary, since it came up:** the registry was never sklearn-only. Four entries (`brier`, `mae`, `mse`, `rmse`) are hand-curated and are not sklearn scorers. Adding scipy correlations on Day 5 is the same pattern, not a departure. What stays fixed is that the model picks WHICH metric while the registry owns WHICH WAY — direction drives what banks as best, and a model getting it backwards optimises away from the goal for a whole run with nothing looking wrong.

**Then the day turned, because a run on CLINC scored 0.9890 against a 0.9890 baseline.** That reads as total failure and is nothing of the kind: a minimal prompt already gets one error in a hundred on 20-intent CLINC, so there was nothing for prompt iteration to find. Which exposed the real gap — **the prompt path had no ceiling**, so no prompt result was readable. Toxicity had gone 0.8611 to 0.8824 and nobody could say whether that was most of the available gain or a tenth of it. Exactly what Day 1 fixed for tabular, never built for prompts.

**The prompt ceiling sweep, and why its list is shaped differently.** Model families transfer between tabular datasets; prompt WORDING does not. So the sweep fixes the FORM and builds the content mechanically from what the harness already knows — the task line, the label set, rows sampled from training data. Six techniques: minimal, define-the-labels, few-shot, reasoning, expert-role, define-plus-few-shot. Nothing hand-written per dataset, which is what keeps it a fair floor rather than a target someone tuned. Few-shot examples come from TRAINING rows only, stated in the code because it is the one place a ceiling sweep could quietly cheat: holdout examples would raise the bar using answers the agent may never see, and every capture fraction measured against it would be wrong.

**Measured (6 techniques x 200 records x 2 datasets, ~80 minutes):**

| dataset | baseline | ceiling | best technique | headroom |
|---|---|---|---|---|
| toxicity_jigsaw | 0.8398 | 0.8681 | define-plus-few-shot | 0.028 |
| hate_speech_davidson | 0.5862 | 0.6822 | few-shot | 0.096 |

**And it changes what Day 3 can claim.** The agent scored 0.8824 on toxicity across three independent runs (0.8822 / 0.8824 / 0.8814). The best standard technique reaches 0.8681. **The agent goes PAST a competent technique sweep, not merely past a bare baseline** — the same shape as laptop price in v0.4, and a materially stronger claim than "+0.021". Two caveats on the record: the ceiling is 200 records against the agent's 300, so not perfectly like for like until the re-score work applies to both; and a ceiling is a lower bound by construction.

**Two findings from the sweep worth keeping.** `define-the-labels` ALONE scored WORSE than minimal on both datasets (0.8136 vs 0.8398; 0.5417 vs 0.5862) — telling a 12B to be precise about boundaries without showing it any makes it overthink. And the best technique DIFFERS by dataset: toxicity wants define-plus-few-shot, Davidson wants few-shot alone and adding the definition costs it 0.09. So there is no single prompt shape to hardcode, which is the first honest argument this project has for why an agent iterating per dataset is the right product at all.

**A new example, chosen by measurement rather than by name.** `hate_speech_davidson`: 29.5% of the raw rows had the three annotators disagree about whether a tweet was hate speech, merely offensive, or neither. That contested boundary is precisely where prompt wording decides the answer, and it shows in the numbers — 0.096 of headroom against CLINC's ~0.011. CLINC stays, because multiclass is worth exercising, but it is documented as demonstrating the plumbing and not the capability.

**Also fixed: `make eval-ceilings` produced no output for 76 minutes while working perfectly.** `print` to a redirected file is block-buffered. The same "silence reads as a hang" failure fixed for the prompt path that morning, reintroduced in the eval harness because it prints rather than logs. All 19 print sites now flush.

**Gates:** 762 unit tests, 5 new integration tests exercising both shipped examples on real data, ruff clean repo-wide, mypy --strict clean.

### 2026-08-09 | Sprint 3 Day 3 | Five live runs, and what only running finds

**Task:** the toxicity example end to end, which turned into the day the prompt path was actually made to work. Five runs on gemma4:12b against 1500 balanced Jigsaw comments. Every run found something no unit test had imagined, and every finding is now a unit test.

**The headline: the agent improves a prompt, and the number is stable.** Baseline f1 0.8611 from the minimal prompt. Best across three completed runs: 0.8822, 0.8824, 0.8814 — **+0.021, reproduced three times independently**. What it writes is recognisably good work: it read the actual misses ("piss off", "sick in the head", sarcasm), named the pattern, and added one sentence defining toxicity as personal attacks, insults, profanity or hostility. It also noticed a false positive going the other way, which is the harder half.

**Run 1 died at scoring, on two rows out of three hundred.** Two answers came back unusable, so predictions carried a third value, sklearn read the union of true and predicted as multiclass, `average='binary'` raised, and the baseline failed — which aborts the run. Two bad answers cost everything. The fix was chosen by measuring four options: passing `labels=` still raises (sklearn types the target before it reads labels); letting macro absorb the sentinel scores 0.533 instead of 0.800; naming the true labels gives 0.800; and mapping the sentinel to the POSITIVE class gives 0.857, i.e. **garbage would have paid**. So scoring names which labels count rather than rewriting what the model said, behind an `open_vocabulary` flag only the prompt path sets, with a test asserting tabular scoring is byte-identical.

**Run 2 proved I had fixed the wrong agent.** The supervisor briefed "one-hot encode categorical 'comment', fit HistGradientBoostingClassifier". It read a column of comment text as a high-cardinality categorical, because I had parameterised the CODER and left the supervisor — which sits upstream and decides what gets tried — entirely tabular. In the Day 2 PR I had flagged "Researcher and Critic still use tabular framing" as the known gap. I named the wrong two agents. The prompt family now has its own ladder (read the mistakes, define the hard case, few-shot, output discipline, repair, reframe) opening with "THERE IS NO MODEL TO TRAIN", and the tabular lever ledger and dead-lever guard are switched off for it rather than mistranslated.

**Also from run 2: the budget was calibrated for a different unit of work.** A tabular cell is a fit, seconds. A prompt cell is one model call per record, minutes. Three cells died at the 120s timeout and both sessions blew their 300s kernel budget on timeouts alone, never submitting. Raised to 600s and 1800s for the prompt family.

**Run 3 worked, and run 4 shipped an empty deliverable.** All three iterations submitted and beat the baseline — then `prompts.yaml` came out containing only the baseline. `submit()` writes `prompt.json` correctly and `score_code_job` reads it, but the LIVE path is the coder's, which reads its outputs directly and only ever looked for predictions.csv. The prompt was written to disk and dropped on the floor. Same class of miss as the supervisor, twice in two days: **fix the path you are looking at, miss the one that actually runs.**

**Run 4 exposed two performance defects, one of them self-inflicted.** The answer cache held exactly 298 entries at the start of a 100-minute run and 298 at the end: `cache_path` went into meta.json relative, and the kernel runs in its own temp cwd, so every session built a throwaway cache and re-paid for the baseline in full. And cells were overrunning a 600s limit by 200s, because `get_iopub_msg(timeout=...)` bounds the GAP between messages, not the cell — so the progress heartbeat I added that morning to fix "nine minutes of silence looks like a hang" had quietly disabled the timeout guard. Fix A broke guard B inside a single day.

**Run 5 confirmed three of four fixes and killed the fourth.** Cache 298 -> 3140. Timeouts 818.9/838.4/698.5/811.0s -> four at exactly 600.0s. The Ollama warning fired. But the `BASE` alias failed: the model produced a FOURTH spelling of BASELINE_PROMPT. Alias whack-a-mole was the wrong shape — the live namespace was already appended to every observation and it still fumbled, because a listing to scan is not the same as being told which name was wrong. Replaced with a `NameError` hint that names the miss and the closest defined match, placed before the traceback. General, not prompt-specific: a mistyped variable is an every-path mistake.

**Run 5 also leaked a tabular lever into a prompt run** — iteration 3 was titled `untried lever: categorical-encoding`. When a brief is rejected twice the harness substitutes a deterministic fallback "novel by construction", and that table was still tabular. Compounding it, the anti-baseline-rebrief nudge told the supervisor to prefer "Levers NOT yet tried" and name a lever class, neither of which exists here, so it could not comply, the guard fired twice, and the fallback handed it nonsense. Both now have prompt-family versions.

**Measured, not assumed: Ollama serialises.** 4 calls sequential 13.8s; 8 calls at 8-wide 25.1s. **1.10x from 8 workers** — the pool is a queue. 3.2s per call means a 300-record pass is 16 minutes whatever the concurrency. Not an iterate bug, but it is the difference between a 105-minute run and a 13-minute one, so the prompt path now says so once per run with the number in it.

**Two changes that came out of the arithmetic rather than a crash.** At n=300 and accuracy 0.88 the standard error on a single score is ±0.019 and the 95% interval ±0.037 — while the improvement being measured is +0.020. Meanwhile run-to-run reproducibility of the same move is ±0.001. Those are different quantities: ranking two prompts on the same records is paired and reliable, but quoting "+2 points" from 300 comments is not supported. So candidates are now ranked on a fixed 100-record slice (same records every time, so the comparison stays paired) and **the winner alone is re-scored on the full holdout**, which is the number `prompts.yaml` leads with. Submit cost drops from 900 calls to 600 and the headline figure is measured on more data than any intermediate one. Second change: `ask()` projects an ETA from its first five calls, because an unbounded wait is the actual pain and a known sixteen minutes is a decision.

**One proposal was dropped under review.** Early-exit on a losing candidate sounded like a saving until Tony asked how it differed from patience. It does differ — patience stops the search, early exit stops a measurement — but it would produce partial scores that are not comparable to full ones, which breaks the one-ruler discipline the whole project rests on, and the loop-holdout change already eats most of the saving. Dropped.

**Examples landed:** `toxicity_jigsaw` and `intent_clinc150` are no longer placeholders. Each has a `prepare.py` that downloads and builds its eval set (no account needed for either) plus a README explaining the prep choices. The data is gitignored — the toxicity set is 750 genuinely abusive comments and a script rebuilds it in seconds, which is not something to carry in a public repo.

**Gates:** 743 unit tests (was 720), ruff clean repo-wide, mypy --strict clean.

**Standing lesson from the day:** every one of these bugs was findable by a unit test I did not think to write. Live runs are for discovery; the moment something is found it becomes a test and never needs a run again. Five runs was right for a brand-new path where each one found something new, and that rate should now fall.

### 2026-08-08 | Sprint 3 Day 2 | `PromptTarget`: the second problem type, on the same machine

**Task:** v0.5's public promise. Prompts as a target family, with the whole multi-agent apparatus reused rather than rebuilt.

**I had the architecture wrong and Tony corrected it.** My plan was that prompts could not use `run_supervised` because it is typed to `TabularDataset`, so v0.5 would ride the older target-agnostic `Orchestrator` loop and go without the Researcher and Critic. His answer: everyone lives as is, it is the same as tabular, the only difference is that there is no ML model — there are LLM calls run per record. He was right, and the reason my objection dissolved is that **a prompt eval set IS tabular**: input columns plus a column holding the correct answer. `TabularDataset`, the deterministic split, the sealed holdout, the predictions contract and `core.scoring` all apply unchanged. Nothing needed decoupling. The whole day got smaller.

**What actually differs is three strings and a helper.** The coder is not replaced by a "prompter" — it is parameterised. Its job, write cells until you can submit predictions, is the same job whether a cell fits a model or calls an LLM. Four additive constructor arguments, all defaulted so the tabular path is byte-identical: the session preamble, an extra input file, the floor cell, and which instruction set to use. 52 added lines in `coder.py`, and the Supervisor, Researcher, Critic, Summarizer, memory, dedup and dead-ends channel were not touched at all.

**The one real piece of new infrastructure is `ask()`.** Tony asked whether the generated code could just construct the Ollama URL and call it. It could, and I pushed back on four grounds: a cloud key would sit in the same place as model-written code; a model writing its own call can silently change model or temperature between experiments, so two experiments would differ by more than the prompt; a hand-rolled loop is sequential, and one call per row sequentially is what makes a prompt loop too slow to finish; and retries, rate limits and garbled replies each become a dead iteration. So the session is handed `ask(prompt, rows)` — concurrent, cached to sqlite, tool-enforced answers — and the agent writes the prompt rather than the transport. The framing that settled it: the tabular coder does not implement gradient boosting from scratch, it calls the library.

**A property that fell out for free, and it is the good kind.** `build_inputs` already writes train.csv WITH answers and holdout.csv WITHOUT them. So the prompt-path version of fitting on the test set — mining the answer key for few-shot examples — is not something the agent is stopped from doing, it is something it cannot see how to do. I had planned a guard for it. The guard was already there, four releases early, in a function written for a different reason.

**The allowed answers are a tool schema, not an instruction.** Asking a model to reply with one of three labels is a request. Giving it a tool whose only argument is an enum of three labels makes anything else inexpressible. Same shape as the Researcher picking a paper by number rather than writing a DOI, and the same principle: make the bad outcome structurally impossible rather than prompted against.

**Unparseable output counts as wrong and is reported.** A prompt that answers "I think this is probably toxic?" is a worse prompt. Coercion tries an exact match, then a UNIQUE substring — a reply naming two labels stays unparseable, because guessing which one it meant is how a wrong answer becomes right by accident. But an endpoint that fails on EVERY row is an error, not a score of zero: reporting 0.0 would bank a number and teach the next iteration a lie.

**`prompts.yaml` is the deliverable, and Tony's call.** He pointed out that notebooks cannot really tell you which prompt was best, so prompts should be saved to a file with a version tag and a best marker. Right, and it is more than convenience: for a prompt run that file IS the artifact, the way `best_model.joblib` is for tabular. One correction to how he framed it — the harness writes it, never the agent, because an agent writing its own scoreboard can mislabel which one won. That forced a good constraint: `submit(prompt)` writes the predictions and `prompt.json` in the SAME call, so a prompt that did not produce the submitted predictions cannot be recorded. `best: true` respects the Critic; a rejected score can never be marked best, and rejected versions stay in the file with their reason.

**The safety net is the majority answer, not a re-run of the baseline prompt.** v0.4 learned this when the tabular floor trained a gradient-boosted tree and timed out alongside the session it was meant to catch. A floor made of LLM calls has exactly that shape and would be slowest in precisely the situation that triggers it.

**Scoped to local compute, stated honestly.** Generated code in an e2b sandbox would need the model endpoint and key shipped into the sandbox. For a local Ollama floor that is a non-issue. A host-side proxy is the fix when cloud sandboxing matters.

**Gates:** 720 unit tests (was 652; 68 new), ruff clean repo-wide, mypy --strict clean. Two of those tests execute the generated session preamble for real in a temp directory and call `ask`, `evaluate` and `submit` — that contract only ever runs together inside a live session, so it needed a test that runs it together.

**Diff discipline note:** running `ruff format` over `src/` reformats 15 files this change never touched, pre-existing drift between the committed code and the installed ruff. Reverted, and the three touched modules were re-edited without a format pass, taking the diff from 450 changed lines to 345 with 15 deletions.

### 2026-08-08 | Sprint 3 Day 1 | The eval harness, and the first thing it found

**Task:** carry-in item 1. Turn the hand-computed headroom table into a runnable system, so "did this release make the product better" is a command rather than an argument. Internal only: no `iterate eval`, nothing under `src/iterate`, nothing in the wheel. Tony's call on both the ordering and the scope.

**Shape it took, after his pushback.** The first plan was a script that ran a sweep and printed a table. He pushed for a genuinely separate subsystem with its own store, versions and dates, and he was right for a reason the first plan missed: a full sweep is hours and released versions are frozen, so v0.2's number on churn can never change. Recomputing the grid on every release makes the tool too expensive to use, which is how measurement tools die. So `evals/results.db` accumulates and a sweep fills only what is missing. Adding v0.6 costs one row.

**Config versus flags, as a correctness rule rather than a preference.** Everything that changes what a number MEANS lives in a tracked `config.toml` — model, budget, patience, repeats — and is hashed into a fingerprint stamped on every cell. Flags only select which rows to fill. If the budget were a flag, one sweep would run v0.4 at ten iterations and the next v0.5 at fifteen, the table would show v0.5 ahead, and the table would be fiction.

**The decision that makes cross-version reading possible at all:** the memory-db reader uses raw SQL and `json.loads`, never the current pydantic models. Those models are `extra="forbid"`, so a v0.1 database fails validation the moment a field is added (`Candidate.citations` in v0.4, `Experiment.digest` before it). Parsing through them would mean the harness could only read runs produced by the version it shipped with, which is exactly the comparison it exists to make.

**THE FINDING, and it corrects a public claim.** The measured ceiling on laptop price is **rmse 248.85**, not the 329.55 the Aug 2 hand sweep implied. A plain `Ridge` — one line, no tuning, no feature engineering — reaches it in under a second. The v0.4 run's best was 321.56, so against a properly measured ceiling it captured **55% of available headroom, not 110%**, and its result is beaten outright by the simplest linear model in the sweep. Same sealed split, same `load_csv` seed, same `ModelTarget` pipeline, same scoring, so it is a like-for-like comparison. No leaking column: the only numeric features are `laptop_ID` (r=0.07) and `Inches`. Ridge wins because one-hot over `Product` (618 distinct values in 1303 rows), `Cpu` and `Gpu` is a wide sparse problem that L2 handles and default-depth trees do not. This is the single dataset the v0.4 capability claim rests on.

**The sweep is not uniformly better than the hand sweep, which changed a design.** It found much MORE headroom on laptop price and diamonds, and LESS on churn, heart and mobile — those three now measure at exactly zero, against 1.6%, 0.6% and 2.1% by hand. The reason is a real limitation: `brute_force_sweep_v1` varies the MODEL, not the FEATURES, and the hand sweep varied both. So the stored ceiling is now a running maximum: `put_ceiling` keeps the better of old and new, and re-measuring can only raise the bar. Overwriting would have silently lowered the bar the agent is judged against on three of six datasets. Feature treatments are `v2` of the sweep, and the `method` column says which kind of ceiling any number is.

**Measured ceilings (gemma4-independent, no LLM, 9 models each):**

| dataset | metric | baseline | ceiling | best model | headroom |
|---|---|---|---|---|---|
| laptop_price | rmse | 411.89 | **248.85** | Ridge | 163.04 |
| diamonds | rmse | 549.06 | 537.14 | LGBM | 11.92 |
| adult_income | f1 | 0.7187 | 0.7259 | XGB | 0.0072 |
| churn | average_precision | 0.6449 | 0.6449 | factory default | 0 |
| heart_risk | average_precision | 0.8967 | 0.8967 | factory default | 0 |
| mobile_price | accuracy | 0.9450 | 0.9450 | factory default | 0 |

**Also landed: the shape corpus,** the second half of carry-in 1 and a different thing entirely — no LLM, no versions, seconds not hours. Five synthetic files with the shapes that actually broke the loader (non-UTF-8, boolean columns, integer regression targets) plus high-cardinality strings and missing values, asserted in CI on every push. Writing it found a bug in the fixture itself: latin-1 has no euro codepoint, so the "European export with a euro sign" file could not be written at all. The real-world version of that file is cp1252, where the euro byte decodes under our latin-1 fallback as a control character — mangled rather than crashed, and not yet covered.

**Isolation choices, and the one that is deliberately shared.** Each cell gets its own memory db, because a shared one would let a previous cell's best carry over as the next cell's baseline and the grid would measure the order the cells ran in. `XDG_CONFIG_HOME` is thrown away per sweep so a saved `iterate setup` backend cannot leak in. The research cache is SHARED on purpose: every version then sees the same papers, so a difference between two cells is the agent rather than what OpenAlex returned that minute.

**Gates:** 652 unit tests (was 583; 69 new), ruff clean over the whole repo, mypy --strict clean over `src/iterate` and `evals`. `make lint`/`format`/`typecheck` now cover `evals/` too, closing the scope gap between them and `make build` that nearly broke the v0.4 release.

**Noted, not fixed:** running `ruff format` with the currently installed version reformats 38 existing tracked files. Pre-existing drift, unrelated to this work, left alone to keep the diff readable.

### 2026-08-02 | Sprint 2 Day 7 | v0.4.0 released

**Task:** Release mechanics per the standing checklist.

**Release gate (step 1):** v0.4 touched the loop in three places, so the trajectory bar was re-run before tagging rather than after — 8 runs across 6 datasets on gemma4:12b. That gate found seven bugs and is written up in the Day 6 entry. Four of eight runs improved on their baseline; of the four that did not, two had zero measured headroom and reporting nothing was the correct answer.

**Build gate (step 2):** 583 unit tests, ruff + mypy --strict clean, CLI startup verified still free of sklearn. One thing caught here that would have failed the release: `make build` runs `ruff check .` over the WHOLE repo while CI runs `ruff check src tests`, so the new `docs/evidence/` reproduction script (33 lint errors, deliberately not production code) would have broken the build. Excluded in pyproject, which is what its own README already claimed.

**Doc sync (step 3):** README status paragraph, release table row to shipped, the roadmap cell, the comparison-table "Literature-aware proposals" row flipped to a tick, and the capability section heading which still read "What v0.2 does" two releases on. LIMITATIONS: five rows flipped to fixed, the metric-panel row rewritten as the closed-registry row, the probability and averaging rows retired, and the false integer-target backlog row replaced. First ever EVAL_LOG entry, carrying the headroom table and a dimension scorecard.

**Version mechanics (step 4):** 0.3.1 -> 0.4.0 in pyproject, `__init__`, and the lockfile; tagged v0.4.0 and pushed. **Published to PyPI 2026-08-02T11:43:51Z**; the built wheel was installed into a clean 3.12 venv and verified before publish (`iterate 0.4.0`, `--metric` optional).

**Launch assets (step 5):** X thread (7 tweets, all under 275) and LinkedIn post (355 words, inside the 323-458 range of v0.1-v0.3) drafted in LAUNCH_POST.md, feature-first per the v0.2 REV2 lesson that a score-led post reads wrong. Demo to be recorded from the published package on `laptop_price.csv --max-iterations 3`, which shows the metric dial, a real citation and the 412 -> 322 win inside about twenty minutes.

**What v0.4 shipped:** all five public promises. Researcher and Critic specialists, literature-grounded proposals with citations that cannot be fabricated, probability metrics (plus 42 more than promised, derived from scikit-learn), a selection-bias watch, and the first input dial to turn since v0.1 — `--metric` is optional.

**Honest state of the evidence:** the headline capability claim rests on one dataset. Laptop price captured 110% of measured headroom by retrieving a 2022 paper on high-cardinality features and beating a hand-parsed baseline. **(Superseded 2026-08-08: the eval harness re-measured that ceiling at rmse 248.85 rather than 329.55, which makes the same run 55% of available headroom, not 110%. See the Sprint 3 Day 1 entry.)** Five of the six other datasets were near-ceiling, which is the real shape of tabular ML rather than a weakness of the agent, but it does mean "captures large gains" is n=1. The posts claim a specific run rather than a general capability, which is the correct framing for what was measured.

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
