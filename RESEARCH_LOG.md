# Research Log

> Every meaningful decision in `iterate` starts with research. This file is the citation trail.
>
> Public. Recruiters reading the repo see literature awareness, not vibes.

---

## Why this file matters

When someone opens this repo, this file shows:

- We don't code by intuition — every architectural decision has a paper or post backing it
- We considered multiple approaches and chose with rationale
- The project is research engineering, not tutorial walkthrough

This is the "papers → citations" pattern that separates research engineers from people who follow LangChain tutorials.

---

## Format per entry

```markdown
## YYYY-MM-DD — Research: <task or bottleneck title>

**Question:** What am I trying to figure out?

**Sources reviewed:**
1. [Title](url) — key insight in 1 sentence
2. [Title](url) — key insight in 1 sentence
3. [Title](url) — key insight in 1 sentence

**Approaches considered:**
- **Approach A — [name]:** brief description. Pros: ___ Cons: ___
- **Approach B — [name]:** brief description. Pros: ___ Cons: ___
- **Approach C — [name]:** brief description. Pros: ___ Cons: ___

**Decision:** Going with Approach X because [reasoning grounded in sources or our constraints]
**Smallest viable implementation:** [concrete description — what gets shipped this session]
**How I'll verify it works:** [test plan — what specifically proves this fix landed]

**Out of scope today:** [other directions the sources suggested but not pursuing now, with rationale]
```

---

## Entries

## 2026-05-24 — Research: domain schemas for the iteration loop

**Question:** How do I model `Metrics`, `FailureCase`, `Candidate`, `Experiment`, `ExperimentResult` so they work for *any* task (ML or LLM) and stay auditable?

**Sources reviewed:**
1. [Pydantic v2 docs — validators](https://docs.pydantic.dev/latest/concepts/validators/) — `field_validator` sees one field; `model_validator(mode="after")` runs post-construction and can check across fields.
2. [Pydantic v2 — models & default_factory](https://docs.pydantic.dev/latest/concepts/models/) — mutable/dynamic defaults (ids, timestamps) must use `default_factory`, else one value is shared across all instances.
3. DDD value-object vs entity distinction — entities carry identity (`id`); value objects are defined by their data (`Metrics`, `FailureCase`).

**Approaches considered:**
- **Metrics — fixed named fields (accuracy/f1/...):** type-safe, but breaks for any metric not predefined (LLM win-rate, custom scores). Cons: not generic — kills the "works for all tasks" goal.
- **Metrics — flexible `values` dict + `primary` + `direction`:** agent fills task-specific metric names into a stable envelope. Pros: generic; stable y-axis for plateau detection + baseline comparison. Cons: no per-key type safety.
- **Composition — nested objects vs id references:** nesting = self-contained snapshot, one-line serialize. IDs = normalized, but presuppose a store that doesn't exist until Week 4.

**Decision:** Flexible `Metrics` envelope (dict + primary + direction) and **nested composition**, keeping an `id` on each model. Stable contract for the Terminator/Memory/Reporter; the LLM picks metric *content*, not schema. LLM-designed eval plans become a *tool* in Week 4, never a self-authored Python schema — a per-experiment schema would give the hill-climber no consistent axis to compare runs.
**Smallest viable implementation:** `src/iterate/schemas/experiment.py` — 5 Pydantic v2 models, `extra="forbid"`, validators (finite metrics, primary∈values, non-empty changes, success⇒metrics, completed⇒result).
**How I'll verify it works:** `tests/unit/test_schemas.py` — each guard raises `ValidationError`; happy-path constructs succeed; two instances get distinct `id`s. Plus `ruff` + `mypy --strict` clean.

**Out of scope today:** `EvalPlan`/`MetricSpec` as an LLM-callable tool (Week 4 discovery agent); id-based retrieval (Week 4 Memory store); bounding `failure_cases` to top-N (Executor's job).

## 2026-05-24 — Research: a tool-capable local model for the agentic default

**Question:** The default `qwen2.5-coder:14b` returns tool calls as plain text, not structured `tool_calls` (verified live, even with `tool_choice="required"`). The agent is tool-driven, so which Ollama model does *structured* tool-calling reliably AND fits a 24 GB M5 MacBook Pro?

**Sources reviewed:**
1. [Ollama qwen3 library](https://ollama.com/library/qwen3) — qwen3 lists `tools` + a thinking mode; sizes 8b=5.2 GB, 14b=9.3 GB, 30b (MoE)=19 GB.
2. [Ollama tool-calling docs](https://docs.ollama.com/capabilities/tool-calling) — only tool-templated models emit structured `tool_calls`; others fall back to text (exactly our qwen2.5-coder symptom).
3. [Ollama qwen3-coder library](https://ollama.com/library/qwen3-coder) — agentic-coding model; smallest is 30b (19 GB, 3.3B active MoE); no ≤14b variant.
4. [Ollama tools models](https://ollama.com/search?c=tools) — tool-capable list (llama3.1, mistral-nemo, qwen3, …).

**Approaches considered:**
- **qwen3:14b (9.3 GB):** built for agentic tool use + thinking mode; same footprint as current model. Pros: comfortable on 24 GB, strong code/reasoning. Cons: 9 GB pull (slow on current network).
- **qwen3:8b (5.2 GB):** same family, faster pull + big headroom. Cons: weaker than 14b.
- **qwen3:30b / qwen3-coder:30b (19 GB MoE, ~3B active):** more capable, still fast. Cons: tight on 24 GB (risky with OS + context).
- **llama3.1:8b / mistral-nemo:12b:** established tool support — fallbacks if qwen3 disappoints.

**Decision:** Default tool-driving model → **qwen3:14b** (reliable structured tool-calling + thinking, same 9 GB footprint). Validate **qwen3:8b** first (faster pull) to confirm the family tool-calls structurally before committing. Keep qwen2.5-coder for pure code-gen sub-tasks.
**Smallest viable implementation:** pull a qwen3 model → run the live tool-call round-trip through `OpenAICompatibleClient` → flip `config.iterate_model` once it passes.
**How I'll verify it works:** `test_live_ollama_smoke` passes AND a tools-provided chat returns `has_tool_calls=True` with parsed dict arguments — the exact behavior qwen2.5-coder failed.
**Result (validated same day, 2026-05-24):** qwen3:14b returned `has_tool_calls=True`, `finish=tool_calls`, args parsed to a dict. `config.iterate_model` default flipped to `qwen3:14b`. Caveat found: qwen3 thinking mode is on by default — it spends tokens before the visible answer, so short `max_tokens` yields empty content (`finish=length`); give generous budgets or toggle thinking off.

**Out of scope today:** the Anthropic adapter (separate non-OpenAI-compatible client, later).

## 2026-05-26 — Research: the BenchmarkTarget contract

**Question:** What's the minimal contract every target (tabular ML, DL/vision, prompt) implements so the orchestrator runs any of them the same way — and how do we establish a baseline fairly?

**Sources reviewed:**
1. [Python `typing.Protocol` / `runtime_checkable`](https://docs.python.org/3/library/typing.html#typing.Protocol) — structural typing; `isinstance` checks member presence, not inheritance.
2. Prior internal contracts — `LLMClient` (precedent: `Protocol` + sync) and the `Experiment`/`Candidate`/`ExperimentResult` schemas.
3. The comparability principle — a metric only means something against a baseline measured the *same way* (same eval, same split).

**Approaches considered:**
- **Single `run(candidate)`, baseline supplied externally:** rejected — reported/external scores aren't comparable, and a "no-op" Candidate can't even be built (the schema validator requires non-empty `changes`).
- **Granular steps (`prepare`/`train`/`evaluate`) driven by the orchestrator:** rejected — leaks the target's internals, more coupling.
- **Two methods, `baseline()` + `run(candidate)`, both returning `ExperimentResult`:** chosen.

**Decision:** `BenchmarkTarget` Protocol (`runtime_checkable`, sync) with `name`, `baseline() -> ExperimentResult`, `run(candidate) -> ExperimentResult`. The target only **measures**; `baseline()` **always re-measures** the starting point through the target's own eval (never adopts a reported score), so every comparison is apples-to-apples; the orchestrator/terminator judges winners + termination; execution venue (local/sandbox/cloud) is out of scope (compute layer).
**Smallest viable implementation:** `src/iterate/targets/base.py` (protocol only) + `tests/unit/test_targets_base.py`.
**How I'll verify it works:** `isinstance(fake, BenchmarkTarget)` is True; a class missing `run()` is False; both methods return `ExperimentResult`. ruff + mypy --strict clean.

**Out of scope today:** concrete `ModelTarget` (Week 2 Days 2-3); discovering an existing model + asking the user for a source artifact when a prior score is claimed (Week 7-8 — see IDEAS).

## 2026-05-26 — Research: tabular data splitting + storage (reproducibility, leakage, persistence)

**Question:** How should the tabular data adapter split the data, and how should the split live (RAM vs disk), so the pipeline is reproducible, leakage-safe, and production-grade — even for small datasets?

**Sources reviewed:**
1. [Stanford CS230 — Splitting into train/dev/test](https://cs230.stanford.edu/blog/split/) — split once via a dedicated step and persist it; never split ad-hoc / by moving files, or you can't reproduce it.
2. [Engineering for Data Science — repeatable splitting via hashing](https://engineeringfordatascience.com/posts/ml_repeatable_splitting_using_hashing/) — a fixed seed only reproduces if the data never changes; if rows are added/reordered the same seed yields a different split. Robust fix: hash a stable row id (`farmhash.fingerprint64(id) % buckets`) so a row always lands in the same split regardless of order/additions.
3. [DVC / data versioning](https://www.datacamp.com/tutorial/data-version-control-dvc) — treat data as immutable; snapshot, keep lineage, make every version reproducible/restorable.

**Approaches considered:**
- **Seed-based `train_test_split`:** reproducible only while the data is static. Fine within a single run (the CSV doesn't change mid-run); fragile across runs when data evolves.
- **Hash-based splitting:** robust to row additions/reordering; the production-grade choice when data evolves between runs. Doesn't by itself guarantee class balance.
- **Hold split in RAM vs persist to disk:** in-RAM is fine for small tabular and is normal; persisting the split (or just indices+seed) is better for reproducibility, crash-recovery, audit, and scale. Holding everything in RAM forever does not scale to DL/large data.
- **Leakage:** split must happen *before* preprocessing; transforms fit on train only, applied to the sealed holdout.

**Decision (v0.1.0):** stratified **seed-based** split (reproducible for a static per-run CSV; preserves class balance for imbalanced targets like churn) + **content-hash the dataset** (`hash_pandas_object` → sha256) recorded on `TabularDataset` as a lightweight data version, so any result traces to the exact data + split. Leakage-safe: the adapter does **load + split only**, no preprocessing.
**Smallest viable implementation:** `src/iterate/adapters/data/tabular.py` → `load_csv()` returns a `TabularDataset` (train + sealed holdout + target/features/seed/test_size/data_hash). Tests cover determinism, stratification, disjoint splits, hash stability + content-sensitivity, missing-target.
**How I'll verify it works:** same seed → identical split; stratified test/train target means match the overall rate; train/holdout indices disjoint; identical data → identical hash, changed data → different hash. ruff + mypy --strict clean.

**Out of scope today (deferred):** persisting the split snapshot to `.iterate/runs/<id>/` → the executor/run layer (Week 2 Day 5); **hash-based splitting** → Week 8, when data evolves between runs (discovery / retraining) and seed-determinism is no longer enough.

## 2026-05-28 — Research: how open should model selection be, and what shape is a candidate?

**Question:** The Proposer will eventually say "use *this* model with *these* params" from research. What is the agent allowed to pick, how do we instantiate it safely, and what shape should a candidate's `changes` take so today's hyperparameters and tomorrow's arbitrary models use the *same* contract?

**Sources reviewed:**
1. [scikit-learn — developing estimators / common API](https://scikit-learn.org/stable/developers/develop.html) — every estimator is a class with a uniform `fit`/`predict` and keyword-only constructor params; `get_params`/`set_params` make params a plain dict. So "model + params" is enough to build any of them.
2. [Python docs — `importlib.import_module`](https://docs.python.org/3/library/importlib.html) + [`inspect.signature`](https://docs.python.org/3/library/inspect.html#inspect.signature) — resolve a class from a dotted path at runtime; introspect its constructor to know whether it accepts `random_state` (so determinism is applied only where supported).
3. [OWASP — dangers of dynamic import / code execution](https://owasp.org/www-community/attacks/Code_Injection) — resolving arbitrary import strings is RCE-adjacent; the mitigation is an allow-list of trusted module prefixes, never importing whatever string arrives.

**Approaches considered:**
- **(a) Curated registry** — a hand-maintained dict of "supported" models. Pros: tightest control. Cons: every new model is a code change; caps the agent at *our* list — kills the "use the model research recommends" value. **Rejected.**
- **(b) Dynamic factory over allow-listed installed libraries** — a candidate names any estimator by import path under `sklearn.*`/`xgboost.*`/`lightgbm.*`; we `importlib`-resolve + instantiate. Pros: any installed estimator, no per-model code; allow-list bounds the RCE surface. Cons: limited to what's *installed*. **Chosen for now (Week 2 Day 4 / v0.1).**
- **(c) Sandboxed code-gen** — the Proposer *writes* the training script; we run it in an e2b sandbox. Pros: literally any model, installed or not, plus custom architectures. Cons: needs the sandbox executor + a strict script/result contract + heavier security. **Bumped early to v0.2** (right after the v0.1 loop) — it's the real unlock, so it shouldn't wait.
- **Candidate shape — flat `{"model": x, ...hyperparams}` vs nested `{"model": x, "params": {…}}`** — flat collides if a hyperparameter is ever named `model`, and muddles "which knob is the selector vs a param." Nested cleanly separates *which model* from *its params*.

**Decision:** Ship **(b)** now as `adapters/models/registry.py::build_estimator(task, spec, *, seed)`, with the **nested** spec `{"model": "<import.path>", "params": {…}}` (both optional; default = `HistGradientBoosting` per task). Allow-list `sklearn.*`/`xgboost.*`/`lightgbm.*`; inject `random_state` only when the constructor accepts it (via `inspect.signature`). Sequence **(c) sandboxed code-gen as v0.2** for "any model at all." `ModelTarget` delegates to the factory — model-family switching and hyperparameter tuning now travel the same path.
**Smallest viable implementation:** `build_estimator` + 8 tests (default-per-task, named model from each allowed lib, `random_state` injected/skipped/not-overridden, disallowed library rejected, non-class path rejected, non-string model rejected); `ModelTarget._evaluate` takes the spec, baseline = `{}`.
**How I'll verify it works:** factory builds RandomForest/LinearRegression/etc. with the right params; `os.system` and other off-list paths raise; `ModelTarget.run` switches model family end-to-end and scores. ruff + mypy --strict clean; suite stays fast.

**Out of scope today (deferred):** the **sandboxed code-gen path (c)** → v0.2 (Week 4–5); the Proposer actually *choosing* the model/params from research → Week 3 loop + Dial-A research (it just consumes this contract); validating that a named classifier matches a classification task → left to fit-time failure for now (the executor will capture it, Day 5).

## 2026-06-01 — Research: sandboxed code-gen execution venue + the code-gen contract (v0.2)

**Question:** v0.2 lets the agent write its own training code so it can use any model, not just the allow-listed installed estimators. Where does that generated code run, and what is the contract between the agent's script and our eval so the sealed-holdout guarantee still holds?

**Sources reviewed:**
1. [e2b code-interpreter docs](https://e2b.dev/docs) — ephemeral cloud sandboxes (fresh micro-VM per run, filesystem + process isolation, torn down after); built for running untrusted/LLM-generated code. Needs an API key; billed by sandbox runtime.
2. [Python `subprocess` + `resource` limits](https://docs.python.org/3/library/subprocess.html) — running a script locally in a child process with a timeout is simple, but offers no real isolation: the child has the user's permissions and filesystem.
3. Our own Week-2 decision trail — the compute layer was always meant to be a pluggable port (local MPS / RTX 4050 / e2b / cloud), and we deferred the `ComputeBackend` protocol until the second backend (the sandbox) actually arrived. That is now.

**Approaches considered:**
- **Run generated code in-process (reuse LocalExecutor as-is):** simplest, but executing arbitrary generated code in our own process risks crashing/contaminating the loop and has zero isolation. Rejected for generated code.
- **e2b sandbox (cloud):** real isolation, contained blast radius, the right default for code generated autonomously with no human approving each script. Costs money + needs a key + adds latency.
- **Local child-process executor:** free, offline, fast, uses the user's own GPU/data, but the generated code runs with the user's permissions (no isolation). Fine as an *explicit opt-in*, wrong as a default.
- **Code-gen contract — return a fitted model vs return predictions:** returning a pickled model invites version/security issues across the sandbox boundary; returning *predictions on the holdout features* (plus optional artifacts) keeps the boundary a plain data file and lets US score through our own eval, preserving the sealed-holdout guarantee.

**Decision (v0.2):** Put execution behind a `ComputeBackend` protocol (extracted Day 1). Ship two backends: `SandboxExecutor` (e2b, the **safe default** for generated code) and a local executor exposed as `--compute local` (**explicit opt-in**, with a warning that generated code runs with the user's permissions). Generated code is the **agent's own, never the user's** (permanent policy). The **contract**: the script receives the train split and the holdout *features* (never the holdout labels), trains on train only, and writes predictions to a known output path; our side reads them back and scores through the existing `Metrics` eval. This keeps the sandbox boundary a plain data handoff and the holdout sealed.
**Smallest viable implementation (Day 1):** `adapters/compute/base.py::ComputeBackend` protocol; `LocalExecutor` conforms; `SandboxExecutor` stub; Orchestrator depends on the protocol. Contract + executors land Days 2-3.
**How I'll verify it works:** protocol conformance tests (Day 1); a generated script that trains CatBoost (not allow-listed) runs in the sandbox and is scored through our eval, holdout labels never crossing the boundary (Day 5 integration).

**Out of scope today:** the real e2b adapter (Day 2), the contract module (Day 3), the CodeProposer (Day 4). Network egress policy inside the sandbox (default deny) to be decided when the e2b adapter lands.

---

## 2026-06-04 — Experiment: what limits exploration depth, the prompt or the model?

**Question:** On the v0.2 code path the agent kept doing the same preprocessing (impute + one-hot) and only swapped the model. Is that a weakness of the local model (`qwen3:14b`), or of our prompt? Resolve by experiment, not opinion.

**Method (A/B on the same harness, churn / f1, 6 iterations, `--fresh`):** held everything constant (data, metric, compute, prompts) and varied only the backend model. Compared local `qwen3:14b` against Groq `llama-3.3-70b-versatile`, reading the deterministic component fingerprint (`codegen.components_used`) of each attempt to see what was actually tried.

**Findings:**
- **Modeling depth is model-bound.** The 70B explored far more algorithms (logistic regression — which won — gradient boosting, and an unprompted *stacking ensemble* with SVM base learners). The 14B mostly repeated RandomForest/HistGB. Better model → more algorithmic diversity and a higher score (0.604 vs 0.580).
- **Feature-engineering depth is prompt-bound.** The 70B used the **identical preprocessing on every one of its 6 iterations** — same blind spot as the 14B. So the monotony was not a capability gap; both models default to "set up a generic pipeline once, then only change the model."
- **Confirmation:** after rewriting the prompt to make feature engineering the primary lever, the *local 14B* engineered a new feature (`TotalCharges_per_tenure`) and reached **f1 0.6166 (+0.049 vs baseline)** — the best result across every run, beating the un-prompted 70B.

**Decisions:**
1. Make the code-proposer prompt **feature-engineering-first** (concrete technique menu), and feed a **deterministic component fingerprint** of each past attempt into history so the agent can see what it has/hasn't tried. (Shipped this session.)
2. Recommend a **cloud backend** for real modeling depth; document the 14B as the floor (LIMITATIONS).
3. Aggressive feature engineering by a weak model also produced silent **near-zero** scores (NaN/inf, single-class predictions) it couldn't foresee while writing the whole pipeline blind — the decisive argument for pulling **cell-by-cell execution into v0.2** (inspect-then-build catches it mid-session).

**Out of scope today:** the LLM-summary version of the digest (v0.4), seeding the code path for run-to-run reproducibility (pending), and a pre-run undefined-name lint (pending).

## 2026-06-07 — Experiment: making a weak local model reliable cell-by-cell, and where the harness ends

**Question:** With the v0.2 cell-by-cell system built, what actually makes a weak driver (`qwen3:14b`) reliable across a multi-experiment run, and is "writes clean stepwise cells" something the harness can deliver or is it model-bound? Resolved by repeated live runs on churn / f1 (`--fresh`, max-iter 5, patience 3), reading every cell of every notebook.

**Method:** iterated the harness against real qwen runs, each time root-causing failures from the captured notebooks + the Ollama server log rather than guessing, then fixing the harness (not the model) and re-running. Held the dataset, metric, and backend constant.

**Findings (each fix traced to an observed failure):**
- **The full-context design wasn't reaching the model.** The Ollama client never set `num_ctx`, so the server ran at its 4096 default and FRONT-truncated the growing session — the system prompt + tool schema were the first thing dropped (server log: `truncating input prompt limit=4096 keep=4`). Pinning `num_ctx` (16384) + a prompt-side budget that elides oldest observations first was the single highest-leverage fix.
- **Silent broken affordances cause loops, not the model.** Auto-install was a no-op in uv venvs (no `pip`) and failed silently, so the agent re-tried an import that could never resolve; the `finish()` tool conflated with a `finish()` Python call NameError-ed otherwise-good cells. Making each affordance either work (pip→uv→ensurepip fallback) or VISIBLY report failure converted loops into course-corrections.
- **Weak models perseverate; bound it structurally.** Two breakers — refuse an identical re-submitted cell (repeated-cell), and escalate when the SAME error signature recurs across cosmetically-different cells (same-error) — kill the 14B's characteristic thrash (e.g. swapping the encoder five times while the real cause, a string column reaching a numeric step, is unchanged).
- **Charge the budget in kernel-seconds, not wall-clock.** Local thinking latency (~1-2 min/turn) would otherwise starve a weak driver of the turns a fast cloud model gets free; bounding by kernel-execution time gives any backend the same working budget.
- **Result:** with these fixes (and monolithic cells), local qwen reached **f1 0.6353 with 5/5 experiments succeeding** (baseline 0.5676) — a new local-qwen high, and the whole curve above baseline. The harness lifts the floor model on **score and reliability**, the infra-over-model thesis holding on the failure axis, not just the score axis.

- **The one thing the harness could NOT deliver: stepwise writing style.** A 14B's default output is a complete script; it stages only when there is nothing to anchor to (the from-scratch iteration). The moment it is handed a working pipeline to edit (every improve iteration), it reverts to one monolithic cell, regardless of prompt wording. Forcing staging via the prompt regressed reliability to **0.5813, 2/5** (the extra cells gave the weak model more rope: positional-index bugs, then thrash). Conclusion: **staged-vs-monolithic is model-bound, not harness-bound** — a stronger backend stages naturally. This is the boundary of "lift the weak model by harness": we can make it *perform* like a strong model (score, reliability), not *write* like one.

**Decisions:**
1. Ship the reliability fixes (num_ctx + budget, install fallback + visibility, verified-finish, improve-nudge, both breakers, finish-shim, input-protection, kernel-time budget, crash-containment) as the v0.2 harness. (Shipped this session; 282 unit tests.)
2. Add the first leg of cross-experiment knowledge transfer: a host-computed data profile + the within-session validation trail in the supervisor's view. (Shipped.)
3. Stop prompt-tuning the coder for cell structure in-session; author the prompt out-of-band to reach the quality bar. If staged R&D *notebooks* are wanted, do it at the deliverable layer (split the winning pipeline into labeled sections), not by constraining the driver. (DECISIONS.md 2026-06-07.)

**Out of scope / pending:** the finalized coder prompt (gating v0.2); seeding the code path for reproducibility; revisiting the concatenated carry-forward if the finalized prompt assumes staged cells; the LLM-summarizer + Critic specialists (v0.4).

## 2026-06-08 — Research: a research-grounded rewrite of the cell-by-cell coder prompt

RESEARCH_LOG: synthesizing the final coder SYSTEM prompt

Question. The shipped coder prompt already says "WORK IN STAGES, ONE step per cell, never preprocess + fit + write in a single cell" and still gets monolithic 181-line cells (F-A), fresh-prepare-every-improve (F-B), dense PREPARE blocks (F-C), and positional column indices (F-D) across gemma4:12b and qwen3:14b. The synthesis question: which structural moves make small, evidence-driven cells the path of least resistance for a weak model, without lying about what the harness enforces, inside a hard 500-950 word, brace-restricted, ASCII, no-dash format contract.

Sources and principles. The decisive principle came from reading the runtime, not from prose theory. I verified in src/iterate/core/coder.py _drive (lines 220-294) that the ONLY pre-execution run_cell gates are: no-tool-call (retry_nudge), empty-code, the identical-normalized-code repeat breaker (lines 267-270), and finish-without-valid-predictions (lines 230-258). The same-error breaker and auto-install notes are POST-execution. There is no content inspection that rejects a fit+write bundle and no detection of re-fitting raw X_train. Two of the three candidate prompts leaned their entire anti-F-A and anti-F-B force on a fabricated "is rejected before it runs" claim. For a weak model this is worse than silence: a monolith RUNS, the threat never fires, and the model then discounts the real breakers too. So the governing principle is parity: every behavior-changing rule in this harness that works (verified finish, repeat breaker, same-error breaker) is enforced in code; staging is the one rule left as pure prose and it is the one that fails. The honest levers are the one real free gate (identical-code repeat breaker) and the one real cost lever (kernel-execution seconds). The prompt must ground deterrence in those, not in a lie.

Second source: prompt-engineering evidence that a weak model imitates the SHAPE of one worked example more than it follows declarative prose, and that primacy and recency dominate for position-biased small models. So the design replaces the five named cell types and the duplicated SEQUENCE recipe with ONE worked example whose shape IS the unit of work, front-loads the role plus the read-decide unit, and puts the starkest structural constraint last.

Structural decisions, grounded in our data. (1) Redefine the atomic cell: each cell does exactly one of inspect / transform-one-group-into-a-NEW-variable / fit-and-score / write, and ends by PRINTING its one result. This makes the monolith structurally incoherent rather than merely discouraged, because PREPARE-as-a-5-action-bundle was the template the model was faithfully obeying. (2) Convert the single compound baseline sentence into an ordered named-cell chain A through H. A compound sentence invites a compound cell; the ordered list makes the staged path the literal reading. This is the highest-leverage F-A fix and it directly answers the diagnosis that F-A is worst exactly at fit+predict+score+write, the steps the shipped example never demonstrated. (3) Carry the worked example all the way to a written prediction, because the model reverts to its complete-script prior the moment it hits a step with no template. The example now shows combine-into-ONE-named-matrix X_tr, the validation carve, the fit, the printed score, the X_holdout transform with the SAME fitted objects, and the write. (4) Make build-on-state the low-effort move structurally: name X_tr as the single carried artifact and show the IMPROVE pass refitting ONLY the changed transform and deriving a new matrix from the live X_tr, which is the positive recipe that replaces the deleted false threat for F-B. (5) Select columns by dtype as a dedicated printed step (num_cols, cat_cols) reused everywhere, which makes positional indices structurally unnecessary and closes the F-D to encoder-thrash chain. (6) Add one genuine read-then-DECIDE micro-loop (Cell B prints categorical level counts, the encoder choice follows from that evidence) to realize the stated goal of progressive insight rather than read-then-proceed sanity checks. (7) Reframe budget as self-interest (a large cell that errors wastes everything in it; a small cell wastes only itself) since a weak model follows self-interested optimization more reliably than a style plea.

Correctness invariants preserved. I kept the leakage rule (fit on X_train, reuse the SAME object on X_holdout), and tightened it with a scoping clause our critiques flagged: simple imputers, scalers, and OneHotEncoder may fit on full X_train before the carve, but TargetEncoder, frequency, and quantile transforms must fit INSIDE the training split or they inflate the validation score. The example builds index-aligned frames throughout, so the concat aligns to y_train and the holdout columns match the trained columns, which I confirmed by running all eight cells end-to-end on synthetic data with a non-default index, injected NaNs, and text columns: prediction count matched holdout rows, X_tr columns matched the holdout columns, and no NaN was injected into the target by misalignment. The host PROFILE handoff, the FE-first encoding menu, the sklearn-native and version-clash guard, the binding improve loop, the error-to-action map, the verified finish and finish-shim distinction, and the input-reset guarantee all survive in substance.

Honesty and contract. No sentence asserts a harness gate that does not exist; a regex check confirms no reject-claim survives. SYSTEM uses only metric, direction, predictions_csv; USER uses only the five allowed placeholders and avoids the metric placeholder that broke one candidate. Both are pure ASCII, no emoji, no em or en dashes, no double-hyphen separators, no markdown fences in SYSTEM. SYSTEM is 947 words, inside the band with a small buffer rather than the zero-headroom 943-949 of the candidates; the length was paid for by collapsing the redundant closing structural rule into the recency line.

## 2026-06-09 — Research: a research-grounded rewrite of the supervisor (strategist) prompt

Question: Run c7ddda92 (gemma4:12b, churn/f1, 10 experiments, all succeeded) had the profile showing F1 plus 73/27 class balance from experiment 1, yet 0 of 10 supervisor briefs touched imbalance handling; the coder found class_weight=balanced on its own in iteration 10 (val 0.548 to 0.619, the run best, holdout 0.5936). Contrast run 0b5cf3 proved transfer compounds once a lever is found (threshold tuning at experiment 5, then 0.567 to 0.619 across 6-10). An older run showed the opposite failure: threshold tuning briefed 5 times consecutively without improvement. What supervisor prompt structure makes a 12B select the metric-appropriate lever early AND pivot after repeated non-improvement?

Sources: arxiv 2507.13949 (primacy bias: position in the decision hierarchy is selection probability for small models); arxiv 2502.11122 (hierarchical expert prompts: numbered priority gates over prose); arxiv 2504.08525 (Task Memory Engine: coverage must be rendered as deterministic strings, not inferred; untried levers are invisible because they never appear in the history text); arxiv 2503.13657 (repetitive-action mode collapse in 12-28 percent of weak-model runs); arxiv 2403.15371 (in-context exploration needs explicit bounds); arxiv 2501.18817 (small models plan well given IF-THEN gates and slot-filling templates); arxiv 2507.18624 (prompts must only reference context that actually renders, or models discount all rules); arxiv 2406.07791 (one-shot exemplars dominate weak-model output style); arxiv 1902.08285 (principled early stopping); local verification of sklearn 1.8.0 (HistGradientBoostingClassifier accepts class_weight since 1.2, so the planned "RandomForest supports it, HistGB needs sample_weight" parenthetical was false on this runtime); shipped infra at src/iterate/core/supervisor.py (lever ledger, seven classes, rendered as the last block of history_section; Technique scoreboard carries counts and best scores).

Findings: (1) Buried means never chosen: "imbalance handling" as the 4th item of a sub-list was selected 0/10 times; the replay shows the model consumed the sub-list in surface order and never reached item 4. (2) The metric-to-lever inference (F1 plus 73/27 implies imbalance-or-threshold) never happened in 10 tries; a 12B needs the conditional precomputed verbatim as a gate, not derivable. (3) A rendered ledger fixes invisibility only if the prompt names it; stored-but-unreferenced context is the same failure mode as stored-but-not-acted-on digests. (4) The two observed failure directions (9-experiment model-swap orbit, 5x threshold fixation) share one missing mechanism: persistence bounded both ways, where one failure keeps a lever in the pool but two consecutive non-improving attempts on a class force a switch. (5) Examples outweigh rules: the canonical exemplar must be the exact brief wanted at experiment 2, with the keep-clause and a single technique; an "A or B" disjunction lets a weak coder implement both and destroy attribution. (6) Rules must be loop-free under literal execution: repair needs a hard cap, pivot needs stated precedence over the decide procedure, and pivot's fallback must differ from the action that just stalled. (7) Factual errors in prompts are not just noise: the wrong sklearn parenthetical actively steered toward bundling a model swap with the imbalance lever.

Decisions: Shipped the ladder skeleton: five numbered rungs (BASELINE, METRIC LEVER, REPAIR, UNTRIED CLASS, REFINE BEST), PIVOT outranking every rung, COMPOUND keep-best-add-one, a two-slot brief (so far: / next:) doubling as the notebook Hypothesis cell, and a ledger-checkable STOP. METRIC LEVER sits at rung 2 with a precomputed trigger (imbalance-sensitive metric, minority class below roughly 40 percent, baseline score exists, class untried) and names class_weight=balanced on the current model as the single move; threshold tuning is the rung's one allowed second firing (a refine, not a repeat), which unburies the proven 0b5cf3 sibling despite the class-granular ledger. Corrected the sklearn fact. Named the shipped Levers line with a safe default (absent line means everything untried) and avoided referencing classes the shipped _LEVER_MARKERS does not compute, so model swaps are gated textually to rung 5 instead of to a nonexistent ledger class. Trimmed the brief field to one worked example and dropped the cardinality-4 target-encoding exemplar that taught a wrong justification pattern. Changed STOP from OR to AND (plateau AND exhausted ledger), the direct c7ddda92 fix, flagged for explicit sign-off because it alters a keep-item. Logged infra caveats rather than over-claiming in the prompt: the bare "threshold" marker can false-positive the imbalance class, and ledger annotations (attempt counts, recency) remain candidate refinements.

## 2026-09-13 - Research: the vision recipe for `DLModelTarget` (v0.6)

**Question:** v0.6 adds image classification by transfer learning on the same loop, with a 12B local model writing the training cells. Which dataset can honestly carry the claim, which backbone and library, what do MPS on the Mac and CUDA on a 6 GB RTX 4050 each need, how is the 4050 reached, and which levers actually move a 3 to 5 epoch fine-tune so the supervisor's ladder is ranked by evidence? Six research angles ran in parallel, one synthesis merged them, and the eight claims the design rests on were each handed to a skeptic told to refute them. All eight held.

**Sources reviewed:**
1. [Kornblith, Shlens, Le 2019](https://arxiv.org/pdf/1805.08974) and [Kolesnikov et al. 2020, BiT](https://arxiv.org/pdf/1912.11370) - Flowers102 has 1 of 8,189 images in the ImageNet-1k training set and zero test near-duplicates; ResNet-50 goes 93.2 to 97.5 from probe to fine-tune; fine-tuning beat a linear probe in 179 of 192 dataset and model pairs.
2. [fastai noisy_imagewoof.csv](https://raw.githubusercontent.com/fastai/imagenette/master/noisy_imagewoof.csv) with [Russakovsky et al., ILSVRC](https://ar5iv.labs.arxiv.org/html/1409.0575) - Imagewoof and Imagenette carry 1,350 images per class, exactly ILSVRC-2012's 1,300 training plus 50 validation images. Every image is an ImageNet image.
3. [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Zenodo 7711810, EuroSAT](https://zenodo.org/records/7711810), [Helber et al. 2019](https://ar5iv.labs.arxiv.org/html/1709.00029) - one 329 MiB download with no account and no license text, numeric labels only; 27,000 Sentinel-2 tiles, MIT, zero ImageNet overlap by construction since Sentinel-2A launched in 2015.
4. [torchvision models](https://docs.pytorch.org/vision/stable/models.html), [ATen MPSFallback.mm](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/mps/MPSFallback.mm), [torch/csrc/Exceptions.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/Exceptions.cpp), [issue 97236](https://github.com/pytorch/pytorch/issues/97236) - pinned weight enums since `DEFAULT` may change; `PYTORCH_ENABLE_MPS_FALLBACK` is read at static init so it must precede the import; `torch.OutOfMemoryError` is the class a cell runner catches; seeded MPS training is not bitwise reproducible and the issue is open.
5. [Kumar et al. 2022, LP-FT](https://arxiv.org/abs/2202.10054), [Li et al. 2020](https://ar5iv.labs.arxiv.org/html/2002.11770), [FixRes](https://ar5iv.labs.arxiv.org/html/1906.06423), [mixup](https://ar5iv.labs.arxiv.org/html/1710.09412) - head from the probe then fine-tune everything beats both alone; the optimal fine-tune learning rate spans 0.001 to 0.5 with domain distance; testing at 1.15 to 1.3x the train resolution is worth about 1.4 points; mixup is worth 0.2 at 90 epochs and 1.5 at 200, a long-schedule lever.

**Approaches considered:**
- **Dataset: Imagewoof (the sprint plan), Pets, Flowers102, EuroSAT.** Imagewoof rejected: every image is an ImageNet image (source 2), and measured here it leaves under two points to fine-tuning. Pets rejected as headline: 1.58% test overlap and one point of headroom. Flowers102 headline, EuroSAT certification, both with the overlap verdict stated.
- **Backbone library: torchvision only, or timm.** timm adds a second dependency and a second cache that makes an HTTP call even when warm. Rejected for v0.6; every backbone the ladder names ships in torchvision, with the weight enum pinned rather than `DEFAULT`.
- **Data loading and precision on MPS: DataLoader workers and autocast, or one pre-decoded tensor in fp32.** Spawn workers cost seconds per epoch on macOS and autocast has no measured gain. One uint8 tensor at training resolution, batches sliced from it, autocast off, seed recorded, no reproducibility claim on MPS.
- **The 4050 laptop: native Windows or WSL2.** The Windows PyPI torch wheel is CPU only, Triton has no Windows build, and this harness has POSIX-only tty checks. WSL2 with the Linux CUDA wheel, sshd inside WSL over Tailscale on the host, harness and kernel on the same side of the hop.

**Decision:** Flowers102 headline, EuroSAT certification, torchvision only, resnet18 `IMAGENET1K_V1` first with resnet50, efficientnet_b0 and convnext_tiny as swaps, one pre-decoded tensor per session, `torch.OutOfMemoryError` caught in the cell runner, AMP plus channels_last plus `cudnn.benchmark` as one switch on CUDA only, the 4050 over WSL2. The measurement that sets the budgets (M5, MPS, resnet18, batch 64, 3 epochs): Flowers102 probe 0.892 to fine-tune 0.958 at 21 s per epoch, EuroSAT 0.897 to 0.974 at 14 s, Imagewoof 0.853 to 0.870 at 28 s; every fine-tune fits the existing 600 s cell. Supervisor ladder by expected gain per cell: unfreeze with the probe as head, learning rate and schedule, layer-wise decay, input resolution, backbone swap, BatchNorm freeze, label smoothing, test-time augmentation validated on the holdout, light augmentation; EMA, class weights on balanced data and weight-decay retuning are named no-ops.
**Smallest viable implementation (Day 1):** the image adapter behind the tabular seam (path column or class-folder tree, absolute paths, image bytes in the data version, byte-named copies with no suffix and one fixed mtime, rows shuffled once, a header-only profile), the user's own split for every family (`--train` plus `--holdout`, Tony's call), two prepare scripts with checksums, `pillow` in core and a `[vision]` extra. The folder switch came on Day 2 and the target on Day 4; the preamble and prompts are planned for Fri Sep 18.
**How I'll verify it works:** synthetic-PNG tests for portrait, grayscale, corrupt, missing and cross-split duplicate images and for a class-sorted source not reaching the kernel in class order; the adapter run on the three real datasets, where it counted byte-identical images across the split in two of them; the Day 4 ceiling sweep puts a measured number beside every example before a run is read.

**Out of scope today:** the target and the code path (Day 4 and Fri Sep 18); a 4050 memory table (measured on Sun Sep 20, never cited from a 3060); a deliverable beyond the notebook, since `ExperimentResult.artifacts` holds strings and a fine-tuned head is not one; species names for Flowers102 unless a verified index turns up; timm.

---

**Corrections, 2026-09-16, measured building the target.** On MPS an out-of-memory error is a plain `RuntimeError` ("MPS backend out of memory"), not `torch.OutOfMemoryError`, which only CUDA raises; the runner reads the message. And by default the MPS pool never raises: its high watermark ratio defaults to 1.7 of the recommended working set, past this Mac's 24 GB, so the machine pages instead. Even at 1.0 a convnext_tiny recipe at 384 px grew swap by 9.7 GB on this Mac before raising, so the runner starts torch with the high ratio at 0.7 and the low at 0.56: every recipe measured fits under it, and the 384 px one raises after 1.8 GB. Setting the high ratio alone fails at startup, because the low default of 1.4 then sits above it.

---

## 2026-09-16 - Research: a number label for images, and a baseline trained from zero

**Question:** Tony's two calls at the image-run plan gate on 2026-09-16: every mode predicts classes and numbers, and the image baseline is a plain CNN trained from zero with basic prep, not a pretrained model. So v0.6 needs a public image dataset whose label is a number, where a small CNN at 64 px clearly beats guessing the mean and a pretrained fine-tune clearly beats the CNN, with an open licence, a download with no account, and no ImageNet images. And the CNN needs a recipe that is stable enough to be a floor.

**Sources reviewed:**
1. [NASA Tropical Cyclone Wind Estimation on Source Cooperative](https://source.coop/nasa/tropical-storm-competition) and [the dataset DOI](https://doi.org/10.34911/rdnt.xs53up) - 70,257 training frames from 494 storms, wind speed in knots, CC-BY-4.0, every file served on its own with no login; the official test set continues 227 of its 371 storms from the training set, all of those frames later in time.
2. [arXiv 2404.08325](https://arxiv.org/html/2404.08325) and [DrivenData's winners write-up](https://www.drivendata.co/blog/wind-dependent-variables-winners/) - a pretrained ResNet-18 at 224 px, given the current frame and the two before it, reported at 10.5 kt on the competition's test set; winners at 6.25 to 6.50 kt with stacks of past frames on that same test set, which continues 227 of its 371 storms from training.
3. [TCIR, KDD 2018](https://www.csie.ntu.edu.tw/~htlin/paper/doc/kdd18tcir.pdf) - a CNN on 64 by 64 crops reaches 10.59 kt, the closest evidence that storm intensity survives a small input.
4. [SKIPP'D, arXiv 2207.00913](https://arxiv.org/pdf/2207.00913) and [its Hugging Face mirror](https://huggingface.co/datasets/solarbench/SKIPPD) - sky images at a native 64 px with solar power output, CC-BY, a two-layer CNN at 2.43 kW RMSE; no verified stronger model on the official test days.
5. [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html) - 10,073 photographs with a quality score, the design pass's first pick; the quality signal is blur, noise and compression, detail a 64 px resize removes.
6. [Nutrition5k, arXiv 2103.03375](https://arxiv.org/pdf/2103.03375), [UTKFace](https://susanqq.github.io/UTKFace/), [Galaxy Zoo DECaLS on Zenodo](https://zenodo.org/records/4573248) - 3,490 overhead dish photos, below the size bar; face ages for non-commercial research only; galaxy vote fractions in about 104 GB of images.

**Approaches considered:**
- **KonIQ-10k:** a real human judgement with an honest random split. Rejected before any download: at the 64 px the baseline trains at, the label's signal is mostly gone, so the baseline would sit near the mean and headroom would be measured against a constant.
- **SKIPP'D:** the 64 px signal is proven by the paper's small CNN. Kept as the fallback: the download is twice the storm subset's and the headroom above a small CNN is unproven.
- **Storm wind speed:** a whole-image label on frames nothing like ImageNet's photographs, with a published ladder above a small CNN. The risk was the 64 px resize, so it was measured before any code depended on it.
- **Nutrition5k, UTKFace, Galaxy Zoo:** too small, a licence that forbids redistribution, and too large.
- **The baseline recipe:** the plain CNN with a constant learning rate, or with a cosine decay. Measured on EuroSAT at 64 px over 20 epochs, the constant rate swung between 0.823 and 0.937 across the last three epochs; the cosine decay ended at 0.950 and 0.945 on two seeds, with its last five epochs inside half a point.

**Decision:** the storm set, every 6th frame of all 494 storms, 11,908 frames and about 250 MB, split by storm with a fixed seed into 395 training storms and 99 held out. The baseline is three blocks of 3 by 3 convolution, batch norm, ReLU and pooling at 32, 64 and 128 channels, a pooled linear head, 64 px, 20 epochs, AdamW at 1e-3 with cosine decay, batch 64, no augmentation; a number label is trained on its training mean and spread and mapped back before scoring. Measured on an Apple M5 with MPS before the build, error bars resampling whole holdout storms:

| | RMSE, knots | error bar |
|---|---|---|
| guess the training mean | 26.6 | 1.7 |
| the plain CNN, 64 px | 13.1 | 0.7 |
| the plain CNN, 128 px | 16.5 | 1.2 |
| resnet18 probe, 160 px | 13.2 | 0.6 |
| resnet18 fine-tune, 3 epochs, 128 px | 9.3 | 0.4 |

The CNN beats the mean by 13.4 knots, 95% interval 10.9 to 15.8, and the fine-tune beats the CNN by 3.8, 2.8 to 4.8. The CNN got worse at 128 px, which is why the baseline stays at 64. The same CNN on the image classification examples: EuroSAT 0.950 and Flowers102 0.554 at 64 px. With a constant rate, Flowers102 scored 0.567 at 64 px and 0.545 at 128 px, at four times the time.

**Smallest viable implementation:** `simple_cnn` beside the pretrained backbones, the fixed baseline recipe with no time cap, number labels through the target, the probe as a ridge fit, `examples/cyclone_wind/prepare.py` with pinned sha256 values for both label files and for the manifest of every frame, and the `holdout` key in the eval corpus so the storm split is sealed as made.
**How I'll verify it works:** the prepare script rebuilt the measured subset offline with the manifest pin matching and both CSVs byte-identical; the ceiling sweep's baseline row reproduced the pre-build measurement to the fourth decimal, 13.1203.

**Out of scope today:** SKIPP'D as a second number-label example; stacks of past frames, which is how the competition winners won; a per-group error bar inside the sweep, which cannot see storms: a formula that treats every frame as independent came out two to five times smaller than the storm-level one, depending on the row and the formula, so a number-label sweep stores none and the example's README carries the storm-level error bar instead.

---

## 2026-09-25 - Research: what a winner costs to serve (v0.7, Day 1)

**Question:** v0.7's flagship is a serving budget: the best score you can afford to serve, never score per dollar. Before a wall can refuse anything, the harness needs a price it can stand behind for every winner a run can produce: a scikit-learn pipeline it never sees (only its predictions reach the host), a torch network it knows by recipe, and a prompt on a model it knows by name. What does each cost a month at a request rate, on the cheapest machine or API that can serve it, and which parts of that number are measured, published, or assumed?

**Sources reviewed:**
1. [AWS EC2 on-demand prices via Vantage](https://instances.vantage.sh/aws/ec2/t3.small), [GCP via gcloud-compute.com](https://gcloud-compute.com/e2-small.html) and [Azure's retail prices API](https://prices.azure.com/api/retail/prices) - on-demand Linux hourly prices for two CPU boxes per cloud, a T4 box on each cloud, and the L4 and A10 boxes where the cloud sells one, read 2026-09-25. The AWS and GCP official pages are script-rendered, so those rows are mirrors and say so; Azure's API needs no key and answered directly. [AWS's bulk price index](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json) needs no key either; [GCP's catalog](https://cloudbilling.googleapis.com/v1/services) needs one.
2. [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Together serverless models](https://docs.together.ai/docs/serverless-models) and [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing) - per-million token prices for the models a prompt run can name today. Groq's self-serve Llama models were retired in August 2026 and its pricing page renders nothing, so a Groq target is unpriced. DeepSeek prices are peak and off-peak; the table stores peak and says so.
3. [torchvision's classification table](https://docs.pytorch.org/vision/stable/models.html) - resnet18 11.7M weights and 1.81 GFLOPS, resnet50 25.6M and 4.09, convnext_tiny 28.6M and 4.46 at 224 px (torchvision's GFLOPS column counts multiply-adds, which is the unit the Pricer uses); the weights match the built networks on torchvision 0.29 to the parameter, and the served network drops the 1,000-class ImageNet classifier, so the Pricer subtracts it.
4. [NVIDIA's ResNet-50 v1.5 PyTorch README](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md) - a T4 at batch 1, FP32, 10.7 ms an image average latency, the one published GPU number that matches the plain torch file this tool delivers. The L4 and A10 numbers in [MLPerf v3.1](https://raw.githubusercontent.com/mlcommons/inference_results_v3.1/main/closed/NVIDIA/results/L4x1_TRT/resnet50/SingleStream/performance/run_1/mlperf_log_summary.txt) and v1.1 are INT8 TensorRT at 0.35 and 0.47 ms, a different deployment, so those rows carry no rate.
5. [Optimizing PyTorch inference on CPU](https://towardsdatascience.com/optimizing-pytorch-model-inference-on-cpu/) - resnet50 at 224 px, batch 1, FP32 eager on an AWS c7i.xlarge: about 44 ms. Not a vendor number, but the one absolute cloud-CPU figure found, and within 12% of the 49.8 ms measured here.
6. [llama.cpp's benchmark thread](https://github.com/ggml-org/llama.cpp/discussions/15013) - a 7B model at 4-bit decodes at about 46 tokens a second on a T4 and 108 on an A10G, community-submitted. Kept for a later calibration, not used: the local-model capacity is not claimed.

**Approaches considered:**
- **Score per dollar:** fold cost into the objective. Rejected in May and again here: it prefers a cheaper worse model over an affordable better one. The objective stays the score; the cost is a wall, and this entry is only about the number the wall reads.
- **Measure the winner's latency inside the run:** the kernel has the model. For tables it means a harness-owned `submit()` that changes the prompt contract a 12B has learned; for images the FIT line is what the 12B reads and every key added to it is a key it will try to pass to `fit()`; and a latency on the user's Mac still needs a mapping to a cloud box. Deferred: the facts the host already has price every family today.
- **Publish throughput per box from vendor benchmarks:** only the T4 has a plain-PyTorch batch-1 number, and no cloud CPU has one. Multiply-adds alone under-predict small images by three times (resnet18 at 64 px measures 6.9 ms; scaling its own 224 px time by multiply-adds says 2.1), because fixed overhead dominates. So the CPU rate is a measured table per backbone and size, taken as a 2-vCPU box, and a custom stack scales from resnet18's measured rate.
- **A rate for every GPU row from the INT8 numbers:** would over-state capacity twenty to thirty-fold for the torch file a run delivers. Those rows price one box and say the capacity is not estimated.

**Decision:** a dated snapshot ships in the package (`core/serving_prices.json`), every row with its source and the day it was read, and a pure-arithmetic Pricer (`core/serving.py`) turns the winner's facts into a profile with a basis line for every number. A month is 730 hours; the default rate is 1,000 requests an hour. An API model costs tokens times price per request, with the tokens per record measured by the prompt runtime on the holdout submit (the kernel writes them into `prompt.json`; when every answer came from the cache the prompt text stands in and the basis says so). A machine-hosted model takes the cheapest row whose memory fits the weights, with instances rounded up to cover the rate; a row with no benchmark rate cannot promise to cover it, so it is offered only when no rated row fits at all, and then prices one box and says the capacity is not estimated. Measured on this Mac, batch 1, two threads, torch 2.14, for the CPU reference table:

| backbone | 64 px | 128 px | 224 px |
|---|---|---|---|
| resnet18 | 6.9 ms | 11.1 ms | 26.3 ms |
| resnet50 | 13.2 ms | 21.7 ms | 49.8 ms |
| convnext_tiny | 166 ms | 265 ms | 527 ms |

And for tables, the slowest single-row predict per estimator family over laptop_price, mobile_price, heart_risk and adult_income, scikit-learn 1.8.0, two threads: linear 2.9 ms, tree ensemble 12.2 ms, boosting 4.5 ms, nearest neighbours 3.2 ms, svm 2.9 ms, mlp 2.8 ms; doubled as the margin. The pipeline overhead dominates every family, since a batch of 100 rows costs 0.02 to 0.16 ms a row.

**Smallest viable implementation:** the schema, the snapshot, the Pricer, the token capture in the prompt preamble, `--requests-per-hour`, and the `serving:` line and blocks. No budget, no refusal, no lever, no prompt change.

**How I'll verify it works:** the arithmetic to the cent on a fake snapshot; the shipped snapshot parses, is dated, and every row is sourced; the backbone table equals the vision runner's; a priced block lands in `best.json` from the real CLI with a stubbed loop, and in `prompts.yaml` through the writer; the printed `ask:` line is byte-identical to before.

**Out of scope today:** the wall and the price gate (Day 2); `iterate cost` (Day 3); a measured latency inside the run; a price refresh script; tokens a second for local models on cloud GPUs; quantization; sizes for the agent's own timm networks.

---

## 2026-09-26 - Research: pricing against the clouds' own lists (v0.7, Day 2)

**Question:** Day 1 priced every winner against a snapshot of 13 machines read once. Tony's call: a static list is baseless for a production tool. So: where does each cloud publish its prices in a form a CLI can read without an account, how big is it, what does it carry, and what does it not; and where does the size of any network the Researcher might name come from?

**Sources reviewed:**
1. [AWS's price list files](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json) - AWS publishes every service's prices as files, no key. The EC2 file for one region is 481 MB as JSON and 303 MB as CSV (`Content-Length`, read 2026-09-26; `Last-Modified` the day before), so it is a refresh, never a run-time fetch. The CSV opens with five metadata lines, then columns including `TermType`, `Unit`, `PricePerUnit`, `Instance Type`, `Current Generation`, `vCPU`, `Memory`, `Tenancy`, `Operating System`, `CapacityStatus`, `GPU`, `GPU Memory` and `Pre Installed S/W`, which is everything the Pricer needs. The 18 KB `region_index.json` lists the per-region files.
2. [Azure's retail prices API](https://prices.azure.com/api/retail/prices) - open, no key, 1,000 rows a page with a `NextPageLink`, filterable by `serviceName`, `armRegionName` and `priceType`. Measured on eastus: 9 pages, 8,770 rows, a few seconds (3 s and 12 s on two walks), 1,677 Linux hourly SKUs after dropping Spot, Low Priority and Windows. It carries `armSkuName`, `retailPrice` and `unitOfMeasure` and no vCPU or memory at all, so specs come from [Azure's VM naming convention](https://learn.microsoft.com/azure/virtual-machines/vm-naming-conventions) through a family table.
3. [GCP's Cloud Billing catalog](https://cloud.google.com/billing/v1/how-tos/catalog-api) - needs an API key, and prices per vCPU-hour and GB-hour rather than per machine. Not in Day 2; the shipped rows stand and the line says so.
4. [timm's published benchmark tables](https://github.com/huggingface/pytorch-image-models/tree/main/results) - `benchmark-infer-fp32-nchw-pt240-cpu-i9_10940x-dynamo.csv` is most timm models at batch size 1 with `param_count`, `infer_gmacs` and `infer_step_time` in milliseconds on one CPU, 1,435 rows over 1,120 networks (one row has a blank field and is dropped), torch 2.4 with `torch.compile`. `results-imagenet.csv` has parameter counts but no multiply-adds. 926 of the 1,101 names in `timm_models.txt` are in the benchmark table.
5. [Vantage's instances.json](https://instances.vantage.sh/instances.json) - an open dataset built from AWS's API with specs and per-region prices in one small file. Considered as the AWS source and not used: it is a secondary, and AWS's own file works once streamed.

**Approaches considered:**
- **Fetch at run time from every cloud:** AWS's file makes that impossible; a run cannot download 300 MB to print one line. Rejected for AWS, kept for Azure in the background because its walk is 3 seconds.
- **A model reading pricing pages:** rejected. A 12B reading a page invents a price, and every number here carries a source a user can check.
- **Vantage for AWS:** small and current, but a secondary. Rejected while AWS's own file works; it is the fallback if that file ever stops.
- **Azure specs from a second API:** the resource SKU API has them but needs an account. Rejected for a table read from the SKU name, which drops what it cannot size rather than guess. Measured: 929 of 1,677 SKUs sized, 748 dropped: constrained-vCPU sizes of known families, names outside the pattern (confidential `_cc_v5`, M `_v3` size codes, the RTX PRO 6000 SKUs), B v2 sizes, the D v1 and v2 names whose number is a size code, the M, L, FX, HB, DS, A and ND families, and the fractional A10 sizes.
- **Capacity on machines nobody measured:** one request uses about two threads, so a box serves one worker per two vCPUs, half a worker on a single vCPU, and its capacity is the 2-vCPU rate times that. A rule, said in the basis, not a measurement.
- **timm's times for named networks:** compiled on an i9, eager here. Measured gap on convnext_tiny at 224 px: 14.5 ms in timm's table against 527 ms eager on this Mac, while resnet50 is 14.9 against 49.8. So timm's times are scaled relative to resnet50 measured here and the basis says a plain deployment can be slower.

**Decision:** `core/prices.py`. `iterate prices refresh` streams AWS's CSV for the region and reduces it to on-demand Linux shared-tenancy current-generation machines, walks Azure's API, and fetches timm's table, into `~/.cache/iterate/prices/` with a date and the URL per file; `iterate prices show` prints what the Pricer would read. `load` prefers the cache per cloud and falls back to the shipped file, and every priced line prints the provenance per cloud (`prices: aws refreshed 2026-09-26 (us-east-1), gcp shipped 2026-09-25 (us-central1)`), Tony's call that offline the shipped rows stand with their date. `--cloud` prices one cloud, `--region` its region; omitted, AWS and GCP, cheapest first. `--cloud azure` also refreshes Azure in the background at the start of a run when the cache is a day old. Measured after the refresh on this Mac: AWS us-east-1 1,289 machines in under two minutes end to end (1 minute 48 seconds on the first walk), Azure eastus 929 machines, timm 1,434 rows over 1,119 networks; resnet18 at 128 px now lands on AWS's cheapest 2 GB box at $12.26 a month, a 0.5 GB box is kept out by the memory rule, and convnext_tiny at 224 px at 50,000 requests an hour scales to eight of them at $98.11.

**Smallest viable implementation:** the module, the two commands, the two flags, the background Azure refresh, timm sizes for the agent's own networks now and for the names the Researcher gives from Day 3 (`facts_from_model_name`, wired into the wall then), the provenance line.

**How I'll verify it works:** the reducers on fixture rows that include everything they must drop; the SKU parser table-driven over the families it knows and the ones it refuses; the cache read, write, staleness and merge order; the fetchers against stub clients; the flags and the two commands through the real CLI; one real refresh here for the numbers above.

**Out of scope today:** GCP through a key (designed, first cut); a background refresh for AWS (its file); a rate for any GPU but the T4; API model prices as a feed (none exists); a measured eager-versus-compiled gap per network family.
