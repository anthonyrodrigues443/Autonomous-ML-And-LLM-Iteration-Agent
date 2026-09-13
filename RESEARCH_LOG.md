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

**Question:** v0.6 adds the third target family: image classification by transfer learning on a pretrained backbone, on the same loop, with a 12B local model writing the training cells. Which dataset can honestly carry the claim, which backbone and library, what does MPS on the Mac and CUDA on a 6 GB RTX 4050 each need, how is the 4050 reached from the Mac, and which levers actually move a short fine-tune, so the supervisor's ladder is ranked by evidence rather than by folklore. Six research angles ran in parallel (datasets, backbones, MPS, CUDA on 6 GB, levers, codebase fit), a synthesis merged them, and the eight claims the design rests on were each handed to a skeptic told to refute them. All eight held. One measurement was taken on this Mac to set the budgets.

**Sources reviewed:**
1. [Kornblith, Shlens, Le 2019, Do Better ImageNet Models Transfer Better?](https://arxiv.org/pdf/1805.08974) - Table H.1 counts ImageNet-train duplicates per transfer set (Flowers102: 1 of 2,040 train, 0 of 6,149 test; Pets: 227 train, 58 test); Table I.1 ResNet-50 on Flowers102 scores 93.2 as a linear probe and 97.5 fine-tuned; fine-tuning beat a logistic probe in 179 of 192 dataset and model pairs.
2. [Kolesnikov et al. 2020, Big Transfer](https://arxiv.org/pdf/1912.11370) - Table 7 finds zero near-duplicates of the Flowers102 test set inside ILSVRC-2012; the BiT-HyperRule resizes images under 96 px to 160 and crops 128.
3. [fastai noisy_imagewoof.csv](https://raw.githubusercontent.com/fastai/imagenette/master/noisy_imagewoof.csv) and [Russakovsky et al., ILSVRC](https://ar5iv.labs.arxiv.org/html/1409.0575) - Imagewoof and Imagenette carry 1,350 images per class, which is exactly ILSVRC-2012's 1,300 training images plus its 50 validation images per class. Every image is an ImageNet image.
4. [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and its [README](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt) - 8,189 images, 102 classes, 40 to 258 per class; the archive downloads with no account; no license text anywhere on the page; species names exist only as an alphabetical web table with no index.
5. [Zenodo 7711810, EuroSAT](https://zenodo.org/records/7711810) and [Helber et al. 2019](https://ar5iv.labs.arxiv.org/html/1709.00029) - 27,000 Sentinel-2 tiles at 64 x 64 px, 10 land-use classes, MIT license, one 94.7 MB zip; the old DFKI host is dead. Sentinel-2A launched in 2015, three years after ILSVRC-2012 was fixed, so overlap is zero by construction.
6. [arXiv 2305.13456](https://arxiv.org/html/2305.13456v1) - on frozen ImageNet ResNet-50 features, EuroSAT kNN accuracy rises from 82.1 to 91.2 when the 64 px tiles are upsampled to 224: input resolution is a real lever on this dataset.
7. [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) - CC BY-SA 4.0, 755 MiB, but 1.58% of its test images are ImageNet-train duplicates and ResNet-50 gains only 91.5 to 92.5 from fine-tuning. The fallback headline if an explicit license is ever required.
8. [torchvision models](https://docs.pytorch.org/vision/stable/models.html) and [PyPI torchvision](https://pypi.org/project/torchvision/) - the multi-weight API, `DEFAULT` may change across versions, `weights.transforms()` encodes each weight's preprocessing, `weights.meta['_file_size']`; torchvision 0.29.0 pins torch 2.14.0 and supports Python 3.11 and 3.12; the macOS arm64 torch wheel is 127 MB and carries MPS, the Linux PyPI wheel is already a CUDA build, the Windows PyPI wheel is CPU only.
9. [PyPI timm](https://pypi.org/project/timm/) and [HF hub environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) - timm adds huggingface_hub and safetensors and a second weight cache that makes an HTTP call even when warm unless `HF_HUB_OFFLINE=1`.
10. [ATen MPSFallback.mm](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/mps/MPSFallback.mm) - `PYTORCH_ENABLE_MPS_FALLBACK` is read at static initialisation, so it must be in the kernel's environment before torch is imported; a notebook cell is too late.
11. [torch 2.9.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0), [PR 99272](https://github.com/pytorch/pytorch/pull/99272), [PR 139390](https://github.com/pytorch/pytorch/pull/139390), [issue 97236](https://github.com/pytorch/pytorch/issues/97236) - MPS autocast exists (float16 from 2.5, bfloat16 from 2.6) with no benchmark showing a gain; 2.9 fixed a BatchNorm gradient bug on MPS; seeded MPS training is not bitwise reproducible across runs and the issue is still open.
12. [torch/csrc/Exceptions.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/Exceptions.cpp) and [v2.4.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.4.0) - `torch.OutOfMemoryError` is a direct `RuntimeError` subclass, the same object as `torch.cuda.OutOfMemoryError`, from 2.4 on. That is the name a cell runner matches.
13. [CUDA semantics notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html) and [PyTorch memory format tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) - `empty_cache` only returns memory nothing references, so a failed cell's traceback keeps its tensors alive; AMP, channels_last and `cudnn.benchmark` are documented as one package for Tensor Core GPUs.
14. [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [Microsoft WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking), [Tailscale KB 1295](https://tailscale.com/kb/1295/install-windows-wsl2) - only the Windows driver is installed, nothing NVIDIA inside WSL; mirrored networking exposes a WSL sshd on the host's interfaces; Tailscale runs on the Windows host only.
15. [Kumar et al. 2022, LP-FT](https://arxiv.org/abs/2202.10054), [VTAB](https://ar5iv.labs.arxiv.org/html/1910.04867), [Li et al. 2020](https://ar5iv.labs.arxiv.org/html/2002.11770), [SpotTune](https://ar5iv.labs.arxiv.org/html/1811.08737) - initialising the head from the probe then fine-tuning everything beats both alone; at 1,000 examples per task fine-tuning scores 65.6 against a linear 57.3; the optimal fine-tune learning rate spans 0.001 to 0.5 depending on how far the domain sits from ImageNet; fine-tuning only the last one to three blocks is worse than fine-tuning everything on all five datasets tested.
16. [FixRes](https://ar5iv.labs.arxiv.org/html/1906.06423), [mixup](https://ar5iv.labs.arxiv.org/html/1710.09412), [ResNet strikes back](https://ar5iv.labs.arxiv.org/html/2110.00476), [Muller et al.](https://ar5iv.labs.arxiv.org/html/1906.02629), [Shanmugam et al., TTA](https://ar5iv.labs.arxiv.org/html/2011.11156) - testing at 1.15 to 1.3x the training resolution is worth about 1.4 points for free; mixup is worth 0.2 at 90 epochs and 1.5 at 200, so it is a long-schedule lever; label smoothing is worth 0.3 to 0.4, under the noise floor of a small holdout; test-time augmentation is worth about 1.1 points on ImageNet and about a third of the labels it changes are wrong, so it has to be validated on the holdout, not assumed.

**The measurement (this Mac, M5, 24 GB, MPS, torch 2.14.0, resnet18, batch 64, images pre-decoded to one uint8 tensor, 3 epochs, no DataLoader):**

| dataset | px | train / holdout | probe acc | fine-tune acc, 3 epochs | s per epoch | peak MPS memory |
|---|---|---|---|---|---|---|
| imagewoof | 128 | 9,025 / 3,929 | 0.813 | 0.846 | 17 | 2.4 GB |
| imagewoof | 160 | 9,025 / 3,929 | 0.853 | 0.870 | 28 | 3.5 GB |
| eurosat | 64 | 21,600 / 5,400 | 0.897 | 0.974 | 14 | 1.2 GB |
| eurosat | 128 | 21,600 / 5,400 | 0.919 | 0.981 | 53 | 2.4 GB |
| flowers102 | 160 | 6,551 / 1,638 | 0.892 | 0.958 | 21 | 2.4 GB |
| flowers102 | 224 | 6,551 / 1,638 | 0.932 | 0.972 | 39 | 4.4 GB |

Decoding and resizing 27,000 JPEGs into the cached tensor costs 4 to 6 s; hashing every image's bytes for the data version costs under 2 s. Every fine-tune above fits inside the existing 600 s cell and the 1,800 s session holds eight or more of them, so the tabular budgets carry over unchanged. The Imagewoof rows show what ImageNet overlap looks like as a number: a probe on features that already trained on those exact photographs starts high and leaves under two points for fine-tuning to find. EuroSAT at its native 64 px leaves nearly eight.

**Approaches considered:**
- **Dataset, Imagewoof or Imagenette:** the fast.ai subsets named in the sprint plan. Rejected: every image is an ILSVRC-2012 image (source 3), so a transfer-learning result there measures the backbone remembering its own training set, and the measurement above shows it.
- **Dataset, Oxford-IIIT Pets:** explicit license, real photographs. Rejected as the headline: 1.58% test overlap and one point of fine-tune headroom (source 7); the agent would have almost nothing to win. It is the fallback if Oxford's missing license text for Flowers102 ever becomes a problem.
- **Dataset, Oxford Flowers102:** the classic transfer benchmark, no overlap (sources 1, 2), 4.3 points of published headroom on the official split, one 329 MiB download. Chosen as the headline example. Two facts stated rather than hidden: Oxford publishes no license, so the prepare script downloads and nothing is redistributed; and Oxford ships numeric labels only, so the CSV carries `class_NNN` rather than a community name map that could be wrong.
- **Dataset, EuroSAT:** satellite tiles, zero overlap by construction, MIT license, 95 MB. Chosen as the certification dataset: the domain shift makes the probe weak at native resolution (0.897 measured) and the upsampling lever real (source 6, and 0.919 measured at 128 px), so the agent has both a fine-tune gain and a resolution gain to find.
- **Backbone library, torchvision only versus timm:** timm's 1,800 models cost a second dependency, a second cache and an HTTP call per load (source 9). Rejected for v0.6; every backbone the ladder names ships in torchvision.
- **Weights, `DEFAULT` versus a pinned enum:** `DEFAULT` may silently change across versions (source 8) and would move every recorded score. Pinned: resnet18 `IMAGENET1K_V1` as the starting backbone, resnet50 `IMAGENET1K_V2`, efficientnet_b0 `IMAGENET1K_V1`, convnext_tiny `IMAGENET1K_V1` as the swaps.
- **Data loading, DataLoader with workers versus one pre-decoded tensor:** spawn-based workers cost seconds per epoch on macOS and unified memory makes a host-device copy free. Pre-decoded uint8 tensor at the training resolution, sliced into fixed-size batches, flips and crops applied on-tensor. The measurement above used exactly this.
- **MPS precision and determinism:** autocast stays off by default (supported, no measured gain, and 2.9 was still fixing its conv dtype bugs, source 11); `use_deterministic_algorithms` stays off (no MPS guarantee, an open 8x slowdown report). The notebook records the seed and the LIMITATIONS register says MPS is not bitwise reproducible.
- **The 4050 laptop, native Windows versus WSL2:** the Windows PyPI torch wheel is CPU only, the CUDA wheel is 2 to 2.6 GB, Triton has no Windows build, and this harness has POSIX-only tty checks. WSL2 with the Linux CUDA wheel, sshd inside WSL exposed by mirrored networking, Tailscale on the Windows host, and the harness and the kernel on the same side of the SSH hop.

**Decision:** Flowers102 headline, EuroSAT certification, torchvision only with pinned weights, resnet18 first, one pre-decoded tensor per session, `PYTORCH_ENABLE_MPS_FALLBACK=1` set by the harness before the kernel starts, `torch.OutOfMemoryError` caught in the cell runner with the traceback dropped before `empty_cache`, AMP plus channels_last plus `cudnn.benchmark` as one switch on CUDA only, and the 4050 reached over WSL2 plus SSH. The supervisor's vision ladder, ranked by expected holdout gain per cell at 3 to 5 epochs: 1 unfreeze with the head initialised from the probe, 2 learning rate and schedule, 3 layer-wise learning-rate decay, 4 input resolution, 5 backbone swap, 6 BatchNorm statistics frozen, 7 label smoothing, 8 test-time augmentation validated on the holdout, 9 light augmentation only, and a named no-op list the supervisor declines (model EMA, class weights on balanced data, weight-decay retuning, fixed last-k unfreezing). The published gains behind rungs 7 to 9 sit under the standard error of a 1,600-image holdout, about 1.1 points, so a brief on those rungs has to clear that bar before it counts.

**Smallest viable implementation (Day 1):** the image adapter behind the tabular seam (`adapters/data/images.py`: detect the one path column, read a class-folder tree into the same shape, resolve paths absolute, hash the image bytes into the data version, count byte-identical images across the split, copy every image under a name derived from its bytes, with one fixed mtime and no suffix, and shuffle row order once so neither a path, a name, a timestamp nor an order can carry the class, profile sizes, modes and formats from headers alone); the user's own split as a first-class input for every family (`load_split`, `--train` plus `--holdout`, Tony's call, see DECISIONS); two prepare scripts that flatten the archives into `images/NNNNN.jpg` with the label only in `data.csv`, checksums verified; `pillow` in core and a `[vision]` extra holding torch and torchvision. The target, the preamble, the CLI switch for folders and the prompts follow on Days 2 and 3.

**How I'll verify it works:** unit tests on synthetic PNGs (portrait, grayscale, corrupt, duplicated across the split); the adapter run on the three real datasets prints a profile and a data version, and it already counted three byte-identical images across Imagewoof's train and holdout split, which is the vision form of a duplicated row; the prepare scripts assert that no filename carries a class token; the ceiling sweep on Day 2 puts a measured number next to every example before any run is read.

**Out of scope today:** the target and the code path (Days 2 and 3); a memory table for the 4050 (measured on Day 5, never cited from a 3060); the vision deliverable beyond the notebook, since `ExperimentResult.artifacts` holds strings and a fine-tuned head is not one; species names for Flowers102 unless a verified index turns up; timm.

---
