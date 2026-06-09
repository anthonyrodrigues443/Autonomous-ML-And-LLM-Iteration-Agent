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
