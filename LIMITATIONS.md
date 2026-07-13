# Known Limitations

> iterate is built incrementally and honest about what each version does **not** do yet.
> This register tracks every deliberate limitation and the version that lifts it. It pairs
> with the roadmap in [BUILD_LOG.md](BUILD_LOG.md). A few limits are **permanent by design**
> (security policy), and are marked as such rather than scheduled for removal.

Version map (from the roadmap): v0.2 sandboxed code-gen + the multi-agent cell-by-cell
core (Supervisor, coding agent, Summarizer) · v0.3 interactive CLI · v0.4 Researcher +
Critic specialists + agent picks metric/model · v0.5 prompts · v0.6 DL/vision · v0.7
cost-constrained · v0.8 infer features/target · v0.9 MCP discovery · v0.10 benchmark + UI
· v1.0 one-sentence input.

## Metrics & evaluation

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Fixed 8-metric panel only: `accuracy, f1, precision, recall` / `rmse, mae, mse, r2` (`core/scoring.py`). Anything else raises "unknown metric". | **v0.4** | Pairs with "agent picks the metric." The `Metrics` schema is already generic (values dict + primary + direction); only `score()` is capped. |
| No probability-based metrics (ROC-AUC, log-loss, PR-AUC). We score predicted **labels** (`predict`), not probabilities (`predict_proba`). | **v0.4** | The most notable gap: AUC/PR-AUC are the standard metrics for imbalanced problems like churn. Needs probability capture, not just a longer list. |
| `f1`/`precision`/`recall` averaging hardcoded (binary if ≤2 classes, else macro). No weighted/micro. | **v0.4** | |
| Single train/holdout split; no cross-validation. The same holdout is reused to select across all iterations (mild selection bias). | **v0.4** | The Critic agent + eval hardening land here; nested/CV selection vs final-test split. |
| Single primary metric (single-objective); no multi-objective trade-offs. | **v0.7** | Cost-constrained optimization: best score within a serving budget. |

## Models & preprocessing

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Spec path limited to allow-listed installed libraries (scikit-learn / XGBoost / LightGBM). | **v0.2** | Superseded by the code-gen path (Day 4): the CodeProposer has **no library allow-list** — it writes a `train_and_predict` and imports whatever it wants; we install its imports before running. The spec path keeps the allow-list as the cheap/fast/reliable option. |
| Spec-path preprocessing is fixed (median impute + one-hot). | **v0.2** | The code-gen path lets the agent preprocess freely. Spec-path flexibility itself: TBD. |

## Data & inputs (the "shrink the inputs" dial)

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Tabular CSV only as input. | **v0.5 / v0.6** | Prompts (v0.5), vision/DL (v0.6). |
| Classification + regression tasks only. | **v0.5 / v0.6** | Expands with prompt + vision targets. |
| Single local CSV (`load_csv`); no Kaggle / HuggingFace / DB / MCP sources. | **v0.9** | MCP discovery finds the data itself. |
| Single target column; no multi-target / multi-label. | TBD | When needed; not scheduled. |
| `--metric` must be given explicitly. | **v0.4** | Agent picks the metric. |
| `--target` + features must be given. | **v0.8** | Infer from the data + a one-line description. |
| Full one-sentence input not yet possible. | **v1.0** | Autonomous discovery. |

## Agent architecture

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Multi-agent core is Supervisor + coding agent + Summarizer; no Researcher or Critic specialist yet. | **v0.4** | The v0.2 default path is already multi-agent (re-sequenced from the original v0.4 plan, see DECISIONS.md). Researcher (literature grounding) and Critic (eval hardening) land at v0.4. The `--spec` path stays single-proposer. |
| On a weak local model, expect 1-2 iterations per run to end as honestly-labeled duplicates or measured null results. | **by-design (the capability floor)** | A deterministic guard stack (grounded briefs, no-op gates, duplicate hashing, floor submissions) detects, labels, and converts weak-model waste rather than hiding it; certified across 21 instrumented gemma4:12b runs. The residue that remains is the model's floor, not silent process failure: each such iteration is stamped in memory and annotated in its notebook. |
| Local-model tool-calling occasionally drops a tool call (qwen3:14b). | mitigated; ~v0.4 | Mitigated with retries; cloud backend is the reliable path; multi-agent specialists reduce per-call load. |
| Local 14B is unreliable at acting on its own error tracebacks (observed: claims to fix a missing import, ships the same `NameError` again). Modeling/exploration depth scales with the backend model. | use a cloud backend | A/B (2026-06-04) confirmed: a 70B explored models far more, and a feature-engineering prompt unlocked real FE. 14B is the floor; `--backend groq/openai/…` goes deeper. The v0.2 cell-by-cell session + breakers (repeated-cell, same-error) + v0.4 Critic reduce reliance on one blind call. |
| Thinking mode (`--think`, ollama backends) applies to the CODER only and is OFF by default. | **by-design** | Tested live (2026-06-08): a thinking trace crowds out the single tool call that strict roles (supervisor, summarizer) must emit, and on the coder it made cells MORE monolithic (plans in-head, dumps one cell). Kept as an opt-in debugging instrument — the trace renders as a "Model reasoning" cell in the notebook. |
| Staged R&D cells are validated on the gemma4:12b floor only; qwen3:14b has not been re-run since the worked-example prompt landed. | **monitor** | The 2026-06-07 "staging is model-bound" conclusion was overturned for gemma by the worked-example prompt (monolithic cells 31–35% → <1%); whether the same prompt stages qwen is untested. |
| Proposals come from the LLM's own knowledge, not literature retrieval. | **v0.4** | The Researcher specialist adds arxiv / papers-with-code grounding, and **records its citations** (`Candidate.citations` + `source="researcher"`) so it neither re-reads papers nor re-runs the same idea. |
| Per-iteration history fed to the proposer is description + score + recent stdout — not a distilled insight, so context grows with run length. | **lifted in v0.2** (cell-by-cell path) | The Summarizer (pulled forward from v0.4) digests each finished experiment once (~150 tokens: techniques, data insights, what helped/hurt, takeaway); the supervisor reads all digests + a technique scoreboard + a lever ledger, never the notebooks. The one-shot `--spec` path keeps the old shape. The Critic remains v0.4. |
| The Summarizer costs one extra LLM call per finished experiment, and a failed call silently degrades the digest to its deterministic skeleton (components + score + validation trail, no insight fields). | **by-design** | Degrading beats crashing: a digest is a nice-to-have for the next brief, never worth failing a recorded run. Observed once live (1/10 digests skeleton-only); a deterministic takeaway is now synthesized so the field is never empty. |
| "Auditable report" = persistent memory + a runnable notebook per experiment (Day 6), not a prose report doc. | ~v0.10 | The notebook deliverable covers "what it tried + a runnable winner"; a dedicated prose Reporter is a later backlog item. |
| No `iterate history` / `best` / `why-failed` query commands. | post-v0.1 | The data is in Memory; the CLI surface is a small near-term add. |

## Execution / sandbox (v0.2)

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| e2b network egress-deny not enforced (needs a custom sandbox template). | **v0.2.x** | Flagged in `compute/runner.py`. |
| Import→package resolution for install-on-demand is a hand-kept alias map (`sklearn`→`scikit-learn`, …) with import-name fallback. | **TBD** | Provisional architecture — to be revisited (logged in DECISIONS.md). Backstop: a wrong/missing package fails its install → captured failure + retry, so the map only needs the common stack. |
| Local install-on-demand requires explicit consent (`--install` / setup); without it, a missing import on `--compute local` is a captured failure. | **by-design** | `--compute local` never silently mutates your environment (typosquat / dependency-conflict risk). With consent it installs into iterate's own env; e2b always installs in its disposable sandbox. |
| The agent prints diagnostics *inside* `train_and_predict` on the one-shot (`--spec`-adjacent) path, so a pure "just explore the data" turn there still costs one scored iteration. | **addressed in v0.2 cell-by-cell** | The v0.2 cell-by-cell session (`--code`, default) lets it inspect-then-build within an experiment for free (kernel-time budget); the one-shot path keeps the old behavior. |
| One-shot code-gen writes the whole pipeline blind; aggressive feature engineering can silently produce near-zero scores (NaN/inf, single-class predictions) the agent can't foresee. | **addressed in v0.2 cell-by-cell** | The cell-by-cell session catches it mid-session (the agent inspects intermediate output + a model-ready assert runs before `.fit()`), instead of writing the pipeline in one shot. |
| The cell-by-cell carry-forward hands the next experiment a concatenated blob of the winning session's cells (not a labeled, staged pipeline). | **works; typed handoff at v0.4** | The staged coder prompt treats the starting point as REFERENCE ONLY (find its score, rebuild as small cells, beat it), which the validation runs show working. A typed `Session` handoff replaces the blob when the v0.4 specialists need structured access. |
| Code-gen winners return predictions, not a pickled model. | **permanent** | Security/portability; the readable artifact is the v0.2 notebook deliverable. |
| No cloud-GPU compute backend (local + e2b only). | ~v0.6 | When DL / large-model training needs it. |

## Permanent by design (not limitations to remove)

- **Never execute user-provided source code.** A `--source` notebook/script is read as text only and reconstructed; the sandbox runs the agent's OWN generated code, never the user's. (Security policy.)
- **The harness/loop is bounded.** Human-approval gates and bounded autonomy are features, not limits to remove.
