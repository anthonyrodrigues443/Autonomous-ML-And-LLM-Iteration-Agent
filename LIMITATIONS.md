# Known Limitations

> iterate is built incrementally and honest about what each version does **not** do yet.
> This register tracks every deliberate limitation and the version that lifts it. It pairs
> with the roadmap in [BUILD_LOG.md](BUILD_LOG.md). A few limits are **permanent by design**
> (security policy), and are marked as such rather than scheduled for removal.

Version map (from the roadmap): v0.2 sandboxed code-gen + the multi-agent cell-by-cell
core (Supervisor, coding agent, Summarizer) · v0.3 interactive runs (terminal UI, chat,
pause/resume/stop, notebook Q&A, standing rules) · v0.4 Researcher + Critic specialists +
agent picks metric/model · v0.5 prompts · v0.6 DL/vision · v0.7 cost-constrained · v0.9
infer features/target + MCP discovery (absorbs v0.8) · v1.0 one-sentence input + benchmark
+ dashboard (absorbs v0.10). "backlog" below = tracked for after v1.0.

## Metrics & evaluation

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Metric vocabulary is what scikit-learn can score (54 names: 12 curated plus every applicable sklearn scorer). Anything outside it still raises "unknown metric". | **widened v0.4**; closed by design | Direction is DERIVED from sklearn's own scorer registry, never hand-written and never supplied by a model: sklearn's scorers are all higher-is-better by construction, with loss metrics carrying a -1 sign. That is what keeps the set open to whatever the Researcher proposes while still making it impossible for an agent to invert the loop, since direction drives what banks as best and when a run stops. Clustering scorers are excluded (they compare two label assignments, not a prediction against a target). The always-computed PANEL stays at 12 so history is comparable and cheap; a selected metric outside it is computed on top. |
| A probability metric needs an estimator that can produce probabilities. On the `--spec` path an estimator without `predict_proba` (SVC at its default, Ridge) scores labels only, so a probability primary fails with a reason rather than silently reporting nothing. | **by-design** | The code-gen path is unaffected: the agent chooses its own model and the finish gate rejects a submission missing `probabilities.csv`, so it is told before the iteration is lost. |
| Single train/holdout split; no cross-validation. The same holdout is reused to select across all iterations (mild selection bias). | Critic **v0.4**; CV backlog | The Critic agent (suspicious-win flags, selection-bias watch) lands at v0.4; a k-fold/CV selection option is tracked on the backlog. |
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
| Single local CSV (`load_csv`); no Kaggle / HuggingFace / DB / MCP sources. | **v0.9** (partial) | v0.9 ships MCP discovery over filesystem + Postgres; Kaggle/HuggingFace loaders and the Notion/github servers are tracked on the backlog. |
| Single target column; no multi-target / multi-label. | TBD | When needed; not scheduled. |
| `--metric` must be given explicitly. | **v0.4** | Agent picks the metric. |
| `--target` + features must be given. | **v0.9** | Infer from the data + a one-line description (the v0.8 milestone, shipping inside the v0.9 release). |
| Full one-sentence input not yet possible. | **v1.0** | Autonomous discovery. |

## Agent architecture

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Multi-agent core is Supervisor + coding agent + Summarizer + Researcher; no Critic specialist yet. | Critic **v0.4** | The v0.2 default path is already multi-agent (re-sequenced from the original v0.4 plan, see DECISIONS.md). Researcher (literature grounding) and Critic (eval hardening) land at v0.4. The `--spec` path stays single-proposer. |
| On a weak local model, expect 1-2 iterations per run to end as honestly-labeled duplicates or measured null results. | **by-design (the capability floor)** | A deterministic guard stack (grounded briefs, no-op gates, duplicate hashing, floor submissions) detects, labels, and converts weak-model waste rather than hiding it; certified across 21 instrumented gemma4:12b runs. The residue that remains is the model's floor, not silent process failure: each such iteration is stamped in memory and annotated in its notebook. |
| Local-model tool-calling occasionally drops a tool call (qwen3:14b). | mitigated; ~v0.4 | Mitigated with retries; cloud backend is the reliable path; multi-agent specialists reduce per-call load. |
| Local 14B is unreliable at acting on its own error tracebacks (observed: claims to fix a missing import, ships the same `NameError` again). Modeling/exploration depth scales with the backend model. | use a cloud backend | A/B (2026-06-04) confirmed: a 70B explored models far more, and a feature-engineering prompt unlocked real FE. 14B is the floor; `--backend groq/openai/…` goes deeper. The v0.2 cell-by-cell session + breakers (repeated-cell, same-error) + v0.4 Critic reduce reliance on one blind call. |
| Thinking mode (`--think`, ollama backends) applies to the CODER only and is OFF by default. | **by-design** | Tested live (2026-06-08): a thinking trace crowds out the single tool call that strict roles (supervisor, summarizer) must emit, and on the coder it made cells MORE monolithic (plans in-head, dumps one cell). Kept as an opt-in debugging instrument — the trace renders as a "Model reasoning" cell in the notebook. |
| Staged R&D cells are validated on the gemma4:12b floor only; qwen3:14b has not been re-run since the worked-example prompt landed. | **monitor** | The 2026-06-07 "staging is model-bound" conclusion was overturned for gemma by the worked-example prompt (monolithic cells 31–35% → <1%); whether the same prompt stages qwen is untested. |
| Literature grounding is retrieval over OpenAlex + arXiv only, and reaches the run only when the supervisor asks for it (plus always on iteration 1, capped per run). | **by-design** | Both are keyless and free, so `pip install iterate-ai` needs no account. papers-with-code, named in the original plan, was shut down and now redirects to HuggingFace; OpenAlex replaced it and covers ~320M works against arXiv's ~2.4M preprints. A citation is always an identifier the API returned: the model selects a paper by INDEX and the harness resolves it, so a DOI that was never fetched cannot be emitted. `--no-research` skips the specialist entirely for offline or fast runs. |
| Per-iteration history fed to the proposer is description + score + recent stdout — not a distilled insight, so context grows with run length. | **lifted in v0.2** (cell-by-cell path) | The Summarizer (pulled forward from v0.4) digests each finished experiment once (~150 tokens: techniques, data insights, what helped/hurt, takeaway); the supervisor reads all digests + a technique scoreboard + a lever ledger, never the notebooks. The one-shot `--spec` path keeps the old shape. The Critic remains v0.4. |
| The Summarizer costs one extra LLM call per finished experiment, and a failed call silently degrades the digest to its deterministic skeleton (components + score + validation trail, no insight fields). | **by-design** | Degrading beats crashing: a digest is a nice-to-have for the next brief, never worth failing a recorded run. Observed once live (1/10 digests skeleton-only); a deterministic takeaway is now synthesized so the field is never empty. As of v0.4 the skeleton is read off `core.dossier`, which observes strictly more than the skeleton carries (data facts the cells printed, error signatures, session shape). Those extra fields are deliberately NOT seeded into the digest's insight fields: they would flow into the supervisor's planning context, and additive planning context is what regressed the floor model in the June EDA-ledger experiment. Wiring them in is gated on a before/after run, not on their being available. |
| `core.dossier`'s data-fact extraction is heuristic: a printed line counts as a data fact if it carries a shape tuple or names a data property. It will miss a fact phrased unusually and can keep a line that merely mentions "rows". | **by-design** | The guarantee it does make is the one that matters for a fallback: facts are quoted verbatim from stdout, never paraphrased or inferred, so a dossier can be incomplete but cannot be wrong. Recall is a nice-to-have; invention would make it unusable as the Summarizer's fallback. |
| "Auditable report" = persistent memory + a runnable notebook per experiment (Day 6), not a prose report doc. | ~v0.10 | The notebook deliverable covers "what it tried + a runnable winner"; a dedicated prose Reporter is a later backlog item. |
| No `iterate history` / `best` / `why-failed` query commands. | backlog | The data is in Memory, and v0.3's in-run Q&A covers the live case; the offline CLI surface is a small add tracked on the backlog. |

## Execution / sandbox (v0.2)

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| e2b network egress-deny not enforced (needs a custom sandbox template). | backlog | Flagged in `compute/runner.py`; the sandbox still isolates the host, this hardens what the sandbox itself can reach. |
| Import→package resolution for install-on-demand is a hand-kept alias map (`sklearn`→`scikit-learn`, …) with import-name fallback. | **TBD** | Provisional architecture — to be revisited (logged in DECISIONS.md). Backstop: a wrong/missing package fails its install → captured failure + retry, so the map only needs the common stack. |
| Local install-on-demand requires explicit consent (`--install` / setup); without it, a missing import on `--compute local` is a captured failure. | **by-design** | `--compute local` never silently mutates your environment (typosquat / dependency-conflict risk). With consent it installs into iterate's own env; e2b always installs in its disposable sandbox. |
| The agent prints diagnostics *inside* `train_and_predict` on the one-shot (`--spec`-adjacent) path, so a pure "just explore the data" turn there still costs one scored iteration. | **addressed in v0.2 cell-by-cell** | The v0.2 cell-by-cell session (`--code`, default) lets it inspect-then-build within an experiment for free (kernel-time budget); the one-shot path keeps the old behavior. |
| One-shot code-gen writes the whole pipeline blind; aggressive feature engineering can silently produce near-zero scores (NaN/inf, single-class predictions) the agent can't foresee. | **addressed in v0.2 cell-by-cell** | The cell-by-cell session catches it mid-session (the agent inspects intermediate output + a model-ready assert runs before `.fit()`), instead of writing the pipeline in one shot. |
| The cell-by-cell carry-forward hands the next experiment a concatenated blob of the winning session's cells (not a labeled, staged pipeline). | **works; typed handoff on the backlog** | The staged coder prompt treats the starting point as REFERENCE ONLY (find its score, rebuild as small cells, beat it), which the validation runs show working. A typed `Session` handoff is tracked on the backlog. |
| Code-gen winners return predictions, not a pickled model. | **permanent** | Security/portability; the readable artifact is the v0.2 notebook deliverable. |
| No cloud-GPU compute backend (local + e2b only). | ~v0.6 | When DL / large-model training needs it. |

## Interactive runs (v0.3)

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| Messages take effect at the next safe boundary (a cell finishing, or just before planning), never mid-cell. The ack is instant; delivery can lag a slow cell or a slow LLM turn. | **by-design** | An interrupt is a kill, not a pause; the cell boundary is where state is coherent, gates run, and the kernel survives. |
| Only the bare words `pause` / `resume` / `stop` are controls (trailing punctuation tolerated). "please stop the run" routes as guidance, with only the ack text hinting otherwise. | monitor | The intent router deliberately has no stop authority: a misclassified sentence must never be able to end a run. |
| Message intent is classified by the same (possibly small, local) model that supervises. A failed or garbled classification defaults to the least-destructive route: a steer for the work in front of us, and the console says so. | **by-design** | Interpretation degrades, it never crashes or blocks the run. |
| Standing rules are run-scoped (they do not survive a restart) and lean-capped: 3 rules, 90 chars each; steers cap at 200 chars per message. | persistence: backlog | Dense injected context measurably regresses small models (the reverted EDA-ledger experiment); the caps are a feature. |
| In-run Q&A answers from the CURRENT run's experiments only; it cannot answer about previous runs. | TBD | Deliberate: iteration numbers restart every run, and cross-run answers were confidently wrong. An offline query surface is on the backlog. |
| Token-level streaming of the model's output did not ship (planned for v0.3, cut for the release). | backlog | The transcript streams per cell and per event, not per token. |
| The `--spec` fast lane stays non-interactive. | **by-design** | The frozen v0.1 path is kept cheap and predictable. |
| The interactive UI needs a foreground terminal; piped, scripted, CI, and backgrounded (`... &`) runs are plain and non-interactive. | **by-design** | A backgrounded process reading its terminal would be suspended by the OS; iterate detects this and stays hands-off. |

## Permanent by design (not limitations to remove)

- **Never execute user-provided source code.** A `--source` notebook/script is read as text only and reconstructed; the sandbox runs the agent's OWN generated code, never the user's. (Security policy.)
- **The harness/loop is bounded.** Human-approval gates and bounded autonomy are features, not limits to remove.
