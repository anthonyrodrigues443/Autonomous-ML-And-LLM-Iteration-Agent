# Known Limitations

> iterate is built incrementally and honest about what each version does **not** do yet.
> This register tracks every deliberate limitation and the version that lifts it. It pairs
> with the roadmap in [BUILD_LOG.md](BUILD_LOG.md). A few limits are **permanent by design**
> (security policy), and are marked as such rather than scheduled for removal.

Version map (from the roadmap): v0.2 sandboxed code-gen · v0.3 interactive CLI · v0.4
multi-agent + agent picks metric/model · v0.5 prompts · v0.6 DL/vision · v0.7
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
| Single-agent (one Proposer in a deterministic loop). | **v0.4** | Multi-agent: specialist agents (Researcher, Proposer, Critic) report to a supervisor. |
| Local-model tool-calling occasionally drops a tool call (qwen3:14b). | mitigated; ~v0.4 | Mitigated with retries; cloud backend is the reliable path; multi-agent specialists reduce per-call load. |
| Proposals come from the LLM's own knowledge, not literature retrieval. | **v0.4** | The Researcher specialist adds arxiv / papers-with-code grounding. |
| "Auditable report" = persistent memory + run summary, not a generated report doc. | ~v0.10 | A dedicated Reporter is a later backlog item. |
| No `iterate history` / `best` / `why-failed` query commands. | post-v0.1 | The data is in Memory; the CLI surface is a small near-term add. |

## Execution / sandbox (v0.2)

| Limitation (today) | Lifted at | Notes |
|---|---|---|
| e2b network egress-deny not enforced (needs a custom sandbox template). | **v0.2.x** | Flagged in `compute/runner.py`. |
| e2b path not live-verified yet (no key in dev). | **v0.2** | Verified when a key is added (Day 5). |
| Import→package resolution for install-on-demand is a hand-kept alias map (`sklearn`→`scikit-learn`, …) with import-name fallback. | **TBD** | Provisional architecture — to be revisited (logged in DECISIONS.md). Backstop: a wrong/missing package fails its install → captured failure + retry, so the map only needs the common stack. |
| Auto-install of the agent's imports runs only in the disposable sandbox, never the user's local venv. | **permanent (local) / by-design** | `--compute local` will not silently mutate your environment (typosquat / dependency-conflict risk). On local, a missing import is a captured failure; install-on-demand is a sandbox feature. |
| Code-gen winners return predictions, not a pickled model. | **permanent** | Security/portability; the readable artifact is the v0.2 notebook deliverable. |
| No cloud-GPU compute backend (local + e2b only). | ~v0.6 | When DL / large-model training needs it. |

## Permanent by design (not limitations to remove)

- **Never execute user-provided source code.** A `--source` notebook/script is read as text only and reconstructed; the sandbox runs the agent's OWN generated code, never the user's. (Security policy.)
- **The harness/loop is bounded.** Human-approval gates and bounded autonomy are features, not limits to remove.
