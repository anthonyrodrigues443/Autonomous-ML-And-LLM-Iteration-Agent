# Decision Log

A running record of points where my judgment redirected the AI's default. The split in [WORKFLOW.md](WORKFLOW.md) says I direct and the AI implements — this is the evidence for that claim, kept deliberately so it shows the full range: roadmap-level strategy down to one-line craft fixes. Nothing here is the AI's idea that I rubber-stamped; each is a call I made against what the AI was about to do.

Entries are factual — what the AI defaulted to, what I decided instead, and why. The reasoning is meant to carry the weight, not adjectives.

## Strategic and architectural calls

| Date | The AI's default | My call | Why |
|---|---|---|---|
| 2026-05-27 | Build breadth-first — every problem type first, turn the agent on around Week 7. | Re-sequence to **agent-first**: the agentic loop is the v0.1 milestone; everything after is capability expansion. | Building breadth before the agent buries the differentiator. Sequence toward the thing that matters first. |
| 2026-05-28 | Leave sandboxed code-gen ("the agent writes + runs any training code") late in the roadmap. | Bump it **early to v0.2**, right after the v0.1 loop. | "Run the model the research actually recommends" is the real unlock — it shouldn't wait behind six other versions. |
| 2026-05-27 | Frame cost-constrained optimization as *the* moat. | Reframe as the **specialized combination**; cost-aware serving is the *flagship*, not the only moat. | One feature is not defensible. The defensibility is the combination plus the domain specialization. |
| 2026-05-28 | Candidate spec as a flat `{"model": x, ...hyperparams}`. | Nested `{"model": x, "params": {...}}`. | Clean separation of *which model* from *its params*, no key collisions, and the exact shape the agent will emit from research. |
| 2026-05-26 | Treat a reported/prior baseline score as the number to beat. | **Never trust a reported score** — always re-measure the baseline through our own eval. | A number computed a different way is not comparable; you cannot honestly claim an improvement against it. |
| 2026-05-29 | Take `--baseline` + `--source` as context only, no behavior; defer reproduction to Week 10. | **Make the inputs do something in v0.1** — by end of Week 3, source must reconstruct the baseline approach through our eval and then improve from there. | A baseline number we never use is dead weight. If you take an input, it has to drive something. |
| 2026-05-29 | Implied user-provided source code would be executed in the v0.2 sandbox to reproduce a baseline. | **Never execute user-provided source code — ever.** Source is read-only text the LLM uses to reconstruct the approach; the v0.2 sandbox runs the agent's OWN generated code, never the user's. | User notebooks/scripts are untrusted (malware/RCE-adjacent); a sandbox reduces but doesn't eliminate the risk. Reconstruct-from-text is the right permanent policy. |
| 2026-05-30 | Defer all interactive-CLI features (pause, mid-run chat, streaming, Ctrl-C) to the Week-12 Streamlit UI. | **Split into two milestones.** v0.2 picks up the *cheap* interactive wins (live progress display, streaming LLM responses, graceful Ctrl-C) alongside sandboxed code-gen; v0.3 is a new focused milestone for *full mid-run chat* (pause/resume/conversational state). Streamlit UI shifts to v0.10. | Waiting until Week 12 for any interactivity is too late; lumping it all into v0.2 balloons that milestone. Two focused milestones ship each piece when it's actually ready. |

## Craft and detail calls

| Date | The AI's default | My call | Why |
|---|---|---|---|
| 2026-05-27 | Treat slow model tests as "probably my laptop," or fix on a hunch. | Fix **only if measurement proves** it's a real bug, not the hardware. | Don't optimize on vibes — and a 10s fit on 120 rows is never the hardware. Measuring found a ~200x thread-oversubscription bug. |
| 2026-05-25 | Use emoji status markers in the public docs. | Drop decorative emoji; words for status, plain `✓`/`✗` in matrices. | Emoji-as-status reads "AI-generated / beginner" in serious infrastructure repos. |
| 2026-05-27 | Leave the BUILD_LOG inconsistent (Week 1 had a "Done?" column; Weeks 2–3 didn't). | Add the column everywhere. | Asymmetry in the public trail looks careless. |
| 2026-05-26 | Date a research entry to the day it was written up. | Correct it to the day the research actually happened. | The decision trail is only useful if it's accurate. |
| 2026-05-27 | Mixed day-numbering in the session notes (plan-position vs calendar). | Reconcile to calendar-day headers with plan-position in parentheses. | One unambiguous timeline. |
| 2026-05-30 | When `/v1` can't disable qwen3 thinking, bundle the workaround into `OpenAICompatibleClient`. | Build a separate **`OllamaClient`** adapter on Ollama's native `/api/chat` with `think:false`; the OpenAI client stays clean for cloud backends. | One backend's quirk shouldn't pollute the shared client. The `LLMClient` protocol is the seam — adding a sibling adapter is the right shape. |
| 2026-05-30 | Hardcode prompt strings inside `core/proposer.py`. | Centralize every prompt in `src/iterate/prompts/prompts.yaml` (a loader exposes `PROMPTS`); modules reference structured keys. | Prompts change far more often than logic; one file = one place to review and iterate wording. |
| 2026-05-30 | Accept `--baseline X` alone (informational-only number; we'd still use the factory default for actual measurement). | **`--baseline` requires `--source`.** A number with no source describes nothing we can run, so the input is dead weight on its own. | Either it drives behavior or it shouldn't be a flag. Explicit CLI error is better than silent ignoring. |
| 2026-05-30 | `--fresh` should delete the existing memory db (true reset). | Archive it instead — rename to `<name>.YYYYMMDD-HHMMSS.bak`, then create a new clean db. Triggered by `--fresh`, `--source`, or `--baseline + --source` — any explicit "new chapter" signal. | Recoverable cheap safety net; honors the "start over" intent without losing forever. The user can `rm` the `.bak` if they really want it gone. |

---

Each entry plays out in the public trail: [BUILD_LOG.md](BUILD_LOG.md) for what shipped and [RESEARCH_LOG.md](RESEARCH_LOG.md) for the sourced reasoning behind the bigger calls.
