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

## Craft and detail calls

| Date | The AI's default | My call | Why |
|---|---|---|---|
| 2026-05-27 | Treat slow model tests as "probably my laptop," or fix on a hunch. | Fix **only if measurement proves** it's a real bug, not the hardware. | Don't optimize on vibes — and a 10s fit on 120 rows is never the hardware. Measuring found a ~200x thread-oversubscription bug. |
| 2026-05-25 | Use emoji status markers in the public docs. | Drop decorative emoji; words for status, plain `✓`/`✗` in matrices. | Emoji-as-status reads "AI-generated / beginner" in serious infrastructure repos. |
| 2026-05-27 | Leave the BUILD_LOG inconsistent (Week 1 had a "Done?" column; Weeks 2–3 didn't). | Add the column everywhere. | Asymmetry in the public trail looks careless. |
| 2026-05-26 | Date a research entry to the day it was written up. | Correct it to the day the research actually happened. | The decision trail is only useful if it's accurate. |
| 2026-05-27 | Mixed day-numbering in the session notes (plan-position vs calendar). | Reconcile to calendar-day headers with plan-position in parentheses. | One unambiguous timeline. |

---

Each entry plays out in the public trail: [BUILD_LOG.md](BUILD_LOG.md) for what shipped and [RESEARCH_LOG.md](RESEARCH_LOG.md) for the sourced reasoning behind the bigger calls.
