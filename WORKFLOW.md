# How this project is built

`iterate` is built in the open, one short focused session at a time. This page is about the *method*, not the code: how decisions get made and who makes them. [BUILD_LOG.md](BUILD_LOG.md) shows **what** shipped each day; this shows **how**.

## The short version

I direct; an AI coding agent (Claude Code) implements; nothing merges without my review.

The decisions that shape the project are mine — architecture, scope, sequencing, trade-offs, what's broken and why. The AI drafts the detailed plan and writes the code from that direction. Every line passes my review before it's logged and merged. The planning *is* the decision-making; the typing is the commodity.

## Who owns what

| Concern | Owner |
|---|---|
| Architecture, scope, sequencing | Me |
| Product + strategy trade-offs | Me |
| Catching what's wrong, deciding the fix | Me |
| Final review + merge veto | Me |
| Drafting the step-level plan (from my direction) | AI |
| Writing the implementation + tests | AI |
| Running the toolchain (lint, types, tests) | AI |

This split is deliberate. The highest-value engineering work is judgment — choosing the right problem, picking the right trade-off, noticing when something is off. That is where I spend my attention. Generating an agreed design into code is delegated and reviewed.

## The session loop

Every build session runs the same loop:

1. **Research** — before any code, the decision gets researched (libraries, papers, alternatives) and the reasoning is written down in [RESEARCH_LOG.md](RESEARCH_LOG.md), in my own words.
2. **Plan** — the AI drafts a concrete, step-level plan from the day's goal in the BUILD_LOG roadmap.
3. **Review + redirect** — I read the plan and approve it, push back on it, or change the approach. This is where the real decisions land.
4. **Implement** — the AI writes the code and tests against the agreed plan.
5. **Review** — I read the diff. If something is off, it goes back to step 4.
6. **Log + ship** — the public docs are updated, a pull request is opened, reviewed, and merged. One reviewable PR per unit of work.

## What keeps it from being AI slop

A workflow that leans on an AI is only as good as its guardrails. The ones this project holds to:

- **Research before code.** Every meaningful decision has a RESEARCH_LOG entry with sources and the alternatives I rejected — not vibes.
- **Measure, don't assume.** When the model tests felt slow, the lazy call was "it's my laptop." Measuring instead found a ~200x thread-oversubscription bug that would have crippled the agentic loop.
- **Correctness by construction.** Data is split *before* any preprocessing, so there is no leakage path; dynamic model imports are allow-listed to trusted libraries, so there is no arbitrary-code execution.
- **Reversible by default.** Incremental versioned releases, one PR per slice, a full audit trail in git plus the logs.
- **I have to understand it.** I do not merge code I cannot explain. If a review turns up something I do not follow, it gets explained before it ships.

## Direction beats typing

The decisions that shaped this project are mine, and they range from roadmap-level strategy to one-line craft fixes. Two for the flavour:

- **Re-planned the roadmap to agent-first.** The plan had drifted breadth-first — build every problem type, *then* turn the agent on around Week 7. I caught that this buried the whole point and re-sequenced so the agentic loop is the v0.1 milestone. A direction call, not a code change.
- **"Only fix it if it is genuinely not the computer."** On the slow tests I declined to let a performance fix happen on a hunch: fix it only if measurement proves it is a real bug, not the hardware. It was real (~200x). The discipline was the contribution.

The full running record — every point where my judgment redirected the AI's default, strategic calls down to small fixes — is in **[DECISIONS.md](DECISIONS.md)**. That list is the honest answer to "whose thinking is this?"

## Why in the open

The roadmap, the daily goals, the decision trail, and the dead-ends are all public — [BUILD_LOG.md](BUILD_LOG.md) (what shipped, honestly, including what did not) and [RESEARCH_LOG.md](RESEARCH_LOG.md) (why, with sources). The process is part of the portfolio: it shows how I think, not only what I produced.
