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
