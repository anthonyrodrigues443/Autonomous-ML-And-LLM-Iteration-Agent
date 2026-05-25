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
