# Build Log

> Daily task tracking for `iterate`. Build sessions log here. Recruiters reading the repo see process, not just final code.
>
> Public file. Honest about what worked + what didn't.

---

## Format per session entry

```markdown
### YYYY-MM-DD | Phase N | Session summary

**Task:** [what you set out to do today]

**What shipped:**
- Files: src/iterate/foo.py, tests/unit/test_foo.py
- Commits: <sha>
- Behavior: [what now works that didn't before]

**What didn't:**
- [honest list of what got punted, broken, or harder than expected]

**Decisions:**
- [any architectural choice made + why — link to RESEARCH_LOG entry if applicable]

**Next session:**
- [what's queued for tomorrow]
```

---

## Week 1 Backlog (P0 = blocking, ordered)

| # | Task | Files | Done? |
|---|------|-------|------|
| 1 | Project metadata: `pyproject.toml`, deps pinned, ruff + mypy config | `pyproject.toml` | ✅ |
| 2 | `.env.example` with Ollama default + optional cloud backend keys (Groq/Together/Deepseek/Anthropic/OpenAI) + e2b + Kaggle | `.env.example` | ✅ |
| 3 | Empty `src/iterate/` package skeleton (folders + `__init__.py`) | `src/iterate/**/` | ✅ |
| 4 | Pydantic schemas — `Experiment`, `ExperimentResult`, `Metrics`, `FailureCase`, `Candidate` | `src/iterate/schemas/experiment.py` | ⏳ |
| 5 | `LLMClient` protocol — what every backend implements | `src/iterate/llm/base.py` | ⏳ |
| 6 | `OpenAICompatibleClient` — first real working LLM call (default: Ollama localhost:11434 + qwen2.5-coder:14b) | `src/iterate/llm/openai_compatible.py` | ⏳ |
| 7 | Smoke test — verify Ollama call actually works end-to-end | `tests/unit/test_openai_compatible.py` | ⏳ |
| 8 | CLI scaffold — `iterate --help` runs (no commands yet, just typer setup) | `src/iterate/cli.py` | ⏳ |
| 9 | First commit message convention doc (semantic commits) | `BUILD_LOG.md` (this section) | ⏳ |

---

## Week 4 Backlog (preview — locks in once Wk 1-3 ship)

The Week 4 phase shifts the agent from "runs experiments in isolation" to "discovers data + history via MCP servers and grounds itself before iterating."

| # | Task | Files |
|---|------|-------|
| 4.1 | MCP client — connects to multiple servers via stdio/HTTP | `src/iterate/mcp/client.py` |
| 4.2 | MCP server registry — config-driven lifecycle (spawn/kill/health-check) | `src/iterate/mcp/registry.py` |
| 4.3 | MCP-to-OpenAI tool bridge — translate MCP tool defs to OpenAI tool schemas (works with Ollama, Groq, etc.) | `src/iterate/mcp/tool_bridge.py` |
| 4.4 | Wire filesystem MCP server (read local notebooks/docs/logs) | config + docs |
| 4.5 | Wire postgres MCP server (DB introspection + read-only sampling) | config + docs |
| 4.6 | Wire notion MCP server (search past experiment pages, write new ones) | config + docs |
| 4.7 | Discovery flow — `iterate init --discover` introspects all MCPs, surfaces summary, pauses for human review | `src/iterate/core/discovery.py` |
| 4.8 | Researcher (arxiv + papers-with-code) — uses MCP for paper retrieval where available | `src/iterate/core/researcher.py` |
| 4.9 | Proposer — uses memory + MCP-discovered context to rank candidates | `src/iterate/core/proposer.py` |
| 4.10 | Memory store integration — every experiment + tool call logged for audit | `src/iterate/core/memory.py` |
| 4.11 | Logging adapter via Notion MCP — write experiment cards to Notion | `src/iterate/adapters/logging/notion.py` |
| 4.12 | Logging adapter for plain markdown (fallback when no Notion) | `src/iterate/adapters/logging/markdown.py` |

---

## Done

### 2026-05-23 | Week 0 | Project scoped, repo scaffolded

**Task:** Lock in project scope + push initial folder structure.

**What shipped:**
- Folder structure (src/, tests/, examples/, etc.)
- `.gitignore` with project-specific entries (LAUNCH_POST, PRD, BIZ, GTM, BOTTLENECKS, EVAL_LOG, PROGRESS_NOTES, data/, models/, .iterate/)
- `README.md` (public hero)
- `BUILD_LOG.md` (this file)
- `RESEARCH_LOG.md` (citation trail template)
- `pyproject.toml`, `Makefile`, `Dockerfile` (placeholder), `.env.example`
- All `__init__.py` files for the `iterate` package skeleton

**What didn't:**
- No actual `iterate` code yet — pure scaffolding.

**Decisions:**
- Name: `iterate` (open-source, single-word brandable)
- Architecture: hexagonal — core + targets + adapters + llm separated cleanly
- v1 covers BOTH `ModelTarget` (sklearn/XGBoost first) AND `PromptTarget` (LLM-as-judge)
- LLM backends pluggable from day 1 (Claude default, Llama/Deepseek via adapters)
- Memory store will use sqlite (no external infra dependency)

**Next session (2026-05-24):**
- Task #4 (Pydantic schemas) → Task #5 (LLMClient protocol) → Task #6 (Anthropic client) → Task #7 (smoke test)

---

## Commit message convention

```
<type>(<scope>): <short summary>

[optional body explaining why, what changed, and any non-obvious choices]

[optional footer — refs to RESEARCH_LOG entries, closes BOTTLENECKS#N, etc.]
```

**Types:**
- `feat:` — new functionality
- `fix:` — bug fix
- `perf:` — performance work
- `refactor:` — no behavior change
- `test:` — tests only
- `docs:` — docs only
- `chore:` — tooling, config, deps
- `research:` — RESEARCH_LOG entry only (no code, locked-in research session)

**Examples:**
- `feat(llm): anthropic client with tool-use loop helper`
- `fix(memory): retrieve_relevant returned duplicates on partial match — added DISTINCT`
- `perf(researcher): cache arxiv API results to disk (eliminated re-fetch on retry)`
- `research(targets): chose Protocol over ABC for BenchmarkTarget — see RESEARCH_LOG 2026-05-24`

---

## Backlog (lower-priority, tracked)

Items not in this week's top P0 but worth keeping visible.

(empty)
