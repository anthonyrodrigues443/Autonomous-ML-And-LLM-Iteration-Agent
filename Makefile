.PHONY: help install install-dev test test-unit test-integration lint format typecheck clean demo build publish eval eval-ceilings eval-report eval-list eval-shapes

help:
	@echo "iterate — Makefile targets"
	@echo ""
	@echo "  make install         Install runtime dependencies (uv)"
	@echo "  make install-dev     Install dev + all optional deps"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests (uses VCR cassettes)"
	@echo "  make lint            Run ruff check"
	@echo "  make format          Run ruff format"
	@echo "  make typecheck       Run mypy (strict)"
	@echo "  make clean           Remove caches + build artifacts"
	@echo "  make demo            Run the churn demo (requires .env)"
	@echo ""
	@echo "  make eval            Cross-version eval sweep (internal; resumes by default)"
	@echo "  make eval-ceilings   Measure brute-force ceilings per dataset (no LLM)"
	@echo "  make eval-report     Rebuild evals/RESULTS.md from the store"
	@echo "  make eval-list       Show the eval corpus and what the store holds"
	@echo "  make eval-shapes     Fast adversarial dataset-shape checks"

install:
	uv sync

install-dev:
	uv sync --all-extras

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/unit/ -m unit

test-integration:
	uv run pytest tests/integration/ -m integration

# Scope matches `make build` below, which runs `ruff check .` over the whole repo.
# When these two disagree a release fails on a file the day-to-day lint never saw.
lint:
	uv run ruff check src/ tests/ evals/

format:
	uv run ruff format src/ tests/ evals/

typecheck:
	uv run mypy src/iterate evals

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

demo:
	uv run iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1

# Internal measurement. Never shipped — see evals/README.md.
eval:
	uv run python -m evals.run sweep $(ARGS)

eval-ceilings:
	uv run python -m evals.run ceilings $(ARGS)

eval-report:
	uv run python -m evals.run report $(ARGS)

eval-list:
	uv run python -m evals.run list $(ARGS)

eval-shapes:
	uv run pytest tests/unit/test_evals_shapes.py -q

# Release: verify, build fresh artifacts, publish (publish prompts for the PyPI token).
build: clean
	uv run pytest -m "not integration" -q
	uv run ruff check .
	uv run mypy src/iterate
	uv build

publish: build
	uv publish
