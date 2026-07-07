.PHONY: help install install-dev test test-unit test-integration lint format typecheck clean demo build publish

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

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/iterate

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

demo:
	uv run iterate run --data examples/churn_tabular/data.clean.csv --target Churn --metric f1

# Release: verify, build fresh artifacts, publish (publish prompts for the PyPI token).
build: clean
	uv run pytest -m "not integration" -q
	uv run ruff check .
	uv run mypy src/iterate
	uv build

publish: build
	uv publish
