# Dockerfile — placeholder, hardened in Week 5 if needed for sandbox / deploy.
# Local dev should use uv venv, not Docker.

FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies (production only, no dev extras)
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "iterate"]
CMD ["--help"]
