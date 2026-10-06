# Image for the recommendation API and the Locust load generator.
# Data (data/) and model artifacts (artifacts/) are mounted at runtime.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-default-groups --group loadtest
COPY src ./src
COPY loadtest ./loadtest
RUN uv sync --frozen --no-default-groups --group loadtest

EXPOSE 8000
CMD ["uvicorn", "streamline.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
