COMPOSE := docker compose -f infra/docker-compose.yml
RUN := uv run
SPEEDUP ?= 1000
START_DATE ?= 2015-09-04
SEEDS ?= 42 43 44
WORKERS ?= 1
SKEW_HOURS ?= 168
REQUESTS ?= 8000
API_WORKERS ?= 1

.PHONY: help install up down logs data eda simulate stream backfill skew-check train rank test test-integration lint fmt typecheck check serve loadtest clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## install python deps and git hooks
	uv sync
	$(RUN) python scripts/fix_macos_openmp.py
	$(RUN) pre-commit install

up:  ## start Redpanda, Redis, Postgres, MLflow
	$(COMPOSE) up -d --build --wait
	@echo "MLflow: http://localhost:5001  Redpanda Console: http://localhost:8080"

down:  ## stop the stack (keeps volumes)
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

data:  ## download RetailRocket events.csv and build parquet
	$(RUN) python -m streamline.ingest.download
	$(RUN) python -m streamline.ingest.events

eda: data  ## dataset summary -> reports/eda.md
	$(RUN) python -m streamline.ingest.eda

simulate:  ## replay events into Redpanda (SPEEDUP=, START_DATE=)
	$(RUN) python -m streamline.ingest.simulator --speedup $(SPEEDUP) --start-date $(START_DATE)

stream:  ## Bytewax job: clickstream topic -> Redis online store (WORKERS=)
	$(RUN) python -m bytewax.run "streamline.features.stream:get_flow()" -w $(WORKERS)

backfill: data  ## point-in-time features for every event -> data/offline (Parquet)
	$(RUN) python -m streamline.features.backfill

skew-check:  ## replay real events through Redpanda/Bytewax/Redis and diff vs offline
	$(RUN) python -m streamline.features.skew_check --start-date $(START_DATE) --hours $(SKEW_HOURS)

train: data  ## retrieval (baselines + two-tower) then ranking; logged to MLflow -> reports/
	$(RUN) python -m streamline.training.train --seeds $(SEEDS)
	$(RUN) python -m streamline.training.train_ranker --requests $(REQUESTS)

rank: data  ## FAISS + LightGBM ranker only (reuses cached two-tower models) -> reports/ranking.md
	$(RUN) python -m streamline.training.train_ranker --requests $(REQUESTS)

test:  ## unit tests
	$(RUN) pytest

test-integration:  ## tests that need `make up`
	$(RUN) pytest -m integration

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

fmt:
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck:
	$(RUN) mypy

check: lint typecheck test  ## everything CI runs

serve:  ## FastAPI recommendation service on :8000 (needs `make rank` artifacts + `make stream`)
	$(RUN) uvicorn streamline.serving.app:app --host 0.0.0.0 --port 8000 --workers $(API_WORKERS) --log-level warning

loadtest:  ## Locust load test (Phase 4)
	@echo "Not built yet: load testing lands in Phase 4." && exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
