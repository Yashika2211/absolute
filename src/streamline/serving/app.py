"""FastAPI recommendation service.

Run: uv run uvicorn streamline.serving.app:app --port 8000   (or `make serve`)

  GET /recommend/{user_id}?k=10[&as_of_ms=...]
  GET /health
  GET /metrics   (Prometheus)

Time: the data is a 2015 replay, so by default "now" is the feature stream's
watermark (latest event-time ingested) + 1 ms. STREAMLINE_CLOCK=wall uses the
wall clock instead, as a live deployment would.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from streamline.config import REPO_ROOT, get_settings
from streamline.features.online import OnlineStore
from streamline.serving.recommender import ServingRecommender

ARTIFACTS = Path(os.environ.get("STREAMLINE_ARTIFACTS", str(REPO_ROOT / "artifacts" / "serving")))

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25)
REQUESTS = Counter("streamline_requests_total", "Recommendation requests", ["source"])
LATENCY = Histogram(
    "streamline_stage_seconds", "Latency per serving stage", ["stage"], buckets=LATENCY_BUCKETS
)
CANDIDATES = Histogram(
    "streamline_candidates", "Candidates scored per request", buckets=(0, 100, 250, 400, 500, 600)
)
WATERMARK = Gauge("streamline_feature_watermark_ms", "Latest event time in the online store")

RecommenderFactory = Callable[[], ServingRecommender]


def _default_factory() -> ServingRecommender:
    import redis

    store = OnlineStore(redis.Redis.from_url(get_settings().redis_url))
    return ServingRecommender.from_artifacts(ARTIFACTS, store)


def create_app(factory: RecommenderFactory = _default_factory, clock: str | None = None) -> FastAPI:
    clock = clock or os.environ.get("STREAMLINE_CLOCK", "event")
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["rec"] = factory()
        yield

    app = FastAPI(title="Streamline", lifespan=lifespan)

    def now_ms(rec: ServingRecommender) -> int:
        if clock == "wall":
            return int(time.time() * 1000)
        watermark = rec.store.watermark()
        if watermark is None:
            raise HTTPException(503, "online store is empty: run the stream job first")
        WATERMARK.set(watermark)
        return watermark + 1

    @app.get("/recommend/{user_id}")
    def recommend(
        user_id: int,
        k: int = Query(10, ge=1, le=100),
        as_of_ms: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        rec_engine: ServingRecommender = state["rec"]
        try:
            as_of = as_of_ms if as_of_ms is not None else now_ms(rec_engine)
            rec = rec_engine.recommend(user_id, as_of, k)
        except HTTPException:
            REQUESTS.labels("error").inc()
            raise
        except Exception as exc:  # keep serving; surface as 500 and count it
            REQUESTS.labels("error").inc()
            raise HTTPException(500, f"recommendation failed: {type(exc).__name__}") from exc
        REQUESTS.labels(rec.source).inc()
        CANDIDATES.observe(rec.n_candidates)
        for stage, ms in rec.timings_ms.items():
            LATENCY.labels(stage).observe(ms / 1000)
        return {
            "user_id": rec.user_id,
            "as_of_ms": rec.as_of_ms,
            "source": rec.source,
            "items": [
                {"item_id": item, "score": rec.scores[i] if rec.scores else None}
                for i, item in enumerate(rec.items)
            ],
            "n_candidates": rec.n_candidates,
            "timings_ms": {k: round(v, 3) for k, v in rec.timings_ms.items()},
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        rec_engine: ServingRecommender | None = state.get("rec")
        if rec_engine is None:
            raise HTTPException(503, "models not loaded")
        try:
            rec_engine.store.r.ping()
        except Exception as exc:
            raise HTTPException(503, f"redis unavailable: {type(exc).__name__}") from exc
        return {
            "status": "ok",
            "clock": clock,
            "watermark_ms": rec_engine.store.watermark(),
            "items_indexed": len(rec_engine.retriever.index),
            "ranker_features": len(rec_engine.ranker.features),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
