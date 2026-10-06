"""Locust load test for GET /recommend/{user_id}.

Traffic mix: ~90% real users active in the last 7 days of the online store
(warm path: Redis -> two-tower -> FAISS -> LightGBM) and ~10% unknown users
(cold path: popularity). Server-side stage timings from each response are
collected and written to reports/loadtest_stages.json when the test stops.

Run via `make loadtest` (headless) or `uv run locust -f loadtest/locustfile.py`.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import redis
from locust import FastHttpUser, events, task

from streamline.config import REPO_ROOT, get_settings
from streamline.ingest.events import load_events

COLD_SHARE = float(os.environ.get("COLD_SHARE", "0.1"))
# TARGET_RPS > 0: Poisson arrivals at that total rate (exponential gaps per user, so
#   requests don't arrive in synchronized bursts) -- the SLO test.
# TARGET_RPS = 0: every user fires back-to-back (saturation / max throughput test).
TARGET_RPS = float(os.environ.get("TARGET_RPS", "0"))
N_USERS = int(os.environ.get("LOCUST_USERS", "1"))
STAGES: dict[str, list[float]] = defaultdict(list)
SOURCES: dict[str, int] = defaultdict(int)


def _warm_users() -> list[int]:
    watermark = int(redis.Redis.from_url(get_settings().redis_url).get("sl:watermark") or 0)
    recent = load_events().filter(pl.col("ts_ms") > watermark - 7 * 86_400_000)
    return recent["user_id"].unique().sort().to_list()


USERS = _warm_users()
RNG = random.Random(7)


class Shopper(FastHttpUser):
    def wait_time(self) -> float:
        return RNG.expovariate(TARGET_RPS / N_USERS) if TARGET_RPS > 0 else 0.0

    @task
    def recommend(self) -> None:
        user = 10**9 + RNG.randrange(10**6) if RNG.random() < COLD_SHARE else RNG.choice(USERS)
        with self.client.get(
            f"/recommend/{user}?k=10", name="/recommend", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
                return
            body: dict[str, Any] = resp.json()
            SOURCES[body["source"]] += 1
            for stage, ms in body["timings_ms"].items():
                STAGES[f"{body['source']}:{stage}"].append(ms)


@events.test_stop.add_listener
def _dump(**_: Any) -> None:
    out = Path(os.environ.get("STAGES_OUT", str(REPO_ROOT / "reports" / "loadtest_stages.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sources": SOURCES, "stages": STAGES}))
