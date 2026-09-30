from __future__ import annotations

import polars as pl
import pytest

from streamline.ingest.events import clean

DAY_MS = 86_400_000


def make_events(rows: list[tuple[int, int, int, str]]) -> pl.DataFrame:
    """Build a canonical events frame from (ts_ms, user_id, item_id, event) tuples."""
    df = pl.DataFrame(
        {
            "ts_ms": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "item_id": [r[2] for r in rows],
            "event": [r[3] for r in rows],
            "transaction_id": [None] * len(rows),
        },
        schema_overrides={"transaction_id": pl.Int64},
    )
    return clean(df)


@pytest.fixture
def toy_events() -> pl.DataFrame:
    """Four weeks of events for a handful of users, one week per block."""
    rows = []
    for week in range(4):
        base = week * 7 * DAY_MS
        for user in range(1, 6):
            rows.append((base + user * 1000, user, 100 + (user + week) % 4, "view"))
            rows.append((base + user * 1000 + 10, user, 200 + week, "view"))
        rows.append((base + 50_000, 1, 100, "addtocart"))
        rows.append((base + 60_000, 1, 100, "transaction"))
    return make_events(rows)


def random_events(seed: int, n_users: int = 25, n_items: int = 15, n: int = 2000) -> pl.DataFrame:
    """Bursty synthetic clickstream that exercises feature edge cases: sessions split
    by gaps just under / exactly at / just over 30 minutes, same-millisecond events,
    and activity spread over ~2 days so 24h windows roll over."""
    import numpy as np

    from streamline.features.definitions import SESSION_GAP_MS

    rng = np.random.default_rng(seed)
    gaps = rng.choice(
        [
            0,
            1,
            1_000,
            60_000,
            299_999,
            300_000,
            SESSION_GAP_MS - 1,
            SESSION_GAP_MS,
            SESSION_GAP_MS + 1,
            3_600_000,
            6 * 3_600_000,
        ],
        size=n,
        p=[0.05, 0.05, 0.2, 0.2, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05],
    )
    users = rng.integers(0, n_users, size=n)
    clock = {int(u): int(rng.integers(0, 3_600_000)) for u in range(n_users)}
    rows = []
    for gap, user in zip(gaps, users, strict=True):
        clock[int(user)] += int(gap)
        event = rng.choice(["view", "addtocart", "transaction"], p=[0.8, 0.15, 0.05])
        rows.append((clock[int(user)], int(user), int(rng.integers(0, n_items)), str(event)))
    return make_events(rows)
