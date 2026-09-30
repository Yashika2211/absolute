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
