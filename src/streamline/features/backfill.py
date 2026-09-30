"""Offline backfill: point-in-time features for every event in the log.

Each output row is one event plus the user and item features as of that event's
timestamp, computed from events strictly before it. This is the training log
for the ranker (Phase 3); it is written as date-partitioned Parquet and queried
with DuckDB (features/offline_store.py).

Usage: uv run python -m streamline.features.backfill
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import polars as pl

from streamline.config import get_settings
from streamline.features.offline import item_features, user_features
from streamline.ingest.events import load_events


def backfill(events: pl.DataFrame) -> pl.DataFrame:
    base = events.with_row_index("event_idx").with_columns(as_of_ms=pl.col("ts_ms"))
    users = user_features(events, base.select("user_id", "as_of_ms"))
    items = item_features(events, base.select("item_id", "as_of_ms"))
    return (
        base.drop("as_of_ms")
        .hstack(users.drop("user_id", "as_of_ms"))
        .hstack(items.drop("item_id", "as_of_ms"))
    ).with_columns(date=pl.from_epoch("ts_ms", time_unit="ms").dt.date())


def write_partitioned(df: pl.DataFrame, root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    df.write_parquet(root, partition_by="date")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=settings.offline_features_dir)
    args = parser.parse_args()

    events = load_events()
    start = time.perf_counter()
    table = backfill(events)
    computed = time.perf_counter() - start
    write_partitioned(table, args.out)
    print(
        f"Backfilled {table.height:,} events x {table.width - events.width - 2} features "
        f"in {computed:.1f}s ({table.height / computed:,.0f} rows/s) -> {args.out}"
    )


if __name__ == "__main__":
    main()
