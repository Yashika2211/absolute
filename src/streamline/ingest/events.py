"""Load RetailRocket events into a clean, typed Polars frame.

Canonical schema used everywhere downstream:
    ts_ms (Int64, epoch millis), user_id (Int64), item_id (Int64),
    event (Enum: view/addtocart/transaction), transaction_id (Int64, nullable)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from streamline.config import get_settings

EVENT_TYPES = ("view", "addtocart", "transaction")
EventType = pl.Enum(list(EVENT_TYPES))

SCHEMA: dict[str, pl.DataType] = {
    "ts_ms": pl.Int64(),
    "user_id": pl.Int64(),
    "item_id": pl.Int64(),
    "event": EventType,
    "transaction_id": pl.Int64(),
}


def read_raw_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        schema_overrides={
            "timestamp": pl.Int64,
            "visitorid": pl.Int64,
            "event": pl.Utf8,
            "itemid": pl.Int64,
            "transactionid": pl.Int64,
        },
    )
    return clean(
        df.rename(
            {
                "timestamp": "ts_ms",
                "visitorid": "user_id",
                "itemid": "item_id",
                "transactionid": "transaction_id",
            }
        )
    )


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Cast to the canonical schema, drop exact duplicates, sort by time (stable)."""
    return (
        df.select(list(SCHEMA))
        .cast(SCHEMA)  # type: ignore[arg-type]
        .unique(maintain_order=True)
        .sort(["ts_ms", "user_id", "item_id"], maintain_order=True)
    )


def load_events(path: Path | None = None) -> pl.DataFrame:
    """Load the cached parquet, building it from the raw CSV on first use."""
    settings = get_settings()
    path = path or settings.events_parquet
    if not path.exists():
        build_parquet(settings.raw_events_csv, path)
    return pl.read_parquet(path).cast(SCHEMA)  # type: ignore[arg-type]


def build_parquet(csv_path: Path, out_path: Path) -> pl.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `make data` first.")
    df = read_raw_csv(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw events.csv to parquet")
    parser.parse_args()
    settings = get_settings()
    df = build_parquet(settings.raw_events_csv, settings.events_parquet)
    print(f"Wrote {settings.events_parquet} ({df.height:,} rows)")


if __name__ == "__main__":
    main()
