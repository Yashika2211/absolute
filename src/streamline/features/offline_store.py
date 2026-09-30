"""Offline store: DuckDB over the date-partitioned Parquet feature log."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from streamline.config import get_settings


class OfflineStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().offline_features_dir
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} not found. Run `make backfill` first.")
        self.con = duckdb.connect()
        glob = str(self.root / "**" / "*.parquet")
        self.con.execute(
            "CREATE VIEW event_features AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true)"
        )

    def query(self, sql: str, params: list[object] | None = None) -> pl.DataFrame:
        return self.con.execute(sql, params or []).pl()

    def between(self, start_ms: int, end_ms: int) -> pl.DataFrame:
        """Feature rows for events with start_ms <= ts < end_ms, in event order.

        Filters on the `date` partition first so DuckDB skips unrelated files.
        """
        return self.query(
            """
            SELECT * EXCLUDE (date) FROM event_features
            WHERE date BETWEEN CAST(epoch_ms(?::BIGINT) AS DATE)
                           AND CAST(epoch_ms(?::BIGINT) AS DATE)
              AND ts_ms >= ? AND ts_ms < ?
            ORDER BY event_idx
            """,
            [start_ms, end_ms, start_ms, end_ms],
        )
