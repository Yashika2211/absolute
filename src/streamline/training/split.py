"""Time-based train/val/test split at global cutoffs (never random).

    train: ts <  val_start
    val:   val_start  <= ts < test_start
    test:  test_start <= ts

Models are evaluated with a "fit on the past, predict the next window" protocol:
for validation, fit on train and predict val; for the final test, refit on
train + val (everything before test_start) and predict test.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DAY_MS = 86_400_000


@dataclass(frozen=True)
class TimeSplit:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    val_start_ms: int
    test_start_ms: int

    @property
    def train_val(self) -> pl.DataFrame:
        return pl.concat([self.train, self.val])


def time_split(df: pl.DataFrame, val_days: int = 14, test_days: int = 14) -> TimeSplit:
    if val_days <= 0 or test_days <= 0:
        raise ValueError("val_days and test_days must be positive")
    end = int(df["ts_ms"].max()) + 1  # type: ignore[arg-type]
    test_start = end - test_days * DAY_MS
    val_start = test_start - val_days * DAY_MS
    if val_start <= int(df["ts_ms"].min()):  # type: ignore[arg-type]
        raise ValueError("not enough history for the requested val/test windows")

    ts = pl.col("ts_ms")
    return TimeSplit(
        train=df.filter(ts < val_start),
        val=df.filter((ts >= val_start) & (ts < test_start)),
        test=df.filter(ts >= test_start),
        val_start_ms=val_start,
        test_start_ms=test_start,
    )
