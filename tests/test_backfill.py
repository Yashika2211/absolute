from pathlib import Path

import polars as pl

from conftest import random_events
from streamline.features.backfill import backfill, write_partitioned
from streamline.features.definitions import ITEM_FEATURES, USER_FEATURES
from streamline.features.offline import user_features
from streamline.features.offline_store import OfflineStore


def test_backfill_rows_are_point_in_time(tmp_path: Path) -> None:
    events = random_events(5)
    table = backfill(events)
    assert table.height == events.height
    assert set(USER_FEATURES) | set(ITEM_FEATURES) <= set(table.columns)

    # every row's features equal a fresh computation at that event's own timestamp
    row = table.sample(1, seed=1).row(0, named=True)
    fresh = user_features(
        events, pl.DataFrame({"user_id": [row["user_id"]], "as_of_ms": [row["ts_ms"]]})
    ).row(0, named=True)
    assert all(row[k] == fresh[k] for k in USER_FEATURES)

    # a user's first event has no history
    first = table.sort("ts_ms").group_by("user_id", maintain_order=True).first()
    assert (first["user_events_1h"] == 0).all()
    assert (first["session_events"] == 0).all()


def test_offline_store_roundtrip(tmp_path: Path) -> None:
    events = random_events(6)
    table = backfill(events)
    write_partitioned(table, tmp_path / "features")
    store = OfflineStore(tmp_path / "features")

    lo = int(events["ts_ms"].quantile(0.25))  # type: ignore[arg-type]
    hi = int(events["ts_ms"].quantile(0.75))  # type: ignore[arg-type]
    got = store.between(lo, hi)
    expected = table.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi)).drop("date")
    assert got.height == expected.height
    assert got.select(expected.columns).cast(expected.schema).equals(expected)  # type: ignore[arg-type]
    assert store.query("SELECT count(*) AS n FROM event_features")["n"][0] == table.height
