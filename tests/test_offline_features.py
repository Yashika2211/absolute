import numpy as np
import polars as pl
import pytest

from conftest import random_events
from streamline.features.definitions import ITEM_WINDOWS, USER_FEATURES, user_features_at
from streamline.features.offline import item_features, user_features


def _queries(events: pl.DataFrame, entity: str, seed: int, n: int = 400) -> pl.DataFrame:
    """Query points at event timestamps (the hardest case) plus random times."""
    rng = np.random.default_rng(seed)
    sample = events.sample(n // 2, seed=seed)
    lo, hi = int(events["ts_ms"].min()), int(events["ts_ms"].max())  # type: ignore[arg-type]
    return pl.DataFrame(
        {
            entity: np.concatenate(
                [sample[entity].to_numpy(), rng.choice(events[entity].unique().to_numpy(), n // 2)]
            ),
            "as_of_ms": np.concatenate(
                [sample["ts_ms"].to_numpy(), rng.integers(lo, hi + 3_600_000, n // 2)]
            ),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_user_features_match_reference(seed: int) -> None:
    events = random_events(seed)
    got = user_features(events, _queries(events, "user_id", seed))
    by_user = {u: g.sort("ts_ms") for (u,), g in events.group_by("user_id")}
    for row in got.iter_rows(named=True):
        g = by_user[row["user_id"]]
        expected = user_features_at(
            g["ts_ms"].to_list(),
            g["item_id"].to_list(),
            g["event"].cast(pl.Utf8).to_list(),
            row["as_of_ms"],
        )
        assert {k: row[k] for k in USER_FEATURES} == expected, row


@pytest.mark.parametrize("seed", [0, 1])
def test_item_features_match_brute_force(seed: int) -> None:
    events = random_events(seed)
    got = item_features(events, _queries(events, "item_id", seed))
    for row in got.iter_rows(named=True):
        for w in ITEM_WINDOWS:
            expected = events.filter(
                (pl.col("item_id") == row["item_id"])
                & (pl.col("event") == w.event)
                & (pl.col("ts_ms") >= row["as_of_ms"] - w.window_ms)
                & (pl.col("ts_ms") < row["as_of_ms"])
            ).height
            assert row[w.name] == expected, (w.name, row)


def test_no_leakage_from_future_events() -> None:
    """Features at t must not change when events at or after t are added or removed."""
    events = random_events(7)
    t = int(events["ts_ms"].quantile(0.5))  # type: ignore[arg-type]
    past = events.filter(pl.col("ts_ms") < t)
    rng = np.random.default_rng(0)
    future_noise = (
        pl.DataFrame(
            {
                "ts_ms": rng.integers(t, t + 3_600_000, 500),
                "user_id": rng.integers(0, 25, 500),
                "item_id": rng.integers(0, 15, 500),
            }
        )
        .with_columns(
            event=pl.lit("addtocart").cast(events.schema["event"]),
            transaction_id=pl.lit(None, pl.Int64),
        )
        .select(events.columns)
    )

    users = pl.DataFrame({"user_id": np.arange(25), "as_of_ms": np.full(25, t)})
    items = pl.DataFrame({"item_id": np.arange(15), "as_of_ms": np.full(15, t)})
    for frame in (events, past, pl.concat([events, future_noise])):
        assert user_features(frame, users).equals(user_features(events, users))
        assert item_features(frame, items).equals(item_features(events, items))


def test_empty_history() -> None:
    events = random_events(0).head(0)
    got = user_features(events, pl.DataFrame({"user_id": [1], "as_of_ms": [10]}))
    assert got["session_events"].to_list() == [0]
    assert got["user_events_1h"].to_list() == [0]


@pytest.mark.parametrize("seed", [0, 1])
def test_user_histories_match_reference(seed: int) -> None:
    from streamline.features.definitions import user_history_at
    from streamline.features.offline import user_histories

    events = random_events(seed, n=3000)
    got = user_histories(events, _queries(events, "user_id", seed))
    ordered = events.with_columns(pl.col("event").cast(pl.Utf8)).sort(
        ["user_id", "ts_ms", "item_id", "event"]
    )
    by_user = {u: g for (u,), g in ordered.group_by("user_id", maintain_order=True)}
    for row in got.iter_rows(named=True):
        g = by_user[row["user_id"]]
        expected = user_history_at(
            g["ts_ms"].to_list(), g["item_id"].to_list(), g["event"].to_list(), row["as_of_ms"]
        )
        assert (row["hist_ts"], row["hist_items"], row["hist_events"]) == expected
