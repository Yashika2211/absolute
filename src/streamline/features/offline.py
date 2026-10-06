"""Offline feature engine: features for any (entity, as_of) points, vectorised.

Events are sorted by a combined (entity << 42 | ts) key, so "events of entity e
with ts in [a, b)" is two binary searches. Session features are prefix sums over
each user's sessions, looked up at the user's last event before as_of.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from streamline.features.definitions import (
    HISTORY_LEN,
    ITEM_WINDOWS,
    LAST_EVENT_CAP_MS,
    SESSION_GAP_MS,
    USER_WINDOWS,
    WindowCount,
)

TS_BITS = 42  # epoch millis fit in 42 bits until the year 2109
MAX_ENTITY = 1 << (63 - TS_BITS)


def _keys(entity: np.ndarray, ts: np.ndarray) -> np.ndarray:
    if len(entity) and (entity.max() >= MAX_ENTITY or entity.min() < 0):
        raise ValueError(f"entity ids must be in [0, {MAX_ENTITY})")
    if len(ts) and ts.min() < 0:
        raise ValueError("timestamps must be non-negative")
    keys: np.ndarray = (entity.astype(np.int64) << TS_BITS) | ts.astype(np.int64)
    return keys


def _sorted_keys(events: pl.DataFrame, entity_col: str, event: str | None) -> np.ndarray:
    if event is not None:
        events = events.filter(pl.col("event") == event)
    return np.sort(_keys(events[entity_col].to_numpy(), events["ts_ms"].to_numpy()))


def _window_counts(
    events: pl.DataFrame,
    entity_col: str,
    windows: tuple[WindowCount, ...],
    entity: np.ndarray,
    as_of: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    cache: dict[str | None, np.ndarray] = {}
    hi_q = _keys(entity, as_of)
    for w in windows:
        if w.event not in cache:
            cache[w.event] = _sorted_keys(events, entity_col, w.event)
        keys = cache[w.event]
        lo_q = _keys(entity, np.maximum(as_of - w.window_ms, 0))
        out[w.name] = np.searchsorted(keys, hi_q, "left") - np.searchsorted(keys, lo_q, "left")
    return out


def _session_features(
    events: pl.DataFrame, user: np.ndarray, as_of: np.ndarray
) -> dict[str, np.ndarray]:
    ev = events.select("user_id", "ts_ms", "item_id", "event").sort(["user_id", "ts_ms"])
    n = ev.height
    if n == 0:
        zeros = np.zeros(len(user), dtype=np.int64)
        return {
            "session_events": zeros,
            "session_distinct_items": zeros,
            "session_carts": zeros,
            "session_duration_ms": zeros,
            "last_event_age_ms": np.full(len(user), LAST_EVENT_CAP_MS, dtype=np.int64),
        }
    ev_user = ev["user_id"].to_numpy()
    ev_ts = ev["ts_ms"].to_numpy()
    idx = np.arange(n)

    new_session = np.ones(n, dtype=bool)
    new_session[1:] = (ev_user[1:] != ev_user[:-1]) | (np.diff(ev_ts) > SESSION_GAP_MS)
    session_id = np.cumsum(new_session)
    start = np.maximum.accumulate(np.where(new_session, idx, 0))

    carts = np.cumsum((ev["event"] == "addtocart").to_numpy())
    first_seen = (
        ev.with_columns(session=pl.Series(session_id))
        .select((pl.int_range(pl.len()).over(["session", "item_id"]) == 0).alias("first"))["first"]
        .to_numpy()
    )
    distinct = np.cumsum(first_seen)

    def upto(prefix: np.ndarray, last: np.ndarray, first: np.ndarray) -> np.ndarray:
        before = np.where(first > 0, prefix[np.maximum(first - 1, 0)], 0)
        result: np.ndarray = prefix[last] - before
        return result

    keys = _keys(ev_user, ev_ts)
    last = np.searchsorted(keys, _keys(user, as_of), "left") - 1  # last event < as_of
    safe = np.maximum(last, 0)
    has_event = (last >= 0) & (ev_user[safe] == user)
    age = np.where(has_event, as_of - ev_ts[safe], LAST_EVENT_CAP_MS)
    active = has_event & (age <= SESSION_GAP_MS)

    first = start[safe]
    return {
        "session_events": np.where(active, safe - first + 1, 0),
        "session_distinct_items": np.where(active, upto(distinct, safe, first), 0),
        "session_carts": np.where(active, upto(carts, safe, first), 0),
        "session_duration_ms": np.where(active, ev_ts[safe] - ev_ts[first], 0),
        "last_event_age_ms": np.minimum(age, LAST_EVENT_CAP_MS),
    }


def user_features(events: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """`queries` has user_id and as_of_ms; returns it with all user features appended."""
    user = queries["user_id"].to_numpy()
    as_of = queries["as_of_ms"].to_numpy()
    cols = _window_counts(events, "user_id", USER_WINDOWS, user, as_of)
    cols |= _session_features(events, user, as_of)
    return queries.with_columns(
        [pl.Series(name, values, dtype=pl.Int64) for name, values in cols.items()]
    )


def item_features(events: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """`queries` has item_id and as_of_ms; returns it with all item features appended."""
    cols = _window_counts(
        events,
        "item_id",
        ITEM_WINDOWS,
        queries["item_id"].to_numpy(),
        queries["as_of_ms"].to_numpy(),
    )
    return queries.with_columns(
        [pl.Series(name, values, dtype=pl.Int64) for name, values in cols.items()]
    )


def user_histories(events: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """`queries` has user_id and as_of_ms; adds hist_ts / hist_items / hist_events:
    the user's last HISTORY_LEN events strictly before as_of, oldest first.

    Ties at the same millisecond are ordered by (item_id, event) so the online
    store (Redis orders equal scores by member) returns the identical sequence.
    """
    ev = events.select(
        "user_id", "ts_ms", "item_id", pl.col("event").cast(pl.Utf8).alias("event")
    ).sort(["user_id", "ts_ms", "item_id", "event"])
    user = queries["user_id"].to_numpy()
    as_of = queries["as_of_ms"].to_numpy()
    keys = _keys(ev["user_id"].to_numpy(), ev["ts_ms"].to_numpy())
    end = np.searchsorted(keys, _keys(user, as_of), "left")
    first = np.searchsorted(keys, _keys(user, np.zeros_like(as_of)), "left")
    start = np.maximum(first, end - HISTORY_LEN)
    ts, items, kinds = ev["ts_ms"].to_numpy(), ev["item_id"].to_numpy(), ev["event"].to_list()
    return queries.with_columns(
        hist_ts=pl.Series(
            [ts[a:b].tolist() for a, b in zip(start, end, strict=True)], dtype=pl.List(pl.Int64)
        ),
        hist_items=pl.Series(
            [items[a:b].tolist() for a, b in zip(start, end, strict=True)], dtype=pl.List(pl.Int64)
        ),
        hist_events=pl.Series(
            [kinds[a:b] for a, b in zip(start, end, strict=True)], dtype=pl.List(pl.Utf8)
        ),
    )
