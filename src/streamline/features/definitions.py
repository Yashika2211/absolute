"""Feature definitions: the single source of truth for online and offline features.

Every feature is a function of (entity, as_of_ms) over the events with
ts_ms < as_of_ms (strictly before), so a value never includes the event it is
used to predict. Two engines implement these definitions:

  * online  (features/online.py):  Redis, updated by the Bytewax stream job
  * offline (features/offline.py): vectorised Polars/NumPy over the event log

tests/test_feature_skew.py proves both engines return identical values.

Window counts cover [as_of - window, as_of). A session is a run of a user's
events with gaps <= SESSION_GAP_MS; the session is "current" at as_of if the
user's last event is within SESSION_GAP_MS of as_of, otherwise session
features are 0.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

SESSION_GAP_MS = 30 * MINUTE_MS
LAST_EVENT_CAP_MS = DAY_MS  # "no event in the last day" is reported as the cap

Entity = Literal["user", "item"]


@dataclass(frozen=True)
class WindowCount:
    name: str
    entity: Entity
    window_ms: int
    event: str | None = None  # None = all event types


USER_WINDOWS: tuple[WindowCount, ...] = (
    WindowCount("user_events_5m", "user", 5 * MINUTE_MS),
    WindowCount("user_events_1h", "user", HOUR_MS),
    WindowCount("user_carts_1h", "user", HOUR_MS, "addtocart"),
)

ITEM_WINDOWS: tuple[WindowCount, ...] = (
    WindowCount("item_views_5m", "item", 5 * MINUTE_MS, "view"),
    WindowCount("item_views_1h", "item", HOUR_MS, "view"),
    WindowCount("item_views_24h", "item", DAY_MS, "view"),
    WindowCount("item_carts_1h", "item", HOUR_MS, "addtocart"),
    WindowCount("item_carts_24h", "item", DAY_MS, "addtocart"),
    WindowCount("item_purchases_24h", "item", DAY_MS, "transaction"),
)

SESSION_FEATURES: tuple[str, ...] = (
    "session_events",
    "session_distinct_items",
    "session_carts",
    "session_duration_ms",
    "last_event_age_ms",
)

USER_FEATURES: tuple[str, ...] = tuple(w.name for w in USER_WINDOWS) + SESSION_FEATURES
ITEM_FEATURES: tuple[str, ...] = tuple(w.name for w in ITEM_WINDOWS)

# How long the online store must retain events for an entity (event time).
USER_RETENTION_MS = max(w.window_ms for w in USER_WINDOWS)
ITEM_RETENTION_MS = max(w.window_ms for w in ITEM_WINDOWS)


def user_features_at(
    ts: Sequence[int], items: Sequence[int], events: Sequence[str], as_of_ms: int
) -> dict[str, int]:
    """Reference implementation of all user features.

    Inputs are one user's events sorted by ts (any events at or after as_of_ms
    are ignored). Used by the online read path and as the test oracle.
    """
    end = bisect.bisect_left(ts, as_of_ms)  # events strictly before as_of
    out: dict[str, int] = {}
    for w in USER_WINDOWS:
        start = bisect.bisect_left(ts, as_of_ms - w.window_ms, 0, end)
        if w.event is None:
            out[w.name] = end - start
        else:
            out[w.name] = sum(1 for e in events[start:end] if e == w.event)

    zeros = dict.fromkeys(SESSION_FEATURES, 0)
    if end == 0:
        return out | zeros | {"last_event_age_ms": LAST_EVENT_CAP_MS}
    last = ts[end - 1]
    age = min(as_of_ms - last, LAST_EVENT_CAP_MS)
    if as_of_ms - last > SESSION_GAP_MS:
        return out | zeros | {"last_event_age_ms": age}

    first = end - 1
    while first > 0 and ts[first] - ts[first - 1] <= SESSION_GAP_MS:
        first -= 1
    return out | {
        "session_events": end - first,
        "session_distinct_items": len(set(items[first:end])),
        "session_carts": sum(1 for e in events[first:end] if e == "addtocart"),
        "session_duration_ms": last - ts[first],
        "last_event_age_ms": age,
    }
