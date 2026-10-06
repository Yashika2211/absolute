import fakeredis
import pytest

from streamline.features.definitions import HOUR_MS, MINUTE_MS, SESSION_GAP_MS
from streamline.features.online import OnlineStore
from streamline.ingest.simulator import ClickEvent


@pytest.fixture
def store() -> OnlineStore:
    return OnlineStore(fakeredis.FakeRedis(), prefix="t")


def ev(ts: int, user: int = 1, item: int = 10, event: str = "view") -> ClickEvent:
    return ClickEvent(ts_ms=ts, user_id=user, item_id=item, event=event)


def test_counts_and_watermark(store: OnlineStore) -> None:
    store.write_events([ev(0), ev(MINUTE_MS, event="addtocart"), ev(2 * MINUTE_MS, item=11)])
    assert store.watermark() == 2 * MINUTE_MS
    [user] = store.user_features([1], as_of_ms=3 * MINUTE_MS)
    assert user["user_events_5m"] == 3
    assert user["user_carts_1h"] == 1
    assert user["session_distinct_items"] == 2
    [item] = store.item_features([10], as_of_ms=3 * MINUTE_MS)
    assert item["item_views_5m"] == 1
    assert item["item_carts_1h"] == 1


def test_replay_is_idempotent(store: OnlineStore) -> None:
    events = [ev(0), ev(1_000, item=11), ev(2_000, event="addtocart")]
    store.write_events(events)
    first = store.user_features([1], 5_000), store.item_features([10, 11], 5_000)
    store.write_events(events)  # at-least-once redelivery
    assert (store.user_features([1], 5_000), store.item_features([10, 11], 5_000)) == first


def test_trimming_keeps_current_session_beyond_window(store: OnlineStore) -> None:
    # one long session: an event every 20 minutes for 3 hours (never a 30m gap)
    events = [ev(i * 20 * MINUTE_MS, item=i) for i in range(10)]
    store.write_events(events)
    as_of = 9 * 20 * MINUTE_MS + 1
    [f] = store.user_features([1], as_of)
    assert f["session_events"] == 10  # older than 1h but still in the session
    assert f["user_events_1h"] == 3


def test_trimming_drops_old_sessions(store: OnlineStore) -> None:
    store.write_events([ev(0), ev(3 * HOUR_MS)])
    retained = store.r.zcard("t:u:1")
    assert retained == 1
    [f] = store.user_features([1], 3 * HOUR_MS + SESSION_GAP_MS + 1)
    assert f["session_events"] == 0


def test_clear_only_touches_prefix(store: OnlineStore) -> None:
    store.r.set("other", 1)
    store.write_events([ev(0)])
    assert store.clear() > 0
    assert store.r.get("other") == b"1"
    assert store.watermark() is None


def test_history_is_capped_ordered_and_idempotent(store: OnlineStore) -> None:
    from streamline.features.definitions import HISTORY_LEN

    events = [ev(i * 1000, item=i) for i in range(HISTORY_LEN + 10)]
    # same-millisecond tie where string order ("10" < "9") differs from numeric order
    events += [ev(10**9, item=10), ev(10**9, item=9)]
    store.write_events(events)
    store.write_events(events)  # replay
    [(ts, items, kinds)] = store.user_histories([1], as_of_ms=10**9 + 1)
    assert len(items) == HISTORY_LEN
    assert items[-2:] == [9, 10]
    assert ts == sorted(ts)
    assert kinds == ["view"] * HISTORY_LEN
    # strictly before as_of: the tie at 10**9 is excluded when as_of == 10**9
    [(_, items_before, _)] = store.user_histories([1], as_of_ms=10**9)
    assert items_before[-1] == HISTORY_LEN + 9


def test_item_features_subset(store: OnlineStore) -> None:
    store.write_events([ev(0, item=10), ev(1, item=10, event="addtocart")])
    full = store.item_features([10, 11], as_of_ms=100)
    part = store.item_features([10, 11], as_of_ms=100, features=["item_carts_1h", "item_views_24h"])
    assert part == [{k: f[k] for k in ("item_views_24h", "item_carts_1h")} for f in full]
    assert part[0] == {"item_views_24h": 1, "item_carts_1h": 1}
    assert store.item_features([10], as_of_ms=100, features=[]) == [{}]
