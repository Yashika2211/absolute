from streamline.features.definitions import (
    LAST_EVENT_CAP_MS,
    MINUTE_MS,
    SESSION_GAP_MS,
    user_features_at,
)

TS = [0, 1 * MINUTE_MS, 2 * MINUTE_MS, 40 * MINUTE_MS, 41 * MINUTE_MS]
ITEMS = [1, 2, 1, 3, 3]
EVENTS = ["view", "addtocart", "view", "view", "addtocart"]


def test_window_counts_exclude_as_of_instant() -> None:
    # the event at exactly 41m is excluded; [36m, 41m) holds only the 40m event
    f = user_features_at(TS, ITEMS, EVENTS, as_of_ms=41 * MINUTE_MS)
    assert f["user_events_5m"] == 1
    assert f["user_events_1h"] == 4
    assert f["user_carts_1h"] == 1


def test_session_splits_on_gap() -> None:
    f = user_features_at(TS, ITEMS, EVENTS, as_of_ms=42 * MINUTE_MS)
    # 2m -> 40m is a 38 minute gap, so the current session is {40m, 41m}
    assert f["session_events"] == 2
    assert f["session_distinct_items"] == 1
    assert f["session_carts"] == 1
    assert f["session_duration_ms"] == MINUTE_MS
    assert f["last_event_age_ms"] == MINUTE_MS


def test_gap_exactly_at_threshold_continues_session() -> None:
    ts = [0, SESSION_GAP_MS]
    f = user_features_at(ts, [1, 2], ["view", "view"], as_of_ms=SESSION_GAP_MS + 1)
    assert f["session_events"] == 2
    # and the session is still current exactly SESSION_GAP_MS after the last event
    f = user_features_at(ts, [1, 2], ["view", "view"], as_of_ms=2 * SESSION_GAP_MS)
    assert f["session_events"] == 2
    f = user_features_at(ts, [1, 2], ["view", "view"], as_of_ms=2 * SESSION_GAP_MS + 1)
    assert f["session_events"] == 0


def test_expired_session_and_no_history() -> None:
    f = user_features_at(TS, ITEMS, EVENTS, as_of_ms=41 * MINUTE_MS + SESSION_GAP_MS + 1)
    assert f["session_events"] == 0
    assert f["last_event_age_ms"] == SESSION_GAP_MS + 1
    f = user_features_at([], [], [], as_of_ms=10)
    assert f["user_events_1h"] == 0
    assert f["last_event_age_ms"] == LAST_EVENT_CAP_MS
