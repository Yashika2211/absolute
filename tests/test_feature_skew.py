"""Training/serving skew: the online store (Redis, fed by the Bytewax job) and the
offline engine must return identical feature values for the same entity + time."""

import fakeredis
import numpy as np
import polars as pl
import pytest
from bytewax.testing import TestingSource, run_main

from conftest import random_events
from streamline.features.definitions import (
    DAY_MS,
    ITEM_FEATURES,
    MINUTE_MS,
    SESSION_GAP_MS,
    USER_FEATURES,
)
from streamline.features.offline import item_features, user_features
from streamline.features.online import OnlineStore
from streamline.features.stream import OnlineStoreSink, build_flow
from streamline.ingest.simulator import iter_events


def _offline(events: pl.DataFrame, users: list[int], items: list[int], as_of: int) -> tuple:
    u = user_features(events, pl.DataFrame({"user_id": users, "as_of_ms": [as_of] * len(users)}))
    i = item_features(events, pl.DataFrame({"item_id": items, "as_of_ms": [as_of] * len(items)}))
    return (
        [{k: row[k] for k in USER_FEATURES} for row in u.iter_rows(named=True)],
        [{k: row[k] for k in ITEM_FEATURES} for row in i.iter_rows(named=True)],
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_online_matches_offline_while_streaming(seed: int) -> None:
    """Replay events one timestamp at a time; before ingesting the events at t,
    the online value at t must equal the offline value at t."""
    events = random_events(seed)
    store = OnlineStore(fakeredis.FakeRedis(), prefix=f"s{seed}")
    rng = np.random.default_rng(seed)
    all_users = events["user_id"].unique().to_list()
    all_items = events["item_id"].unique().to_list()

    checked = 0
    for (t,), batch in events.group_by("ts_ms", maintain_order=True):
        if rng.random() < 0.15:
            users = rng.choice(all_users, 5).tolist()
            items = rng.choice(all_items, 5).tolist()
            online = store.user_features(users, t), store.item_features(items, t)
            assert online == _offline(events, users, items, t), f"skew at t={t}"
            checked += 1
        store.write_events(iter_events(batch))
    assert checked > 50


def test_bytewax_dataflow_matches_offline() -> None:
    """Run the real dataflow (with a malformed message) into Redis, then compare
    every entity at several times after the stream ends."""
    events = random_events(3)
    payloads: list[bytes | None] = [e.to_json() for e in iter_events(events)]
    payloads.insert(len(payloads) // 2, b"{not json")
    payloads.insert(10, None)

    server = fakeredis.FakeServer()
    run_main(
        build_flow(
            TestingSource(payloads, batch_size=64),
            OnlineStoreSink(lambda: fakeredis.FakeRedis(server=server), prefix="flow"),
        )
    )
    store = OnlineStore(fakeredis.FakeRedis(server=server), prefix="flow")
    end = int(events["ts_ms"].max())  # type: ignore[arg-type]
    assert store.watermark() == end

    users = events["user_id"].unique().sort().to_list()
    items = events["item_id"].unique().sort().to_list()
    for offset in (
        1,
        2 * MINUTE_MS,
        10 * MINUTE_MS,
        SESSION_GAP_MS + 1,
        2 * 60 * MINUTE_MS,
        DAY_MS,
    ):
        as_of = end + offset
        online = store.user_features(users, as_of), store.item_features(items, as_of)
        assert online == _offline(events, users, items, as_of), f"skew at end+{offset}"
