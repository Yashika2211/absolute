import polars as pl

from conftest import make_events, random_events
from streamline.features.definitions import MINUTE_MS
from streamline.training.requests import sample_requests

EVENTS = make_events(
    [
        (0, 1, 10, "view"),
        (100 * MINUTE_MS, 1, 11, "view"),  # request candidate for user 1
        (110 * MINUTE_MS, 1, 12, "addtocart"),  # within 30m horizon
        (200 * MINUTE_MS, 1, 13, "view"),  # outside horizon
        (100 * MINUTE_MS, 2, 20, "view"),  # user 2's first event: not eligible
        (150 * MINUTE_MS, 3, 30, "view"),
        (155 * MINUTE_MS, 3, 31, "view"),
    ]
)


def test_requests_are_point_in_time() -> None:
    reqs = sample_requests(EVENTS, start_ms=50 * MINUTE_MS, end_ms=180 * MINUTE_MS, n=10)
    rows = {r["user_id"]: (i, r) for i, r in enumerate(reqs.frame.iter_rows(named=True))}
    assert set(rows) == {1, 3}  # user 2's only event is their first: no history
    i, r = rows[1]
    # user 1 is requested at one of their eligible events; inputs are strictly before it
    expected = {
        100 * MINUTE_MS: ([10], {11, 12}, {10}),
        110 * MINUTE_MS: ([10, 11], {12}, {10, 11}),
    }
    hist, relevant, seen = expected[r["as_of_ms"]]
    assert (r["hist_items"], reqs.relevant[i], reqs.seen[i]) == (hist, relevant, seen)
    j, r3 = rows[3]
    assert r3["as_of_ms"] == 155 * MINUTE_MS
    assert (r3["hist_items"], reqs.relevant[j]) == ([30], {31})


def test_labels_clipped_to_window() -> None:
    reqs = sample_requests(EVENTS, start_ms=50 * MINUTE_MS, end_ms=105 * MINUTE_MS, n=10)
    [r] = reqs.frame.to_dicts()
    assert r["user_id"] == 1
    assert reqs.relevant[0] == {11}  # the 110m event is past end_ms


def test_requests_ignore_future_events() -> None:
    """Nothing at or after the window end may affect requests, histories or labels."""
    events = random_events(4)
    lo = int(events["ts_ms"].quantile(0.3))  # type: ignore[arg-type]
    hi = int(events["ts_ms"].quantile(0.7))  # type: ignore[arg-type]
    base = sample_requests(events, lo, hi, n=20, seed=1)
    future = events.filter(pl.col("ts_ms") >= hi)
    altered = pl.concat([events.filter(pl.col("ts_ms") < hi), future.sample(fraction=0.3, seed=0)])
    again = sample_requests(altered, lo, hi, n=20, seed=1)
    assert again.frame.equals(base.frame)
    assert again.relevant == base.relevant and again.seen == base.seen
    assert all(t < hi for t in base.frame["as_of_ms"])


def test_one_request_per_user_and_deterministic() -> None:
    events = random_events(2)
    lo, hi = int(events["ts_ms"].quantile(0.2)), int(events["ts_ms"].max()) + 1  # type: ignore[arg-type]
    a = sample_requests(events, lo, hi, n=15, seed=3)
    b = sample_requests(events, lo, hi, n=15, seed=3)
    assert a.frame.equals(b.frame)
    assert a.frame["user_id"].is_unique().all()
    assert len(a) == 15
