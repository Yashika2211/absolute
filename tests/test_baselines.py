from conftest import DAY_MS, make_events
from streamline.training.baselines import Popularity, RecentThenPopular, popularity_ranking

HISTORY = make_events(
    [
        (0, 1, 10, "view"),
        (0, 2, 10, "view"),
        (0, 3, 10, "view"),  # item 10: 3 old views
        (20 * DAY_MS, 1, 11, "view"),
        (20 * DAY_MS, 2, 11, "view"),  # item 11: 2 recent views
        (20 * DAY_MS + 1, 1, 12, "transaction"),  # item 12: 1 recent purchase (weight 5)
    ]
)


def test_popularity_weights_and_window() -> None:
    assert popularity_ranking(HISTORY, window_days=None) == [12, 10, 11]
    # 14-day window drops the old views of item 10 entirely
    assert popularity_ranking(HISTORY, window_days=14) == [12, 11]


def test_popularity_recommends_same_list() -> None:
    model = Popularity(window_days=None)
    model.fit(HISTORY)
    assert model.recommend([1, 99], k=2) == [[12, 10], [12, 10]]


def test_recent_then_popular() -> None:
    model = RecentThenPopular(window_days=None)
    model.fit(HISTORY)
    recs = model.recommend([1, 3, 99], k=3)
    assert recs[0] == [12, 11, 10]  # own items, most recent first
    assert recs[1] == [10, 12, 11]  # own item then popular backfill, no duplicates
    assert recs[2] == [12, 10, 11]  # unknown user -> pure popularity
