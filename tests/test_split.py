import polars as pl
import pytest

from streamline.training.split import DAY_MS, time_split


def test_split_is_disjoint_and_ordered(toy_events: pl.DataFrame) -> None:
    s = time_split(toy_events, val_days=7, test_days=7)

    assert s.train.height + s.val.height + s.test.height == toy_events.height
    assert s.train["ts_ms"].max() < s.val_start_ms <= s.val["ts_ms"].min()  # type: ignore[operator]
    assert s.val["ts_ms"].max() < s.test_start_ms <= s.test["ts_ms"].min()  # type: ignore[operator]
    assert s.test_start_ms - s.val_start_ms == 7 * DAY_MS
    assert s.train_val.height == s.train.height + s.val.height


def test_split_rejects_short_history(toy_events: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        time_split(toy_events, val_days=30, test_days=30)
