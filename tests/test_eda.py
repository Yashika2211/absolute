import pytest

from conftest import make_events
from streamline.ingest.eda import summarize, to_markdown


def test_summarize_counts() -> None:
    df = make_events(
        [
            (0, 1, 10, "view"),
            (1, 1, 10, "addtocart"),
            (2, 1, 11, "view"),
            (86_400_000, 2, 10, "view"),
        ]
    )
    s = summarize(df)
    assert s["events"] == 4
    assert s["events_by_type"] == {"view": 3, "addtocart": 1, "transaction": 0}
    assert s["users"] == 2
    assert s["items"] == 2
    assert s["user_item_pairs"] == 3
    assert s["sparsity"] == pytest.approx(1 - 3 / 4)
    assert s["days"] == 1.0
    assert s["users_with_one_event_pct"] == 50.0
    assert s["view_to_cart_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert "| Users (visitors) | 2 |" in to_markdown(s)
