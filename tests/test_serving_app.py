import fakeredis
import pytest
from fastapi.testclient import TestClient

from streamline.features.online import OnlineStore
from streamline.ingest.simulator import ClickEvent
from streamline.serving.app import create_app
from streamline.serving.recommender import Recommendation


class FakeRecommender:
    """Stands in for ServingRecommender: same interface, no models."""

    def __init__(self) -> None:
        self.store = OnlineStore(fakeredis.FakeRedis(), prefix="app")
        self.retriever = type("R", (), {"index": [0] * 7})()
        self.ranker = type("K", (), {"features": ["a", "b"]})()
        self.calls: list[tuple[int, int, int]] = []

    def recommend(self, user_id: int, as_of_ms: int, k: int) -> Recommendation:
        self.calls.append((user_id, as_of_ms, k))
        if user_id == 666:
            raise RuntimeError("boom")
        if user_id == 0:
            return Recommendation(0, as_of_ms, [9, 8][:k], [], "popular", 0, {"total": 0.1})
        return Recommendation(
            user_id,
            as_of_ms,
            [1, 2, 3][:k],
            [0.9, 0.5, 0.1][:k],
            "ranked",
            3,
            {"retrieval": 1.0, "rank": 0.5, "total": 2.0},
        )


@pytest.fixture
def client_and_rec():  # type: ignore[no-untyped-def]
    rec = FakeRecommender()
    with TestClient(create_app(lambda: rec, clock="event")) as client:  # type: ignore[arg-type, return-value]
        yield client, rec


def test_recommend_uses_event_time_watermark(client_and_rec) -> None:  # type: ignore[no-untyped-def]
    client, rec = client_and_rec
    assert client.get("/recommend/5").status_code == 503  # nothing ingested yet
    rec.store.write_events([ClickEvent(ts_ms=1_000, user_id=5, item_id=1, event="view")])
    body = client.get("/recommend/5?k=2").json()
    assert body["as_of_ms"] == 1_001
    assert body["items"] == [{"item_id": 1, "score": 0.9}, {"item_id": 2, "score": 0.5}]
    assert body["source"] == "ranked"
    assert rec.calls[-1] == (5, 1_001, 2)


def test_explicit_as_of_and_cold_user(client_and_rec) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_rec
    body = client.get("/recommend/0?k=2&as_of_ms=42").json()
    assert body["as_of_ms"] == 42
    assert body["source"] == "popular"
    assert [i["item_id"] for i in body["items"]] == [9, 8]


def test_validation_errors_and_failures(client_and_rec) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_rec
    assert client.get("/recommend/5?k=0&as_of_ms=1").status_code == 422
    assert client.get("/recommend/abc?as_of_ms=1").status_code == 422
    assert client.get("/recommend/666?as_of_ms=1").status_code == 500


def test_health_and_metrics(client_and_rec) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_rec
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["items_indexed"] == 7
    client.get("/recommend/5?as_of_ms=10")
    client.get("/recommend/666?as_of_ms=10")
    text = client.get("/metrics").text
    assert 'streamline_requests_total{source="ranked"}' in text
    assert 'streamline_requests_total{source="error"}' in text
    assert 'streamline_stage_seconds_bucket{le="0.0025",stage="total"}' in text
