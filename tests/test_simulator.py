import time
import uuid
from typing import Any

import pytest

from conftest import make_events
from streamline.ingest.simulator import ClickEvent, KafkaSink, iter_events, replay


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.now += secs


def _events(*ts: int) -> list[ClickEvent]:
    return [ClickEvent(ts_ms=t, user_id=1, item_id=2, event="view") for t in ts]


def test_json_roundtrip() -> None:
    ev = ClickEvent(ts_ms=5, user_id=7, item_id=9, event="transaction", transaction_id=3)
    assert ClickEvent.from_json(ev.to_json()) == ev
    assert ev.key() == b"7"


def test_replay_reproduces_gaps_at_speedup() -> None:
    clock, sent = FakeClock(), []
    n = replay(_events(0, 10_000, 30_000), sent.append, speedup=10, clock=clock, sleep=clock.sleep)
    assert n == 3
    # 10s and 20s event-time gaps at 10x -> 1s and 2s of wall time
    assert clock.sleeps == pytest.approx([1.0, 2.0])
    assert [e.ts_ms for e in sent] == [0, 10_000, 30_000]


def test_replay_unpaced() -> None:
    clock = FakeClock()
    replay(_events(0, 10_000_000), lambda e: None, speedup=0, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == []


def test_replay_rejects_unsorted() -> None:
    with pytest.raises(ValueError):
        replay(_events(10, 5), lambda e: None, speedup=0)


def test_iter_events_sorted() -> None:
    df = make_events([(3, 1, 10, "view"), (1, 2, 11, "addtocart")])
    assert [(e.ts_ms, e.event) for e in iter_events(df)] == [(1, "addtocart"), (3, "view")]


class FakeProducer:
    def __init__(self, fail_first: int = 0) -> None:
        self.messages: list[tuple[str, bytes, bytes]] = []
        self.fail_first = fail_first

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any) -> None:
        if self.fail_first:
            self.fail_first -= 1
            raise BufferError
        self.messages.append((topic, key, value))

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        return 0


def test_kafka_sink_keys_by_user_and_retries_full_buffer() -> None:
    producer = FakeProducer(fail_first=2)
    sink = KafkaSink("unused", "clicks", producer=producer)
    ev = ClickEvent(ts_ms=1, user_id=42, item_id=7, event="view")
    sink(ev)
    assert producer.messages == [("clicks", b"42", ev.to_json())]
    assert sink.flush() == 0


@pytest.mark.integration
def test_roundtrip_through_redpanda() -> None:
    from confluent_kafka import Consumer

    from streamline.config import get_settings
    from streamline.ingest.simulator import ensure_topic

    settings = get_settings()
    topic = f"test-{uuid.uuid4().hex[:8]}"
    ensure_topic(settings.kafka_bootstrap, topic, partitions=3)
    events = [ClickEvent(ts_ms=i, user_id=i % 5, item_id=i, event="view") for i in range(100)]
    sink = KafkaSink(settings.kafka_bootstrap, topic)
    replay(events, sink, speedup=0)
    assert sink.flush() == 0

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": topic,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    received: list[ClickEvent] = []
    deadline = time.monotonic() + 30
    while len(received) < 100 and time.monotonic() < deadline:
        msg = consumer.poll(1.0)
        if msg is not None and msg.error() is None:
            received.append(ClickEvent.from_json(msg.value()))
    consumer.close()

    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    admin.delete_topics([topic])[topic].result()
    assert sorted(received, key=lambda e: e.ts_ms) == events
    for user in range(5):  # per-user order is preserved by keying
        ts = [e.ts_ms for e in received if e.user_id == user]
        assert ts == sorted(ts)
