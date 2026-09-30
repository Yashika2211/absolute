"""Replay RetailRocket events into Redpanda in event-time order.

Events keep their original timestamps (event time); the simulator only controls
*when* they are published. `--speedup 1000` compresses 1000s of history into 1s of
wall time; `--speedup 0` publishes as fast as possible. Messages are keyed by
user_id so each user's events land on one partition, preserving per-user order.

Usage: uv run python -m streamline.ingest.simulator --speedup 1000 --start-date 2015-09-04
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from streamline.config import get_settings
from streamline.ingest.events import load_events


@dataclass(frozen=True)
class ClickEvent:
    ts_ms: int
    user_id: int
    item_id: int
    event: str
    transaction_id: int | None = None

    def key(self) -> bytes:
        return str(self.user_id).encode()

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, raw: bytes | str) -> ClickEvent:
        return cls(**json.loads(raw))


def iter_events(df: pl.DataFrame) -> Iterator[ClickEvent]:
    for row in df.sort("ts_ms", maintain_order=True).iter_rows(named=True):
        yield ClickEvent(
            ts_ms=row["ts_ms"],
            user_id=row["user_id"],
            item_id=row["item_id"],
            event=str(row["event"]),
            transaction_id=row["transaction_id"],
        )


def replay(
    events: Iterable[ClickEvent],
    sink: Callable[[ClickEvent], None],
    speedup: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, float], None] | None = None,
    progress_every: int = 50_000,
) -> int:
    """Publish events so that event-time gaps are reproduced at `speedup` x.

    Returns the number of events sent. speedup <= 0 disables pacing.
    """
    sent = 0
    first_ts: int | None = None
    wall_start = clock()
    for ev in events:
        if first_ts is None:
            first_ts = ev.ts_ms
        if ev.ts_ms < first_ts:
            raise ValueError("events must be sorted by timestamp")
        if speedup > 0:
            due = (ev.ts_ms - first_ts) / 1000.0 / speedup
            lag = due - (clock() - wall_start)
            if lag > 0:
                sleep(lag)
        sink(ev)
        sent += 1
        if on_progress and sent % progress_every == 0:
            on_progress(sent, clock() - wall_start)
    return sent


class KafkaSink:
    """Thin wrapper over confluent_kafka.Producer, keyed by user_id."""

    def __init__(self, bootstrap: str, topic: str, producer: Any | None = None) -> None:
        if producer is None:
            from confluent_kafka import Producer

            producer = Producer(
                {
                    "bootstrap.servers": bootstrap,
                    "linger.ms": 5,
                    "compression.type": "lz4",
                    "enable.idempotence": True,
                }
            )
        self.producer = producer
        self.topic = topic
        self.errors = 0

    def _on_delivery(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self.errors += 1

    def __call__(self, ev: ClickEvent) -> None:
        while True:
            try:
                self.producer.produce(
                    self.topic, key=ev.key(), value=ev.to_json(), on_delivery=self._on_delivery
                )
                break
            except BufferError:  # local queue full: let it drain
                self.producer.poll(0.1)
        self.producer.poll(0)

    def flush(self, timeout: float = 30.0) -> int:
        remaining: int = self.producer.flush(timeout)
        return remaining


def ensure_topic(bootstrap: str, topic: str, partitions: int = 6) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        return
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=partitions, replication_factor=1)]
    )
    futures[topic].result()


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--speedup", type=float, default=1000.0, help="0 = as fast as possible")
    parser.add_argument("--start-date", help="replay events from this UTC date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="stop after N events")
    parser.add_argument("--topic", default=settings.events_topic)
    parser.add_argument("--bootstrap", default=settings.kafka_bootstrap)
    parser.add_argument("--partitions", type=int, default=6)
    args = parser.parse_args()

    df = load_events()
    if args.start_date:
        df = df.filter(pl.col("ts_ms") >= _date_ms(args.start_date))
    if args.limit:
        df = df.sort("ts_ms").head(args.limit)

    ensure_topic(args.bootstrap, args.topic, args.partitions)
    sink = KafkaSink(args.bootstrap, args.topic)
    print(f"Replaying {df.height:,} events to {args.topic} at {args.speedup:g}x")

    def progress(n: int, secs: float) -> None:
        print(f"  {n:,} events in {secs:.1f}s ({n / max(secs, 1e-9):,.0f} ev/s)", flush=True)

    start = time.monotonic()
    try:
        sent = replay(iter_events(df), sink, args.speedup, on_progress=progress)
    except KeyboardInterrupt:
        sent = -1
        print("Interrupted, flushing...")
    unflushed = sink.flush()
    elapsed = time.monotonic() - start
    print(f"Done: sent={sent:,} unflushed={unflushed} errors={sink.errors} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
