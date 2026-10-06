"""End-to-end skew check on the live stack (`make up` first).

Replays a slice of real events into a fresh Redpanda topic, runs the Bytewax job
into an isolated Redis key prefix, then compares every user and item feature
against the offline engine at several times after the stream ends.

Usage: uv run python -m streamline.features.skew_check --start-date 2015-09-04 --hours 24
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl
import redis
from bytewax.testing import run_main

from streamline.config import get_settings
from streamline.features.definitions import (
    DAY_MS,
    HOUR_MS,
    ITEM_FEATURES,
    MINUTE_MS,
    SESSION_GAP_MS,
    USER_FEATURES,
)
from streamline.features.offline import item_features, user_features, user_histories
from streamline.features.online import OnlineStore
from streamline.features.stream import get_flow
from streamline.ingest.events import load_events
from streamline.ingest.simulator import KafkaSink, _date_ms, ensure_topic, iter_events, replay

CHECK_OFFSETS_MS = (1, 2 * MINUTE_MS, 10 * MINUTE_MS, SESSION_GAP_MS + 1, 2 * HOUR_MS, DAY_MS)


@dataclass
class SkewReport:
    start_date: str
    hours: int
    events: int
    users: int
    items: int
    as_of_points: int
    values_compared: int
    mismatches: int
    stream_seconds: float
    stream_events_per_sec: float
    examples: list[dict[str, Any]]


def compare(
    store: OnlineStore, events: pl.DataFrame, as_of: int, batch: int = 2000
) -> tuple[int, int, list[dict[str, Any]]]:
    compared, mismatches = 0, 0
    examples: list[dict[str, Any]] = []

    def online_history(users: list[int], t: int) -> list[dict[str, Any]]:
        return [{"user_history": h} for h in store.user_histories(users, t)]

    def offline_history(ev: pl.DataFrame, q: pl.DataFrame) -> list[dict[str, Any]]:
        rows = user_histories(ev, q).iter_rows(named=True)
        return [{"user_history": (r["hist_ts"], r["hist_items"], r["hist_events"])} for r in rows]

    for entity, names, online_fn, offline_fn in (
        (
            "user_id",
            USER_FEATURES,
            store.user_features,
            lambda e, q: user_features(e, q).to_dicts(),
        ),
        (
            "item_id",
            ITEM_FEATURES,
            store.item_features,
            lambda e, q: item_features(e, q).to_dicts(),
        ),
        ("user_id", ("user_history",), online_history, offline_history),
    ):
        ids = events[entity].unique().sort().to_list()
        for s in range(0, len(ids), batch):
            chunk = ids[s : s + batch]
            online = online_fn(chunk, as_of)
            offline = offline_fn(
                events, pl.DataFrame({entity: chunk, "as_of_ms": [as_of] * len(chunk)})
            )
            for entity_id, on, off in zip(chunk, online, offline, strict=True):
                for name in names:
                    compared += 1
                    if on[name] != off[name]:
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                {
                                    entity: entity_id,
                                    "as_of_ms": as_of,
                                    "feature": name,
                                    "online": on[name],
                                    "offline": off[name],
                                }
                            )
    return compared, mismatches, examples


def run(start_date: str, hours: int) -> SkewReport:
    settings = get_settings()
    start = _date_ms(start_date)
    events = load_events().filter(
        (pl.col("ts_ms") >= start) & (pl.col("ts_ms") < start + hours * HOUR_MS)
    )
    run_id = uuid.uuid4().hex[:8]
    topic, prefix = f"skew-{run_id}", f"skew{run_id}"

    ensure_topic(settings.kafka_bootstrap, topic, partitions=6)
    sink = KafkaSink(settings.kafka_bootstrap, topic)
    replay(iter_events(events), sink, speedup=0)
    if sink.flush() or sink.errors:
        raise RuntimeError("failed to publish all events")

    client = redis.Redis.from_url(settings.redis_url)
    store = OnlineStore(client, prefix)
    try:
        t0 = time.perf_counter()
        run_main(
            get_flow(  # type: ignore[no-untyped-call]
                tail=False, prefix=prefix, topic=topic
            )
        )
        stream_seconds = time.perf_counter() - t0

        end = store.watermark()
        expected_end = int(events["ts_ms"].max())  # type: ignore[arg-type]
        if end != expected_end:
            raise RuntimeError(f"stream incomplete: watermark {end} != {expected_end}")

        compared = mismatches = 0
        examples: list[dict[str, Any]] = []
        for offset in CHECK_OFFSETS_MS:
            c, m, ex = compare(store, events, end + offset)
            compared, mismatches = compared + c, mismatches + m
            examples.extend(ex[: 10 - len(examples)])
    finally:
        store.clear()
        from confluent_kafka.admin import AdminClient

        admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
        admin.delete_topics([topic])[topic].result()

    return SkewReport(
        start_date=start_date,
        hours=hours,
        events=events.height,
        users=events["user_id"].n_unique(),
        items=events["item_id"].n_unique(),
        as_of_points=len(CHECK_OFFSETS_MS),
        values_compared=compared,
        mismatches=mismatches,
        stream_seconds=round(stream_seconds, 2),
        stream_events_per_sec=round(events.height / stream_seconds),
        examples=examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2015-09-04")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()
    report = run(args.start_date, args.hours)
    out = get_settings().reports_dir / "skew.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2) + "\n")
    print(json.dumps(asdict(report), indent=2))
    if report.mismatches:
        raise SystemExit(f"SKEW DETECTED: {report.mismatches} mismatching values")


if __name__ == "__main__":
    main()
