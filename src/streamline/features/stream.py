"""Bytewax dataflow: clickstream topic -> online feature store (Redis).

Run:  uv run python -m bytewax.run "streamline.features.stream:get_flow()"
      (or `make stream`; add -w N for N worker threads)

The feature logic lives in the store's idempotent Lua write (features/online.py),
so the dataflow is stateless: it can be restarted or replayed from the earliest
offset without double counting, and scaled by adding workers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import bytewax.operators as op
from bytewax.connectors.kafka import operators as kop
from bytewax.dataflow import Dataflow
from bytewax.inputs import Source
from bytewax.outputs import DynamicSink, StatelessSinkPartition

from streamline.config import get_settings
from streamline.features.online import OnlineStore
from streamline.ingest.simulator import ClickEvent

log = logging.getLogger(__name__)

ClientFactory = Callable[[], Any]


def parse_event(raw: bytes | None) -> ClickEvent | None:
    """Decode one message; malformed payloads are logged and dropped."""
    if raw is None:
        return None
    try:
        return ClickEvent.from_json(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        log.warning("dropping malformed event: %r", raw[:200])
        return None


class _StorePartition(StatelessSinkPartition[ClickEvent]):
    def __init__(self, store: OnlineStore) -> None:
        self.store = store

    def write_batch(self, items: list[ClickEvent]) -> None:
        self.store.write_events(items)


class OnlineStoreSink(DynamicSink[ClickEvent]):
    def __init__(self, client_factory: ClientFactory, prefix: str = "sl") -> None:
        self.client_factory = client_factory
        self.prefix = prefix

    def build(
        self, step_id: str, worker_index: int, worker_count: int
    ) -> StatelessSinkPartition[ClickEvent]:
        return _StorePartition(OnlineStore(self.client_factory(), self.prefix))


def redis_factory(url: str) -> ClientFactory:
    def make() -> Any:
        import redis

        return redis.Redis.from_url(url)

    return make


def build_flow(
    source: Source[bytes | None] | None, sink: OnlineStoreSink, **kafka: Any
) -> Dataflow:
    """With `source=None`, read from Kafka using `kafka` kwargs (brokers, topics, tail)."""
    flow = Dataflow("streamline_features")
    if source is None:
        msgs = kop.input("kafka_in", flow, **kafka)
        op.inspect("kafka_errors", msgs.errs, lambda _step, err: log.error("kafka: %s", err))
        raw = op.map("value", msgs.oks, lambda msg: msg.value)
    else:
        raw = op.input("input", flow, source)
    events = op.filter_map("parse", raw, parse_event)
    op.output("online_store", events, sink)
    return flow


def get_flow(tail: bool = True, prefix: str = "sl", topic: str | None = None) -> Dataflow:
    settings = get_settings()
    return build_flow(
        None,
        OnlineStoreSink(redis_factory(settings.redis_url), prefix),
        brokers=[settings.kafka_bootstrap],
        topics=[topic or settings.events_topic],
        tail=tail,
    )
