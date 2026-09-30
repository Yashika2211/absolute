"""Online feature store on Redis.

Layout (all scores are event-time millis):
  {p}:u:{user}          ZSET  member "ts:item:event"   the user's recent events
  {p}:us:{user}         HASH  last_ts, session_start   bookkeeping for trimming
  {p}:i:{item}:{event}  ZSET  member "ts:user"         the item's recent events by type
  {p}:watermark         STR   max event ts ingested    the stream's "now"

Writes go through one Lua script per event, so they are atomic and idempotent:
ZADD of an existing member is a no-op, so replaying a topic (at-least-once
delivery) cannot double count. Events are trimmed by event time, keeping each
user's current session plus the longest user window, and the longest item
window. Reads evaluate the shared definitions over the retained events, which
is exact for any as_of at or after the entity's latest ingested event (i.e.
"now" in serving).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from streamline.features.definitions import (
    ITEM_FEATURES,
    ITEM_RETENTION_MS,
    ITEM_WINDOWS,
    LAST_EVENT_CAP_MS,
    SESSION_GAP_MS,
    USER_FEATURES,
    USER_RETENTION_MS,
    user_features_at,
)
from streamline.ingest.simulator import ClickEvent

# wall-clock TTL for garbage collection only; correctness never depends on it
TTL_MS = LAST_EVENT_CAP_MS + 3_600_000

WRITE_EVENT_LUA = """
local ts = tonumber(ARGV[1])
local gap, user_ret = tonumber(ARGV[4]), tonumber(ARGV[5])
local item_ret, ttl = tonumber(ARGV[6]), tonumber(ARGV[7])

redis.call('ZADD', KEYS[1], ts, ARGV[2])
local last = tonumber(redis.call('HGET', KEYS[2], 'last_ts'))
local start = tonumber(redis.call('HGET', KEYS[2], 'session_start'))
if last == nil or start == nil then
  last, start = ts, ts
elseif ts > last then
  if ts - last > gap then start = ts end
  last = ts
end
redis.call('HSET', KEYS[2], 'last_ts', last, 'session_start', start)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', '(' .. math.min(last - user_ret, start))
redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)

redis.call('ZADD', KEYS[3], ts, ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', '(' .. (ts - item_ret))
redis.call('PEXPIRE', KEYS[3], ttl)

local wm = tonumber(redis.call('GET', KEYS[4]))
if wm == nil or ts > wm then redis.call('SET', KEYS[4], ts) end
return 1
"""


class OnlineStore:
    def __init__(self, client: Any, prefix: str = "sl") -> None:
        self.r = client
        self.prefix = prefix
        self._write = client.register_script(WRITE_EVENT_LUA)

    # keys -----------------------------------------------------------------
    def _user_key(self, user: int) -> str:
        return f"{self.prefix}:u:{user}"

    def _session_key(self, user: int) -> str:
        return f"{self.prefix}:us:{user}"

    def _item_key(self, item: int, event: str) -> str:
        return f"{self.prefix}:i:{item}:{event}"

    @property
    def watermark_key(self) -> str:
        return f"{self.prefix}:watermark"

    # writes ---------------------------------------------------------------
    def write_events(self, events: Iterable[ClickEvent]) -> int:
        pipe = self.r.pipeline(transaction=False)
        n = 0
        for ev in events:
            self._write(
                keys=[
                    self._user_key(ev.user_id),
                    self._session_key(ev.user_id),
                    self._item_key(ev.item_id, ev.event),
                    self.watermark_key,
                ],
                args=[
                    ev.ts_ms,
                    f"{ev.ts_ms}:{ev.item_id}:{ev.event}",
                    f"{ev.ts_ms}:{ev.user_id}",
                    SESSION_GAP_MS,
                    USER_RETENTION_MS,
                    ITEM_RETENTION_MS,
                    TTL_MS,
                ],
                client=pipe,
            )
            n += 1
        if n:
            pipe.execute()
        return n

    def watermark(self) -> int | None:
        value = self.r.get(self.watermark_key)
        return int(value) if value is not None else None

    # reads ----------------------------------------------------------------
    def user_features(self, users: Sequence[int], as_of_ms: int) -> list[dict[str, int]]:
        pipe = self.r.pipeline(transaction=False)
        for user in users:
            pipe.zrangebyscore(self._user_key(user), "-inf", f"({as_of_ms}", withscores=True)
        out = []
        for rows in pipe.execute():
            ts, items, events = [], [], []
            for member, score in rows:
                _, item, event = (member.decode() if isinstance(member, bytes) else member).split(
                    ":"
                )
                ts.append(int(score))
                items.append(int(item))
                events.append(event)
            features = user_features_at(ts, items, events, as_of_ms)
            out.append({name: features[name] for name in USER_FEATURES})
        return out

    def item_features(self, items: Sequence[int], as_of_ms: int) -> list[dict[str, int]]:
        pipe = self.r.pipeline(transaction=False)
        for item in items:
            for w in ITEM_WINDOWS:
                assert w.event is not None, "item windows are per event type"
                pipe.zcount(self._item_key(item, w.event), as_of_ms - w.window_ms, f"({as_of_ms}")
        counts = pipe.execute()
        n = len(ITEM_WINDOWS)
        return [
            dict(zip(ITEM_FEATURES, (int(c) for c in counts[i * n : (i + 1) * n]), strict=True))
            for i in range(len(items))
        ]

    def clear(self) -> int:
        """Delete every key under this store's prefix."""
        deleted = 0
        for key in self.r.scan_iter(match=f"{self.prefix}:*", count=1000):
            deleted += self.r.delete(key)
        return int(deleted)
