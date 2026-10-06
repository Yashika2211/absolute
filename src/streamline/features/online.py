"""Online feature store on Redis.

Layout (all scores are event-time millis):
  {p}:u:{user}          ZSET  member "ts:item:event"   the user's recent events
  {p}:us:{user}         HASH  last_ts, session_start   bookkeeping for trimming
  {p}:uh:{user}         ZSET  member "ts:item:event"   the user's last HISTORY_LEN events
                                (zero-padded so equal scores sort like the offline engine)
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
    HISTORY_LEN,
    ITEM_RETENTION_MS,
    ITEM_WINDOWS,
    LAST_EVENT_CAP_MS,
    SESSION_GAP_MS,
    USER_FEATURES,
    USER_RETENTION_MS,
    user_features_at,
    user_history_at,
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

redis.call('ZADD', KEYS[5], ts, ARGV[8])
redis.call('ZREMRANGEBYRANK', KEYS[5], 0, -(tonumber(ARGV[9]) + 1))

local wm = tonumber(redis.call('GET', KEYS[4]))
if wm == nil or ts > wm then redis.call('SET', KEYS[4], ts) end
return 1
"""


ITEM_COUNTS_LUA = """
-- ARGV: prefix, as_of, n_types, type names..., n_windows, (type_position, window_ms)...,
--       then item ids. Keys are built here so the client sends ids, not 3 key strings
--       per item. (Not Redis Cluster safe: keys are undeclared. Single node by design.)
local prefix, as_of, n_types = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3])
local types = {}
for t = 1, n_types do types[t] = ARGV[3 + t] end
local p = 4 + n_types
local n_windows = tonumber(ARGV[p])
local win_type, win_ms = {}, {}
for w = 1, n_windows do
  win_type[w] = types[tonumber(ARGV[p + 2 * w - 1])]
  win_ms[w] = tonumber(ARGV[p + 2 * w])
end
local upper = '(' .. as_of
local out = {}
for i = p + 2 * n_windows + 1, #ARGV do
  local base = prefix .. ':i:' .. ARGV[i] .. ':'
  for w = 1, n_windows do
    out[#out + 1] = redis.call('ZCOUNT', base .. win_type[w], as_of - win_ms[w], upper)
  end
end
return out
"""

# event types that have per-item sets (window args refer to their 1-based position)
_ITEM_TYPES: tuple[str, ...] = tuple(dict.fromkeys(w.event for w in ITEM_WINDOWS if w.event))


class OnlineStore:
    def __init__(self, client: Any, prefix: str = "sl") -> None:
        self.r = client
        self.prefix = prefix
        self._write = client.register_script(WRITE_EVENT_LUA)
        self._item_counts = client.register_script(ITEM_COUNTS_LUA)

    # keys -----------------------------------------------------------------
    def _user_key(self, user: int) -> str:
        return f"{self.prefix}:u:{user}"

    def _session_key(self, user: int) -> str:
        return f"{self.prefix}:us:{user}"

    def _history_key(self, user: int) -> str:
        return f"{self.prefix}:uh:{user}"

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
                    self._history_key(ev.user_id),
                ],
                args=[
                    ev.ts_ms,
                    f"{ev.ts_ms}:{ev.item_id}:{ev.event}",
                    f"{ev.ts_ms}:{ev.user_id}",
                    SESSION_GAP_MS,
                    USER_RETENTION_MS,
                    ITEM_RETENTION_MS,
                    TTL_MS,
                    f"{ev.ts_ms:013d}:{ev.item_id:010d}:{ev.event}",
                    HISTORY_LEN,
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

    def user_histories(
        self, users: Sequence[int], as_of_ms: int
    ) -> list[tuple[list[int], list[int], list[str]]]:
        """Last HISTORY_LEN events before as_of per user: (ts, items, events), oldest first."""
        pipe = self.r.pipeline(transaction=False)
        for user in users:
            pipe.zrangebyscore(self._history_key(user), "-inf", f"({as_of_ms}")
        out = []
        for rows in pipe.execute():
            ts, items, events = [], [], []
            for member in rows:
                t, item, event = (member.decode() if isinstance(member, bytes) else member).split(
                    ":"
                )
                ts.append(int(t))
                items.append(int(item))
                events.append(event)
            out.append(user_history_at(ts, items, events, as_of_ms))
        return out

    def item_features(
        self, items: Sequence[int], as_of_ms: int, features: Sequence[str] | None = None
    ) -> list[dict[str, int]]:
        """Item window counts in one server-side call (one round trip, no per-command
        client overhead). `features` limits the work to the windows a model uses."""
        windows = [w for w in ITEM_WINDOWS if features is None or w.name in features]
        if not items or not windows:
            return [{} for _ in items]
        window_args = [
            x for w in windows for x in (_ITEM_TYPES.index(w.event or "") + 1, w.window_ms)
        ]
        counts = self._item_counts(
            keys=[],
            args=[
                self.prefix,
                as_of_ms,
                len(_ITEM_TYPES),
                *_ITEM_TYPES,
                len(windows),
                *window_args,
                *items,
            ],
        )
        names, n = [w.name for w in windows], len(windows)
        return [
            dict(zip(names, (int(c) for c in counts[i * n : (i + 1) * n]), strict=True))
            for i in range(len(items))
        ]

    def clear(self) -> int:
        """Delete every key under this store's prefix."""
        deleted = 0
        for key in self.r.scan_iter(match=f"{self.prefix}:*", count=1000):
            deleted += self.r.delete(key)
        return int(deleted)
