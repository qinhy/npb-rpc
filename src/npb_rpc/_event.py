from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EventBackend = Literal["valkey", "redis"]
EventState = Literal["created", "succeeded", "failed", "cancelled"]
TerminalState = Literal["succeeded", "failed", "cancelled"]
EventStatus = Literal["created", "succeeded", "failed", "cancelled", "expired"]


class RpcEventError(RuntimeError):
    pass


class RpcEventExpired(RpcEventError):
    pass


class RpcEventAlreadySet(RpcEventError):
    pass


class RpcEventPoolExhausted(RpcEventError):
    pass


def _default_url(backend: EventBackend) -> str:
    if url := os.environ.get("NPB_RPC_EVENT_URL"):
        return url
    return f"{backend}://127.0.0.1:6379/0"


def _as_valkey_url(url: str) -> str:
    if url.startswith("redis://"):
        return "valkey://" + url[8:]
    if url.startswith("rediss://"):
        return "valkeys://" + url[9:]
    return url


def _as_redis_url(url: str) -> str:
    if url.startswith("valkey://"):
        return "redis://" + url[9:]
    if url.startswith("valkeys://"):
        return "rediss://" + url[10:]
    return url


@lru_cache(maxsize=32)
def _cached_client(backend: EventBackend, url: str, pid: int) -> Any:
    del pid
    kwargs = dict(
        decode_responses=True,
        socket_timeout=None,       # XREAD BLOCK 0
        socket_connect_timeout=5,
    )

    if backend == "valkey":
        try:
            import valkey
            return valkey.from_url(_as_valkey_url(url), **kwargs)
        except ImportError:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "Install `valkey` (preferred) or `redis`."
                ) from exc
            return redis.Redis.from_url(_as_redis_url(url), **kwargs)

    try:
        import redis
    except ImportError as exc:
        raise ImportError("Install `redis`.") from exc
    return redis.Redis.from_url(_as_redis_url(url), **kwargs)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


_NEXT_SLOT = r"""
local n = tonumber(redis.call("GET", KEYS[1]) or "-1") + 1
local size = tonumber(ARGV[1])
if n >= size then n = 0 end
redis.call("SET", KEYS[1], n)
return n
"""

_TRY_CLAIM = r"""
local meta = KEYS[1]
local stream = KEYS[2]

local now = tonumber(ARGV[1])
local lease_ms = tonumber(ARGV[2])
local retention_ms = tonumber(ARGV[3])
local maxlen = ARGV[4]

local generation = redis.call("HGET", meta, "generation")

local function notify(gen, state, err)
    return redis.call(
        "XADD", stream, "MAXLEN", "=", maxlen, "*",
        "generation", gen,
        "state", state,
        "error", err
    )
end

local function claim(gen)
    redis.call(
        "HSET", meta,
        "generation", gen,
        "state", "created",
        "error", "",
        "reusable_at", 0,
        "lease_until", now + lease_ms,
        "updated_at", now
    )
    notify(gen, "created", "")
    return gen
end

if not generation then
    return claim(1)
end

local gen = tonumber(generation)
local state = redis.call("HGET", meta, "state")

if state == "created" then
    local lease_until = tonumber(redis.call("HGET", meta, "lease_until") or "0")

    -- Convert abandoned active work into a terminal failure first.
    if lease_until > 0 and lease_until <= now then
        redis.call(
            "HSET", meta,
            "state", "failed",
            "error", "event owner lease expired",
            "reusable_at", now + retention_ms,
            "lease_until", 0,
            "updated_at", now
        )
        notify(gen, "failed", "event owner lease expired")

        -- With zero retention it can be reused immediately.
        if retention_ms == 0 then
            return claim(gen + 1)
        end
    end

    return 0
end

if state == "succeeded" or state == "failed" or state == "cancelled" then
    local reusable_at = tonumber(redis.call("HGET", meta, "reusable_at") or "0")
    if reusable_at <= now then
        return claim(gen + 1)
    end
end

return 0
"""

_COMPLETE = r"""
local meta = KEYS[1]
local stream = KEYS[2]

local expected = tonumber(ARGV[1])
local new_state = ARGV[2]
local err = ARGV[3]
local now = tonumber(ARGV[4])
local retention_ms = tonumber(ARGV[5])
local maxlen = ARGV[6]

local generation = redis.call("HGET", meta, "generation")
if not generation or tonumber(generation) ~= expected then
    return {"expired", generation or ""}
end

local state = redis.call("HGET", meta, "state")
if state ~= "created" then
    return {"already", state or ""}
end

redis.call(
    "HSET", meta,
    "state", new_state,
    "error", err,
    "reusable_at", now + retention_ms,
    "lease_until", 0,
    "updated_at", now
)

local id = redis.call(
    "XADD", stream, "MAXLEN", "=", maxlen, "*",
    "generation", expected,
    "state", new_state,
    "error", err
)

return {"ok", id}
"""

_REFRESH_LEASE = r"""
local generation = redis.call("HGET", KEYS[1], "generation")
if not generation or tonumber(generation) ~= tonumber(ARGV[1]) then
    return -1
end
if redis.call("HGET", KEYS[1], "state") ~= "created" then
    return 0
end
redis.call(
    "HSET", KEYS[1],
    "lease_until", tonumber(ARGV[2]) + tonumber(ARGV[3]),
    "updated_at", tonumber(ARGV[2])
)
return 1
"""

_RELEASE = r"""
local generation = redis.call("HGET", KEYS[1], "generation")
if not generation or tonumber(generation) ~= tonumber(ARGV[1]) then
    return -1
end

local state = redis.call("HGET", KEYS[1], "state")
if state == "created" then
    return 0
end

redis.call(
    "HSET", KEYS[1],
    "reusable_at", tonumber(ARGV[2]),
    "updated_at", tonumber(ARGV[2])
)
return 1
"""


class RpcEventResult(BaseModel):
    state: TerminalState
    error: str = ""


class RpcEvent(BaseModel):
    """Bounded, reusable cross-process completion event.

    Identity is (slot, generation).  Slots are reused in a fixed circular pool;
    generation prevents an old RpcEvent from attaching to a newer event that
    reused the same slot.
    """

    backend: EventBackend = "redis"
    url: str = ""
    prefix: str = Field(default="npb-rpc:event", min_length=1)

    slot: int = Field(ge=0)
    generation: int = Field(ge=1)

    retention_seconds: float = Field(default=60.0, ge=0)
    lease_seconds: float = Field(default=3600.0, gt=0)
    stream_maxlen: int = Field(default=8, ge=2)

    @model_validator(mode="after")
    def _defaults(self) -> "RpcEvent":
        if not self.url:
            self.url = _default_url(self.backend)
        if not self.prefix.strip():
            raise ValueError("prefix must be non-empty")
        return self

    @property
    def _meta_key(self) -> str:
        # Same Redis Cluster hash tag for the two keys touched by Lua.
        return f"{self.prefix}:{{slot-{self.slot}}}:meta"

    @property
    def _stream_key(self) -> str:
        return f"{self.prefix}:{{slot-{self.slot}}}:stream"

    @property
    def _counter_key(self) -> str:
        return f"{self.prefix}:next"

    def _client(self) -> Any:
        return _cached_client(self.backend, self.url, os.getpid())

    @classmethod
    def create(
        cls,
        *,
        backend: EventBackend = "redis",
        url: str | None = None,
        prefix: str = "npb-rpc:event",
        pool_size: int = 1024, # 100_000,
        retention_seconds: float = 60.0,
        lease_seconds: float = 3600.0,
        stream_maxlen: int = 8,
    ) -> "RpcEvent":
        if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")
        if not prefix or not prefix.strip():
            raise ValueError("prefix must be non-empty")
        if not math.isfinite(retention_seconds) or retention_seconds < 0:
            raise ValueError("retention_seconds must be finite and >= 0")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and > 0")
        if not isinstance(stream_maxlen, int) or isinstance(stream_maxlen, bool) or stream_maxlen < 2:
            raise ValueError("stream_maxlen must be an integer >= 2")

        resolved_url = url or _default_url(backend)
        client = _cached_client(backend, resolved_url, os.getpid())
        counter_key = f"{prefix}:next"

        start = int(client.eval(_NEXT_SLOT, 1, counter_key, pool_size))
        now = _now_ms()
        lease_ms = math.ceil(lease_seconds * 1000)
        retention_ms = math.ceil(retention_seconds * 1000)

        for offset in range(pool_size):
            slot = (start + offset) % pool_size
            tag = f"{prefix}:{{slot-{slot}}}"
            meta_key = f"{tag}:meta"
            stream_key = f"{tag}:stream"

            generation = int(
                client.eval(
                    _TRY_CLAIM,
                    2,
                    meta_key,
                    stream_key,
                    now,
                    lease_ms,
                    retention_ms,
                    stream_maxlen,
                )
            )
            if generation > 0:
                return cls(
                    backend=backend,
                    url=resolved_url,
                    prefix=prefix,
                    slot=slot,
                    generation=generation,
                    retention_seconds=retention_seconds,
                    lease_seconds=lease_seconds,
                    stream_maxlen=stream_maxlen,
                )

        raise RpcEventPoolExhausted(
            f"RpcEvent pool exhausted: prefix={prefix!r}, pool_size={pool_size}"
        )

    def set(
        self,
        *,
        state: TerminalState = "succeeded",
        error: str = "",
    ) -> str:
        if state not in ("succeeded", "failed", "cancelled"):
            raise ValueError(f"invalid terminal state: {state!r}")
        if not isinstance(error, str):
            raise TypeError("error must be a string")

        result = self._client().eval(
            _COMPLETE,
            2,
            self._meta_key,
            self._stream_key,
            self.generation,
            state,
            error,
            _now_ms(),
            math.ceil(self.retention_seconds * 1000),
            self.stream_maxlen,
        )

        status = self._text(result[0])
        value = self._text(result[1])

        if status == "expired":
            raise RpcEventExpired(
                f"RpcEvent was recycled: slot={self.slot}, generation={self.generation}"
            )
        if status == "already":
            raise RpcEventAlreadySet(
                f"RpcEvent already terminal: {value!r}"
            )
        return value

    def succeed(self) -> str:
        return self.set(state="succeeded")

    def fail(self, error: str) -> str:
        return self.set(state="failed", error=error)

    def cancel(self, error: str = "") -> str:
        return self.set(state="cancelled", error=error)

    def wait(self, timeout: float | None = None) -> RpcEventResult | None:
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
                raise TypeError("timeout must be numeric or None")
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and >= 0 or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        client = self._client()
        last_id = "0-0"

        while True:
            result = self._snapshot(client)
            if result is not None:
                return result

            if timeout == 0:
                return None

            if deadline is None:
                block_ms = 0
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._snapshot(client)
                block_ms = max(1, math.ceil(remaining * 1000))

            rows = client.xread(
                streams={self._stream_key: last_id},
                count=self.stream_maxlen,
                block=block_ms,
            )

            if not rows:
                # One final atomic-ish state observation at the timeout edge.
                return self._snapshot(client)

            _, entries = rows[0]
            for entry_id, values in entries:
                last_id = self._text(entry_id)
                generation = int(self._text(values.get("generation", "0")))

                if generation < self.generation:
                    continue
                if generation > self.generation:
                    raise RpcEventExpired(
                        f"RpcEvent was recycled: slot={self.slot}, "
                        f"generation={self.generation} -> {generation}"
                    )

                state = self._text(values.get("state", ""))
                if state == "created":
                    continue
                if state in ("succeeded", "failed", "cancelled"):
                    return RpcEventResult(
                        state=state,
                        error=self._text(values.get("error", "")),
                    )

                raise RpcEventError(f"invalid RpcEvent state: {state!r}")

    def status(self) -> EventStatus:
        generation, state, _ = self._read_meta(self._client())
        if generation != self.generation:
            return "expired"
        if state not in ("created", "succeeded", "failed", "cancelled"):
            raise RpcEventError(f"invalid RpcEvent state: {state!r}")
        return state

    def exists(self) -> bool:
        generation, _, _ = self._read_meta(self._client())
        return generation == self.generation

    def is_set(self) -> bool:
        return self.status() in ("succeeded", "failed", "cancelled")

    def refresh_lease(self, lease_seconds: float | None = None) -> bool:
        lease = self.lease_seconds if lease_seconds is None else lease_seconds
        if not isinstance(lease, (int, float)) or isinstance(lease, bool):
            raise TypeError("lease_seconds must be numeric")
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("lease_seconds must be finite and > 0")

        result = int(
            self._client().eval(
                _REFRESH_LEASE,
                1,
                self._meta_key,
                self.generation,
                _now_ms(),
                math.ceil(lease * 1000),
            )
        )

        if result < 0:
            raise RpcEventExpired(
                f"RpcEvent was recycled: slot={self.slot}, generation={self.generation}"
            )
        return result == 1

    heartbeat = refresh_lease

    def release(self) -> bool:
        """Make a terminal event immediately reusable; never deletes the slot."""
        result = int(
            self._client().eval(
                _RELEASE,
                1,
                self._meta_key,
                self.generation,
                _now_ms(),
            )
        )
        if result < 0:
            return False
        if result == 0:
            raise RpcEventError("cannot release an active RpcEvent")
        return True

    def delete(self) -> bool:
        """Compatibility alias: logical release, not physical Redis DEL."""
        return self.release()

    def _snapshot(self, client: Any) -> RpcEventResult | None:
        generation, state, error = self._read_meta(client)

        if generation != self.generation:
            raise RpcEventExpired(
                f"RpcEvent was recycled: slot={self.slot}, generation={self.generation}"
            )

        if state == "created":
            return None
        if state in ("succeeded", "failed", "cancelled"):
            return RpcEventResult(state=state, error=error)
        raise RpcEventError(f"invalid RpcEvent state: {state!r}")

    def _read_meta(self, client: Any) -> tuple[int | None, str, str]:
        generation, state, error = client.hmget(
            self._meta_key,
            "generation",
            "state",
            "error",
        )
        if generation is None:
            return None, "", ""
        return (
            int(self._text(generation)),
            self._text(state or ""),
            self._text(error or ""),
        )

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)