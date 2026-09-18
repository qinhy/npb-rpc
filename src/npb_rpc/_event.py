from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


EventBackend = Literal["valkey", "redis"]
EventState = Literal["succeeded", "failed", "cancelled"]


def _default_url(backend: EventBackend) -> str:
    configured = os.environ.get("NPB_RPC_EVENT_URL")
    if configured:
        return configured
    if backend == "valkey":
        return "valkey://127.0.0.1:6379/0"
    return "redis://127.0.0.1:6379/0"


def _as_valkey_url(url: str) -> str:
    if url.startswith("redis://"):
        return "valkey://" + url[len("redis://") :]
    if url.startswith("rediss://"):
        return "valkeys://" + url[len("rediss://") :]
    return url


def _as_redis_url(url: str) -> str:
    if url.startswith("valkey://"):
        return "redis://" + url[len("valkey://") :]
    if url.startswith("valkeys://"):
        return "rediss://" + url[len("valkeys://") :]
    return url


@lru_cache(maxsize=32)
def _cached_client(backend: EventBackend, url: str, process_id: int) -> Any:
    """Return one lazy connection-pool client per process/backend/url.

    ``process_id`` is deliberately part of the cache key so a client created
    before ``fork()`` is never reused by the child process.
    """
    del process_id

    if backend == "valkey":
        try:
            import valkey
        except ImportError:
            # Valkey speaks RESP and can also be used through redis-py.
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "RpcEvent backend='valkey' requires `valkey` or `redis`. "
                    "Install with `pip install valkey` (preferred) or `pip install redis`."
                ) from exc
            return redis.Redis.from_url(
                _as_redis_url(url),
                decode_responses=True,
            )

        return valkey.from_url(
            _as_valkey_url(url),
            decode_responses=True,
        )

    try:
        import redis
    except ImportError as exc:
        raise ImportError(
            "RpcEvent backend='redis' requires redis-py. "
            "Install with `pip install redis`."
        ) from exc

    return redis.Redis.from_url(
        _as_redis_url(url),
        decode_responses=True,
    )


class RpcEventResult(BaseModel):
    """Terminal value stored in a one-shot RpcEvent."""

    state: EventState = "succeeded"
    error: str = ""


class RpcEvent(BaseModel):
    """Transportable one-shot completion event backed by Valkey or Redis.

    The model itself is safe to put inside an NPB RPC request/response.  It
    contains only serializable connection information and an event key.  The
    actual Valkey/Redis client is created lazily in whichever process calls
    ``set()``, ``wait()``, ``is_set()`` or ``delete()``.

    Typical usage::

        done = RpcEvent.create()
        request.done_event = done

        # Producer process
        if request.done_event:
            request.done_event.set()

        # Dependent process
        for event in request.wait_for:
            result = event.wait()

        # Owner cleanup
        done.delete()
    """

    backend: EventBackend = "redis"
    url: str = ""
    key: str = Field(min_length=1)
    ttl_seconds: int = Field(default=3600, ge=1)

    @model_validator(mode="after")
    def _fill_default_url(self) -> RpcEvent:
        if not self.url:
            self.url = _default_url(self.backend)
        return self

    @classmethod
    def create(
        cls,
        *,
        backend: EventBackend = "redis",
        url: str | None = None,
        ttl_seconds: int = 3600,
        prefix: str = "npb-rpc:event",
    ) -> RpcEvent:
        """Create a new event descriptor without contacting Valkey/Redis."""
        if not prefix or not prefix.strip():
            raise ValueError("prefix must be a non-empty string")
        return cls(
            backend=backend,
            url=url or _default_url(backend),
            key=f"{prefix}:{uuid4().hex}",
            ttl_seconds=ttl_seconds,
        )

    def _client(self) -> Any:
        return _cached_client(
            self.backend,
            self.url,
            os.getpid(),
        )

    def set(
        self,
        *,
        state: EventState = "succeeded",
        error: str = "",
    ) -> str:
        """Set the one-shot event and return the backend stream-entry ID.

        The event is persisted as a one-entry stream.  Therefore a waiter that
        starts *after* ``set()`` still observes the completion immediately.
        A TTL is applied at the same time as a fallback for abandoned events.
        """
        client = self._client()
        pipe = client.pipeline(transaction=True)
        pipe.xadd(
            self.key,
            {
                "state": state,
                "error": error,
            },
            maxlen=1,
            approximate=False,
        )
        pipe.expire(self.key, self.ttl_seconds)
        stream_id, _ = pipe.execute()
        return str(stream_id)

    def wait(self, timeout: float | None = None) -> RpcEventResult | None:
        """Wait for the event without application-side polling.

        ``timeout`` is expressed in seconds:

        - ``None``: block forever.
        - ``0``: return immediately if the event is not set.
        - ``> 0``: block for at most that many seconds.

        Returns ``None`` only when a finite timeout expires before the event is
        set/deleted.
        """
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
                raise TypeError("timeout must be numeric or None")
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and >= 0 or None")

        streams = {self.key: "0-0"}
        client = self._client()

        if timeout is None:
            rows = client.xread(streams=streams, count=1, block=0)
        elif timeout == 0:
            rows = client.xread(streams=streams, count=1)
        else:
            rows = client.xread(
                streams=streams,
                count=1,
                block=max(1, math.ceil(timeout * 1000.0)),
            )

        if not rows:
            return None

        _, entries = rows[0]
        if not entries:
            return None

        _, values = entries[0]
        state = values.get("state", "succeeded")
        error = values.get("error", "")

        # Be tolerant if a client was configured without decode_responses=True.
        if isinstance(state, bytes):
            state = state.decode("utf-8")
        if isinstance(error, bytes):
            error = error.decode("utf-8")

        return RpcEventResult(
            state=state,
            error=error,
        )

    def is_set(self) -> bool:
        """Return True when the completion value currently exists."""
        return bool(self._client().xlen(self.key))

    def refresh_ttl(self, ttl_seconds: int | None = None) -> bool:
        """Refresh the event TTL after it has been set."""
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        return bool(self._client().expire(self.key, ttl))

    def delete(self) -> bool:
        """Explicitly remove the event from Valkey/Redis.

        Usually the creator/owner should call this only after every dependent
        consumer no longer needs the completion event.  If cleanup is missed,
        ``ttl_seconds`` remains the safety fallback.
        """
        return bool(self._client().delete(self.key))
