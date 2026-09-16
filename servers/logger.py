from __future__ import annotations

import logging
import os
import queue
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Mapping

try:
    import redis
except ImportError:  # Module can still be imported without the optional Redis dependency.
    redis = None  # type: ignore[assignment]


# LogRecord attributes created by logging itself. Any other attributes are treated
# as values supplied through logger.info(..., extra={...}).
_STANDARD_LOG_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
) | {"message", "asctime"}

_LEGACY_PREFIX_PATTERN = re.compile(r"^\[([^\]]+)\]\s*(.*)$", re.DOTALL)


@dataclass(slots=True)
class _RedisLogEntry:
    stream_name: str
    message: str
    level: str
    timestamp: float
    module: str
    function: str
    line: int
    process: int
    thread: int
    exception: str | None
    stack_info: str | None
    extra: dict[str, str]


@dataclass(slots=True)
class _FlushRequest:
    done: threading.Event


class RedisStreamHandler(logging.Handler):
    """Asynchronous :mod:`logging` handler backed by Redis Streams.

    ``emit()`` only snapshots the ``LogRecord`` and performs a non-blocking
    ``Queue.put_nowait``. Redis I/O, stream-key construction, and batching are
    performed by a worker thread.

    Stream naming
    -------------
    Normally the stream is ``key_prefix + record.name``. For example::

        logging.getLogger("camera.rgb")

    writes to ``camera.rgb`` by default, or ``log:camera.rgb`` when
    ``key_prefix="log:"``.

    ``legacy_bracket_keys=True`` keeps compatibility with the old convention::

        LOG.info("[Camera:rgb] frame received")

    which writes ``frame received`` to ``Camera:rgb`` by default.

    Shutdown
    --------
    ``close()`` stops accepting new records, places a sentinel *after* all
    already-queued records, waits up to ``shutdown_timeout`` for the worker to
    flush them, and then closes Redis. Shutdown is therefore bounded: a Redis
    outage cannot make process exit wait forever.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "",
        max_queue_size: int = 10_000,
        max_stream_len: int = 100_000,
        batch_size: int = 100,
        flush_interval: float = 0.25,
        redis_socket_timeout: float = 2.0,
        shutdown_timeout: float = 3.0,
        legacy_bracket_keys: bool = True,
        include_source: bool = True,
        redis_client: Any | None = None,
        ping_on_start: bool = True,
    ) -> None:
        super().__init__()

        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if max_stream_len <= 0:
            raise ValueError("max_stream_len must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")
        if shutdown_timeout < 0:
            raise ValueError("shutdown_timeout must be >= 0")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.max_stream_len = max_stream_len
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.shutdown_timeout = shutdown_timeout
        self.legacy_bracket_keys = legacy_bracket_keys
        self.include_source = include_source

        self.hostname = socket.gethostname()
        self.pid = os.getpid()

        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._sentinel = object()
        self._state_lock = threading.Lock()
        self._closing = False
        self._worker_error: BaseException | None = None

        self._dropped = 0
        self._written = 0
        self._last_drop_warning = 0.0

        if redis_client is None:
            if redis is None:
                raise RuntimeError(
                    "RedisStreamHandler requires the 'redis' package; "
                    "install it with: pip install redis"
                )
            self.redis = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=redis_socket_timeout,
                socket_timeout=redis_socket_timeout,
            )
        else:
            self.redis = redis_client

        if ping_on_start and not self.redis.ping():
            raise RuntimeError("Failed to connect to Redis")

        self._worker_thread = threading.Thread(
            target=self._worker,
            name="RedisStreamHandlerWorker",
            daemon=True,
        )
        self._worker_thread.start()

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def written_count(self) -> int:
        return self._written

    @property
    def worker_error(self) -> BaseException | None:
        return self._worker_error

    def emit(self, record: logging.LogRecord) -> None:
        """Snapshot a LogRecord and enqueue it without waiting for Redis."""
        if self._closing:
            return

        try:
            message = record.getMessage()
            stream_name = record.name or "root"

            if self.legacy_bracket_keys:
                match = _LEGACY_PREFIX_PATTERN.match(message.strip())
                if match:
                    stream_name = match.group(1).strip() or stream_name
                    message = match.group(2).strip()

            exception = None
            if record.exc_info:
                exception = self.formatter.formatException(record.exc_info) if self.formatter else logging.Formatter().formatException(record.exc_info)

            custom_extra = {
                key: self._stringify(value)
                for key, value in record.__dict__.items()
                if key not in _STANDARD_LOG_RECORD_ATTRS
            }

            entry = _RedisLogEntry(
                stream_name=stream_name,
                message=message,
                level=record.levelname,
                timestamp=record.created,
                module=record.module,
                function=record.funcName,
                line=record.lineno,
                process=record.process,
                thread=record.thread,
                exception=exception,
                stack_info=record.stack_info,
                extra=custom_extra,
            )

            self._queue.put_nowait(entry)

        except queue.Full:
            self._dropped += 1
            self._warn_queue_full(record)
        except Exception:
            self.handleError(record)

    def _warn_queue_full(self, record: logging.LogRecord) -> None:
        # Never log this through logging itself: that would recurse into us.
        now = time.monotonic()
        if now - self._last_drop_warning >= 1.0:
            self._last_drop_warning = now
            try:
                print(
                    "[RedisStreamHandler] queue full; "
                    f"dropped={self._dropped} latest={record.getMessage()!r}",
                    file=sys.stderr,
                )
            except Exception:
                pass

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return repr(value)

    def _make_redis_key(self, stream_name: str) -> str:
        return f"{self.key_prefix}{stream_name}"

    def _to_redis_record(self, entry: _RedisLogEntry) -> dict[str, str]:
        record = {
            "msg": entry.message,
            "level": entry.level,
            "ts": repr(entry.timestamp),
        }

        if self.include_source:
            record.update(
                {
                    "hostname": self.hostname,
                    "pid": str(entry.process),
                    "module": entry.module,
                    "func": entry.function,
                    "line": str(entry.line),
                    "thread": str(entry.thread),
                }
            )

        if entry.exception:
            record["exception"] = entry.exception

        if entry.stack_info:
            record["stack"] = entry.stack_info

        for key, value in entry.extra.items():
            record[f"extra:{key}"] = value

        return record

    def _write_batch(self, batch: list[_RedisLogEntry]) -> None:
        if not batch:
            return

        try:
            pipe = self.redis.pipeline(transaction=False)
            for entry in batch:
                pipe.xadd(
                    self._make_redis_key(entry.stream_name),
                    self._to_redis_record(entry),
                    maxlen=self.max_stream_len,
                    approximate=True,
                )
            pipe.execute()
            self._written += len(batch)

        except Exception as exc:
            # We deliberately do not feed this error back through logging,
            # otherwise the handler could recursively log its own failure.
            self._worker_error = exc
            try:
                print(
                    "[RedisStreamHandler] failed to write "
                    f"batch of {len(batch)} record(s): {exc!r}",
                    file=sys.stderr,
                )
            except Exception:
                pass

    def _worker(self) -> None:
        pending: list[_RedisLogEntry] = []
        deadline: float | None = None

        while True:
            timeout: float | None
            if pending and deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            else:
                timeout = None

            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._write_batch(pending)
                pending.clear()
                deadline = None
                continue

            if item is self._sentinel:
                self._write_batch(pending)
                return

            if isinstance(item, _FlushRequest):
                try:
                    self._write_batch(pending)
                    pending.clear()
                    deadline = None
                finally:
                    item.done.set()
                continue

            if not isinstance(item, _RedisLogEntry):
                continue

            pending.append(item)

            if len(pending) == 1:
                deadline = time.monotonic() + self.flush_interval

            if len(pending) >= self.batch_size:
                self._write_batch(pending)
                pending.clear()
                deadline = None

    def flush(self) -> None:
        """Flush records queued before this call, with a bounded wait."""
        # Handler.__init__ creates _closed. During/after close there is no
        # useful work to do and enqueuing another control item could block.
        if self._closing or getattr(self, "_closed", False):
            return

        if not self._worker_thread.is_alive():
            return

        request = _FlushRequest(done=threading.Event())
        deadline = time.monotonic() + self.shutdown_timeout

        try:
            remaining = max(0.0, deadline - time.monotonic())
            self._queue.put(request, timeout=remaining)
            remaining = max(0.0, deadline - time.monotonic())
            request.done.wait(timeout=remaining)
        except Exception:
            pass

    def close(self) -> None:
        """Drain the queue and stop the worker without unbounded shutdown."""
        with self._state_lock:
            if self._closing or getattr(self, "_closed", False):
                return
            self._closing = True

        deadline = time.monotonic() + self.shutdown_timeout

        try:
            if self._worker_thread.is_alive():
                try:
                    remaining = max(0.0, deadline - time.monotonic())
                    self._queue.put(self._sentinel, timeout=remaining)
                except queue.Full:
                    pass

                remaining = max(0.0, deadline - time.monotonic())
                self._worker_thread.join(timeout=remaining)

                if self._worker_thread.is_alive():
                    try:
                        print(
                            "[RedisStreamHandler] shutdown timeout reached; "
                            "some queued logs may not have been written",
                            file=sys.stderr,
                        )
                    except Exception:
                        pass
        finally:
            try:
                self.redis.close()
            except Exception:
                pass
            super().close()


def configure_redis_logging(
    redis_url: str = "redis://localhost:6379/0",
    *,
    logger: logging.Logger | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
    console_format: str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    key_prefix: str = "",
    max_queue_size: int = 10_000,
    max_stream_len: int = 100_000,
    batch_size: int = 100,
    flush_interval: float = 0.25,
    redis_socket_timeout: float = 2.0,
    shutdown_timeout: float = 3.0,
    legacy_bracket_keys: bool = True,
    include_source: bool = True,
    ping_on_start: bool = True,
) -> RedisStreamHandler:
    """Install Redis logging on ``logger`` (root logger by default).

    Example::

        REDIS_HANDLER = configure_redis_logging()
        LOG = logging.getLogger("nng_dai_camera")
        LOG.info("camera started")

        try:
            1 / 0
        except Exception:
            LOG.exception("camera failed", extra={"stream": "rgb"})

    Attaching to the root logger means existing ``logging.getLogger(...)``
    instances propagate to Redis without needing per-module changes.
    """
    target = logger if logger is not None else logging.getLogger()
    target.setLevel(level)

    handler = RedisStreamHandler(
        redis_url=redis_url,
        key_prefix=key_prefix,
        max_queue_size=max_queue_size,
        max_stream_len=max_stream_len,
        batch_size=batch_size,
        flush_interval=flush_interval,
        redis_socket_timeout=redis_socket_timeout,
        shutdown_timeout=shutdown_timeout,
        legacy_bracket_keys=legacy_bracket_keys,
        include_source=include_source,
        ping_on_start=ping_on_start,
    )
    handler.setLevel(level)
    target.addHandler(handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(console_format))
        target.addHandler(console_handler)

    return handler


def tail_streams(
    pattern: str = "*",
    ignore_patterns: tuple[str, ...] = (),
    redis_url: str = "redis://localhost:6379/0",
    level: str | None = None,
    discovery_interval: float = 1.0,
) -> None:
    """Tail matching Redis Streams and print new entries as they arrive."""
    if redis is None:
        raise RuntimeError("tail_streams requires the 'redis' package; install it with: pip install redis")

    client = redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=None,
    )

    last_ids: dict[str, str] = {}
    wanted_level = level.upper() if level else None
    next_discovery = 0.0

    def is_ignored(stream: str) -> bool:
        return any(
            fnmatchcase(stream, ignored_pattern)
            for ignored_pattern in ignore_patterns
        )

    def print_entry(
        stream: str,
        entry_id: str,
        fields: Mapping[str, str],
    ) -> None:
        entry_level = fields.get("level", "").upper()

        if wanted_level and entry_level != wanted_level:
            return

        print(
            f"{entry_id} "
            f"[{entry_level or '-'}] "
            f"[{stream}] "
            f"{fields.get('msg', '')}"
        )

    try:
        while True:
            now = time.monotonic()

            if now >= next_discovery:
                discovered_streams = set(
                    client.scan_iter(
                        match=pattern,
                        count=500,
                        _type="stream",
                    )
                )

                current_streams = {
                    stream
                    for stream in discovered_streams
                    if not is_ignored(stream)
                }

                for stream in set(last_ids) - current_streams:
                    del last_ids[stream]

                # Preserve the old behavior: when a stream is first discovered,
                # show its current latest record once, then follow new records.
                for stream in sorted(current_streams - set(last_ids)):
                    latest = client.xrevrange(stream, count=1)

                    if latest:
                        entry_id, fields = latest[0]
                        print_entry(stream, entry_id, fields)
                        last_ids[stream] = entry_id
                    else:
                        last_ids[stream] = "0-0"

                next_discovery = now + discovery_interval

            if not last_ids:
                time.sleep(discovery_interval)
                continue

            results = client.xread(
                streams=last_ids,
                count=100,
                block=max(100, int(discovery_interval * 1000)),
            )

            entries: list[tuple[str, str, Mapping[str, str]]] = []

            for stream, stream_entries in results:
                for entry_id, fields in stream_entries:
                    last_ids[stream] = entry_id
                    entries.append((entry_id, stream, fields))

            entries.sort(
                key=lambda item: tuple(
                    int(part) for part in item[0].split("-", 1)
                )
            )

            for entry_id, stream, fields in entries:
                print_entry(stream, entry_id, fields)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        client.close()

# ---------------------------------------------------------------------------
# Auto configuration
# ---------------------------------------------------------------------------
_REDIS_HANDLER: RedisStreamHandler | None = None


def _auto_configure() -> None:
    global _REDIS_HANDLER

    if _REDIS_HANDLER is not None:
        return

    root = logging.getLogger()

    # Protect against duplicate configuration.
    for handler in root.handlers:
        if isinstance(handler, RedisStreamHandler):
            _REDIS_HANDLER = handler
            return

    redis_url = os.getenv(
        "LOG_REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )

    key_prefix = os.getenv(
        "LOG_REDIS_PREFIX",
        "",
    )

    try:
        _REDIS_HANDLER = configure_redis_logging(
            redis_url=redis_url,
            key_prefix=key_prefix,
            ping_on_start=False,
        )
    except Exception as exc:
        print(
            f"[LOGGER WARNING] Redis logging unavailable: {exc}",
            file=sys.stderr,
        )


_auto_configure()


__all__ = [
    "logging",
    "RedisStreamHandler",
    "tail_streams",
]

if __name__ == "__main__":
    configure_redis_logging()
    LOG = logging.getLogger("RedisLogger:init")
    LOG.info("start")
