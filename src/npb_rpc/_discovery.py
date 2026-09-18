"""Pluggable service discovery with filesystem and Redis backends."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._errors import DiscoveryError, ServiceNotFoundError

BackendName = Literal["zmq", "nng", "iceoryx2"]

_SERVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_service_name(value: str) -> str:
    """Validate a filesystem-safe logical service name."""
    if not isinstance(value, str) or not _SERVICE_NAME.fullmatch(value):
        raise ValueError(
            "service name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    return value


def _validate_instance_id(value: str) -> str:
    if not isinstance(value, str) or not _INSTANCE_ID.fullmatch(value):
        raise ValueError(
            "instance ID must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    """One discoverable RPC server instance."""

    service: str
    instance_id: str
    endpoint: str
    backend: BackendName
    pid: int
    hostname: str
    started_at: float
    heartbeat_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_service_name(self.service)
        _validate_instance_id(self.instance_id)
        if not isinstance(self.endpoint, str) or "://" not in self.endpoint:
            raise ValueError("endpoint must be a transport URL")
        if self.backend not in {"zmq", "nng", "iceoryx2"}:
            raise ValueError("backend must be 'zmq', 'nng', or 'iceoryx2'")
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid < 0:
            raise ValueError("pid must be a non-negative integer")
        if not isinstance(self.hostname, str) or not self.hostname:
            raise ValueError("hostname must be a non-empty string")
        if not isinstance(self.started_at, (int, float)):
            raise TypeError("started_at must be numeric")
        if not isinstance(self.heartbeat_at, (int, float)):
            raise TypeError("heartbeat_at must be numeric")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ServiceRecord:
        if not isinstance(value, dict):
            raise TypeError("service record must be an object")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("service record metadata must be an object")
        return cls(
            service=value["service"],
            instance_id=value["instance_id"],
            endpoint=value["endpoint"],
            backend=value["backend"],
            pid=value["pid"],
            hostname=value["hostname"],
            started_at=value["started_at"],
            heartbeat_at=value["heartbeat_at"],
            metadata=dict(metadata),
        )


class DiscoveryBackend(ABC):
    """Synchronous registry interface used by discovered RPC clients and servers."""

    @abstractmethod
    def register(self, record: ServiceRecord) -> None: ...

    @abstractmethod
    def heartbeat(self, record: ServiceRecord) -> None: ...

    @abstractmethod
    def unregister(self, service: str, instance_id: str) -> None: ...

    @abstractmethod
    def list_services(self) -> list[str]: ...

    @abstractmethod
    def list_instances(self, service: str) -> list[ServiceRecord]: ...

    @abstractmethod
    def resolve(self, service: str) -> ServiceRecord: ...

    def close(self) -> None:
        """Release backend resources, if any."""
        return None


class FilesystemDiscovery(DiscoveryBackend):
    """Local-host discovery using atomic JSON registry files and heartbeats."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        heartbeat_timeout: float = 6.0,
        prune_stale: bool = True,
    ) -> None:
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be > 0")
        default_root = Path(tempfile.gettempdir()) / "npb-rpc" / "registry"
        self.root = Path(root) if root is not None else default_root
        self.heartbeat_timeout = heartbeat_timeout
        self.prune_stale = prune_stale
        self._round_robin: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _record_path(self, service: str, instance_id: str) -> Path:
        validate_service_name(service)
        _validate_instance_id(instance_id)
        return self.root / service / f"{instance_id}.json"

    @staticmethod
    def _atomic_write(path: Path, record: ServiceRecord) -> None:
        try:
            serialized = json.dumps(record.to_dict(), separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise DiscoveryError(f"service record is not JSON serializable: {exc}") from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                dir=path.parent,
                text=True,
            )
        except OSError as exc:
            raise DiscoveryError(f"failed to prepare service record {path}: {exc}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except OSError as exc:
            raise DiscoveryError(f"failed to write service record {path}: {exc}") from exc
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    def register(self, record: ServiceRecord) -> None:
        self._atomic_write(
            self._record_path(record.service, record.instance_id),
            record,
        )

    def heartbeat(self, record: ServiceRecord) -> None:
        self.register(record)

    def unregister(self, service: str, instance_id: str) -> None:
        path = self._record_path(service, instance_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DiscoveryError(f"failed to remove service record {path}: {exc}") from exc
        try:
            path.parent.rmdir()
        except OSError:
            pass

    @staticmethod
    def _read_record(path: Path) -> ServiceRecord | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return ServiceRecord.from_dict(value)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None

    def _prune_if_still_stale(self, path: Path, service: str, cutoff: float) -> None:
        current = self._read_record(path)
        if current is None or (
            current.service == service and current.heartbeat_at < cutoff
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise DiscoveryError(f"failed to prune stale service record {path}: {exc}") from exc

    def list_services(self) -> list[str]:
        """Return sorted service names that have at least one healthy instance."""
        if not self.root.exists():
            return []

        try:
            directories = tuple(self.root.iterdir())
        except OSError as exc:
            raise DiscoveryError(f"failed to read service registry {self.root}: {exc}") from exc

        services: list[str] = []
        for directory in directories:
            if not directory.is_dir():
                continue
            try:
                validate_service_name(directory.name)
            except ValueError:
                continue
            if self.list_instances(directory.name):
                services.append(directory.name)
        services.sort()
        return services

    def list_instances(self, service: str) -> list[ServiceRecord]:
        validate_service_name(service)
        directory = self.root / service
        if not directory.exists():
            return []

        cutoff = time.time() - self.heartbeat_timeout
        healthy: list[ServiceRecord] = []
        try:
            paths = tuple(directory.glob("*.json"))
        except OSError as exc:
            raise DiscoveryError(f"failed to read service registry {directory}: {exc}") from exc
        for path in paths:
            record = self._read_record(path)
            if record is None or record.service != service:
                continue
            if record.heartbeat_at >= cutoff:
                healthy.append(record)
            elif self.prune_stale:
                self._prune_if_still_stale(path, service, cutoff)

        healthy.sort(key=lambda item: item.instance_id)
        return healthy

    def resolve(self, service: str) -> ServiceRecord:
        instances = self.list_instances(service)
        if not instances:
            raise ServiceNotFoundError(
                f"no healthy instances are registered for service {service!r}"
            )
        with self._lock:
            index = self._round_robin[service] % len(instances)
            self._round_robin[service] += 1
        return instances[index]


class RedisDiscovery(DiscoveryBackend):
    """Distributed discovery backed by Redis keys with TTL-based liveness."""

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        heartbeat_timeout: float = 6.0,
        prefix: str = "npb-rpc:registry",
        scan_count: int = 100,
    ) -> None:
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be > 0")
        if scan_count <= 0:
            raise ValueError("scan_count must be > 0")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")

        prefix = prefix.rstrip(":")
        if not prefix:
            raise ValueError("prefix must not be empty")
        if any(character in prefix for character in "*?[]"):
            raise ValueError("prefix must not contain Redis glob characters '*', '?', '[', or ']'")

        try:
            import redis
            from redis.exceptions import RedisError
        except ImportError as exc:
            raise DiscoveryError(
                "Redis discovery requires the optional 'redis' package; "
                "install it with `pip install redis` or your project's Redis extra"
            ) from exc

        self.root = self.url = url
        self.heartbeat_timeout = heartbeat_timeout
        self.prefix = prefix
        self.scan_count = scan_count
        self._ttl_ms = max(1, math.ceil(heartbeat_timeout * 1000.0))

        # Keep Redis optional at package-import time. Connections are also lazy;
        # Redis is contacted only when an operation is executed.
        self._redis_error = RedisError
        self._redis = redis.Redis.from_url(url, decode_responses=True)

        self._round_robin: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _record_key(self, service: str, instance_id: str) -> str:
        validate_service_name(service)
        _validate_instance_id(instance_id)
        return f"{self.prefix}:{service}:{instance_id}"

    def _service_pattern(self, service: str) -> str:
        validate_service_name(service)
        return f"{self.prefix}:{service}:*"

    def _all_pattern(self) -> str:
        return f"{self.prefix}:*"

    @staticmethod
    def _serialize(record: ServiceRecord) -> str:
        try:
            return json.dumps(record.to_dict(), separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise DiscoveryError(f"service record is not JSON serializable: {exc}") from exc

    @staticmethod
    def _deserialize(value: str | None) -> ServiceRecord | None:
        if value is None:
            return None
        try:
            decoded = json.loads(value)
            return ServiceRecord.from_dict(decoded)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def register(self, record: ServiceRecord) -> None:
        """Register an instance and create/refresh its liveness TTL."""
        key = self._record_key(record.service, record.instance_id)
        value = self._serialize(record)
        try:
            # Value and expiration are installed atomically by one Redis SET.
            self._redis.set(key, value, px=self._ttl_ms)
        except self._redis_error as exc:
            raise DiscoveryError(
                f"failed to register service {record.service!r} in Redis: {exc}"
            ) from exc

    def heartbeat(self, record: ServiceRecord) -> None:
        """Refresh both the service record and its Redis TTL."""
        self.register(record)

    def unregister(self, service: str, instance_id: str) -> None:
        key = self._record_key(service, instance_id)
        try:
            self._redis.delete(key)
        except self._redis_error as exc:
            raise DiscoveryError(
                f"failed to unregister service {service!r} from Redis: {exc}"
            ) from exc

    def list_services(self) -> list[str]:
        """Return sorted service names that currently have Redis instance keys."""
        prefix = f"{self.prefix}:"
        services: set[str] = set()

        try:
            for key in self._redis.scan_iter(
                match=self._all_pattern(),
                count=self.scan_count,
            ):
                if not isinstance(key, str) or not key.startswith(prefix):
                    continue

                remainder = key[len(prefix):]
                service, separator, instance_id = remainder.partition(":")
                if not separator:
                    continue

                try:
                    validate_service_name(service)
                    _validate_instance_id(instance_id)
                except ValueError:
                    continue

                services.add(service)
        except self._redis_error as exc:
            raise DiscoveryError(f"failed to list Redis services: {exc}") from exc

        return sorted(services)

    def list_instances(self, service: str) -> list[ServiceRecord]:
        """Return currently live instances for a service.

        Redis TTL is the source of truth for liveness. A key that expires between
        SCAN and MGET simply produces ``None`` and is ignored.
        """
        validate_service_name(service)

        try:
            keys = list(
                self._redis.scan_iter(
                    match=self._service_pattern(service),
                    count=self.scan_count,
                )
            )
            if not keys:
                return []
            values = self._redis.mget(keys)
        except self._redis_error as exc:
            raise DiscoveryError(
                f"failed to list instances for service {service!r} from Redis: {exc}"
            ) from exc

        # SCAN may return duplicates while the keyspace changes, so de-duplicate
        # by instance_id before returning a stable order.
        healthy: dict[str, ServiceRecord] = {}
        for value in values:
            record = self._deserialize(value)
            if record is None or record.service != service:
                continue
            healthy[record.instance_id] = record

        return sorted(healthy.values(), key=lambda item: item.instance_id)

    def resolve(self, service: str) -> ServiceRecord:
        instances = self.list_instances(service)
        if not instances:
            raise ServiceNotFoundError(
                f"no healthy instances are registered for service {service!r}"
            )

        with self._lock:
            index = self._round_robin[service] % len(instances)
            self._round_robin[service] += 1

        return instances[index]

    def close(self) -> None:
        """Close the Redis connection pool."""
        try:
            self._redis.close()
        except self._redis_error as exc:
            raise DiscoveryError(f"failed to close Redis discovery client: {exc}") from exc
