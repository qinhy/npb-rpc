from __future__ import annotations

import time

import pytest

from npb_rpc import FilesystemDiscovery, ServiceNotFoundError, ServiceRecord


def record(
    *,
    service: str = "sum",
    instance_id: str = "one",
    endpoint: str = "tcp://127.0.0.1:5555",
    backend: str = "zmq",
    heartbeat_at: float | None = None,
) -> ServiceRecord:
    now = time.time() if heartbeat_at is None else heartbeat_at
    return ServiceRecord(
        service=service,
        instance_id=instance_id,
        endpoint=endpoint,
        backend=backend,
        pid=123,
        hostname="test",
        started_at=now - 10,
        heartbeat_at=now,
        metadata={"zone": "local"},
    )


def test_register_resolve_and_unregister(tmp_path) -> None:
    discovery = FilesystemDiscovery(tmp_path, heartbeat_timeout=5)
    value = record()

    discovery.register(value)
    resolved = discovery.resolve("sum")

    assert resolved == value
    discovery.unregister("sum", "one")
    with pytest.raises(ServiceNotFoundError):
        discovery.resolve("sum")


def test_stale_record_is_not_resolved_and_is_pruned(tmp_path) -> None:
    discovery = FilesystemDiscovery(tmp_path, heartbeat_timeout=1)
    discovery.register(record(heartbeat_at=time.time() - 20))

    with pytest.raises(ServiceNotFoundError):
        discovery.resolve("sum")

    assert list(tmp_path.rglob("*.json")) == []


def test_multiple_instances_are_resolved_round_robin(tmp_path) -> None:
    discovery = FilesystemDiscovery(tmp_path, heartbeat_timeout=5)
    discovery.register(record(instance_id="one"))
    discovery.register(
        record(
            instance_id="two",
            endpoint="ipc://npb-rpc-sum",
            backend="nng",
        )
    )

    assert discovery.resolve("sum").instance_id == "one"
    assert discovery.resolve("sum").instance_id == "two"
    assert discovery.resolve("sum").instance_id == "one"


def test_list_services_returns_only_services_with_healthy_instances(tmp_path) -> None:
    discovery = FilesystemDiscovery(tmp_path, heartbeat_timeout=5)
    discovery.register(record(service="sum"))
    discovery.register(record(service="math", instance_id="two"))
    discovery.register(
        record(
            service="stale",
            instance_id="old",
            heartbeat_at=time.time() - 20,
        )
    )

    assert discovery.list_services() == ["math", "sum"]


@pytest.mark.parametrize("service", ["", "../escape", "has spaces", "/absolute"])
def test_invalid_service_name_is_rejected(tmp_path, service: str) -> None:
    discovery = FilesystemDiscovery(tmp_path)
    with pytest.raises(ValueError):
        discovery.list_instances(service)
