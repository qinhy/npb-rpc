from __future__ import annotations

from typing import TypeVar

from npb import BinaryModel
from npb_rpc import (
    DiscoveredRpcClient,
    FilesystemDiscovery,
    NngRpcClient,
    ZmqRpcClient,
)

try:
    from .msg import (
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraStatusResponse,
        EmptyRequest,
    )
except ImportError:  # Support running the files directly from one directory.
    from msg import (
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraStatusResponse,
        EmptyRequest,
    )


ResponseT = TypeVar("ResponseT", bound=BinaryModel)


def _client_type(backend: str):
    if backend == "nng":
        return NngRpcClient
    if backend == "zmq":
        return ZmqRpcClient
    raise ValueError(f"unsupported RPC backend: {backend!r}")


def resolve_service_instance(
    discovery: FilesystemDiscovery,
    service: str,
    server_name: str,
):
    """Resolve one exact healthy service instance from the discovery registry."""
    matches = [
        instance
        for instance in discovery.list_instances(service)
        if instance.instance_id == server_name
    ]
    if not matches:
        raise RuntimeError(
            f"server {server_name!r} for service {service!r} was not found"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"server {server_name!r} for service {service!r} is ambiguous"
        )
    return matches[0]


def _call(
    method: str,
    request: BinaryModel,
    response_type: type[ResponseT],
    *,
    endpoint: str | None,
    backend: str,
    discovery: FilesystemDiscovery | None,
    service: str,
    server_name: str | None,
) -> ResponseT:
    """Call either a direct endpoint or a service discovered from the registry."""
    if endpoint is not None:
        client_type = _client_type(backend)
        with client_type.connect(endpoint) as client:
            return client.call(method, request, response_type)

    if discovery is None:
        raise ValueError("either endpoint or discovery must be supplied")

    if server_name is not None:
        instance = resolve_service_instance(discovery, service, server_name)
        client_type = _client_type(instance.backend)
        with client_type.connect(instance.endpoint) as client:
            return client.call(method, request, response_type)

    with DiscoveredRpcClient(discovery) as client:
        return client.call(service, method, request, response_type)


def client_camera_status(
    endpoint: str | None = None,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = "camera",
    server_name: str | None = None,
) -> CameraStatusResponse:
    return _call(
        "camera.status",
        EmptyRequest(),
        CameraStatusResponse,
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )


def client_camera_frame(
    request: CameraFrameRequest,
    endpoint: str | None = None,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = "camera",
    server_name: str | None = None,
) -> CameraFrameResponse:
    return _call(
        "camera.get_frame",
        request,
        CameraFrameResponse,
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )


def client_camera_frame_set(
    request: CameraFrameSetRequest,
    endpoint: str | None = None,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = "camera",
    server_name: str | None = None,
) -> CameraFrameSetResponse:
    return _call(
        "camera.frame",
        request,
        CameraFrameSetResponse,
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )
