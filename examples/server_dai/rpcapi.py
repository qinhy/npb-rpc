from __future__ import annotations

from typing import TypeVar

from npb import BinaryModel
from npb_rpc import DiscoveredRpcClient, FilesystemDiscovery, NngRpcClient, ZmqRpcClient

try:
    from . import msg as _msg
except ImportError:  # Support running the files directly from one directory.
    import msg as _msg

# Keep the original message classes available from this module.
CameraCloseRequest = _msg.CameraCloseRequest
CameraControlResponse = _msg.CameraControlResponse
CameraFrameRequest = _msg.CameraFrameRequest
CameraFrameResponse = _msg.CameraFrameResponse
CameraFrameSetRequest = _msg.CameraFrameSetRequest
CameraFrameSetResponse = _msg.CameraFrameSetResponse
CameraOpenRequest = _msg.CameraOpenRequest
CameraStatusResponse = _msg.CameraStatusResponse
EmptyRequest = _msg.EmptyRequest

ResponseT = TypeVar("ResponseT", bound=BinaryModel)


def _client_type(backend: str):
    if backend == "nng":
        return NngRpcClient
    if backend == "zmq":
        return ZmqRpcClient
    raise ValueError(f"unsupported RPC backend: {backend!r}")


def resolve_service_instance(discovery: FilesystemDiscovery, service: str, server_name: str):
    """Resolve one exact healthy service instance from the discovery registry."""
    matches = [
        item for item in discovery.list_instances(service) if item.instance_id == server_name
    ]
    if len(matches) != 1:
        reason = "was not found" if not matches else "is ambiguous"
        raise RuntimeError(f"server {server_name!r} for service {service!r} {reason}")
    return matches[0]


def _call(
    method: str,
    request: BinaryModel,
    response_type: type[ResponseT],
    *,
    endpoint: str | None = None,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = "camera",
    server_name: str | None = None,
) -> ResponseT:
    """Call a direct endpoint or a service discovered from the registry."""
    if endpoint is not None:
        with _client_type(backend).connect(endpoint) as client:
            return client.call(method, request, response_type)

    if discovery is None:
        raise ValueError("either endpoint or discovery must be supplied")

    if server_name is not None:
        instance = resolve_service_instance(discovery, service, server_name)
        with _client_type(instance.backend).connect(instance.endpoint) as client:
            return client.call(method, request, response_type)

    with DiscoveredRpcClient(discovery) as client:
        return client.call(service, method, request, response_type)


def client_camera_open(
    request: CameraOpenRequest, endpoint: str | None = None, **rpc
) -> CameraControlResponse:
    """Open (or switch to) a DepthAI device, optionally by IP/DeviceID/USB path."""
    return _call("camera.open", request, CameraControlResponse, endpoint=endpoint, **rpc)


def client_camera_close(
    request: CameraCloseRequest | None = None, endpoint: str | None = None, **rpc
) -> CameraControlResponse:
    """Close the active camera while leaving the RPC service available."""
    return _call(
        "camera.close", request or CameraCloseRequest(), CameraControlResponse,
        endpoint=endpoint, **rpc
    )


def client_camera_status(
    endpoint: str | None = None, **rpc
) -> CameraStatusResponse:
    return _call("camera.status", EmptyRequest(), CameraStatusResponse, endpoint=endpoint, **rpc)


def client_camera_frame(
    request: CameraFrameRequest, endpoint: str | None = None, **rpc
) -> CameraFrameResponse:
    return _call("camera.get_frame", request, CameraFrameResponse, endpoint=endpoint, **rpc)


def client_camera_frame_set(
    request: CameraFrameSetRequest, endpoint: str | None = None, **rpc
) -> CameraFrameSetResponse:
    return _call("camera.frame", request, CameraFrameSetResponse, endpoint=endpoint, **rpc)
