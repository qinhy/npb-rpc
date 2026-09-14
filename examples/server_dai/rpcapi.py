from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from npb import BinaryModel
from npb_rpc import DiscoveredRpcClient, FilesystemDiscovery, NngRpcClient, ZmqRpcClient

try:
    from . import msg as _msg
    from .interface import ApiMethod, CAMERA_API
except ImportError:  # Support running the files directly from one directory.
    import msg as _msg
    from interface import ApiMethod, CAMERA_API


# Backward compatibility: keep the message classes importable from rpcapi.py.
CameraCloseRequest = _msg.CameraCloseRequest
CameraControlResponse = _msg.CameraControlResponse
CameraFrameRequest = _msg.CameraFrameRequest
CameraFrameResponse = _msg.CameraFrameResponse
CameraFrameSetRequest = _msg.CameraFrameSetRequest
CameraFrameSetResponse = _msg.CameraFrameSetResponse
CameraOpenRequest = _msg.CameraOpenRequest
CameraStatusResponse = _msg.CameraStatusResponse
CameraCalibrationResponse = _msg.CameraCalibrationResponse
EmptyRequest = _msg.EmptyRequest

RequestT = TypeVar("RequestT", bound=BinaryModel)
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
        item
        for item in discovery.list_instances(service)
        if item.instance_id == server_name
    ]
    if len(matches) != 1:
        reason = "was not found" if not matches else "is ambiguous"
        raise RuntimeError(
            f"server {server_name!r} for service {service!r} {reason}"
        )
    return matches[0]


def _call_raw(
    method: str,
    request: BinaryModel,
    response_type: type[ResponseT],
    *,
    endpoint: str | None = None,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = CAMERA_API.service,
    server_name: str | None = None,
) -> ResponseT:
    """Call a raw RPC method through a direct endpoint or service discovery."""
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


def _call(
    method: ApiMethod[RequestT, ResponseT],
    request: RequestT,
    *,
    endpoint: str | None = None,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = CAMERA_API.service,
    server_name: str | None = None,
) -> ResponseT:
    """Call a typed API method using its central interface definition."""
    return _call_raw(
        method.rpc,
        request,
        method.response,
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )


@dataclass(frozen=True, slots=True)
class RpcTarget:
    """Reusable RPC destination shared by CLI/web/application clients."""

    endpoint: str | None = None
    backend: str = "nng"
    discovery: FilesystemDiscovery | None = None
    service: str = CAMERA_API.service
    server_name: str | None = None

    def call(
        self,
        method: ApiMethod[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        return _call(
            method,
            request,
            endpoint=self.endpoint,
            backend=self.backend,
            discovery=self.discovery,
            service=self.service,
            server_name=self.server_name,
        )

    def call_raw(
        self,
        method: str,
        request: BinaryModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        """Escape hatch for services that do not yet have an ApiMethod contract."""
        return _call_raw(
            method,
            request,
            response_type,
            endpoint=self.endpoint,
            backend=self.backend,
            discovery=self.discovery,
            service=self.service,
            server_name=self.server_name,
        )


def client_camera_open(
    request: CameraOpenRequest,
    endpoint: str | None = None,
    **rpc,
) -> CameraControlResponse:
    """Open (or switch to) a DepthAI device, optionally by IP/DeviceID/USB path."""
    return _call(CAMERA_API.open, request, endpoint=endpoint, **rpc)


def client_camera_close(
    request: CameraCloseRequest | None = None,
    endpoint: str | None = None,
    **rpc,
) -> CameraControlResponse:
    """Close the active camera while leaving the RPC service available."""
    return _call(
        CAMERA_API.close,
        request or CameraCloseRequest(),
        endpoint=endpoint,
        **rpc,
    )


def client_camera_status(
    endpoint: str | None = None,
    **rpc,
) -> CameraStatusResponse:
    return _call(
        CAMERA_API.status,
        EmptyRequest(),
        endpoint=endpoint,
        **rpc,
    )


def client_camera_frame(
    request: CameraFrameRequest,
    endpoint: str | None = None,
    **rpc,
) -> CameraFrameResponse:
    return _call(
        CAMERA_API.get_frame,
        request,
        endpoint=endpoint,
        **rpc,
    )


def client_camera_frame_set(
    request: CameraFrameSetRequest,
    endpoint: str | None = None,
    **rpc,
) -> CameraFrameSetResponse:
    return _call(
        CAMERA_API.frame,
        request,
        endpoint=endpoint,
        **rpc,
    )


def client_camera_get_calib(
    endpoint: str | None = None,
    **rpc,
) -> CameraCalibrationResponse:
    return _call(
        CAMERA_API.get_calib,
        EmptyRequest(),
        endpoint=endpoint,
        **rpc,
    )
