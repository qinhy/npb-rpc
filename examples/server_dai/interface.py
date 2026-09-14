from __future__ import annotations
"""Unified RPC + HTTP interface for the DepthAI camera service."""

from dataclasses import dataclass
from functools import cache
import inspect
import io
import logging
from typing import Any, Generic, Literal, Protocol, TypeVar, get_type_hints
import zipfile

import numpy as np
from npb import BinaryModel
from npb_rpc import DiscoveredRpcClient, FilesystemDiscovery, NngRpcClient, ZmqRpcClient

try:
    from .msg import (
        CameraCalibrationResponse,
        CameraCloseRequest,
        CameraControlResponse,
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraOpenRequest,
        CameraStatusResponse,
        EmptyRequest,
    )
    from .worker import CameraSupervisor
    
except ImportError:  # Support running files directly from this directory.
    from msg import (
        CameraCalibrationResponse,
        CameraCloseRequest,
        CameraControlResponse,
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraOpenRequest,
        CameraStatusResponse,
        EmptyRequest,
    )
    from worker import CameraSupervisor

LOG = logging.getLogger("dai_camera.interface")
STREAMS = ("rgb", "left", "right")
FRAME_KEYS = (*STREAMS, *(f"{name}.thumbnail" for name in STREAMS))
CLIENT_TYPES = {"nng": NngRpcClient, "zmq": ZmqRpcClient}

RequestT = TypeVar("RequestT", bound=BinaryModel)
ResponseT = TypeVar("ResponseT", bound=BinaryModel)
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
WebResponseKind = Literal["model", "jpeg", "zip"]


# ---------------------------------------------------------------------------
# Unified interface declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WebApi:
    method: HttpMethod
    path: str
    response: WebResponseKind = "model"

    def __post_init__(self) -> None:
        path = self.path.strip("/")
        if not path:
            raise ValueError("web API path must not be empty")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class RpcSpec:
    rpc: str
    web: WebApi | None = None


@dataclass(frozen=True, slots=True)
class ApiMethod(Generic[RequestT, ResponseT]):
    name: str
    rpc: str
    request: type[RequestT]
    response: type[ResponseT]
    web: WebApi | None = None


def api(
    rpc: str,
    http: HttpMethod | None = None,
    path: str | None = None,
    response: WebResponseKind = "model",
):
    if (http is None) != (path is None):
        raise ValueError("http and path must be supplied together")
    spec = RpcSpec(rpc, WebApi(http, path, response) if http and path else None)

    def decorate(fn):
        fn.__rpc_spec__ = spec
        return fn

    return decorate


class CameraInterface(Protocol):
    """Single source of truth for RPC, generated client, server, and HTTP."""

    service = "camera"

    @api("camera.open", "GET", "open")
    def open(self, request: CameraOpenRequest) -> CameraControlResponse: ...

    @api("camera.close", "GET", "close")
    def close(self, request: CameraCloseRequest) -> CameraControlResponse: ...

    @api("camera.status", "GET", "status")
    def status(self, request: EmptyRequest) -> CameraStatusResponse: ...

    @api("camera.frame", "GET", "frames", "zip")
    def frames(self, request: CameraFrameSetRequest) -> CameraFrameSetResponse: ...

    @api("camera.get_frame", "GET", "frame", "jpeg")
    def get_frame(self, request: CameraFrameRequest) -> CameraFrameResponse: ...

    @api("camera.get_calib", "GET", "get_calib")
    def get_calib(self, request: EmptyRequest) -> CameraCalibrationResponse: ...


# ---------------------------------------------------------------------------
# Camera service implementation
# ---------------------------------------------------------------------------

def _frame_fields(frames: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in FRAME_KEYS:
        name = key.replace(".", "_")
        frame = frames.get(key)
        fields[name] = np.frombuffer(frame.jpeg, dtype=np.uint8) if frame else np.empty(0, dtype=np.uint8)
        fields[f"{name}_sequence"] = frame.sequence if frame else 0
        fields[f"{name}_captured_ns"] = frame.captured_ns if frame else 0
    return fields


def _empty_frames(error: str) -> CameraFrameSetResponse:
    return CameraFrameSetResponse(
        ok=False,
        camera_online=False,
        generation=0,
        restart_count=0,
        error=error,
        **_frame_fields({}),
    )


def _frame_response(
    request: CameraFrameRequest,
    *,
    online: bool,
    frame: Any | None = None,
    error: str = "",
) -> CameraFrameResponse:
    return CameraFrameResponse(
        ok=frame is not None,
        camera_online=online,
        stream=request.stream,
        thumbnail=request.thumbnail,
        sequence=frame.sequence if frame else 0,
        captured_ns=frame.captured_ns if frame else 0,
        jpeg=np.frombuffer(frame.jpeg, dtype=np.uint8) if frame else np.empty(0, dtype=np.uint8),
        error=error,
    )


class CameraService(CameraInterface):
    """Typed interface implementation over the long-lived CameraSupervisor."""

    def __init__(self, camera: CameraSupervisor, logger: logging.Logger | None = None) -> None:
        self.camera = camera
        self.log = logger or LOG

    def _control(
        self,
        opening: bool,
        request: CameraOpenRequest | CameraCloseRequest,
    ) -> CameraControlResponse:
        try:
            if opening:
                result = self.camera.open_camera(request.device, timeout_s=request.timeout_s)  # type: ignore[attr-defined]
            else:
                result = self.camera.close_camera(timeout_s=request.timeout_s)
            ok, online, device, generation, error = result
            return CameraControlResponse(
                ok=ok,
                requested_open=opening,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            action = "open" if opening else "close"
            self.log.exception("camera.%s failed", action)
            device = str(getattr(request, "device", "")) if opening else ""
            return CameraControlResponse(
                ok=False,
                requested_open=opening,
                online=False,
                device=device,
                generation=0,
                error=f"{action} error: {type(exc).__name__}: {exc}",
            )

    def open(self, request: CameraOpenRequest) -> CameraControlResponse:
        return self._control(True, request)

    def close(self, request: CameraCloseRequest) -> CameraControlResponse:
        return self._control(False, request)

    def status(self, request: EmptyRequest) -> CameraStatusResponse:
        del request
        try:
            return self.camera.status()
        except Exception as exc:
            self.log.exception("camera.status failed")
            return CameraStatusResponse(
                requested_open=False,
                online=False,
                device="",
                generation=0,
                restart_count=0,
                frames_published=0,
                last_frame_ns=0,
                error=f"status error: {type(exc).__name__}: {exc}",
            )

    def frames(self, request: CameraFrameSetRequest) -> CameraFrameSetResponse:
        del request
        try:
            frames, online, generation, restarts, _published, _last_ns, camera_error = self.camera.snapshot_all()
            missing = [key for key in FRAME_KEYS if key not in frames]
            detail = "missing: " + ", ".join(missing) if missing else ""
            error = f"{camera_error}; {detail}" if camera_error and detail else camera_error or detail
            return CameraFrameSetResponse(
                ok=not missing,
                camera_online=online,
                generation=generation,
                restart_count=restarts,
                error=error if missing or not online else "",
                **_frame_fields(frames),
            )
        except Exception as exc:
            self.log.exception("camera.frame failed")
            return _empty_frames(f"frame error: {type(exc).__name__}: {exc}")

    def get_frame(self, request: CameraFrameRequest) -> CameraFrameResponse:
        try:
            status = self.camera.status()
            if request.stream not in STREAMS:
                return _frame_response(request, online=status.online, error=f"unknown stream: {request.stream!r}")
            frame = self.camera.get_frame(request.stream, request.thumbnail)
            if frame is None:
                return _frame_response(
                    request,
                    online=status.online,
                    error=status.error or "frame is not available yet",
                )
            return _frame_response(
                request,
                online=status.online,
                frame=frame,
                error=status.error if not status.online else "",
            )
        except Exception as exc:
            self.log.exception("camera.get_frame failed")
            return _frame_response(
                request,
                online=False,
                error=f"frame error: {type(exc).__name__}: {exc}",
            )

    def get_calib(self, request: EmptyRequest) -> CameraCalibrationResponse:
        del request
        try:
            return self.camera.get_calib()
        except Exception as exc:
            self.log.exception("camera.get_calib failed")
            return CameraCalibrationResponse.empty(error=str(exc))


@cache
def api_methods(interface: type) -> tuple[ApiMethod, ...]:
    result: list[ApiMethod] = []
    for name, fn in interface.__dict__.items():
        spec: RpcSpec | None = getattr(fn, "__rpc_spec__", None)
        if spec is None:
            continue

        params = list(inspect.signature(fn).parameters)
        if params != ["self", "request"]:
            raise TypeError(f"{interface.__name__}.{name} must be (self, request)")

        hints = get_type_hints(fn)
        request = hints.get("request")
        response = hints.get("return")
        if not isinstance(request, type) or not issubclass(request, BinaryModel):
            raise TypeError(f"{interface.__name__}.{name}: request must be a BinaryModel type")
        if not isinstance(response, type) or not issubclass(response, BinaryModel):
            raise TypeError(f"{interface.__name__}.{name}: return must be a BinaryModel type")

        result.append(ApiMethod(name, spec.rpc, request, response, spec.web))
    return tuple(result)


# ---------------------------------------------------------------------------
# Generic client generation
# ---------------------------------------------------------------------------


def _client_type(backend: str):
    try:
        return CLIENT_TYPES[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported RPC backend: {backend!r}") from exc


def resolve_service_instance(discovery: FilesystemDiscovery, service: str, server_name: str):
    matches = [x for x in discovery.list_instances(service) if x.instance_id == server_name]
    if len(matches) != 1:
        reason = "was not found" if not matches else "is ambiguous"
        raise RuntimeError(f"server {server_name!r} for service {service!r} {reason}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class RpcTarget:
    endpoint: str | None = None
    backend: str = "nng"
    discovery: FilesystemDiscovery | None = None
    service: str = ""
    server_name: str | None = None

    def call_raw(self, method: str, request: BinaryModel, response_type: type[ResponseT]) -> ResponseT:
        if self.endpoint is not None:
            with _client_type(self.backend).connect(self.endpoint) as client:
                return client.call(method, request, response_type)
        if self.discovery is None:
            raise ValueError("either endpoint or discovery must be supplied")
        if self.server_name is not None:
            target = resolve_service_instance(self.discovery, self.service, self.server_name)
            with _client_type(target.backend).connect(target.endpoint) as client:
                return client.call(method, request, response_type)
        if not self.service:
            raise ValueError("service is required when using discovery")
        with DiscoveredRpcClient(self.discovery) as client:
            return client.call(self.service, method, request, response_type)

    def call(self, method: ApiMethod[RequestT, ResponseT], request: RequestT) -> ResponseT:
        return self.call_raw(method.rpc, request, method.response)


def build_client_class(interface: type, name: str | None = None):
    service = interface.service

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        backend: str = "nng",
        discovery: FilesystemDiscovery | None = None,
        service: str = service,
        server_name: str | None = None,
    ) -> None:
        self.target = RpcTarget(endpoint, backend, discovery, service, server_name)

    namespace: dict[str, Any] = {"__init__": __init__}

    for method in api_methods(interface):
        def make_method(method: ApiMethod):
            def call(self, request):
                return self.target.call(method, request)

            call.__name__ = method.name
            call.__qualname__ = f"{name or interface.__name__ + 'Client'}.{method.name}"
            call.__annotations__ = {"request": method.request, "return": method.response}
            return call

        namespace[method.name] = make_method(method)

    return type(name or f"{interface.__name__}Client", (interface,), namespace)


CameraClient = build_client_class(CameraInterface, "CameraClient")


# ---------------------------------------------------------------------------
# Generic RPC registration
# ---------------------------------------------------------------------------


def add_rpc(server: Any, interface: type, implementation: Any) -> Any:
    """Register every decorated interface method on an NNG/ZMQ RPC server."""

    for method in api_methods(interface):
        fn = getattr(implementation, method.name)

        def make_handler(method: ApiMethod, fn: Any):
            def handler(request, context):
                del context
                return fn(request)

            handler.__name__ = f"rpc_{method.name}"
            handler.__annotations__ = {"request": method.request, "return": method.response}
            return handler

        server.method(
            method.rpc,
            request=method.request,
            response=method.response,
        )(make_handler(method, fn))

    return server


# ---------------------------------------------------------------------------
# Generic FastAPI generation
# ---------------------------------------------------------------------------


def add_routes(
    app: Any,
    interface: type,
    *,
    endpoint: str | None = None,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str | None = None,
    server_name: str | None = None,
    route_prefix: str | None = None,
    route_name_prefix: str = "",
    tag: str | None = None,
):
    try:
        from fastapi import Depends, HTTPException, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is required only when add_routes() is used") from exc

    service = service or interface.service
    Client = build_client_class(interface)
    client = Client(
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )
    route_prefix = route_prefix or f"/{service}" + (f"/{server_name}" if server_name else "")
    route_prefix = "/" + route_prefix.strip("/")
    route_tag = tag or (f"{service}:{server_name}" if server_name else service)
    instance_name = server_name or "discovered"

    def call_rpc(fn: Any):
        try:
            return fn()
        except RuntimeError as exc:
            message = str(exc)
            if "was not found" in message:
                raise HTTPException(404, message) from exc
            if "ambiguous" in message:
                raise HTTPException(409, message) from exc
            raise HTTPException(502, f"RPC failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(502, f"RPC failed: {exc}") from exc

    def encode(result: BinaryModel, kind: WebResponseKind):
        if kind == "model":
            return result
        if kind == "jpeg":
            if not getattr(result, "ok", True):
                raise HTTPException(503, getattr(result, "error", "frame unavailable"))
            return Response(result.jpeg.tobytes(), media_type="image/jpeg")
        if kind == "zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as archive:
                for field in type(result).model_fields:
                    value = getattr(result, field)
                    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 1 and value.size:
                        archive.writestr(f"{field}.jpg", value.tobytes())
            return Response(buf.getvalue(), media_type="application/zip")
        raise ValueError(f"unsupported web response kind: {kind!r}")

    def make_route(method: ApiMethod, client_method: Any):
        web = method.web
        assert web is not None

        def invoke(request: BinaryModel):
            return encode(call_rpc(lambda: client_method(request)), web.response)

        if web.method == "GET" and not method.request.model_fields:
            def route():
                return invoke(method.request())
        elif web.method == "GET":
            def route(request=Depends(method.request)):
                return invoke(request)
            route.__annotations__ = {"request": method.request}
        else:
            def route(request):
                return invoke(request)
            route.__annotations__ = {"request": method.request}

        route.__name__ = f"web_{method.name}"
        if web.response == "model":
            route.__annotations__["return"] = method.response
        return route

    for method in api_methods(interface):
        web = method.web
        if web is None:
            continue
        route = make_route(method, getattr(client, method.name))
        media_type = {"jpeg": "image/jpeg", "zip": "application/zip"}.get(web.response)
        responses = {200: {"content": {media_type: {}}}} if media_type else None
        app.add_api_route(
            f"{route_prefix}/{web.path}",
            route,
            methods=[web.method],
            tags=[route_tag],
            name=f"{route_name_prefix}{service}:{instance_name}:{web.path}",
            response_model=method.response if web.response == "model" else None,
            responses=responses,
        )

    return client


def add_camera_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_routes(app, CameraInterface, **kwargs)
