from __future__ import annotations

"""Public RPC + HTTP interface for the DepthAI camera service."""

from dataclasses import dataclass
import io
import logging
from typing import Any, Generic, Literal, Protocol, TypeVar
import zipfile

import numpy as np
from npb import BinaryModel
from npb_rpc import DiscoveredRpcClient, FilesystemDiscovery, NngRpcClient, RpcContext, ZmqRpcClient

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
except ImportError:  # Allow running the files directly.
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

LOG = logging.getLogger("dai_camera.interface")
STREAMS = ("rgb", "left", "right")
FRAME_KEYS = (*STREAMS, *(f"{name}.thumbnail" for name in STREAMS))
CLIENT_TYPES = {"nng": NngRpcClient, "zmq": ZmqRpcClient}

RequestT = TypeVar("RequestT", bound=BinaryModel)
ResponseT = TypeVar("ResponseT", bound=BinaryModel)
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
WebResponseKind = Literal["model", "jpeg", "zip"]


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
class ApiMethod(Generic[RequestT, ResponseT]):
    rpc: str
    request: type[RequestT]
    response: type[ResponseT]
    web: WebApi | None = None

    def server_method(self, server: Any):
        return server.method(self.rpc, request=self.request, response=self.response)


class CameraApiInterface:
    service = "camera"
    open = ApiMethod("camera.open", CameraOpenRequest, CameraControlResponse, WebApi("POST", "open"))
    close = ApiMethod("camera.close", CameraCloseRequest, CameraControlResponse, WebApi("POST", "close"))
    status = ApiMethod("camera.status", EmptyRequest, CameraStatusResponse, WebApi("GET", "status"))
    frame = ApiMethod("camera.frame", CameraFrameSetRequest, CameraFrameSetResponse, WebApi("GET", "frames", "zip"))
    get_frame = ApiMethod("camera.get_frame", CameraFrameRequest, CameraFrameResponse, WebApi("GET", "frame", "jpeg"))
    get_calib = ApiMethod("camera.get_calib", EmptyRequest, CameraCalibrationResponse, WebApi("GET", "get_calib"))

    @classmethod
    def methods(cls) -> tuple[ApiMethod, ...]:
        return cls.open, cls.close, cls.status, cls.frame, cls.get_frame, cls.get_calib

    @classmethod
    def web_methods(cls) -> tuple[ApiMethod, ...]:
        return tuple(method for method in cls.methods() if method.web is not None)

    @classmethod
    def by_rpc(cls, name: str) -> ApiMethod:
        for method in cls.methods():
            if method.rpc == name:
                return method
        raise KeyError(f"unknown {cls.service!r} RPC method: {name!r}")

    @classmethod
    def by_web_path(cls, path: str) -> ApiMethod:
        path = path.strip("/")
        for method in cls.web_methods():
            if method.web and method.web.path == path:
                return method
        raise KeyError(f"unknown {cls.service!r} web API path: {path!r}")

    def client(self, **kwargs: Any) -> "CameraClient":
        kwargs.setdefault("service", self.service)
        return CameraClient(**kwargs)

    def add_rpc(self, server: Any, camera: "CameraBackend", **kwargs: Any) -> Any:
        return add_camera_rpc(server, camera, **kwargs)

    def add_routes(self, app: Any, **kwargs: Any) -> "CameraClient":
        kwargs.setdefault("service", self.service)
        return add_camera_routes(app, **kwargs)


CAMERA_API = CameraApiInterface()


# ---------------------------------------------------------------------------
# Client
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
    service: str = CAMERA_API.service
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
        with DiscoveredRpcClient(self.discovery) as client:
            return client.call(self.service, method, request, response_type)

    def call(self, method: ApiMethod[RequestT, ResponseT], request: RequestT) -> ResponseT:
        return self.call_raw(method.rpc, request, method.response)


class CameraClient:
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        backend: str = "nng",
        discovery: FilesystemDiscovery | None = None,
        service: str = CAMERA_API.service,
        server_name: str | None = None,
    ) -> None:
        self.target = RpcTarget(endpoint, backend, discovery, service, server_name)

    def call(self, method: ApiMethod[RequestT, ResponseT], request: RequestT) -> ResponseT:
        return self.target.call(method, request)

    def open(self, device: str = "", *, timeout_s: float = 10.0) -> CameraControlResponse:
        return self.call(CAMERA_API.open, CameraOpenRequest(device=device, timeout_s=timeout_s))

    def close(self, *, timeout_s: float = 5.0) -> CameraControlResponse:
        return self.call(CAMERA_API.close, CameraCloseRequest(timeout_s=timeout_s))

    def status(self) -> CameraStatusResponse:
        return self.call(CAMERA_API.status, EmptyRequest())

    def frames(self) -> CameraFrameSetResponse:
        return self.call(CAMERA_API.frame, CameraFrameSetRequest())

    def get_frame(self, stream: str = "rgb", *, thumbnail: bool = False) -> CameraFrameResponse:
        return self.call(CAMERA_API.get_frame, CameraFrameRequest(stream=stream, thumbnail=thumbnail))

    def get_calib(self) -> CameraCalibrationResponse:
        return self.call(CAMERA_API.get_calib, EmptyRequest())


# ---------------------------------------------------------------------------
# RPC server
# ---------------------------------------------------------------------------


class CameraBackend(Protocol):
    def open_camera(self, device: str = "", *, timeout_s: float = 10.0) -> tuple[bool, bool, str, int, str]: ...
    def close_camera(self, *, timeout_s: float = 5.0) -> tuple[bool, bool, str, int, str]: ...
    def status(self) -> CameraStatusResponse: ...
    def snapshot_all(self) -> tuple[dict[str, Any], bool, int, int, int, int, str]: ...
    def get_frame(self, stream: str, thumbnail: bool) -> Any | None: ...
    def get_calib(self) -> CameraCalibrationResponse: ...


def _frame_fields(frames: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in FRAME_KEYS:
        name = key.replace(".", "_")
        frame = frames.get(key)
        fields[name] = np.frombuffer(frame.jpeg, dtype=np.uint8) if frame else np.empty(0, dtype=np.uint8)
        fields[f"{name}_sequence"] = frame.sequence if frame else 0
        fields[f"{name}_captured_ns"] = frame.captured_ns if frame else 0
    return fields


def _empty_frame_set_response(error: str) -> CameraFrameSetResponse:
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


def add_camera_rpc(server: Any, camera: CameraBackend, *, logger: logging.Logger | None = None) -> Any:
    log = logger or LOG

    def control(opening: bool, request: CameraOpenRequest | CameraCloseRequest) -> CameraControlResponse:
        try:
            if opening:
                result = camera.open_camera(request.device, timeout_s=request.timeout_s)  # type: ignore[attr-defined]
            else:
                result = camera.close_camera(timeout_s=request.timeout_s)
            ok, online, device, generation, error = result
            return CameraControlResponse(ok=ok, requested_open=opening, online=online, device=device, generation=generation, error=error)
        except Exception as exc:
            action = "open" if opening else "close"
            log.exception("camera.%s handler failed", action)
            device = str(getattr(request, "device", "")) if opening else ""
            return CameraControlResponse(
                ok=False,
                requested_open=opening,
                online=False,
                device=device,
                generation=0,
                error=f"{action} handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.open.server_method(server)
    def camera_open(request: CameraOpenRequest, context: RpcContext) -> CameraControlResponse:
        del context
        return control(True, request)

    @CAMERA_API.close.server_method(server)
    def camera_close(request: CameraCloseRequest, context: RpcContext) -> CameraControlResponse:
        del context
        return control(False, request)

    @CAMERA_API.status.server_method(server)
    def camera_status(request: EmptyRequest, context: RpcContext) -> CameraStatusResponse:
        del request, context
        try:
            return camera.status()
        except Exception as exc:
            log.exception("camera.status handler failed")
            return CameraStatusResponse(
                requested_open=False,
                online=False,
                device="",
                generation=0,
                restart_count=0,
                frames_published=0,
                last_frame_ns=0,
                error=f"status handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.frame.server_method(server)
    def camera_frame(request: CameraFrameSetRequest, context: RpcContext) -> CameraFrameSetResponse:
        del request, context
        try:
            frames, online, generation, restarts, _published, _last_ns, camera_error = camera.snapshot_all()
            missing = [key for key in FRAME_KEYS if key not in frames]
            if missing:
                detail = "missing: " + ", ".join(missing)
                error = f"{camera_error}; {detail}" if camera_error else detail
            else:
                error = camera_error if not online else ""
            return CameraFrameSetResponse(
                ok=not missing,
                camera_online=online,
                generation=generation,
                restart_count=restarts,
                error=error,
                **_frame_fields(frames),
            )
        except Exception as exc:
            log.exception("camera.frame handler failed")
            return _empty_frame_set_response(f"frame handler error: {type(exc).__name__}: {exc}")

    @CAMERA_API.get_frame.server_method(server)
    def camera_get_frame(request: CameraFrameRequest, context: RpcContext) -> CameraFrameResponse:
        del context
        try:
            status = camera.status()
            if request.stream not in STREAMS:
                return _frame_response(request, online=status.online, error=f"unknown stream: {request.stream!r}")
            frame = camera.get_frame(request.stream, request.thumbnail)
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
            log.exception("camera.get_frame handler failed")
            safe_request = CameraFrameRequest(
                stream=str(getattr(request, "stream", "")),
                thumbnail=bool(getattr(request, "thumbnail", False)),
            )
            return _frame_response(
                safe_request,
                online=False,
                error=f"frame handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.get_calib.server_method(server)
    def camera_get_calib(request: EmptyRequest, context: RpcContext) -> CameraCalibrationResponse:
        del request, context
        try:
            return camera.get_calib()
        except Exception as exc:
            log.exception("camera.get_calib handler failed")
            return CameraCalibrationResponse.empty(error=str(exc))

    return server


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


def add_camera_routes(
    app: Any,
    *,
    endpoint: str | None = None,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = CAMERA_API.service,
    server_name: str | None = None,
    route_prefix: str | None = None,
    route_name_prefix: str = "",
    tag: str | None = None,
    default_device: str = "169.254.1.222",
) -> CameraClient:
    try:
        from fastapi import HTTPException, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is required only when add_camera_routes() is used") from exc

    client = CameraClient(
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
    content_types = {"jpeg": "image/jpeg", "zip": "application/zip"}

    def web_call(func: Any):
        try:
            return func()
        except RuntimeError as exc:
            message = str(exc)
            if "was not found" in message:
                raise HTTPException(404, message) from exc
            if "ambiguous" in message:
                raise HTTPException(409, message) from exc
            raise HTTPException(502, f"RPC failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(502, f"RPC failed: {exc}") from exc

    def add_route(api_method: ApiMethod, func: Any) -> None:
        web = api_method.web
        if web is None:
            raise ValueError(f"{api_method.rpc!r} has no HTTP exposure")
        media_type = content_types.get(web.response)
        responses = {200: {"content": {media_type: {}}}} if media_type else None
        app.add_api_route(
            f"{route_prefix}/{web.path}",
            func,
            methods=[web.method],
            tags=[route_tag],
            name=f"{route_name_prefix}{service}:{instance_name}:{web.path}",
            responses=responses,
        )

    def open_camera(device: str = default_device, timeout_s: float = 10.0):
        return web_call(lambda: client.open(device, timeout_s=timeout_s))

    def close_camera(timeout_s: float = 5.0):
        return web_call(lambda: client.close(timeout_s=timeout_s))

    def status():
        return web_call(client.status)

    def frame(stream: str = "rgb", thumbnail: bool = False):
        result = web_call(lambda: client.get_frame(stream, thumbnail=thumbnail))
        if not result.ok:
            raise HTTPException(503, result.error)
        return Response(result.jpeg.tobytes(), media_type="image/jpeg")

    def frames():
        result = web_call(client.frames)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for key in FRAME_KEYS:
                name = key.replace(".", "_")
                image = getattr(result, name)
                if image.size:
                    filename = f"{name}.jpg"
                    archive.writestr(filename, image.tobytes())
        return Response(buf.getvalue(), media_type="application/zip")

    def get_calib():
        return web_call(client.get_calib)

    handlers = {
        CAMERA_API.open: open_camera,
        CAMERA_API.close: close_camera,
        CAMERA_API.status: status,
        CAMERA_API.get_frame: frame,
        CAMERA_API.frame: frames,
        CAMERA_API.get_calib: get_calib,
    }
    for method, handler in handlers.items():
        add_route(method, handler)
    return client
