from __future__ import annotations

"""Integrated public interface for the DepthAI camera service.

This module is intentionally the only public integration point for the service.
It owns:

* RPC method names + request/response schemas
* RPC client transport/discovery
* typed CameraClient calls
* RPC handler registration
* FastAPI route registration

Implementation details stay in worker.py.  msg.py continues to own the wire models.
"""

from dataclasses import dataclass
import io
import logging
from typing import Any, Generic, Literal, Protocol, TypeVar
import zipfile

import numpy as np
from npb import BinaryModel
from npb_rpc import (
    DiscoveredRpcClient,
    FilesystemDiscovery,
    NngRpcClient,
    RpcContext,
    ZmqRpcClient,
)

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


LOG = logging.getLogger("dai_camera.interface")
VALID_STREAMS = ("rgb", "left", "right")

RequestT = TypeVar("RequestT", bound=BinaryModel)
ResponseT = TypeVar("ResponseT", bound=BinaryModel)

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
WebResponseKind = Literal["model", "jpeg", "zip"]


@dataclass(frozen=True, slots=True)
class WebApi:
    """HTTP exposure metadata for one API operation."""

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
    """One authoritative RPC operation and its optional HTTP exposure."""

    rpc: str
    request: type[RequestT]
    response: type[ResponseT]
    web: WebApi | None = None

    def server_method(self, server: Any):
        """Return the npb_rpc decorator for this operation."""
        return server.method(
            self.rpc,
            request=self.request,
            response=self.response,
        )


class CameraApiInterface:
    """Complete public contract and integration facade for the camera service."""

    service = "camera"

    open = ApiMethod(
        rpc="camera.open",
        request=CameraOpenRequest,
        response=CameraControlResponse,
        web=WebApi(method="POST", path="open"),
    )

    close = ApiMethod(
        rpc="camera.close",
        request=CameraCloseRequest,
        response=CameraControlResponse,
        web=WebApi(method="POST", path="close"),
    )

    status = ApiMethod(
        rpc="camera.status",
        request=EmptyRequest,
        response=CameraStatusResponse,
        web=WebApi(method="GET", path="status"),
    )

    # One complete latest snapshot: RGB + left + right + all thumbnails.
    frame = ApiMethod(
        rpc="camera.frame",
        request=CameraFrameSetRequest,
        response=CameraFrameSetResponse,
        web=WebApi(method="GET", path="frames", response="zip"),
    )

    # One selected image.
    get_frame = ApiMethod(
        rpc="camera.get_frame",
        request=CameraFrameRequest,
        response=CameraFrameResponse,
        web=WebApi(method="GET", path="frame", response="jpeg"),
    )

    get_calib = ApiMethod(
        rpc="camera.get_calib",
        request=EmptyRequest,
        response=CameraCalibrationResponse,
        web=WebApi(method="GET", path="get_calib"),
    )

    @classmethod
    def methods(cls) -> tuple[ApiMethod, ...]:
        return (
            cls.open,
            cls.close,
            cls.status,
            cls.frame,
            cls.get_frame,
            cls.get_calib,
        )

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
        normalized = path.strip("/")
        for method in cls.web_methods():
            assert method.web is not None
            if method.web.path == normalized:
                return method
        raise KeyError(f"unknown {cls.service!r} web API path: {path!r}")

    def client(self, **target_kwargs: Any) -> "CameraClient":
        """Create a typed client for this service."""
        target_kwargs.setdefault("service", self.service)
        return CameraClient(**target_kwargs)

    def add_rpc(self, server: Any, camera: "CameraBackend", **kwargs: Any) -> Any:
        """Register every camera RPC handler on an npb_rpc server."""
        return add_camera_rpc(server, camera, **kwargs)

    def add_routes(self, app: Any, **kwargs: Any) -> "CameraClient":
        """Register every camera HTTP route on a FastAPI app."""
        kwargs.setdefault("service", self.service)
        return add_camera_routes(app, **kwargs)


CAMERA_API = CameraApiInterface()


# ---------------------------------------------------------------------------
# RPC client side
# ---------------------------------------------------------------------------


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
    """Resolve one exact healthy instance from FilesystemDiscovery."""
    matches = [
        instance
        for instance in discovery.list_instances(service)
        if instance.instance_id == server_name
    ]
    if len(matches) != 1:
        reason = "was not found" if not matches else "is ambiguous"
        raise RuntimeError(
            f"server {server_name!r} for service {service!r} {reason}"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class RpcTarget:
    """Reusable direct/discovered RPC destination.

    Shared transport/discovery target reusable by camera and other services.
    """

    endpoint: str | None = None
    backend: str = "nng"
    discovery: FilesystemDiscovery | None = None
    service: str = CAMERA_API.service
    server_name: str | None = None

    def call_raw(
        self,
        method: str,
        request: BinaryModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        if self.endpoint is not None:
            with _client_type(self.backend).connect(self.endpoint) as client:
                return client.call(method, request, response_type)

        if self.discovery is None:
            raise ValueError("either endpoint or discovery must be supplied")

        if self.server_name is not None:
            instance = resolve_service_instance(
                self.discovery,
                self.service,
                self.server_name,
            )
            with _client_type(instance.backend).connect(instance.endpoint) as client:
                return client.call(method, request, response_type)

        with DiscoveredRpcClient(self.discovery) as client:
            return client.call(
                self.service,
                method,
                request,
                response_type,
            )

    def call(
        self,
        method: ApiMethod[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        return self.call_raw(
            method.rpc,
            request,
            method.response,
        )


class CameraClient:
    """Typed camera client built directly from the shared API contract."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        backend: str = "nng",
        discovery: FilesystemDiscovery | None = None,
        service: str = CAMERA_API.service,
        server_name: str | None = None,
    ) -> None:
        self.target = RpcTarget(
            endpoint=endpoint,
            backend=backend,
            discovery=discovery,
            service=service,
            server_name=server_name,
        )

    def call(
        self,
        method: ApiMethod[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        return self.target.call(method, request)

    def open(
        self,
        device: str = "",
        *,
        timeout_s: float = 10.0,
    ) -> CameraControlResponse:
        return self.call(
            CAMERA_API.open,
            CameraOpenRequest(device=device, timeout_s=timeout_s),
        )

    def close(self, *, timeout_s: float = 5.0) -> CameraControlResponse:
        return self.call(
            CAMERA_API.close,
            CameraCloseRequest(timeout_s=timeout_s),
        )

    def status(self) -> CameraStatusResponse:
        return self.call(
            CAMERA_API.status,
            EmptyRequest(),
        )

    def frames(self) -> CameraFrameSetResponse:
        return self.call(
            CAMERA_API.frame,
            CameraFrameSetRequest(),
        )

    def get_frame(
        self,
        stream: str = "rgb",
        *,
        thumbnail: bool = False,
    ) -> CameraFrameResponse:
        return self.call(
            CAMERA_API.get_frame,
            CameraFrameRequest(
                stream=stream,
                thumbnail=thumbnail,
            ),
        )

    def get_calib(self) -> CameraCalibrationResponse:
        return self.call(
            CAMERA_API.get_calib,
            EmptyRequest(),
        )


# ---------------------------------------------------------------------------
# RPC server side
# ---------------------------------------------------------------------------


class CameraBackend(Protocol):
    """Minimal implementation surface required by add_camera_rpc()."""

    def open_camera(
        self,
        device: str = "",
        *,
        timeout_s: float = 10.0,
    ) -> tuple[bool, bool, str, int, str]: ...

    def close_camera(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> tuple[bool, bool, str, int, str]: ...

    def status(self) -> CameraStatusResponse: ...

    def snapshot_all(
        self,
    ) -> tuple[dict[str, Any], bool, int, int, int, int, str]: ...

    def get_frame(self, stream: str, thumbnail: bool) -> Any | None: ...

    def get_calib(self) -> CameraCalibrationResponse: ...


def _snapshot_payload(
    frames: dict[str, Any],
    key: str,
) -> tuple[np.ndarray, int, int]:
    frame = frames.get(key)
    if frame is None:
        return np.empty(0, dtype=np.uint8), 0, 0
    return (
        np.frombuffer(frame.jpeg, dtype=np.uint8),
        frame.sequence,
        frame.captured_ns,
    )


def _empty_frame_set_response(error: str) -> CameraFrameSetResponse:
    empty = np.empty(0, dtype=np.uint8)
    return CameraFrameSetResponse(
        ok=False,
        camera_online=False,
        generation=0,
        restart_count=0,
        rgb=empty,
        rgb_sequence=0,
        rgb_captured_ns=0,
        left=empty,
        left_sequence=0,
        left_captured_ns=0,
        right=empty,
        right_sequence=0,
        right_captured_ns=0,
        rgb_thumbnail=empty,
        rgb_thumbnail_sequence=0,
        rgb_thumbnail_captured_ns=0,
        left_thumbnail=empty,
        left_thumbnail_sequence=0,
        left_thumbnail_captured_ns=0,
        right_thumbnail=empty,
        right_thumbnail_sequence=0,
        right_thumbnail_captured_ns=0,
        error=error,
    )


def add_camera_rpc(
    server: Any,
    camera: CameraBackend,
    *,
    logger: logging.Logger | None = None,
) -> Any:
    """Register the complete camera RPC implementation on ``server``.

    server.py therefore does not need to repeat RPC names, request types,
    response types, or handler registration.
    """

    log = logger or LOG

    @CAMERA_API.open.server_method(server)
    def camera_open(
        request: CameraOpenRequest,
        context: RpcContext,
    ) -> CameraControlResponse:
        del context
        try:
            ok, online, device, generation, error = camera.open_camera(
                request.device,
                timeout_s=request.timeout_s,
            )
            return CameraControlResponse(
                ok=ok,
                requested_open=True,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            log.exception("camera.open handler failed")
            return CameraControlResponse(
                ok=False,
                requested_open=True,
                online=False,
                device=str(getattr(request, "device", "")),
                generation=0,
                error=f"open handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.close.server_method(server)
    def camera_close(
        request: CameraCloseRequest,
        context: RpcContext,
    ) -> CameraControlResponse:
        del context
        try:
            ok, online, device, generation, error = camera.close_camera(
                timeout_s=request.timeout_s,
            )
            return CameraControlResponse(
                ok=ok,
                requested_open=False,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            log.exception("camera.close handler failed")
            return CameraControlResponse(
                ok=False,
                requested_open=False,
                online=False,
                device="",
                generation=0,
                error=f"close handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.status.server_method(server)
    def camera_status(
        request: EmptyRequest,
        context: RpcContext,
    ) -> CameraStatusResponse:
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
    def camera_frame(
        request: CameraFrameSetRequest,
        context: RpcContext,
    ) -> CameraFrameSetResponse:
        del request, context
        try:
            (
                frames,
                online,
                generation,
                restarts,
                _published,
                _last_ns,
                camera_error,
            ) = camera.snapshot_all()

            rgb, rgb_seq, rgb_ns = _snapshot_payload(frames, "rgb")
            left, left_seq, left_ns = _snapshot_payload(frames, "left")
            right, right_seq, right_ns = _snapshot_payload(frames, "right")
            rgb_thumb, rgb_thumb_seq, rgb_thumb_ns = _snapshot_payload(
                frames, "rgb.thumbnail"
            )
            left_thumb, left_thumb_seq, left_thumb_ns = _snapshot_payload(
                frames, "left.thumbnail"
            )
            right_thumb, right_thumb_seq, right_thumb_ns = _snapshot_payload(
                frames, "right.thumbnail"
            )

            expected = (
                "rgb",
                "left",
                "right",
                "rgb.thumbnail",
                "left.thumbnail",
                "right.thumbnail",
            )
            missing = [key for key in expected if key not in frames]
            ok = not missing

            if missing:
                detail = "missing: " + ", ".join(missing)
                error = f"{camera_error}; {detail}" if camera_error else detail
            elif not online and camera_error:
                # Complete cached data is still useful while a device reconnects.
                error = camera_error
            else:
                error = ""

            return CameraFrameSetResponse(
                ok=ok,
                camera_online=online,
                generation=generation,
                restart_count=restarts,
                rgb=rgb,
                rgb_sequence=rgb_seq,
                rgb_captured_ns=rgb_ns,
                left=left,
                left_sequence=left_seq,
                left_captured_ns=left_ns,
                right=right,
                right_sequence=right_seq,
                right_captured_ns=right_ns,
                rgb_thumbnail=rgb_thumb,
                rgb_thumbnail_sequence=rgb_thumb_seq,
                rgb_thumbnail_captured_ns=rgb_thumb_ns,
                left_thumbnail=left_thumb,
                left_thumbnail_sequence=left_thumb_seq,
                left_thumbnail_captured_ns=left_thumb_ns,
                right_thumbnail=right_thumb,
                right_thumbnail_sequence=right_thumb_seq,
                right_thumbnail_captured_ns=right_thumb_ns,
                error=error,
            )
        except Exception as exc:
            log.exception("camera.frame handler failed")
            return _empty_frame_set_response(
                f"frame handler error: {type(exc).__name__}: {exc}"
            )

    @CAMERA_API.get_frame.server_method(server)
    def camera_get_frame(
        request: CameraFrameRequest,
        context: RpcContext,
    ) -> CameraFrameResponse:
        del context
        try:
            if request.stream not in VALID_STREAMS:
                return CameraFrameResponse(
                    ok=False,
                    camera_online=camera.status().online,
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=f"unknown stream: {request.stream!r}",
                )

            status = camera.status()
            frame = camera.get_frame(request.stream, request.thumbnail)
            if frame is None:
                return CameraFrameResponse(
                    ok=False,
                    camera_online=status.online,
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=status.error or "frame is not available yet",
                )

            return CameraFrameResponse(
                ok=True,
                camera_online=status.online,
                stream=request.stream,
                thumbnail=request.thumbnail,
                sequence=frame.sequence,
                captured_ns=frame.captured_ns,
                jpeg=np.frombuffer(frame.jpeg, dtype=np.uint8),
                error=status.error if not status.online else "",
            )
        except Exception as exc:
            log.exception("camera.get_frame handler failed")
            return CameraFrameResponse(
                ok=False,
                camera_online=False,
                stream=str(getattr(request, "stream", "")),
                thumbnail=bool(getattr(request, "thumbnail", False)),
                sequence=0,
                captured_ns=0,
                jpeg=np.empty(0, dtype=np.uint8),
                error=f"frame handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.get_calib.server_method(server)
    def camera_get_calib(
        request: EmptyRequest,
        context: RpcContext,
    ) -> CameraCalibrationResponse:
        del request, context
        try:
            return camera.get_calib()
        except Exception as exc:
            log.exception("camera.get_calib handler failed")
            return CameraCalibrationResponse.empty(error=str(exc))

    return server


# ---------------------------------------------------------------------------
# FastAPI side
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
    """Expose the complete camera interface through a FastAPI application.

    Example::

        add_camera_routes(
            app,
            discovery=DISCOVERY,
            server_name="camera_front",
        )

    The web server does not need to know RPC method strings, request/response
    model pairs, NNG/ZMQ selection, or the JPEG/ZIP adaptation details.
    """

    try:
        from fastapi import HTTPException, Response
    except ImportError as exc:  # Keep FastAPI optional for pure RPC deployments.
        raise RuntimeError(
            "FastAPI is required only when add_camera_routes() is used"
        ) from exc

    client = CameraClient(
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )

    if route_prefix is None:
        route_prefix = f"/{service}"
        if server_name is not None:
            route_prefix += f"/{server_name}"
    route_prefix = "/" + route_prefix.strip("/")

    route_tag = tag or (
        f"{service}:{server_name}" if server_name is not None else service
    )
    instance_name = server_name or "discovered"

    def route_name(api_method: ApiMethod) -> str:
        assert api_method.web is not None
        return (
            f"{route_name_prefix}{service}:{instance_name}:{api_method.web.path}"
        )

    def add_route(api_method: ApiMethod, func: Any) -> None:
        web = api_method.web
        if web is None:
            raise ValueError(f"{api_method.rpc!r} has no HTTP exposure")

        responses = None
        if web.response == "jpeg":
            responses = {200: {"content": {"image/jpeg": {}}}}
        elif web.response == "zip":
            responses = {200: {"content": {"application/zip": {}}}}

        app.add_api_route(
            f"{route_prefix}/{web.path}",
            func,
            methods=[web.method],
            tags=[route_tag],
            name=route_name(api_method),
            responses=responses,
        )

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

    def open_camera(
        device: str = default_device,
        timeout_s: float = 10.0,
    ):
        return web_call(lambda: client.open(device, timeout_s=timeout_s))

    def close_camera(timeout_s: float = 5.0):
        return web_call(lambda: client.close(timeout_s=timeout_s))

    def status():
        return web_call(client.status)

    def frame(stream: str = "rgb", thumbnail: bool = False):
        result = web_call(
            lambda: client.get_frame(
                stream,
                thumbnail=thumbnail,
            )
        )
        if not result.ok:
            raise HTTPException(503, result.error)
        return Response(
            result.jpeg.tobytes(),
            media_type="image/jpeg",
        )

    def frames():
        result = web_call(client.frames)
        images = {
            "rgb.jpg": result.rgb,
            "left.jpg": result.left,
            "right.jpg": result.right,
            "rgb_thumbnail.jpg": result.rgb_thumbnail,
            "left_thumbnail.jpg": result.left_thumbnail,
            "right_thumbnail.jpg": result.right_thumbnail,
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for filename, image in images.items():
                if image.size:
                    archive.writestr(filename, image.tobytes())

        return Response(
            buf.getvalue(),
            media_type="application/zip",
        )

    def get_calib():
        return web_call(client.get_calib)

    add_route(CAMERA_API.open, open_camera)
    add_route(CAMERA_API.close, close_camera)
    add_route(CAMERA_API.status, status)
    add_route(CAMERA_API.get_frame, frame)
    add_route(CAMERA_API.frame, frames)
    add_route(CAMERA_API.get_calib, get_calib)

    # Returning the client is convenient for applications that also need to make
    # programmatic calls to the same selected camera instance.
    return client
