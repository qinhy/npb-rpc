from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from npb import BinaryModel

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
except ImportError:  # Support running the files directly from one directory.
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


RequestT = TypeVar("RequestT", bound=BinaryModel)
ResponseT = TypeVar("ResponseT", bound=BinaryModel)

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
WebResponseKind = Literal["model", "jpeg", "zip"]


@dataclass(frozen=True, slots=True)
class WebApi:
    """HTTP exposure metadata for one RPC operation."""

    method: HttpMethod
    path: str
    response: WebResponseKind = "model"

    def __post_init__(self) -> None:
        normalized = self.path.strip("/")
        if not normalized:
            raise ValueError("web API path must not be empty")
        object.__setattr__(self, "path", normalized)


@dataclass(frozen=True, slots=True)
class ApiMethod(Generic[RequestT, ResponseT]):
    """Single source of truth for one RPC operation and its HTTP exposure."""

    rpc: str
    request: type[RequestT]
    response: type[ResponseT]
    web: WebApi | None = None

    def server_method(self, server: Any):
        """Return the npb_rpc server decorator for this operation."""
        return server.method(
            self.rpc,
            request=self.request,
            response=self.response,
        )


class CameraApiInterface:
    """Public RPC + HTTP contract for the DepthAI camera service."""

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

    frame = ApiMethod(
        rpc="camera.frame",
        request=CameraFrameSetRequest,
        response=CameraFrameSetResponse,
        web=WebApi(method="GET", path="frames", response="zip"),
    )

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


CAMERA_API = CameraApiInterface()
