from __future__ import annotations
"""Unified RPC + HTTP interface for the yolo service."""

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
        EmptyRequest,
        YoloInferenceRequest,
        YoloInferenceSubmitResponse,
        YoloJobRequest,
        YoloJobResultResponse,
        YoloJobStatusResponse,
        YoloStatusResponse,
    )
    from .worker import YoloWorker
    
except ImportError:  # Support running files directly from this directory.
    from msg import (
        EmptyRequest,
        YoloInferenceRequest,
        YoloInferenceSubmitResponse,
        YoloJobRequest,
        YoloJobResultResponse,
        YoloJobStatusResponse,
        YoloStatusResponse,
    )
    from worker import YoloWorker


LOG = logging.getLogger("yolo.interface")
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


class YoloInterface(Protocol):
    """Single source of truth for RPC, generated client, server, and HTTP."""

    service = "yolo"

    @api("yolo.inference", "POST", "inference")
    def inference(
        self,
        request: YoloInferenceRequest,
    ) -> YoloInferenceSubmitResponse: ...

    @api("yolo.job_status", "GET", "job_status")
    def job_status(
        self,
        request: YoloJobRequest,
    ) -> YoloJobStatusResponse: ...

    @api("yolo.job_result", "GET", "job_result")
    def job_result(
        self,
        request: YoloJobRequest,
    ) -> YoloJobResultResponse: ...

    @api("yolo.status", "GET", "status")
    def status(
        self,
        request: EmptyRequest,
    ) -> YoloStatusResponse: ...



class YoloService(YoloInterface):
    """Typed RPC/HTTP façade over the long-lived asynchronous YoloWorker."""

    def __init__(
        self,
        worker: YoloWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.worker = worker
        self.log = logger or LOG

    def inference(
        self,
        request: YoloInferenceRequest,
    ) -> YoloInferenceSubmitResponse:
        """Queue inference and return immediately with a job id."""
        try:
            return self.worker.submit(request)
        except Exception as exc:
            self.log.exception("yolo.inference failed")
            return YoloInferenceSubmitResponse(
                accepted=False,
                input_jpg_path=request.input_jpg_path,
                output_json_path=request.output_json_path,
                error=f"inference submit error: {type(exc).__name__}: {exc}",
            )

    def job_status(
        self,
        request: YoloJobRequest,
    ) -> YoloJobStatusResponse:
        """Return lightweight state for one asynchronous inference job."""
        try:
            return self.worker.job_status(request.job_id)
        except Exception as exc:
            self.log.exception("yolo.job_status failed")
            return YoloJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )

    def job_result(
        self,
        request: YoloJobRequest,
    ) -> YoloJobResultResponse:
        """Return the full result when a job has succeeded."""
        try:
            return self.worker.job_result(request.job_id)
        except Exception as exc:
            self.log.exception("yolo.job_result failed")
            return YoloJobResultResponse(
                found=False,
                job_id=request.job_id,
                error=f"job result error: {type(exc).__name__}: {exc}",
            )

    def status(
        self,
        request: EmptyRequest,
    ) -> YoloStatusResponse:
        """Return worker, queue, and model-cache status."""
        del request

        try:
            return self.worker.status()
        except Exception as exc:
            self.log.exception("yolo.status failed")
            return YoloStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )



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


YoloClient = build_client_class(YoloInterface, "YoloClient")


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


def add_yolo_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_routes(app, YoloInterface, **kwargs)
