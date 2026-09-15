from __future__ import annotations

"""Unified RPC + HTTP interface for the PCD service."""

from dataclasses import dataclass
from functools import cache
import inspect
import logging
from typing import Any, Generic, Literal, Protocol, TypeVar, get_type_hints

from npb import BinaryModel
from npb_rpc import (
    DiscoveredRpcClient,
    FilesystemDiscovery,
    NngRpcClient,
    ZmqRpcClient,
)

try:
    from .msg import (
        EmptyRequest,
        PcdBuildRequest,
        PcdBuildSubmitResponse,
        PcdJobRequest,
        PcdJobResultResponse,
        PcdJobStatusResponse,
        PcdStatusResponse,
    )
    from .worker import PcdWorker
except ImportError:  # Support running files directly from this directory.
    from msg import (
        EmptyRequest,
        PcdBuildRequest,
        PcdBuildSubmitResponse,
        PcdJobRequest,
        PcdJobResultResponse,
        PcdJobStatusResponse,
        PcdStatusResponse,
    )
    from worker import PcdWorker


LOG = logging.getLogger("pcd.interface")
CLIENT_TYPES = {"nng": NngRpcClient, "zmq": ZmqRpcClient}

RequestT = TypeVar("RequestT", bound=BinaryModel)
ResponseT = TypeVar("ResponseT", bound=BinaryModel)
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


# API declaration

@dataclass(frozen=True, slots=True)
class WebApi:
    method: HttpMethod
    path: str

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


def api(rpc: str, http: HttpMethod | None = None, path: str | None = None):
    """Mark an interface method as RPC and optionally expose it through HTTP."""
    if (http is None) != (path is None):
        raise ValueError("http and path must be supplied together")

    spec = RpcSpec(
        rpc=rpc,
        web=WebApi(http, path) if http is not None and path is not None else None,
    )

    def decorate(fn):
        fn.__rpc_spec__ = spec
        return fn

    return decorate


# PCD service contract

class PcdInterface(Protocol):
    """Single source of truth for RPC, generated client, server, and FastAPI."""

    service = "pcd"

    @api("pcd.build", "POST", "build")
    def build(self, request: PcdBuildRequest) -> PcdBuildSubmitResponse: ...

    @api("pcd.job_status", "GET", "job_status")
    def job_status(self, request: PcdJobRequest) -> PcdJobStatusResponse: ...

    @api("pcd.job_result", "GET", "job_result")
    def job_result(self, request: PcdJobRequest) -> PcdJobResultResponse: ...

    @api("pcd.status", "GET", "status")
    def status(self, request: EmptyRequest) -> PcdStatusResponse: ...


# Service implementation

class PcdService(PcdInterface):
    """Typed RPC/HTTP façade over the long-lived asynchronous PcdWorker."""

    def __init__(
        self,
        worker: PcdWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.worker = worker
        self.log = logger or LOG

    def build(self, request: PcdBuildRequest) -> PcdBuildSubmitResponse:
        try:
            return self.worker.submit(request)
        except Exception as exc:
            self.log.exception("pcd.build failed")
            return PcdBuildSubmitResponse(
                accepted=False,
                rgb_jpg_path=request.rgb_jpg_path,
                left_jpg_path=request.left_jpg_path,
                right_jpg_path=request.right_jpg_path,
                output_pcd_path=request.output_pcd_path,
                output_json_path=request.output_json_path,
                error=f"build submit error: {type(exc).__name__}: {exc}",
            )

    def job_status(self, request: PcdJobRequest) -> PcdJobStatusResponse:
        try:
            return self.worker.job_status(request.job_id)
        except Exception as exc:
            self.log.exception("pcd.job_status failed")
            return PcdJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )

    def job_result(self, request: PcdJobRequest) -> PcdJobResultResponse:
        try:
            return self.worker.job_result(request.job_id)
        except Exception as exc:
            self.log.exception("pcd.job_result failed")
            return PcdJobResultResponse(
                found=False,
                job_id=request.job_id,
                error=f"job result error: {type(exc).__name__}: {exc}",
            )

    def status(self, request: EmptyRequest) -> PcdStatusResponse:
        del request
        try:
            return self.worker.status()
        except Exception as exc:
            self.log.exception("pcd.status failed")
            return PcdStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )


# Interface reflection

@cache
def api_methods(interface: type) -> tuple[ApiMethod, ...]:
    """Discover and validate decorated methods declared by an interface."""
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
            raise TypeError(
                f"{interface.__name__}.{name}: request must be a BinaryModel type"
            )
        if not isinstance(response, type) or not issubclass(response, BinaryModel):
            raise TypeError(
                f"{interface.__name__}.{name}: return must be a BinaryModel type"
            )

        result.append(
            ApiMethod(
                name=name,
                rpc=spec.rpc,
                request=request,
                response=response,
                web=spec.web,
            )
        )

    return tuple(result)


# RPC client

def _client_type(backend: str):
    try:
        return CLIENT_TYPES[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported RPC backend: {backend!r}") from exc


def resolve_service_instance(
    discovery: FilesystemDiscovery,
    service: str,
    server_name: str,
):
    """Resolve one explicitly named discovered server instance."""
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
    endpoint: str | None = None
    backend: str = "nng"
    discovery: FilesystemDiscovery | None = None
    service: str = ""
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
            target = resolve_service_instance(
                self.discovery,
                self.service,
                self.server_name,
            )
            with _client_type(target.backend).connect(target.endpoint) as client:
                return client.call(method, request, response_type)

        if not self.service:
            raise ValueError("service is required when using discovery")

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
        return self.call_raw(method.rpc, request, method.response)


# Generated typed client

def build_client_class(interface: type, name: str | None = None):
    """Generate a typed client from an interface declaration."""
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
        self.target = RpcTarget(
            endpoint=endpoint,
            backend=backend,
            discovery=discovery,
            service=service,
            server_name=server_name,
        )

    namespace: dict[str, Any] = {"__init__": __init__}

    for method in api_methods(interface):
        def make_method(method: ApiMethod):
            def call(self, request):
                return self.target.call(method, request)

            client_name = name or interface.__name__ + "Client"
            call.__name__ = method.name
            call.__qualname__ = f"{client_name}.{method.name}"
            call.__annotations__ = {
                "request": method.request,
                "return": method.response,
            }
            return call

        namespace[method.name] = make_method(method)

    return type(
        name or f"{interface.__name__}Client",
        (interface,),
        namespace,
    )


PcdClient = build_client_class(PcdInterface, "PcdClient")


# RPC server registration

def add_rpc(server: Any, interface: type, implementation: Any) -> Any:
    """Register every decorated interface method on an NNG/ZMQ RPC server."""
    for method in api_methods(interface):
        fn = getattr(implementation, method.name)

        def make_handler(method: ApiMethod, fn: Any):
            def handler(request, context):
                del context
                return fn(request)

            handler.__name__ = f"rpc_{method.name}"
            handler.__annotations__ = {
                "request": method.request,
                "return": method.response,
            }
            return handler

        server.method(
            method.rpc,
            request=method.request,
            response=method.response,
        )(make_handler(method, fn))

    return server


# FastAPI generation

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
    """Generate FastAPI routes for all decorated HTTP methods."""
    try:
        from fastapi import Depends, HTTPException
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required only when add_routes() is used"
        ) from exc

    service = service or interface.service
    Client = build_client_class(interface)
    client = Client(
        endpoint=endpoint,
        backend=backend,
        discovery=discovery,
        service=service,
        server_name=server_name,
    )

    route_prefix = route_prefix or (
        f"/{service}" + (f"/{server_name}" if server_name else "")
    )
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

    def make_route(method: ApiMethod, client_method: Any):
        web = method.web
        assert web is not None

        def invoke(request: BinaryModel):
            return call_rpc(lambda: client_method(request))

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
        route.__annotations__["return"] = method.response
        return route

    for method in api_methods(interface):
        web = method.web
        if web is None:
            continue

        route = make_route(method, getattr(client, method.name))
        app.add_api_route(
            f"{route_prefix}/{web.path}",
            route,
            methods=[web.method],
            tags=[route_tag],
            name=f"{route_name_prefix}{service}:{instance_name}:{web.path}",
            response_model=method.response,
        )

    return client


# PCD convenience wrapper

def add_pcd_routes(app: Any, **kwargs: Any):
    return add_routes(app, PcdInterface, **kwargs)
