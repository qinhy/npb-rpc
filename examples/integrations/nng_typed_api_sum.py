"""Generate an NNG server, typed client, and FastAPI routes from one API interface."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from typing import Any

import numpy as np
from npb import BinaryModel, binary_schema
from pydantic import field_serializer, field_validator

from npb_rpc import (
    NngRpcClient,
    NngRpcServer,
    RpcContext,
    RpcSpec,
    api,
    api_methods,
    portable_ipc,
    portable_tcp,
)


@binary_schema("npb-rpc.example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray

    @field_validator("values", mode="before", json_schema_input_type=list[float])
    @classmethod
    def parse_values(cls, value):
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value, dtype=np.float32)

    @field_serializer("values", when_used="json")
    def serialize_values(self, value: np.ndarray) -> list[float]:
        return value.tolist()


@binary_schema("npb-rpc.example.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


class SumInterface:
    @api("array.sum", "POST", "/sum")
    def sum(self, request: SumRequest) -> SumResponse: ...


class SumService(SumInterface):
    def sum(self, request: SumRequest) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))


def build_server(endpoint: str, interface: type, service: Any):
    server = NngRpcServer.bind(endpoint)

    for name, spec, request_type, response_type in api_methods(interface):
        service_method = getattr(service, name)

        def make_handler(method):
            def handler(request, context: RpcContext):
                return method(request)

            return handler

        server.method(
            spec.rpc,
            request=request_type,
            response=response_type,
        )(make_handler(service_method))

    return server


def build_client(interface: type):
    namespace: dict[str, Any] = {}

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    namespace["__init__"] = __init__

    for name, spec, request_type, response_type in api_methods(interface):

        def make_method(rpc_spec: RpcSpec, result_type):
            def method(self, request):
                with NngRpcClient.connect(self.endpoint) as client:
                    return client.call(rpc_spec.rpc, request, result_type)

            return method

        method = make_method(spec, response_type)
        method.__name__ = name
        method.__annotations__ = {
            "request": request_type,
            "return": response_type,
        }
        namespace[name] = method

    return type(f"{interface.__name__}Client", (), namespace)


SumClient = build_client(SumInterface)


def build_fastapi(endpoint: str, interface: type):
    from fastapi import FastAPI

    app = FastAPI(title="npb-rpc typed API example")
    client = build_client(interface)(endpoint)

    for name, spec, request_type, response_type in api_methods(interface):
        if not spec.http:
            continue

        client_method = getattr(client, name)

        def make_route(method):
            def route(request):
                return method(request)

            return route

        route = make_route(client_method)
        route.__name__ = name
        route.__annotations__ = {
            "request": request_type,
            "return": response_type,
        }

        app.add_api_route(
            spec.path,
            route,
            methods=[spec.http],
            response_model=response_type,
        )

    return app


def endpoint_for(transport: str, name: str, host: str = "127.0.0.1") -> str:
    if transport == "ipc":
        return portable_ipc(f"npb-rpc-{name}")
    if transport == "tcp":
        return portable_tcp(name, host=host)
    raise ValueError(f"unsupported transport: {transport}")


def run_server(endpoint: str) -> None:
    server = build_server(endpoint, SumInterface, SumService())
    print(f"serving RPC at {server.endpoint}", flush=True)
    with server:
        server.serve_forever()


def run_client(endpoint: str) -> None:
    client = SumClient(endpoint)
    response = client.sum(
        SumRequest(values=np.arange(1_000_000, dtype=np.float32))
    )
    print(response.total)


def run_api(endpoint: str) -> None:
    import uvicorn

    process = mp.Process(target=run_server, args=(endpoint,))
    process.start()
    try:
        uvicorn.run(build_fastapi(endpoint, SumInterface))
    finally:
        if process.is_alive():
            process.terminate()
        process.join()


def main() -> None:
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Run the typed NNG API example")
    parser.add_argument("role", choices=("server", "client", "api"))
    parser.add_argument("--transport", choices=("tcp", "ipc"), default="ipc")
    parser.add_argument("--name", default="sum", help="logical endpoint name")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host")
    parser.add_argument("--endpoint", help="explicit RPC endpoint override")
    args = parser.parse_args()

    endpoint = args.endpoint or endpoint_for(args.transport, args.name, args.host)
    if args.role == "server":
        run_server(endpoint)
    elif args.role == "client":
        run_client(endpoint)
    else:
        run_api(endpoint)


if __name__ == "__main__":
    main()
