from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, get_type_hints

import numpy as np
from npb import BinaryModel, binary_schema
from npb_rpc import NngRpcClient, NngRpcServer, RpcContext, portable_ipc, portable_tcp
from pydantic import field_serializer, field_validator


# ----------------------------------------------------------------------
# Msgs
# ----------------------------------------------------------------------

@binary_schema("npb-rpc.example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray

    @field_validator("values", mode="before", json_schema_input_type=list[float])
    @classmethod
    def parse_values(cls, v):
        return v if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32)

    @field_serializer("values", when_used="json")
    def serialize_values(self, v: np.ndarray) -> list[float]:
        return v.tolist()


@binary_schema("npb-rpc.example.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


# ----------------------------------------------------------------------
# Unified API definition
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RpcSpec:
    rpc: str
    http: str | None = None
    path: str | None = None


def api(rpc: str, http: str | None = None, path: str | None = None):
    def wrap(fn):
        fn.__rpc_spec__ = RpcSpec(rpc, http, path)
        return fn
    return wrap


class SumInterface:
    @api("array.sum", "POST", "/sum")
    def sum(self, request: SumRequest) -> SumResponse: ...


def methods(interface: type):
    for name, fn in interface.__dict__.items():
        if spec := getattr(fn, "__rpc_spec__", None):
            types = get_type_hints(fn)
            yield name, spec, types["request"], types["return"]


# ----------------------------------------------------------------------
# Implementation
# ----------------------------------------------------------------------

class SumService(SumInterface):
    def sum(self, request: SumRequest) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))


# ----------------------------------------------------------------------
# NNG server
# ----------------------------------------------------------------------

def build_server(endpoint: str, interface: type, service: Any):
    server = NngRpcServer.bind(endpoint)

    for name, spec, req_t, res_t in methods(interface):
        fn = getattr(service, name)

        def make_handler(fn):
            def handler(request, context: RpcContext):
                return fn(request)
            return handler

        server.method(spec.rpc, request=req_t, response=res_t)(make_handler(fn))

    return server


# ----------------------------------------------------------------------
# NNG client class
# ----------------------------------------------------------------------

def build_client(interface: type):
    ns = {}

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    ns["__init__"] = __init__

    for name, spec, req_t, res_t in methods(interface):

        def make_method(spec, res_t):
            def method(self, request):
                with NngRpcClient.connect(self.endpoint) as client:
                    return client.call(spec.rpc, request, res_t)
            return method

        fn = make_method(spec, res_t)
        fn.__name__ = name
        fn.__annotations__ = {"request": req_t, "return": res_t}
        ns[name] = fn

    return type(f"{interface.__name__}Client", (), ns)


SumClient = build_client(SumInterface)


# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------

def build_fastapi(endpoint: str, interface: type):
    from fastapi import FastAPI

    app = FastAPI()
    client = build_client(interface)(endpoint)

    for name, spec, req_t, res_t in methods(interface):
        if not spec.http:
            continue

        fn = getattr(client, name)

        def make_route(fn):
            def route(request):
                return fn(request)
            return route

        route = make_route(fn)
        route.__name__ = name
        route.__annotations__ = {"request": req_t, "return": res_t}

        app.add_api_route(
            spec.path,
            route,
            methods=[spec.http],
            response_model=res_t,
        )

    return app


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def endpoint_for(transport: str, name: str, host="127.0.0.1") -> str:
    if transport == "ipc":
        return portable_ipc(f"npb-rpc-{name}")
    if transport == "tcp":
        return portable_tcp(name, host=host)
    raise ValueError(transport)


def run_server(endpoint: str):
    server = build_server(endpoint, SumInterface, SumService())
    print(f"listening on {endpoint}")
    with server:
        server.serve_forever()


def run_client(endpoint: str):
    client = SumClient(endpoint)
    response = client.sum(
        SumRequest(values=np.arange(1_000_000, dtype=np.float32))
    )
    print(response.total)


def run_api(endpoint: str):
    import uvicorn

    process = mp.Process(target=run_server, args=(endpoint,))
    process.start()

    try:
        uvicorn.run(build_fastapi(endpoint, SumInterface))
    finally:
        process.terminate()
        process.join()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("server", "client", "api"))
    parser.add_argument("transport", choices=("tcp", "ipc"))
    parser.add_argument("server_name")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    endpoint = endpoint_for(args.transport, args.server_name, args.host)

    if args.role == "server":
        run_server(endpoint)
    elif args.role == "client":
        run_client(endpoint)
    else:
        run_api(endpoint)