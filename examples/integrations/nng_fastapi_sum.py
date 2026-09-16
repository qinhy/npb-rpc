"""Expose an NNG sum RPC service through a small FastAPI gateway."""

from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
from npb import BinaryModel, binary_schema
from pydantic import field_serializer, field_validator

from npb_rpc import NngRpcClient, NngRpcServer, RpcContext, portable_ipc, portable_tcp


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


def endpoint_for(transport: str, name: str, host: str = "127.0.0.1") -> str:
    if transport == "ipc":
        return portable_ipc(f"npb-rpc-{name}")
    if transport == "tcp":
        return portable_tcp(name, host=host)
    raise ValueError(f"unsupported transport: {transport}")


def run_server(endpoint: str) -> None:
    server = NngRpcServer.bind(endpoint)

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    print(f"serving RPC at {server.endpoint}", flush=True)
    with server:
        server.serve_forever()


def client_array_sum(request: SumRequest, endpoint: str) -> SumResponse:
    with NngRpcClient.connect(endpoint) as client:
        return client.call("array.sum", request, SumResponse)


def build_fastapi(endpoint: str):
    from fastapi import FastAPI

    app = FastAPI(title="npb-rpc NNG + FastAPI example")

    @app.post("/sum", response_model=SumResponse)
    def sum_values(request: SumRequest) -> SumResponse:
        return client_array_sum(request, endpoint)

    return app


def run_api(endpoint: str) -> None:
    import uvicorn

    process = mp.Process(target=run_server, args=(endpoint,))
    process.start()
    try:
        uvicorn.run(build_fastapi(endpoint))
    finally:
        if process.is_alive():
            process.terminate()
        process.join()


def run_client(endpoint: str) -> None:
    response = client_array_sum(
        SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
        endpoint,
    )
    print(response.total)


def main() -> None:
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Run the NNG + FastAPI sum example")
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
