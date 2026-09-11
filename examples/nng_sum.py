from __future__ import annotations

import argparse
from typing import Any
import multiprocessing as mp

import numpy as np
from npb import BinaryModel, binary_schema
from pydantic import field_serializer, field_validator

from npb_rpc import NngRpcClient, NngRpcServer, RpcContext, portable_ipc, portable_tcp


@binary_schema("npb-rpc.example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray
    
    @field_validator("values", mode="before",
            json_schema_input_type=list[float])
    @classmethod
    def parse_values(cls, value):
        if isinstance(value, np.ndarray): return value
        return np.asarray(value, dtype=np.float32)

    @field_serializer("values", when_used="json")
    def serialize_values(self, value: np.ndarray) -> list[float]:
        return value.tolist()


@binary_schema("npb-rpc.example.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def endpoint_for(transport: str, server_name: str, host="127.0.0.1") -> str:
    if transport == "ipc":
        return portable_ipc(f"npb-rpc-{server_name}")
    if transport == "tcp":
        return portable_tcp(server_name, host=host)
    raise ValueError(f"unsupported transport: {transport}")


def run_server(endpoint: str) -> None:
    server = NngRpcServer.bind(endpoint)

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    print(f"listening on {endpoint}")
    with server:
        server.serve_forever()


def client_array_sum(request: SumRequest, endpoint) -> SumResponse:
    with NngRpcClient.connect(endpoint) as client:
        return client.call("array.sum", request, SumResponse)


def run_client(endpoint: str) -> None:
    response = client_array_sum(
        SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
        endpoint=endpoint)
    print(response.total)


def run_api(endpoint: str) -> None:
    process = mp.Process(target=run_server, args=(endpoint,))
    process.start()

    import uvicorn
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/sum")
    def sum_values(request: SumRequest) -> SumResponse:
        return client_array_sum(request, endpoint=endpoint)

    try:
        uvicorn.run(app)
    finally:
        process.terminate()
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NNG sum example")
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
    elif args.role == "api":
        run_api(endpoint)