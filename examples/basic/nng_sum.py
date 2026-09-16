"""Minimal typed sum RPC example using the NNG backend."""

from __future__ import annotations

import argparse

import numpy as np
from npb import BinaryModel, binary_schema

from npb_rpc import NngRpcClient, NngRpcServer, RpcContext, portable_ipc, portable_tcp


@binary_schema("npb-rpc.example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


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

    print(f"serving at {server.endpoint}", flush=True)
    with server:
        server.serve_forever()


def run_client(endpoint: str) -> None:
    with NngRpcClient.connect(endpoint) as client:
        response = client.call(
            "array.sum",
            SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
            SumResponse,
        )
    print(response.total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal NNG sum example")
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--transport", choices=("tcp", "ipc"), default="ipc")
    parser.add_argument("--name", default="sum", help="logical endpoint name")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host")
    parser.add_argument("--endpoint", help="explicit endpoint override")
    args = parser.parse_args()

    endpoint = args.endpoint or endpoint_for(args.transport, args.name, args.host)
    if args.role == "server":
        run_server(endpoint)
    else:
        run_client(endpoint)


if __name__ == "__main__":
    main()
