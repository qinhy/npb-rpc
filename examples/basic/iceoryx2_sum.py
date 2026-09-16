"""Minimal typed sum RPC example using iceoryx2 shared memory."""

from __future__ import annotations

import argparse
from contextlib import suppress

import numpy as np
from npb import BinaryModel, binary_schema

from npb_rpc import Iceoryx2RpcClient, Iceoryx2RpcServer, RpcContext


@binary_schema("npb-rpc.example.iceoryx2.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.example.iceoryx2.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def endpoint_for(name: str) -> str:
    return f"iceoryx2://npb-rpc-{name}"


def run_server(endpoint: str) -> None:
    with Iceoryx2RpcServer.bind(endpoint) as server:

        @server.method("array.sum", request=SumRequest, response=SumResponse)
        def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
            return SumResponse(total=float(request.values.sum()))

        print(f"serving at {server.endpoint}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


def run_client(endpoint: str) -> None:
    with Iceoryx2RpcClient.connect(endpoint) as client:
        response = client.call(
            "array.sum",
            SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
            SumResponse,
            timeout=5.0,
        )
    print(response.total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal iceoryx2 sum example")
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--name", default="sum", help="logical endpoint name")
    parser.add_argument("--endpoint", help="explicit endpoint override")
    args = parser.parse_args()

    endpoint = args.endpoint or endpoint_for(args.name)
    if args.role == "server":
        run_server(endpoint)
    else:
        run_client(endpoint)


if __name__ == "__main__":
    main()
