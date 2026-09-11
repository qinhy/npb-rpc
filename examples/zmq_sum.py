from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import zmq
from npb import BinaryModel, binary_schema

from npb_rpc import RpcContext, ZmqRpcClient, ZmqRpcServer

TCP_ENDPOINT = "tcp://127.0.0.1:5555"
IPC_ENDPOINT = f"ipc://{Path(tempfile.gettempdir()).resolve() / 'npb-rpc-zmq-sum.sock'}"


@binary_schema("npb-rpc.example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.example.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def endpoint_for(transport: str) -> str:
    if transport == "ipc":
        if not zmq.has("ipc"):
            raise SystemExit(
                "This libzmq build does not support ipc://. "
                "On native Windows, use --transport tcp or the NNG IPC example."
            )
        return IPC_ENDPOINT
    return TCP_ENDPOINT


def run_server(endpoint: str) -> None:
    server = ZmqRpcServer.bind(endpoint)

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    print(f"listening on {endpoint}")
    with server:
        server.serve_forever()


def run_client(endpoint: str) -> None:
    with ZmqRpcClient.connect(endpoint) as client:
        response = client.call(
            "array.sum",
            SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
            SumResponse,
        )

    print(response.total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ZeroMQ sum example")
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument(
        "--transport",
        choices=("tcp", "ipc"),
        default="tcp",
        help="ZeroMQ transport to use (default: tcp)",
    )
    args = parser.parse_args()
    endpoint = endpoint_for(args.transport)

    if args.role == "server":
        run_server(endpoint)
    else:
        run_client(endpoint)
