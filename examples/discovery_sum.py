from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import zmq
from npb import BinaryModel, binary_schema

from npb_rpc import (
    DiscoveredRpcClient,
    DiscoveredRpcServer,
    FilesystemDiscovery,
    NngRpcServer,
    RpcContext,
    ZmqRpcServer,
    portable_ipc,
)


@binary_schema("npb-rpc.example.discovery.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.example.discovery.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def default_endpoint(backend: str, transport: str) -> str:
    if backend == "zmq":
        if transport == "tcp":
            return "tcp://127.0.0.1:5560"
        if not zmq.has("ipc"):
            raise SystemExit(
                "This libzmq build does not support ipc://. "
                "Use --transport tcp or --backend nng on native Windows."
            )
        socket_path = Path(tempfile.gettempdir()).resolve() / "npb-rpc-discovery-zmq.sock"
        return f"ipc://{socket_path}"
    if transport == "tcp":
        return "tcp://127.0.0.1:5561"
    return portable_ipc("npb-rpc-discovery-sum")


def run_server(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    endpoint = args.endpoint or default_endpoint(args.backend, args.transport)
    server_type = ZmqRpcServer if args.backend == "zmq" else NngRpcServer
    server = DiscoveredRpcServer(
        args.service,
        server_type.bind(endpoint),
        discovery,
        advertise_endpoint=args.advertise_endpoint,
    )

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    print(
        f"serving {args.service!r} via {args.backend} at {server.endpoint}; "
        f"registry: {discovery.root}"
    )
    with server:
        server.serve_forever()


def run_client(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    with DiscoveredRpcClient(discovery) as client:
        response = client.call(
            args.service,
            "array.sum",
            SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
            SumResponse,
        )
    print(response.total)


def show_services(discovery: FilesystemDiscovery) -> None:
    services = discovery.list_services()
    if not services:
        print("no healthy services")
        return
    for service in services:
        print(f"{service}:")
        for instance in discovery.list_instances(service):
            print(
                f"  {instance.instance_id}  {instance.backend}  "
                f"{instance.endpoint}  {instance.hostname} pid={instance.pid}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the service discovery example")
    parser.add_argument("role", choices=("server", "client", "list"))
    parser.add_argument("--service", default="sum")
    parser.add_argument("--backend", choices=("zmq", "nng"), default="zmq")
    parser.add_argument("--transport", choices=("tcp", "ipc"), default="tcp")
    parser.add_argument("--endpoint", help="server bind endpoint override")
    parser.add_argument(
        "--advertise-endpoint",
        help="endpoint stored in discovery when it differs from the bind endpoint",
    )
    parser.add_argument("--registry", type=Path, help="filesystem registry directory")
    arguments = parser.parse_args()

    registry = FilesystemDiscovery(arguments.registry)
    if arguments.role == "server":
        run_server(arguments, registry)
    elif arguments.role == "client":
        run_client(arguments, registry)
    else:
        show_services(registry)
