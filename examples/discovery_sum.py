from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zmq
from npb import BinaryModel, binary_schema

from npb_rpc import (
    DiscoveredRpcClient,
    DiscoveredRpcServer,
    FilesystemDiscovery,
    Iceoryx2RpcClient,
    Iceoryx2RpcServer,
    NngRpcClient,
    NngRpcServer,
    RpcContext,
    ZmqRpcClient,
    ZmqRpcServer,
    portable_ipc,
    portable_tcp,
)


@binary_schema("npb-rpc.example.discovery.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.example.discovery.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def endpoint_for(backend: str, transport: str, name: str,
        host: str = "127.0.0.1") -> str:
    if backend == "iceoryx2":
        if transport != "ipc":
            raise SystemExit("iceoryx2 requires --transport ipc")
        return f"iceoryx2://{name}"
    if transport == "tcp":
        return portable_tcp(name, host=host)
    if backend == "zmq" and not zmq.has("ipc"):
        raise SystemExit(
            "This libzmq build does not support ipc://. "
            "Use --transport tcp or --backend nng on native Windows."
        )
    return portable_ipc(name)


def run_server(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    server_name = args.server_name or args.service
    endpoint = args.endpoint or endpoint_for(
        args.backend, args.transport, server_name, args.host)
    server_type = {
        "zmq": ZmqRpcServer, "nng": NngRpcServer, "iceoryx2": Iceoryx2RpcServer
    }[args.backend]
    server = DiscoveredRpcServer(
        args.service,
        server_type.bind(endpoint),
        discovery,
        instance_id=server_name,
        advertise_endpoint=args.advertise_endpoint,
    )

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    print(
        f"serving {args.service!r} as {server_name!r} via {args.backend} "
        f"at {server.endpoint}; registry: {discovery.root}"
    )
    with server:
        server.serve_forever()


def run_client(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    request = SumRequest(values=np.arange(1_000_000, dtype=np.float32))

    if args.server_name:
        instances = discovery.list_instances(args.service)
        matches = [x for x in instances if x.instance_id == args.server_name]
        if not matches:
            raise SystemExit(f"server {args.server_name!r} not found")
        if len(matches) > 1:
            raise SystemExit(f"server {args.server_name!r} is ambiguous")

        instance = matches[0]
        client_type = {
            "zmq": ZmqRpcClient, "nng": NngRpcClient, "iceoryx2": Iceoryx2RpcClient
        }[instance.backend]
        print(
            f"connecting to {instance.instance_id!r} via {instance.backend} "
            f"at {instance.endpoint}"
        )
        with client_type.connect(instance.endpoint) as client:
            response = client.call("array.sum", request, SumResponse)
    else:
        with DiscoveredRpcClient(discovery) as client:
            response = client.call(
                args.service, "array.sum", request, SumResponse)

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
                f"  {instance.instance_id}  "
                f"{instance.backend}  {instance.endpoint}  "
                f"{instance.hostname} pid={instance.pid}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the service discovery example")
    parser.add_argument("role", choices=("server", "client", "list"))
    parser.add_argument("--service", default="sum")
    parser.add_argument("--server-name")
    parser.add_argument("--backend", choices=("zmq", "nng", "iceoryx2"), default="zmq")
    parser.add_argument("--transport", choices=("tcp", "ipc"), default="tcp")
    parser.add_argument("--host", default="127.0.0.1")
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