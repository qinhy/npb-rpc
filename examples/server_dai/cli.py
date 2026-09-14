from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    import zmq
except ImportError:  # Only needed for ZeroMQ IPC capability detection.
    zmq = None

from npb_rpc import FilesystemDiscovery, portable_ipc, portable_tcp

try:
    from .interface import CAMERA_API, CameraClient, resolve_service_instance
    from .server import run_server
except ImportError:  # Support running files directly from this directory.
    from interface import CAMERA_API, CameraClient, resolve_service_instance
    from server import run_server


LOG = logging.getLogger("dai_camera")
VALID_STREAMS = ("rgb", "left", "right")
DEFAULT_SERVICE = CAMERA_API.service


def endpoint_for(
    backend: str,
    transport: str,
    name: str,
    host: str = "127.0.0.1",
) -> str:
    """Build a portable per-instance endpoint."""
    if transport == "tcp":
        return portable_tcp(name, host=host)

    if backend == "zmq":
        if zmq is None:
            raise SystemExit(
                "ZeroMQ backend requested but pyzmq is not installed. "
                "Install pyzmq, or use --backend nng."
            )
        if not zmq.has("ipc"):
            raise SystemExit(
                "This libzmq build does not support ipc://. "
                "Use --transport tcp or --backend nng."
            )

    return portable_ipc(name)


def make_client(
    args: argparse.Namespace,
    discovery: FilesystemDiscovery,
) -> CameraClient:
    if args.endpoint:
        return CameraClient(
            endpoint=args.endpoint,
            backend=args.backend,
            service=args.service,
        )

    return CameraClient(
        discovery=discovery,
        service=args.service,
        server_name=args.server_name,
    )


def run_client(
    args: argparse.Namespace,
    discovery: FilesystemDiscovery,
) -> None:
    client = make_client(args, discovery)

    if args.endpoint:
        print(f"connecting directly via {args.backend} at {args.endpoint}")
    elif args.server_name:
        instance = resolve_service_instance(
            discovery,
            args.service,
            args.server_name,
        )
        print(
            f"connecting to {instance.instance_id!r} via {instance.backend} "
            f"at {instance.endpoint}"
        )
    else:
        print(
            f"discovering a healthy instance of {args.service!r}; "
            f"registry: {discovery.root}"
        )

    if args.open_camera:
        response = client.open(
            args.device or "",
            timeout_s=args.control_timeout,
        )
        print(
            "open:",
            f"ok={response.ok}",
            f"online={response.online}",
            f"device={response.device!r}",
            f"generation={response.generation}",
            f"error={response.error!r}",
        )
        return

    if args.close_camera:
        response = client.close(timeout_s=args.control_timeout)
        print(
            "close:",
            f"ok={response.ok}",
            f"online={response.online}",
            f"device={response.device!r}",
            f"generation={response.generation}",
            f"error={response.error!r}",
        )
        return

    status = client.status()
    print(
        "status:",
        f"requested_open={status.requested_open}",
        f"online={status.online}",
        f"device={status.device!r}",
        f"generation={status.generation}",
        f"restarts={status.restart_count}",
        f"published={status.frames_published}",
        f"last_frame_ns={status.last_frame_ns}",
        f"error={status.error!r}",
    )

    if args.single:
        frame = client.get_frame(
            args.stream,
            thumbnail=args.thumbnail,
        )
        if not frame.ok:
            print(f"frame unavailable: {frame.error}")
            return

        suffix = "_thumbnail" if args.thumbnail else ""
        path = f"{args.stream}{suffix}.jpg"
        with open(path, "wb") as f:
            f.write(frame.jpeg.tobytes())

        print(
            f"wrote {path}: {frame.jpeg.nbytes} bytes, "
            f"seq={frame.sequence}, online={frame.camera_online}, "
            f"captured_ns={frame.captured_ns}"
        )
        return

    frames = client.frames()
    print(
        "frame-set:",
        f"ok={frames.ok}",
        f"online={frames.camera_online}",
        f"generation={frames.generation}",
        f"restarts={frames.restart_count}",
        f"error={frames.error!r}",
    )

    outputs = (
        ("rgb.jpg", frames.rgb, frames.rgb_sequence, frames.rgb_captured_ns),
        ("left.jpg", frames.left, frames.left_sequence, frames.left_captured_ns),
        ("right.jpg", frames.right, frames.right_sequence, frames.right_captured_ns),
        (
            "rgb_thumbnail.jpg",
            frames.rgb_thumbnail,
            frames.rgb_thumbnail_sequence,
            frames.rgb_thumbnail_captured_ns,
        ),
        (
            "left_thumbnail.jpg",
            frames.left_thumbnail,
            frames.left_thumbnail_sequence,
            frames.left_thumbnail_captured_ns,
        ),
        (
            "right_thumbnail.jpg",
            frames.right_thumbnail,
            frames.right_thumbnail_sequence,
            frames.right_thumbnail_captured_ns,
        ),
    )

    for path, jpeg, sequence, captured_ns in outputs:
        if jpeg.size == 0:
            print(f"missing {path}")
            continue
        with open(path, "wb") as f:
            f.write(jpeg.tobytes())
        print(
            f"wrote {path}: {jpeg.nbytes} bytes, "
            f"seq={sequence}, captured_ns={captured_ns}"
        )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fault-tolerant discovered DepthAI camera RPC service"
    )
    parser.add_argument("role", choices=("server", "client", "list"))
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument(
        "--server-name",
        help="instance id to advertise (server) or select (client)",
    )
    parser.add_argument(
        "--backend",
        choices=("nng", "zmq"),
        default="nng",
        help="server backend, or direct-client backend (default: nng)",
    )
    parser.add_argument(
        "--transport",
        choices=("tcp", "ipc"),
        default="ipc",
        help="server transport when --endpoint is not supplied (default: ipc)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP bind host used by portable_tcp (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "server bind endpoint override; for clients, bypass discovery and "
            "connect directly to this endpoint"
        ),
    )
    parser.add_argument(
        "--advertise-endpoint",
        help="endpoint stored in discovery when it differs from the bind endpoint",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="filesystem discovery registry directory",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="camera retry delay in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        default="",
        help=(
            "DepthAI DeviceID, PoE IP address, or USB path. On the server this "
            "is the initial device; with --open-camera it is the device to open."
        ),
    )
    parser.add_argument(
        "--no-auto-open",
        action="store_true",
        help="start the RPC server with the camera closed; use camera.open later",
    )

    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--open-camera",
        action="store_true",
        help="client action: call camera.open and exit",
    )
    control.add_argument(
        "--close-camera",
        action="store_true",
        help="client action: call camera.close and exit",
    )

    parser.add_argument(
        "--control-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for camera.open/camera.close (default: 10)",
    )
    parser.add_argument("--stream", choices=VALID_STREAMS, default="rgb")
    parser.add_argument("--thumbnail", action="store_true")
    parser.add_argument(
        "--single",
        action="store_true",
        help="use camera.get_frame instead of fetching all six images",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )

    discovery = FilesystemDiscovery(args.registry)

    if args.role == "server":
        server_name = args.server_name or args.service
        endpoint = args.endpoint or endpoint_for(
            args.backend,
            args.transport,
            server_name,
            args.host,
        )
        print(
            f"serving {args.service!r} as {server_name!r} via {args.backend} "
            f"at {endpoint}; registry: {discovery.root}"
        )
        run_server(
            endpoint,
            args.reconnect_delay,
            backend=args.backend,
            discovery=discovery,
            service=args.service,
            instance_id=server_name,
            advertise_endpoint=args.advertise_endpoint,
            device=args.device,
            auto_open=not args.no_auto_open,
        )
    elif args.role == "client":
        run_client(args, discovery)
    else:
        show_services(discovery)


if __name__ == "__main__":
    main()
