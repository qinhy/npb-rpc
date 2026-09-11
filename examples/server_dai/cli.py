from __future__ import annotations

import argparse
import logging
import threading
from typing import Literal

from npb_rpc import portable_ipc

from rpcapi import *
from server import *

LOG = logging.getLogger("nng_dai_camera")
STOP = threading.Event()

TCP_ENDPOINT = "tcp://127.0.0.1:5556"
IPC_ENDPOINT = portable_ipc("npb-rpc-dai-camera")
VALID_STREAMS = ("rgb", "left", "right")


def run_client(endpoint: str, stream: Literal["rgb", "left", "right"], thumbnail: bool, single: bool) -> None:
    status = client_camera_status(endpoint=endpoint)
    print(
        "status:",
        f"online={status.online}",
        f"generation={status.generation}",
        f"restarts={status.restart_count}",
        f"published={status.frames_published}",
        f"error={status.error!r}",
    )

    if single:
        # Backward-compatible diagnostic path for one stream.
        frame = client_camera_frame(
            CameraFrameRequest(stream=stream, thumbnail=thumbnail),
            endpoint=endpoint,
        )
        if not frame.ok:
            print(f"frame unavailable: {frame.error}")
            return
        suffix = "_thumbnail" if thumbnail else ""
        path = f"{stream}{suffix}.jpg"
        with open(path, "wb") as f:
            f.write(frame.jpeg.tobytes())
        print(
            f"wrote {path}: {frame.jpeg.nbytes} bytes, "
            f"seq={frame.sequence}, online={frame.camera_online}"
        )
        return

    frames = client_camera_frame_set(CameraFrameSetRequest(), endpoint=endpoint)
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


def endpoint_for(transport: str) -> str:
    return {"tcp": TCP_ENDPOINT, "ipc": IPC_ENDPOINT}[transport]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fault-tolerant NNG DepthAI camera")
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument(
        "--transport",
        choices=("tcp", "ipc"),
        default="ipc",
        help="NNG transport when --endpoint is not supplied (default: ipc)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="override NNG endpoint, e.g. tcp://0.0.0.0:5556",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="camera retry delay in seconds (default: 1.0)",
    )
    parser.add_argument("--stream", choices=VALID_STREAMS, default="rgb")
    parser.add_argument("--thumbnail", action="store_true")
    parser.add_argument(
        "--single",
        action="store_true",
        help="use legacy camera.get_frame instead of fetching all six images",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )
    endpoint = args.endpoint or endpoint_for(args.transport)

    if args.role == "server":
        run_server(endpoint, args.reconnect_delay)
    else:
        run_client(endpoint, args.stream, args.thumbnail, args.single)


if __name__ == "__main__":
    main()
