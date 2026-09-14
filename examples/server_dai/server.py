#!/usr/bin/env python3
"""Fault-tolerant DepthAI MJPEG camera server over npb_rpc.

The RPC server and the camera capture loop have deliberately separate lifetimes.
A DepthAI startup/read/disconnect error only restarts the camera worker; it must
not terminate the RPC server. The server can run directly or advertise itself
through FilesystemDiscovery, using either NNG or ZeroMQ.
"""

from __future__ import annotations

import logging
import threading
from typing import Literal

import numpy as np

from npb_rpc import (
    DiscoveredRpcServer,
    FilesystemDiscovery,
    NngRpcServer,
    RpcContext,
    ZmqRpcServer,
)

try:
    from .msg import (
        CameraCloseRequest,
        CameraControlResponse,
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraOpenRequest,
        CameraStatusResponse,
        CameraCalibrationResponse,
        EmptyRequest,
    )
    from .interface import CAMERA_API
    from .worker import FrameSnapshot, CameraSupervisor, DaiStereoCameraStream

except ImportError:  # Support running the files directly from one directory.
    from msg import (
        CameraCloseRequest,
        CameraControlResponse,
        CameraFrameRequest,
        CameraFrameResponse,
        CameraFrameSetRequest,
        CameraFrameSetResponse,
        CameraOpenRequest,
        CameraStatusResponse,
        CameraCalibrationResponse,
        EmptyRequest,
    )
    from interface import CAMERA_API
    from worker import FrameSnapshot, CameraSupervisor, DaiStereoCameraStream


LOG = logging.getLogger("nng_dai_camera")
STOP = threading.Event()

StreamName = Literal["rgb", "left", "right"]
VALID_STREAMS = ("rgb", "left", "right")

def _snapshot_payload(
    frames: dict[str, FrameSnapshot], key: str
) -> tuple[np.ndarray, int, int]:
    frame = frames.get(key)
    if frame is None:
        return np.empty(0, dtype=np.uint8), 0, 0
    return (
        np.frombuffer(frame.jpeg, dtype=np.uint8),
        frame.sequence,
        frame.captured_ns,
    )


def _empty_frame_response(error: str) -> CameraFrameSetResponse:
    empty = np.empty(0, dtype=np.uint8)
    return CameraFrameSetResponse(
        ok=False,
        camera_online=False,
        generation=0,
        restart_count=0,
        rgb=empty,
        rgb_sequence=0,
        rgb_captured_ns=0,
        left=empty,
        left_sequence=0,
        left_captured_ns=0,
        right=empty,
        right_sequence=0,
        right_captured_ns=0,
        rgb_thumbnail=empty,
        rgb_thumbnail_sequence=0,
        rgb_thumbnail_captured_ns=0,
        left_thumbnail=empty,
        left_thumbnail_sequence=0,
        left_thumbnail_captured_ns=0,
        right_thumbnail=empty,
        right_thumbnail_sequence=0,
        right_thumbnail_captured_ns=0,
        error=error,
    )


def make_server(
    endpoint: str,
    camera: CameraSupervisor,
    *,
    backend: str = "nng",
):
    """Create a backend-specific RPC server and register the camera methods."""
    if backend == "nng":
        server_type = NngRpcServer
    elif backend == "zmq":
        server_type = ZmqRpcServer
    else:
        raise ValueError(f"unsupported RPC backend: {backend!r}")

    server = server_type.bind(endpoint)

    @CAMERA_API.open.server_method(server)
    def camera_open(
        request: CameraOpenRequest,
        context: RpcContext,
    ) -> CameraControlResponse:
        try:
            ok, online, device, generation, error = camera.open_camera(
                request.device,
                timeout_s=request.timeout_s,
            )
            return CameraControlResponse(
                ok=ok,
                requested_open=True,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            LOG.exception("camera.open handler failed")
            return CameraControlResponse(
                ok=False,
                requested_open=True,
                online=False,
                device=str(getattr(request, "device", "")),
                generation=0,
                error=f"open handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.close.server_method(server)
    def camera_close(
        request: CameraCloseRequest,
        context: RpcContext,
    ) -> CameraControlResponse:
        try:
            ok, online, device, generation, error = camera.close_camera(
                timeout_s=request.timeout_s,
            )
            return CameraControlResponse(
                ok=ok,
                requested_open=False,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            LOG.exception("camera.close handler failed")
            return CameraControlResponse(
                ok=False,
                requested_open=False,
                online=False,
                device="",
                generation=0,
                error=f"close handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.status.server_method(server)
    def camera_status(
        request: EmptyRequest,
        context: RpcContext,
    ) -> CameraStatusResponse:
        # RPC methods also have a defensive boundary so a bad camera state can
        # return an error response rather than escape through the server loop.
        try:
            return camera.status()
        except Exception as exc:
            LOG.exception("camera.status handler failed")
            return CameraStatusResponse(
                requested_open=False,
                online=False,
                device="",
                generation=0,
                restart_count=0,
                frames_published=0,
                last_frame_ns=0,
                error=f"status handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.frame.server_method(server)
    def camera_frame(
        request: CameraFrameSetRequest,
        context: RpcContext,
    ) -> CameraFrameSetResponse:
        """Return RGB, stereo, and all thumbnails in one RPC response."""
        try:
            (
                frames,
                online,
                generation,
                restarts,
                _published,
                _last_ns,
                camera_error,
            ) = camera.snapshot_all()

            rgb, rgb_seq, rgb_ns = _snapshot_payload(frames, "rgb")
            left, left_seq, left_ns = _snapshot_payload(frames, "left")
            right, right_seq, right_ns = _snapshot_payload(frames, "right")
            rgb_thumb, rgb_thumb_seq, rgb_thumb_ns = _snapshot_payload(
                frames, "rgb.thumbnail"
            )
            left_thumb, left_thumb_seq, left_thumb_ns = _snapshot_payload(
                frames, "left.thumbnail"
            )
            right_thumb, right_thumb_seq, right_thumb_ns = _snapshot_payload(
                frames, "right.thumbnail"
            )

            expected = (
                "rgb",
                "left",
                "right",
                "rgb.thumbnail",
                "left.thumbnail",
                "right.thumbnail",
            )
            missing = [key for key in expected if key not in frames]
            ok = not missing
            if missing:
                detail = "missing: " + ", ".join(missing)
                error = f"{camera_error}; {detail}" if camera_error else detail
            elif not online and camera_error:
                # Complete cached data is still returned when the device is offline.
                error = camera_error
            else:
                error = ""

            return CameraFrameSetResponse(
                ok=ok,
                camera_online=online,
                generation=generation,
                restart_count=restarts,
                rgb=rgb,
                rgb_sequence=rgb_seq,
                rgb_captured_ns=rgb_ns,
                left=left,
                left_sequence=left_seq,
                left_captured_ns=left_ns,
                right=right,
                right_sequence=right_seq,
                right_captured_ns=right_ns,
                rgb_thumbnail=rgb_thumb,
                rgb_thumbnail_sequence=rgb_thumb_seq,
                rgb_thumbnail_captured_ns=rgb_thumb_ns,
                left_thumbnail=left_thumb,
                left_thumbnail_sequence=left_thumb_seq,
                left_thumbnail_captured_ns=left_thumb_ns,
                right_thumbnail=right_thumb,
                right_thumbnail_sequence=right_thumb_seq,
                right_thumbnail_captured_ns=right_thumb_ns,
                error=error,
            )
        except Exception as exc:
            LOG.exception("camera.frame handler failed")
            return _empty_frame_response(
                f"frame handler error: {type(exc).__name__}: {exc}"
            )

    @CAMERA_API.get_frame.server_method(server)
    def camera_get_frame(
        request: CameraFrameRequest,
        context: RpcContext,
    ) -> CameraFrameResponse:
        try:
            if request.stream not in VALID_STREAMS:
                return CameraFrameResponse(
                    ok=False,
                    camera_online=camera.status().online,
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=f"unknown stream: {request.stream!r}",
                )

            status = camera.status()
            frame = camera.get_frame(request.stream, request.thumbnail)
            if frame is None:
                return CameraFrameResponse(
                    ok=False,
                    camera_online=status.online,
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=status.error or "frame is not available yet",
                )
            return CameraFrameResponse(
                ok=True,
                camera_online=status.online,
                stream=request.stream,
                thumbnail=request.thumbnail,
                sequence=frame.sequence,
                captured_ns=frame.captured_ns,
                jpeg=np.frombuffer(frame.jpeg, dtype=np.uint8),
                # If offline, the JPEG is the most recent cached frame. The caller
                # can decide whether to use it from camera_online/captured_ns.
                error=status.error if not status.online else "",
            )
        except Exception as exc:
            LOG.exception("camera.get_frame handler failed")
            return CameraFrameResponse(
                ok=False,
                camera_online=False,
                stream=str(getattr(request, "stream", "")),
                thumbnail=bool(getattr(request, "thumbnail", False)),
                sequence=0,
                captured_ns=0,
                jpeg=np.empty(0, dtype=np.uint8),
                error=f"frame handler error: {type(exc).__name__}: {exc}",
            )

    @CAMERA_API.get_calib.server_method(server)
    def camera_get_calib(
        request: EmptyRequest,
        context: RpcContext,
    ) -> CameraCalibrationResponse:
        try:
            return camera.get_calib()
        except Exception as exc:
            LOG.exception("camera.get_calib handler failed")
            return CameraCalibrationResponse.empty(error=str(exc))

    return server


def run_server(
    endpoint: str,
    reconnect_delay: float,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = CAMERA_API.service,
    instance_id: str | None = None,
    advertise_endpoint: str | None = None,
    device: str = "",
    auto_open: bool = True,
) -> None:
    """Run the resilient camera service, optionally registered for discovery."""
    STOP.clear()
    camera = CameraSupervisor(
        DaiStereoCameraStream(device_ip=device),
        reconnect_delay=reconnect_delay,
        auto_open=auto_open,
    )
    camera.start()

    try:
        # Camera errors are handled by CameraSupervisor. This outer loop additionally
        # prevents an unexpected transport/server exception from permanently ending
        # the service; it re-binds and re-registers after a short delay.
        while not STOP.is_set():
            try:
                raw_server = make_server(endpoint, camera, backend=backend)
                if discovery is None:
                    server = raw_server
                else:
                    server = DiscoveredRpcServer(
                        service,
                        raw_server,
                        discovery,
                        instance_id=instance_id or service,
                        advertise_endpoint=advertise_endpoint,
                    )

                LOG.info(
                    "%s camera server listening on %s%s",
                    backend.upper(),
                    endpoint,
                    (
                        f" as {instance_id or service!r} for service {service!r}"
                        if discovery is not None
                        else ""
                    ),
                )
                with server:
                    server.serve_forever()
                if not STOP.is_set():
                    raise RuntimeError("RPC serve_forever returned unexpectedly")
            except KeyboardInterrupt:
                STOP.set()
            except Exception:
                if STOP.is_set():
                    break
                LOG.exception("RPC server failed; restarting")
                STOP.wait(1.0)
    finally:
        camera.close()
