#!/usr/bin/env python3
"""Fault-tolerant DepthAI MJPEG camera server over npb_rpc.

The RPC server and the camera capture loop have deliberately separate lifetimes.
A DepthAI startup/read/disconnect error only restarts the camera worker; it must
not terminate the RPC server. The server can run directly or advertise itself
through FilesystemDiscovery, using either NNG or ZeroMQ.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Literal

import depthai as dai
import numpy as np
from npb_rpc import (
    DiscoveredRpcServer,
    FilesystemDiscovery,
    NngRpcServer,
    RpcContext,
    ZmqRpcServer,
)

from depthai_camera_stream import CameraStream

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
        EmptyRequest,
    )
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
        EmptyRequest,
    )


LOG = logging.getLogger("nng_dai_camera")
STOP = threading.Event()

StreamName = Literal["rgb", "left", "right"]
VALID_STREAMS = ("rgb", "left", "right")


@dataclass
class DaiStereoCameraStream:
    """Same DepthAI stream configuration as rgb_with_thumbnail_and_stereo.py."""

    # DeviceID, PoE IP address, or USB path. Empty means automatic selection.
    device_ip: str = ""
    rgb_size: tuple[int, int] = (3872, 3008)
    stereo_size: tuple[int, int] = (1280, 800)
    mjpeg_quality: int = 95
    fps: float = 12.0
    input_type: str = "NV12"
    resize_mode: str = "CROP"
    max_exposure_us: int = 16667

    def build(self, pipeline: dai.Pipeline) -> dict[str, Any]:
        common = dict(
            pipeline=pipeline,
            fps=self.fps,
            max_exposure_us=self.max_exposure_us,
            input_type={"NV12": dai.ImgFrame.Type.NV12}[self.input_type],
            resize_mode={"CROP": dai.ImgResizeMode.CROP}[self.resize_mode],
            queue_size=1,
            queue_blocking=False,
            thumbnail_size=(192, 150),
            thumbnail_fps=self.fps,
            thumbnail_mjpeg_quality=70,
            thumbnail_queue_size=1,
            thumbnail_queue_blocking=False,
        )
        specs = {
            "rgb": (dai.CameraBoardSocket.CAM_A, self.rgb_size, self.mjpeg_quality),
            "left": (
                dai.CameraBoardSocket.CAM_B,
                self.stereo_size,
                self.mjpeg_quality - 5,
            ),
            "right": (
                dai.CameraBoardSocket.CAM_C,
                self.stereo_size,
                self.mjpeg_quality - 5,
            ),
        }
        return {
            name: CameraStream(
                name=name,
                socket=socket,
                size=size,
                mjpeg_quality=quality,
                **common,
            ).build()
            for name, (socket, size, quality) in specs.items()
        }


@dataclass(frozen=True)
class FrameSnapshot:
    jpeg: bytes
    sequence: int
    captured_ns: int


class CameraSupervisor:
    """Own a restartable camera worker and allow runtime open/close/switch."""

    def __init__(
        self,
        config: DaiStereoCameraStream,
        *,
        reconnect_delay: float = 1.0,
        auto_open: bool = True,
    ) -> None:
        self.config = config
        self.reconnect_delay = max(0.05, reconnect_delay)
        self.stop_event = threading.Event()
        self.cv = threading.Condition()
        self.frames: dict[str, FrameSnapshot] = {}
        self.sequence: dict[str, int] = {}
        self.online = False
        self.generation = 0
        self.restart_count = 0
        self.frames_published = 0
        self.last_frame_ns = 0

        self._desired_open = bool(auto_open)
        self._desired_device = config.device_ip.strip()
        self._command_revision = 0
        self._session_cancel: threading.Event | None = None
        self._session_active = False
        self._session_device = ""
        self.error = "camera is opening" if auto_open else "camera closed"

        self.thread = threading.Thread(
            target=self._run,
            name="depthai-camera-supervisor",
            daemon=True,
        )

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def close(self) -> None:
        """Permanently stop the supervisor (process/server shutdown path)."""
        self.stop_event.set()
        with self.cv:
            self._desired_open = False
            self._command_revision += 1
            if self._session_cancel is not None:
                self._session_cancel.set()
            self.cv.notify_all()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            # Do not block RPC/process shutdown forever on a stuck device driver.
            LOG.warning("camera worker did not exit within 5 seconds")
        with self.cv:
            self.online = False
            self._session_device = ""
            self.error = "camera supervisor stopped"
            self.cv.notify_all()

    def open_camera(
        self,
        device: str = "",
        *,
        timeout_s: float = 10.0,
    ) -> tuple[bool, bool, str, int, str]:
        """Open/switch the camera and optionally wait for it to become online."""
        target = device.strip()
        timeout_s = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s

        with self.cv:
            if (
                self._desired_open
                and self._desired_device == target
                and self.online
                and self._session_device == target
            ):
                return True, True, target, self.generation, ""

            if not self._desired_open or self._desired_device != target:
                self._desired_open = True
                self._desired_device = target
                self._command_revision += 1
                self.online = False
                self._session_device = ""
                # Never return cached frames from a different explicitly selected camera.
                self.frames.clear()
                self.sequence.clear()
                self.error = "camera is opening"
                if self._session_cancel is not None:
                    self._session_cancel.set()
                self.cv.notify_all()
            elif not self.online:
                # Already trying this target; wake the worker in case it is waiting to retry.
                self.cv.notify_all()

            while True:
                if (
                    self.online
                    and self._desired_open
                    and self._desired_device == target
                    and self._session_device == target
                ):
                    return True, True, target, self.generation, ""

                if not self._desired_open or self._desired_device != target:
                    return (
                        False,
                        self.online,
                        target,
                        self.generation,
                        "camera open request was superseded",
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self.error or "camera did not become online"
                    return (
                        False,
                        self.online,
                        target,
                        self.generation,
                        f"open timed out after {timeout_s:g}s: {detail}",
                    )
                self.cv.wait(remaining)

    def close_camera(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> tuple[bool, bool, str, int, str]:
        """Close the camera session but keep the supervisor/RPC service alive."""
        timeout_s = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s

        with self.cv:
            target = self._desired_device
            self._desired_open = False
            self._command_revision += 1
            self.online = False
            self._session_device = ""
            self.frames.clear()
            self.sequence.clear()
            self.error = "camera closing" if self._session_active else "camera closed"
            if self._session_cancel is not None:
                self._session_cancel.set()
            self.cv.notify_all()

            while self._session_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (
                        False,
                        False,
                        target,
                        self.generation,
                        f"close timed out after {timeout_s:g}s; device shutdown is still in progress",
                    )
                self.cv.wait(remaining)

            self.error = "camera closed"
            self.cv.notify_all()
            return True, False, target, self.generation, ""

    def _set_offline(self, error: str) -> None:
        with self.cv:
            self.online = False
            self._session_device = ""
            self.error = error
            self.cv.notify_all()

    def _set_online(self, device: str, revision: int) -> bool:
        """Mark online only if this session is still the requested session."""
        with self.cv:
            if (
                self.stop_event.is_set()
                or not self._desired_open
                or revision != self._command_revision
                or device != self._desired_device
            ):
                return False
            self.online = True
            self._session_device = device
            self.error = ""
            self.generation += 1
            self.cv.notify_all()
            return True

    def _publish(self, key: str, packet: Any, revision: int) -> None:
        # Detach bytes from the DepthAI packet immediately. RPC readers never hold
        # or touch DepthAI objects, so a device teardown cannot invalidate a reply.
        jpeg = bytes(packet.getData())
        now_ns = time.time_ns()
        with self.cv:
            # Drop packets racing with camera.close or a switch to another device.
            if (
                not self._desired_open
                or revision != self._command_revision
                or not self.online
            ):
                return
            seq = self.sequence.get(key, 0) + 1
            self.sequence[key] = seq
            self.frames[key] = FrameSnapshot(jpeg, seq, now_ns)
            self.frames_published += 1
            self.last_frame_ns = now_ns

    def get_frame(self, stream: str, thumbnail: bool) -> FrameSnapshot | None:
        key = f"{stream}.thumbnail" if thumbnail else stream
        with self.cv:
            return self.frames.get(key)

    def snapshot_all(
        self,
    ) -> tuple[dict[str, FrameSnapshot], bool, int, int, int, int, str]:
        """Atomically copy the latest six image slots plus camera state."""
        with self.cv:
            return (
                self.frames.copy(),
                self.online,
                self.generation,
                self.restart_count,
                self.frames_published,
                self.last_frame_ns,
                self.error,
            )

    def status(self) -> CameraStatusResponse:
        with self.cv:
            return CameraStatusResponse(
                requested_open=self._desired_open,
                online=self.online,
                device=self._desired_device,
                generation=self.generation,
                restart_count=self.restart_count,
                frames_published=self.frames_published,
                last_frame_ns=self.last_frame_ns,
                error=self.error,
            )

    def _run(self) -> None:
        # The worker persists for the lifetime of the RPC service. camera.close only
        # ends the current DepthAI session; camera.open can start another later.
        while not self.stop_event.is_set():
            with self.cv:
                while not self._desired_open and not self.stop_event.is_set():
                    self.cv.wait(0.5)
                if self.stop_event.is_set():
                    break

                revision = self._command_revision
                target = self._desired_device
                cancel = threading.Event()
                self._session_cancel = cancel
                self._session_active = True
                self.error = "camera is opening"
                self.cv.notify_all()

            failed = False
            try:
                self._camera_session(target, revision, cancel)
            except Exception as exc:
                failed = True
                with self.cv:
                    # A close/switch intentionally tears down the session and must not
                    # be counted as a camera restart failure.
                    if (
                        self._desired_open
                        and revision == self._command_revision
                        and target == self._desired_device
                        and not cancel.is_set()
                    ):
                        self.restart_count += 1
                        self.online = False
                        self._session_device = ""
                        self.error = f"{type(exc).__name__}: {exc}"
                        self.cv.notify_all()
                        LOG.exception(
                            "camera session for %r failed; reconnecting",
                            target or "automatic device",
                        )
            finally:
                with self.cv:
                    self._session_active = False
                    if self._session_cancel is cancel:
                        self._session_cancel = None
                    if not self._desired_open:
                        self.online = False
                        self._session_device = ""
                        self.error = "camera closed"
                    self.cv.notify_all()

            with self.cv:
                should_retry = (
                    failed
                    and self._desired_open
                    and revision == self._command_revision
                    and target == self._desired_device
                    and not self.stop_event.is_set()
                )
                if should_retry:
                    # A close or a new open request wakes this wait immediately.
                    self.cv.wait(self.reconnect_delay)

        self._set_offline("camera supervisor stopped")

    def _camera_session(
        self,
        device: str,
        revision: int,
        cancel: threading.Event,
    ) -> None:
        pipeline: dai.Pipeline | None = None
        device_handle: Any | None = None
        try:
            if device:
                # DepthAI v3: select a specific DeviceID, PoE IP, or USB path.
                device_handle = dai.Device(dai.DeviceInfo(device))
                pipeline = dai.Pipeline(device_handle)
            else:
                pipeline = dai.Pipeline()

            if cancel.is_set() or self.stop_event.is_set():
                return

            streams = self.config.build(pipeline)
            if cancel.is_set() or self.stop_event.is_set():
                return

            pipeline.start()
            if not self._set_online(device, revision):
                return

            LOG.info(
                "DepthAI pipeline started for %s (generation %d)",
                device or "automatic device",
                self.generation,
            )

            while not self.stop_event.is_set() and not cancel.is_set():
                if not pipeline.isRunning():
                    raise RuntimeError("DepthAI pipeline is no longer running")

                got_packet = False
                for name in VALID_STREAMS:
                    if cancel.is_set() or self.stop_event.is_set():
                        break

                    # Non-blocking reads keep camera.close and camera switching
                    # responsive even if one device queue is idle.
                    packet = streams[name].read_latest(block=False)
                    if packet is not None:
                        self._publish(name, packet, revision)
                        got_packet = True

                    thumbnail = streams[name].read_latest(
                        thumbnail=True,
                        block=False,
                    )
                    if thumbnail is not None:
                        self._publish(f"{name}.thumbnail", thumbnail, revision)
                        got_packet = True

                if not got_packet:
                    cancel.wait(0.002)
        finally:
            # Mark the current session offline before the potentially slower device
            # teardown. A close RPC still waits for _session_active to become false.
            with self.cv:
                if revision == self._command_revision:
                    self.online = False
                    self._session_device = ""
                    if self._desired_open and not self.stop_event.is_set():
                        self.error = "camera reconnecting"
                    self.cv.notify_all()

            if pipeline is not None:
                with suppress(Exception):
                    pipeline.stop()
                with suppress(Exception):
                    pipeline.wait()
            if device_handle is not None:
                # Explicitly selected devices are host objects we created ourselves.
                with suppress(Exception):
                    device_handle.close()


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

    @server.method(
        "camera.open",
        request=CameraOpenRequest,
        response=CameraControlResponse,
    )
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

    @server.method(
        "camera.close",
        request=CameraCloseRequest,
        response=CameraControlResponse,
    )
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

    @server.method(
        "camera.status",
        request=EmptyRequest,
        response=CameraStatusResponse,
    )
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

    @server.method(
        "camera.frame",
        request=CameraFrameSetRequest,
        response=CameraFrameSetResponse,
    )
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

    @server.method(
        "camera.get_frame",
        request=CameraFrameRequest,
        response=CameraFrameResponse,
    )
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

    return server


def run_server(
    endpoint: str,
    reconnect_delay: float,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = "camera",
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
