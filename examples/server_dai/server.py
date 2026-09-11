#!/usr/bin/env python3
"""Fault-tolerant DepthAI MJPEG camera server over npb_rpc/NNG.

The NNG RPC server and the camera capture loop have deliberately separate
lifetimes.  A DepthAI startup/read/disconnect error only restarts the camera
worker; it must not terminate the NNG server.
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
from npb_rpc import NngRpcServer, RpcContext

from depthai_camera_stream import CameraStream

from msg import *


LOG = logging.getLogger("nng_dai_camera")
STOP = threading.Event()

StreamName = Literal["rgb", "left", "right"]
VALID_STREAMS = ("rgb", "left", "right")


@dataclass
class DaiStereoCameraStream:
    """Same DepthAI stream configuration as rgb_with_thumbnail_and_stereo.py."""

    device_ip: str = ""  # Kept for config compatibility; original code does not use it.
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
    """Own the camera in a restartable worker and expose immutable snapshots."""

    def __init__(
        self,
        config: DaiStereoCameraStream,
        *,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.config = config
        self.reconnect_delay = max(0.05, reconnect_delay)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frames: dict[str, FrameSnapshot] = {}
        self.sequence: dict[str, int] = {}
        self.online = False
        self.generation = 0
        self.restart_count = 0
        self.frames_published = 0
        self.last_frame_ns = 0
        self.error = "camera has not started yet"
        self.thread = threading.Thread(
            target=self._run,
            name="depthai-camera-supervisor",
            daemon=True,
        )

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            # Do not block NNG/process shutdown forever on a stuck device driver.
            LOG.warning("camera worker did not exit within 5 seconds")

    def _set_offline(self, error: str) -> None:
        with self.lock:
            self.online = False
            self.error = error

    def _set_online(self) -> None:
        with self.lock:
            self.online = True
            self.error = ""
            self.generation += 1

    def _publish(self, key: str, packet: Any) -> None:
        # Detach bytes from the DepthAI packet immediately. RPC readers never hold
        # or touch DepthAI objects, so a device teardown cannot invalidate a reply.
        jpeg = bytes(packet.getData())
        now_ns = time.time_ns()
        with self.lock:
            seq = self.sequence.get(key, 0) + 1
            self.sequence[key] = seq
            self.frames[key] = FrameSnapshot(jpeg, seq, now_ns)
            self.frames_published += 1
            self.last_frame_ns = now_ns

    def get_frame(self, stream: str, thumbnail: bool) -> FrameSnapshot | None:
        key = f"{stream}.thumbnail" if thumbnail else stream
        with self.lock:
            return self.frames.get(key)

    def snapshot_all(
        self,
    ) -> tuple[dict[str, FrameSnapshot], bool, int, int, int, int, str]:
        """Atomically copy the latest six image slots plus camera state."""
        with self.lock:
            return (
                self.frames.copy(),
                self.online,
                self.generation,
                self.restart_count,
                self.frames_published,
                self.last_frame_ns,
                self.error,
            )

    def status(self) -> tuple[bool, int, int, int, int, str]:
        with self.lock:
            return (
                self.online,
                self.generation,
                self.restart_count,
                self.frames_published,
                self.last_frame_ns,
                self.error,
            )

    def _run(self) -> None:
        # This is the critical crash barrier: no normal camera exception is allowed
        # to escape this thread. A failed session is cleaned up and retried forever.
        while not self.stop_event.is_set():
            try:
                self._camera_session()
                if self.stop_event.is_set():
                    break
                raise RuntimeError("DepthAI pipeline stopped")
            except Exception as exc:
                with self.lock:
                    self.restart_count += 1
                self._set_offline(f"{type(exc).__name__}: {exc}")
                LOG.exception("camera session failed; reconnecting")

            self.stop_event.wait(self.reconnect_delay)

        self._set_offline("camera supervisor stopped")

    def _camera_session(self) -> None:
        pipeline: dai.Pipeline | None = None
        try:
            pipeline = dai.Pipeline()
            streams = self.config.build(pipeline)
            pipeline.start()
            self._set_online()
            LOG.info("DepthAI pipeline started (generation %d)", self.generation)

            while not self.stop_event.is_set():
                if not pipeline.isRunning():
                    raise RuntimeError("DepthAI pipeline is no longer running")

                got_packet = False
                for name in VALID_STREAMS:
                    # Non-blocking reads are intentional. The NNG service must not
                    # become dependent on a wedged camera queue/device.
                    packet = streams[name].read_latest(block=False)
                    if packet is not None:
                        self._publish(name, packet)
                        got_packet = True

                    thumbnail = streams[name].read_latest(
                        thumbnail=True,
                        block=False,
                    )
                    if thumbnail is not None:
                        self._publish(f"{name}.thumbnail", thumbnail)
                        got_packet = True

                if not got_packet:
                    self.stop_event.wait(0.002)
        finally:
            self._set_offline("camera reconnecting")
            if pipeline is not None:
                # Disconnects can make stop()/wait() throw too; cleanup must never
                # kill the supervisor.
                with suppress(Exception):
                    pipeline.stop()
                with suppress(Exception):
                    pipeline.wait()

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


def make_server(endpoint: str, camera: CameraSupervisor) -> NngRpcServer:
    server = NngRpcServer.bind(endpoint)

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
            online, generation, restarts, published, last_ns, error = camera.status()
            return CameraStatusResponse(
                online=online,
                generation=generation,
                restart_count=restarts,
                frames_published=published,
                last_frame_ns=last_ns,
                error=error,
            )
        except Exception as exc:
            LOG.exception("camera.status handler failed")
            return CameraStatusResponse(
                online=False,
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
                    camera_online=camera.status()[0],
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=f"unknown stream: {request.stream!r}",
                )

            online, _, _, _, _, camera_error = camera.status()
            frame = camera.get_frame(request.stream, request.thumbnail)
            if frame is None:
                return CameraFrameResponse(
                    ok=False,
                    camera_online=online,
                    stream=request.stream,
                    thumbnail=request.thumbnail,
                    sequence=0,
                    captured_ns=0,
                    jpeg=np.empty(0, dtype=np.uint8),
                    error=camera_error or "frame is not available yet",
                )
            return CameraFrameResponse(
                ok=True,
                camera_online=online,
                stream=request.stream,
                thumbnail=request.thumbnail,
                sequence=frame.sequence,
                captured_ns=frame.captured_ns,
                jpeg=np.frombuffer(frame.jpeg, dtype=np.uint8),
                # If offline, the JPEG is the most recent cached frame. The caller
                # can decide whether to use it from camera_online/captured_ns.
                error=camera_error if not online else "",
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


def run_server(endpoint: str, reconnect_delay: float) -> None:
    camera = CameraSupervisor(
        DaiStereoCameraStream(),
        reconnect_delay=reconnect_delay,
    )
    camera.start()

    try:
        # Camera errors are handled by CameraSupervisor. This outer loop additionally
        # prevents an unexpected transport/server exception from permanently ending
        # the service; it re-binds after a short delay.
        while not STOP.is_set():
            try:
                server = make_server(endpoint, camera)
                LOG.info("NNG camera server listening on %s", endpoint)
                with server:
                    server.serve_forever()
                if not STOP.is_set():
                    raise RuntimeError("NNG serve_forever returned unexpectedly")
            except KeyboardInterrupt:
                STOP.set()
            except Exception:
                if STOP.is_set():
                    break
                LOG.exception("NNG server failed; restarting")
                STOP.wait(1.0)
    finally:
        camera.close()
