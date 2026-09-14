from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Literal

import depthai as dai
from depthai_camera_stream import CameraCalibrationResult, CameraStream
import numpy as np

try:
    from .msg import (
        CameraStatusResponse,
        CameraCalibrationResponse,
    )
except ImportError:  # Support running the files directly from one directory.
    from msg import (
        CameraStatusResponse,
        CameraCalibrationResponse,
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
        self.calibration: CameraCalibrationResponse | None = None

        self._desired_open = bool(auto_open)
        self._desired_device = config.device_ip.strip()
        self._command_revision = 0
        self._session_cancel: threading.Event | None = None
        self._session_active = False
        self._session_device = ""
        self.error = "camera is opening" if auto_open else "camera closed"

        self._thread_lock = threading.Lock()
        self.thread = self._make_thread()

    def _make_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._thread_main,
            name="depthai-camera-supervisor",
            daemon=True,
        )

    def start(self) -> None:
        # A Python Thread object cannot be started twice. Recreate it if a previous
        # worker exited unexpectedly. close() remains a permanent shutdown.
        with self._thread_lock:
            if self.stop_event.is_set():
                raise RuntimeError("camera supervisor has been permanently stopped")
            if self.thread.is_alive():
                return
            if self.thread.ident is not None:
                self.thread = self._make_thread()
            self.thread.start()

    def _thread_main(self) -> None:
        # Last-resort guard for unexpected Python-level failures in _run itself.
        # This cannot catch native SIGSEGV/SIGABRT from depthai; process isolation
        # is required if the RPC/server process must survive those failures.
        while not self.stop_event.is_set():
            try:
                self._run()
                return
            except BaseException as exc:
                if self.stop_event.is_set():
                    return
                LOG.exception("camera supervisor worker crashed; restarting", exc_info=exc)
                with self.cv:
                    self.online = False
                    self._session_active = False
                    self._session_device = ""
                    self._session_cancel = None
                    self.restart_count += 1
                    self.error = f"supervisor error: {type(exc).__name__}: {exc}"
                    self.cv.notify_all()
                self.stop_event.wait(self.reconnect_delay)

    def close(self) -> None:
        """Permanently stop the supervisor (process/server shutdown path)."""
        self.stop_event.set()
        with self.cv:
            self._desired_open = False
            self._command_revision += 1
            if self._session_cancel is not None:
                self._session_cancel.set()
            self.cv.notify_all()
        # join() raises if the Thread was never started, so keep close() idempotent.
        if self.thread.ident is not None:
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

        if self.stop_event.is_set():
            return (
                False,
                False,
                target,
                self.generation,
                "camera supervisor has been permanently stopped",
            )

        # Be tolerant if the caller forgot to start the supervisor, or if a prior
        # Python-level worker failure ended the Thread object.
        if not self.thread.is_alive():
            try:
                self.start()
            except Exception as exc:
                return False, False, target, self.generation, f"supervisor start failed: {exc}"

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

    def get_calib(self) -> CameraCalibrationResponse:
        with self.cv:
            if self.calibration is None:
                return CameraCalibrationResponse.empty(
                    camera_online=self.online,
                    error=self.error or "camera calibration is not available yet",
                )
            return self.calibration
        
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
        streams: dict[str, Any] = {}
        pipeline_started = False
        session_failed = False
        try:
            print(f"try open {device}")
            if device:
                # DepthAI v3: select a specific DeviceID, PoE IP, or USB path.
                device_handle = dai.Device(dai.DeviceInfo(device))
                print(f"device_handle {device_handle}")
                pipeline = dai.Pipeline(device_handle)
            else:
                pipeline = dai.Pipeline()

            if cancel.is_set() or self.stop_event.is_set():
                return

            streams = self.config.build(pipeline)
            print(f"streams {streams}")
            stream: CameraStream = next(iter(streams.values()))
            self.calibration = CameraCalibrationResponse(**dict(
                        ok=True, camera_online=True,**stream.read_calibration_dict()))
            print(f"calibration {self.calibration}")

            if cancel.is_set() or self.stop_event.is_set():
                return

            pipeline.start()
            pipeline_started = True
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
        except BaseException:
            # Once a DepthAI call fails, do not make additional device API calls in
            # cleanup. Some native failure paths close/disconnect the device first,
            # and calling stop/wait/close again can enter unsafe native code.
            session_failed = True
            raise
        finally:
            # Mark the current session offline before the potentially slower teardown.
            # A close RPC still waits for _session_active to become false in _run().
            with self.cv:
                if revision == self._command_revision:
                    self.online = False
                    self._session_device = ""
                    if self._desired_open and not self.stop_event.is_set():
                        self.error = "camera reconnecting"
                    self.cv.notify_all()

            # Drop CameraStream / output-queue wrappers before touching the pipeline or
            # device they reference. This avoids destructors running after Device.close().
            streams.clear()

            # Only perform graceful stop/wait for a session that actually started and
            # is ending normally (camera.close, switch, or server shutdown). If any
            # DepthAI operation raised, simply release Python references and let the
            # library's own ownership/destructors handle the already-failed session.
            if pipeline is not None and pipeline_started and not session_failed:
                with suppress(Exception):
                    pipeline.stop()
                with suppress(Exception):
                    pipeline.wait()

            # IMPORTANT: do not explicitly call device_handle.close() here. Pipeline
            # and Device share native state, and startup/disconnect failure paths may
            # already have closed it. Release in dependency order instead.
            pipeline = None
            device_handle = None

