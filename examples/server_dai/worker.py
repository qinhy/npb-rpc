from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Generic, Literal, TypeVar

import depthai as dai
from depthai_camera_stream import CameraStream

try:
    from .msg import CameraCalibrationResponse, CameraStatusResponse
    from .session_supervisor import RetryPolicy, SessionControl, SessionSupersededError, SessionSupervisor
except ImportError:  # Support running files directly from one directory.
    from msg import CameraCalibrationResponse, CameraStatusResponse
    from session_supervisor import RetryPolicy, SessionControl, SessionSupersededError, SessionSupervisor


LOG = logging.getLogger("nng_dai_camera")
StreamName = Literal["rgb", "left", "right"]
VALID_STREAMS: tuple[StreamName, ...] = ("rgb", "left", "right")
TCalibration = TypeVar("TCalibration")
CameraActionResult = tuple[bool, bool, str, int, str]


@dataclass(frozen=True)
class FrameSnapshot:
    """Detached image data safe to hand to RPC/client threads."""

    jpeg: bytes
    sequence: int
    captured_ns: int


@dataclass(frozen=True)
class CameraStoreSnapshot(Generic[TCalibration]):
    frames: dict[str, FrameSnapshot]
    calibration: TCalibration | None
    frames_published: int
    last_frame_ns: int
    active_revision: int | None


class CameraFrameStore(Generic[TCalibration]):
    """Thread-safe frame/calibration cache guarded by supervisor revision."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_revision: int | None = None
        self._revision_floor = 0
        self._frames: dict[str, FrameSnapshot] = {}
        self._sequence: dict[str, int] = {}
        self._calibration: TCalibration | None = None
        self._frames_published = self._last_frame_ns = 0

    def activate(self, revision: int) -> bool:
        with self._lock:
            if revision < self._revision_floor:
                return False
            self._active_revision = revision
            return True

    def invalidate(
        self,
        *,
        clear_frames: bool = False,
        clear_sequence: bool = False,
        clear_calibration: bool = False,
        revision_floor: int | None = None,
    ) -> None:
        with self._lock:
            self._active_revision = None
            if revision_floor is not None:
                self._revision_floor = max(self._revision_floor, revision_floor)
            if clear_frames:
                self._frames.clear()
            if clear_sequence:
                self._sequence.clear()
            if clear_calibration:
                self._calibration = None

    def deactivate(self, revision: int) -> None:
        with self._lock:
            if self._active_revision == revision:
                self._active_revision = None

    def publish(self, key: str, jpeg: bytes, *, revision: int) -> bool:
        now_ns = time.time_ns()
        with self._lock:
            if self._active_revision != revision:
                return False
            seq = self._sequence.get(key, 0) + 1
            self._sequence[key] = seq
            self._frames[key] = FrameSnapshot(bytes(jpeg), seq, now_ns)
            self._frames_published += 1
            self._last_frame_ns = now_ns
            return True

    def set_calibration(self, calibration: TCalibration, *, revision: int) -> bool:
        with self._lock:
            if self._active_revision != revision:
                return False
            self._calibration = calibration
            return True

    def get_frame(self, key: str) -> FrameSnapshot | None:
        with self._lock:
            return self._frames.get(key)

    def get_calibration(self) -> TCalibration | None:
        with self._lock:
            return self._calibration

    def snapshot(self) -> CameraStoreSnapshot[TCalibration]:
        with self._lock:
            return CameraStoreSnapshot(
                self._frames.copy(),
                self._calibration,
                self._frames_published,
                self._last_frame_ns,
                self._active_revision,
            )


@dataclass
class DaiStereoCameraStream:
    """DepthAI RGB + stereo stream configuration."""

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
            "left": (dai.CameraBoardSocket.CAM_B, self.stereo_size, self.mjpeg_quality - 5),
            "right": (dai.CameraBoardSocket.CAM_C, self.stereo_size, self.mjpeg_quality - 5),
        }
        return {
            name: CameraStream(name=name, socket=socket, size=size, mjpeg_quality=quality, **common).build()
            for name, (socket, size, quality) in specs.items()
        }


class DepthAISessionHandler:
    """Own one DepthAI session; retry/switch policy stays in SessionSupervisor."""

    def __init__(
        self,
        config: DaiStereoCameraStream,
        store: CameraFrameStore[CameraCalibrationResponse],
        *,
        idle_wait: float = 0.002,
    ) -> None:
        self.config = config
        self.store = store
        self.idle_wait = max(0.0, float(idle_wait))

    def run(self, device: str, control: SessionControl) -> None:
        revision = control.revision
        pipeline: dai.Pipeline | None = None
        device_handle: Any | None = None
        streams: dict[str, Any] = {}
        pipeline_started = session_failed = False

        if not self.store.activate(revision):
            return

        try:
            label = device or "automatic device"
            LOG.info("opening DepthAI device %r", label)
            if device:
                device_handle = dai.Device(dai.DeviceInfo(device))
                pipeline = dai.Pipeline(device_handle)
            else:
                pipeline = dai.Pipeline()

            if control.cancelled:
                return

            streams = self.config.build(pipeline)
            calibration = CameraCalibrationResponse(
                ok=True,
                camera_online=True,
                **next(iter(streams.values())).read_calibration_dict(),
            )
            calibration.rgb_resolution = self.config.rgb_size
            calibration.left_resolution = self.config.stereo_size
            calibration.right_resolution = self.config.stereo_size            
            self.store.set_calibration(calibration, revision=revision)
            if control.cancelled:
                return

            pipeline.start()
            pipeline_started = True
            if not control.mark_ready():
                return

            LOG.info("DepthAI pipeline started for %s (revision %d)", label, revision)
            while not control.cancelled:
                if not pipeline.isRunning():
                    raise RuntimeError("DepthAI pipeline is no longer running")

                got_packet = False
                for name in VALID_STREAMS:
                    if control.cancelled:
                        break
                    stream = streams[name]
                    for key, thumbnail in ((name, False), (f"{name}.thumbnail", True)):
                        packet = (
                            stream.read_latest(thumbnail=True, block=False)
                            if thumbnail
                            else stream.read_latest(block=False)
                        )
                        if packet is not None:
                            self.store.publish(key, bytes(packet.getData()), revision=revision)
                            got_packet = True

                if not got_packet:
                    control.wait_cancelled(self.idle_wait)

        except BaseException:
            # After native DepthAI failure, avoid more device API calls in cleanup.
            session_failed = True
            raise
        finally:
            self.store.deactivate(revision)
            streams.clear()  # Release queue wrappers before Pipeline/Device.
            if pipeline is not None and pipeline_started and not session_failed:
                with suppress(Exception):
                    pipeline.stop()
                with suppress(Exception):
                    pipeline.wait()
            # Preserve explicit release order; do not call device_handle.close().
            pipeline = None
            device_handle = None


class CameraSupervisor:
    """Camera-facing compatibility façade over SessionSupervisor."""

    _STATE_ERRORS = {
        "closed": "camera closed",
        "opening": "camera is opening",
        "closing": "camera closing",
        "retrying": "camera reconnecting",
        "stopped": "camera supervisor stopped",
    }

    def __init__(
        self,
        config: DaiStereoCameraStream,
        *,
        reconnect_delay: float = 1.0,
        auto_open: bool = True,
    ) -> None:
        self.config = config
        self.reconnect_delay = max(0.05, float(reconnect_delay))
        self.store: CameraFrameStore[CameraCalibrationResponse] = CameraFrameStore()
        self.handler = DepthAISessionHandler(config, self.store)
        self.supervisor: SessionSupervisor[str] = SessionSupervisor(
            self.handler,
            retry=RetryPolicy(
                initial_delay=self.reconnect_delay,
                multiplier=1.0,
                max_delay=self.reconnect_delay,
                max_attempts=None,
            ),
        )
        self._default_device = config.device_ip.strip()
        self._auto_open_pending = bool(auto_open)

    def _invalidate(self, revision: int, *, clear: bool) -> None:
        self.store.invalidate(
            clear_frames=clear,
            clear_sequence=clear,
            revision_floor=revision + 1,
        )

    def start(self) -> None:
        if self._auto_open_pending:
            self._auto_open_pending = False
            self.supervisor.open(self._default_device, timeout=0)
        else:
            self.supervisor.start()

    def close(self) -> None:
        """Permanently stop the supervisor."""
        self._auto_open_pending = False
        self._invalidate(self.supervisor.status().revision, clear=False)
        try:
            self.supervisor.shutdown(timeout=5.0)
        except TimeoutError:
            LOG.warning("camera session worker did not exit within 5 seconds")

    def open_camera(self, device: str = "", *, timeout_s: float = 10.0) -> CameraActionResult:
        """Open/switch camera and optionally wait for ONLINE."""
        target, timeout_s = device.strip(), max(0.0, float(timeout_s))
        self._auto_open_pending = False
        before = self.supervisor.status()
        if not before.desired_open or before.desired_target != target:
            self._invalidate(before.revision, clear=True)

        try:
            status = self.supervisor.open(target, timeout=timeout_s)
        except SessionSupersededError:
            status = self.supervisor.status()
            return False, status.online, target, status.generation, "camera open request was superseded"
        except TimeoutError:
            status = self.supervisor.status()
            detail = self._camera_error(status) or "camera did not become online"
            return False, status.online, target, status.generation, f"open timed out after {timeout_s:g}s: {detail}"
        except RuntimeError as exc:
            status = self.supervisor.status()
            return False, status.online, target, status.generation, str(exc)

        if status.online and status.active_target == target:
            return True, True, target, status.generation, ""
        return False, status.online, target, status.generation, status.error or f"camera state is {status.state.value}"

    def close_camera(self, *, timeout_s: float = 5.0) -> CameraActionResult:
        """Close the camera session while keeping the supervisor/RPC service alive."""
        timeout_s = max(0.0, float(timeout_s))
        self._auto_open_pending = False
        before = self.supervisor.status()
        target = before.desired_target or ""
        self._invalidate(before.revision, clear=True)

        try:
            status = self.supervisor.close_session(timeout=timeout_s)
        except TimeoutError:
            status = self.supervisor.status()
            error = f"close timed out after {timeout_s:g}s; device shutdown is still in progress"
            return False, False, target, status.generation, error
        return True, False, target, status.generation, ""

    def get_frame(self, stream: str, thumbnail: bool) -> FrameSnapshot | None:
        return self.store.get_frame(f"{stream}.thumbnail" if thumbnail else stream)

    def _camera_error(self, status: Any | None = None) -> str:
        if status is None:
            status = self.supervisor.status()
        if status.error:
            return status.error
        if self._auto_open_pending:
            return "camera is opening"
        return self._STATE_ERRORS.get(status.state.value, "")

    def get_calib(self) -> CameraCalibrationResponse:
        calibration = self.store.get_calibration()
        if calibration is not None:
            return calibration
        status = self.supervisor.status()
        return CameraCalibrationResponse.empty(
            camera_online=status.online,
            error=self._camera_error(status) or "camera calibration is not available yet",
        )

    def snapshot_all(self) -> tuple[dict[str, FrameSnapshot], bool, int, int, int, int, str]:
        status, data = self.supervisor.status(), self.store.snapshot()
        return (
            data.frames,
            status.online,
            status.generation,
            status.restart_count,
            data.frames_published,
            data.last_frame_ns,
            self._camera_error(status),
        )

    def status(self) -> CameraStatusResponse:
        status, data = self.supervisor.status(), self.store.snapshot()
        return CameraStatusResponse(
            requested_open=status.desired_open or self._auto_open_pending,
            online=status.online,
            device=status.desired_target if status.desired_target is not None else self._default_device,
            generation=status.generation,
            restart_count=status.restart_count,
            frames_published=data.frames_published,
            last_frame_ns=data.last_frame_ns,
            error=self._camera_error(status),
        )

    @property
    def online(self) -> bool:
        return self.supervisor.status().online

    @property
    def generation(self) -> int:
        return self.supervisor.status().generation

    @property
    def restart_count(self) -> int:
        return self.supervisor.status().restart_count

    @property
    def frames_published(self) -> int:
        return self.store.snapshot().frames_published

    @property
    def last_frame_ns(self) -> int:
        return self.store.snapshot().last_frame_ns

    @property
    def error(self) -> str:
        return self._camera_error()
