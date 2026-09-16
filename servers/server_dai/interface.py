from __future__ import annotations

"""Unified RPC + HTTP interface for the DepthAI camera service."""

import logging
from typing import Any

import numpy as np
from npb_rpc.utils import add_fastapi_routes

from servers.msg.dai import (
    CameraCalibrationResponse,
    CameraCloseRequest,
    CameraControlResponse,
    CameraFrameRequest,
    CameraFrameResponse,
    CameraFrameSetRequest,
    CameraFrameSetResponse,
    CameraOpenRequest,
    CameraStatusResponse,
    EmptyRequest,
    CameraInterface,
)
from servers.server_dai.worker import CameraSupervisor, FrameSnapshot

LOG = logging.getLogger("dai_camera.interface")
STREAMS = ("rgb", "left", "right")
FRAME_KEYS = (*STREAMS, *(f"{name}.thumbnail" for name in STREAMS))

# ---------------------------------------------------------------------------
# Camera service implementation
# ---------------------------------------------------------------------------

def _frame_fields(frames: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in FRAME_KEYS:
        name = key.replace(".", "_")
        frame = frames.get(key)
        fields[name] = np.frombuffer(frame.jpeg, dtype=np.uint8) if frame else np.empty(0, dtype=np.uint8)
        fields[f"{name}_sequence"] = frame.sequence if frame else 0
        fields[f"{name}_captured_ns"] = frame.captured_ns if frame else 0
    return fields


def _empty_frames(error: str) -> CameraFrameSetResponse:
    return CameraFrameSetResponse(
        ok=False,
        camera_online=False,
        generation=0,
        restart_count=0,
        error=error,
        **_frame_fields({}),
    )


def _frame_response(
    request: CameraFrameRequest,
    *,
    online: bool,
    frame: FrameSnapshot | None = None,
    error: str = "",
) -> CameraFrameResponse:
    return CameraFrameResponse(
        ok=frame is not None,
        camera_online=online,
        stream=request.stream,
        thumbnail=request.thumbnail,
        sequence=frame.sequence if frame else 0,
        captured_ns=frame.captured_ns if frame else 0,
        jpeg=np.frombuffer(frame.jpeg, dtype=np.uint8) if frame else np.empty(0, dtype=np.uint8),
        error=error,
    )


class CameraService(CameraInterface):
    """Typed interface implementation over the long-lived CameraSupervisor."""

    def __init__(self, camera: CameraSupervisor, logger: logging.Logger | None = None) -> None:
        self.camera = camera
        self.log = logger or LOG

    def _control(
        self,
        opening: bool,
        request: CameraOpenRequest | CameraCloseRequest,
    ) -> CameraControlResponse:
        try:
            if opening:
                result = self.camera.open_camera(request.device, timeout_s=request.timeout_s)  # type: ignore[attr-defined]
            else:
                result = self.camera.close_camera(timeout_s=request.timeout_s)
            ok, online, device, generation, error = result
            return CameraControlResponse(
                ok=ok,
                requested_open=opening,
                online=online,
                device=device,
                generation=generation,
                error=error,
            )
        except Exception as exc:
            action = "open" if opening else "close"
            self.log.exception("camera.%s failed", action)
            device = str(getattr(request, "device", "")) if opening else ""
            return CameraControlResponse(
                ok=False,
                requested_open=opening,
                online=False,
                device=device,
                generation=0,
                error=f"{action} error: {type(exc).__name__}: {exc}",
            )

    def open(self, request: CameraOpenRequest) -> CameraControlResponse:
        return self._control(True, request)

    def close(self, request: CameraCloseRequest) -> CameraControlResponse:
        return self._control(False, request)

    def status(self, request: EmptyRequest) -> CameraStatusResponse:
        del request
        try:
            return self.camera.status()
        except Exception as exc:
            self.log.exception("camera.status failed")
            return CameraStatusResponse(
                requested_open=False,
                online=False,
                device="",
                generation=0,
                restart_count=0,
                frames_published=0,
                last_frame_ns=0,
                error=f"status error: {type(exc).__name__}: {exc}",
            )

    def frames(self, request: CameraFrameSetRequest) -> CameraFrameSetResponse:
        del request
        try:
            frames, online, generation, restarts, _published, _last_ns, camera_error = self.camera.snapshot_all()
            missing = [key for key in FRAME_KEYS if key not in frames]
            detail = "missing: " + ", ".join(missing) if missing else ""
            error = f"{camera_error}; {detail}" if camera_error and detail else camera_error or detail
            return CameraFrameSetResponse(
                ok=not missing,
                camera_online=online,
                generation=generation,
                restart_count=restarts,
                error=error if missing or not online else "",
                **_frame_fields(frames),
            )
        except Exception as exc:
            self.log.exception("camera.frame failed")
            return _empty_frames(f"frame error: {type(exc).__name__}: {exc}")

    def get_frame(self, request: CameraFrameRequest) -> CameraFrameResponse:
        try:
            status = self.camera.status()
            if request.stream not in STREAMS:
                return _frame_response(request, online=status.online, error=f"unknown stream: {request.stream!r}")
            frame = self.camera.get_frame(request.stream, request.thumbnail)
            if frame is None:
                return _frame_response(
                    request,
                    online=status.online,
                    error=status.error or "frame is not available yet",
                )
            return _frame_response(
                request,
                online=status.online,
                frame=frame,
                error=status.error if not status.online else "",
            )
        except Exception as exc:
            self.log.exception("camera.get_frame failed")
            return _frame_response(
                request,
                online=False,
                error=f"frame error: {type(exc).__name__}: {exc}",
            )

    def get_calib(self, request: EmptyRequest) -> CameraCalibrationResponse:
        del request
        try:
            return self.camera.get_calib()
        except Exception as exc:
            self.log.exception("camera.get_calib failed")
            return CameraCalibrationResponse.empty(error=str(exc))


# ---------------------------------------------------------------------------
# Generic APIs
# ---------------------------------------------------------------------------


def add_camera_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_fastapi_routes(app, CameraInterface, **kwargs)
