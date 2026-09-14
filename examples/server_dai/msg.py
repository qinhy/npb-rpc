from __future__ import annotations

import numpy as np
from npb import BinaryModel, binary_schema
from pydantic import field_serializer, field_validator


@binary_schema("npb-rpc.dai.camera.empty", version=1)
class EmptyRequest(BinaryModel):
    pass


@binary_schema("npb-rpc.dai.camera.open.request", version=1)
class CameraOpenRequest(BinaryModel):
    """Open a DepthAI device. Empty device means automatic device selection."""

    device: str = ""
    timeout_s: float = 10.0


@binary_schema("npb-rpc.dai.camera.close.request", version=1)
class CameraCloseRequest(BinaryModel):
    """Close the active DepthAI device while keeping the RPC service alive."""

    timeout_s: float = 5.0


@binary_schema("npb-rpc.dai.camera.control.response", version=1)
class CameraControlResponse(BinaryModel):
    ok: bool
    requested_open: bool
    online: bool
    device: str
    generation: int
    error: str


@binary_schema("npb-rpc.dai.camera.status.response", version=2)
class CameraStatusResponse(BinaryModel):
    requested_open: bool
    online: bool
    device: str
    generation: int
    restart_count: int
    frames_published: int
    last_frame_ns: int
    error: str


@binary_schema("npb-rpc.dai.camera.frame.request", version=1)
class CameraFrameRequest(BinaryModel):
    # Keep the wire schema simple; validate allowed stream names in the handler.
    stream: str = "rgb"
    thumbnail: bool = False


@binary_schema("npb-rpc.dai.camera.frame.response", version=2)
class CameraFrameResponse(BinaryModel):
    ok: bool
    camera_online: bool
    stream: str
    thumbnail: bool
    sequence: int
    captured_ns: int
    # npb already has a native binary representation for numpy arrays (the same
    # path used by nng_sum.py).  Do not put Python `bytes` directly in a
    # BinaryModel response: response encoding happens after the RPC handler
    # returns, so an unsupported bytes encoding becomes "RPC handler failed".
    jpeg: np.ndarray
    error: str

    @field_validator("jpeg", mode="before", json_schema_input_type=list[int])
    @classmethod
    def parse_jpeg(cls, value):
        if isinstance(value, np.ndarray):
            return value.astype(np.uint8, copy=False)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.frombuffer(value, dtype=np.uint8)
        return np.asarray(value, dtype=np.uint8)

    @field_serializer("jpeg", when_used="json")
    def serialize_jpeg(self, value: np.ndarray) -> list[int]:
        return value.tolist()


@binary_schema("npb-rpc.dai.camera.frames.request", version=1)
class CameraFrameSetRequest(BinaryModel):
    """Request one complete latest-camera snapshot."""

    pass


@binary_schema("npb-rpc.dai.camera.frames.response", version=1)
class CameraFrameSetResponse(BinaryModel):
    """Latest RGB + stereo MJPEG images and all three MJPEG thumbnails."""

    ok: bool
    camera_online: bool
    generation: int
    restart_count: int

    rgb: np.ndarray
    rgb_sequence: int
    rgb_captured_ns: int

    left: np.ndarray
    left_sequence: int
    left_captured_ns: int

    right: np.ndarray
    right_sequence: int
    right_captured_ns: int

    rgb_thumbnail: np.ndarray
    rgb_thumbnail_sequence: int
    rgb_thumbnail_captured_ns: int

    left_thumbnail: np.ndarray
    left_thumbnail_sequence: int
    left_thumbnail_captured_ns: int

    right_thumbnail: np.ndarray
    right_thumbnail_sequence: int
    right_thumbnail_captured_ns: int

    error: str

    @field_validator(
        "rgb",
        "left",
        "right",
        "rgb_thumbnail",
        "left_thumbnail",
        "right_thumbnail",
        mode="before",
        json_schema_input_type=list[int],
    )
    @classmethod
    def parse_jpeg(cls, value):
        if isinstance(value, np.ndarray):
            return value.astype(np.uint8, copy=False)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.frombuffer(value, dtype=np.uint8)
        return np.asarray(value, dtype=np.uint8)

    @field_serializer(
        "rgb",
        "left",
        "right",
        "rgb_thumbnail",
        "left_thumbnail",
        "right_thumbnail",
        when_used="json",
    )
    def serialize_jpeg(self, value: np.ndarray) -> list[int]:
        return value.tolist()


@binary_schema("npb-rpc.dai.camera.calibration.response", version=1)
class CameraCalibrationResponse(BinaryModel):
    """Normalized calibration information for the active DepthAI device."""

    ok: bool
    camera_online: bool

    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]

    # Shape: (3, 3)
    rgb_intrinsics: np.ndarray
    left_intrinsics: np.ndarray
    right_intrinsics: np.ndarray

    # Typically shape: (4, 4)
    left_to_right_extrinsics: np.ndarray
    left_to_rgb_extrinsics: np.ndarray

    # Shape: (N,)
    rgb_distortion: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray

    distortion_coeff_order: tuple[str, ...]
    stereo_translation_units: str

    board_name: str | None = None
    product_name: str | None = None
    device_id: str | None = None

    stereo_baseline_cm: float | None = None

    rgb_fov_deg: float | None = None
    left_fov_deg: float | None = None
    right_fov_deg: float | None = None

    error: str = ""

    @field_validator(
        "rgb_intrinsics",
        "left_intrinsics",
        "right_intrinsics",
        "left_to_right_extrinsics",
        "left_to_rgb_extrinsics",
        "rgb_distortion",
        "left_distortion",
        "right_distortion",
        mode="before",
    )
    @classmethod
    def parse_float_array(cls, value):
        if isinstance(value, np.ndarray):
            return value.astype(np.float64, copy=False)
        return np.asarray(value, dtype=np.float64)

    @field_serializer(
        "rgb_intrinsics",
        "left_intrinsics",
        "right_intrinsics",
        "left_to_right_extrinsics",
        "left_to_rgb_extrinsics",
        "rgb_distortion",
        "left_distortion",
        "right_distortion",
        when_used="json",
    )
    def serialize_float_array(self, value: np.ndarray):
        return value.tolist()
    
    @classmethod
    def empty(
        cls,
        *,
        camera_online: bool = False,
        error: str = "",
    ) -> "CameraCalibrationResponse":
        return cls(
            ok=False,
            camera_online=camera_online,

            rgb_resolution=(0, 0),
            left_resolution=(0, 0),
            right_resolution=(0, 0),

            rgb_intrinsics=np.empty((0, 0), dtype=np.float64),
            left_intrinsics=np.empty((0, 0), dtype=np.float64),
            right_intrinsics=np.empty((0, 0), dtype=np.float64),

            left_to_right_extrinsics=np.empty((0, 0), dtype=np.float64),
            left_to_rgb_extrinsics=np.empty((0, 0), dtype=np.float64),

            rgb_distortion=np.empty((0,), dtype=np.float64),
            left_distortion=np.empty((0,), dtype=np.float64),
            right_distortion=np.empty((0,), dtype=np.float64),

            distortion_coeff_order=(),
            stereo_translation_units="cm",

            board_name=None,
            product_name=None,
            device_id=None,
            stereo_baseline_cm=None,
            rgb_fov_deg=None,
            left_fov_deg=None,
            right_fov_deg=None,

            error=error,
        )

    