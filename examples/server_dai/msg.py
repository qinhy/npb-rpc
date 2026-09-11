from __future__ import annotations

import numpy as np
from npb import BinaryModel, binary_schema
from pydantic import field_serializer, field_validator


@binary_schema("npb-rpc.dai.camera.empty", version=1)
class EmptyRequest(BinaryModel):
    pass


@binary_schema("npb-rpc.dai.camera.status.response", version=1)
class CameraStatusResponse(BinaryModel):
    online: bool
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
