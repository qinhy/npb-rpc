from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from npb import BinaryModel, binary_schema
from pydantic import BaseModel, Field, model_validator

from npb_rpc.utils import api, build_client_class


# Common

class PcdJobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PcdBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    DNN = "dnn"
    VPI = "vpi"


@binary_schema("npb-rpc.pcd.empty", version=1)
class EmptyRequest(BinaryModel):
    pass


# Build request

@binary_schema("npb-rpc.pcd.build.request", version=1)
class PcdBuildRequest(BinaryModel):
    """Build one RGB-colored point cloud from an RGB + stereo image set.

    Input images and calibration are server-local files. The pipeline is:
    left/right -> rectify -> disparity -> 3D -> original-left -> RGB projection.

    Optionally, an existing YOLO result JSON can split the full cloud into
    per-detection PCD files.
    """

    # Input / output
    rgb_jpg_path: str = Field(min_length=1)
    left_jpg_path: str = Field(min_length=1)
    right_jpg_path: str = Field(min_length=1)
    calibration_json_path: str = Field(min_length=1)
    output_pcd_path: str = Field(min_length=1)
    output_json_path: str | None = Field(default=None, min_length=1)

    # Disparity backend: cpu=OpenCV SGBM, cuda=libSGM,
    # dnn=Fast-FoundationStereo, vpi=NVIDIA VPI.
    backend: PcdBackend = Field(default="cpu", validate_default=True)
    cuda_device: int = Field(default=0, ge=0)

    # Point-cloud generation
    min_disparity: float = Field(default=0.5, ge=0.0)
    min_depth_m: float | None = Field(default=0.01, gt=0.0)
    max_depth_m: float | None = Field(default=5.0, gt=0.0)
    stride: int = Field(default=1, ge=1)
    alpha: float = 0.0
    rgb_image_is_undistorted: bool = False
    binary_pcd: bool = True

    # Optional YOLO segmentation. Both paths must be supplied together.
    detections_json_path: str | None = Field(default=None, min_length=1)
    segments_output_dir: str | None = Field(default=None, min_length=1)
    min_segment_points: int = Field(default=30, ge=0)
    erode_pixels: int = Field(default=0, ge=0)
    exclusive_segments: bool = False
    save_background: bool = False

    @model_validator(mode="after")
    def validate_request(self):
        if (
            self.min_depth_m is not None
            and self.max_depth_m is not None
            and self.max_depth_m < self.min_depth_m
        ):
            raise ValueError(
                "max_depth_m must be greater than or equal to min_depth_m"
            )

        if (self.detections_json_path is None) != (self.segments_output_dir is None):
            raise ValueError(
                "detections_json_path and segments_output_dir must be supplied together"
            )
        return self


# Results

class PcdSegment(BaseModel):
    """One point-cloud segment produced from one YOLO detection."""

    detection_index: int = Field(ge=0)
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    point_count: int = Field(default=0, ge=0)
    pcd_path: str = Field(min_length=1)


class PcdTiming(BaseModel):
    """PCD build timing in milliseconds.

    backend_ms may include predictor/model initialization on a backend-cache miss.
    """

    backend_ms: float = 0.0
    read_ms: float = 0.0
    calibration_ms: float = 0.0
    rectify_ms: float = 0.0
    disparity_ms: float = 0.0
    points_ms: float = 0.0
    projection_ms: float = 0.0
    segmentation_ms: float = 0.0
    write_ms: float = 0.0
    total_ms: float = 0.0


@binary_schema("npb-rpc.pcd.build.result", version=1)
class PcdBuildResult(PcdBuildRequest):
    """Complete result for one PCD build.

    Request fields are inherited so the result records how the cloud was built.
    Large XYZ/RGB arrays stay in output_pcd_path rather than crossing RPC.
    """

    backend_used: str = ""
    device_used: str = ""

    rgb_image_width: int = Field(default=0, ge=0)
    rgb_image_height: int = Field(default=0, ge=0)
    stereo_image_width: int = Field(default=0, ge=0)
    stereo_image_height: int = Field(default=0, ge=0)

    point_count: int = Field(default=0, ge=0)
    num_segments: int = Field(default=0, ge=0)
    segments: list[PcdSegment] = Field(default_factory=list)
    timing: PcdTiming = Field(default_factory=PcdTiming)


# Async jobs

@binary_schema("npb-rpc.pcd.build.submit.response", version=1)
class PcdBuildSubmitResponse(BinaryModel):
    """Returned immediately after a PCD build job is accepted."""

    accepted: bool
    job_id: str = ""
    state: PcdJobState | None = None

    rgb_jpg_path: str = ""
    left_jpg_path: str = ""
    right_jpg_path: str = ""
    output_pcd_path: str = ""
    output_json_path: str | None = None
    error: str = ""


@binary_schema("npb-rpc.pcd.job.request", version=1)
class PcdJobRequest(BinaryModel):
    """Identify one asynchronous PCD job."""

    job_id: str = Field(min_length=1)


@binary_schema("npb-rpc.pcd.job.status.response", version=1)
class PcdJobStatusResponse(BinaryModel):
    """Lightweight state for one asynchronous PCD build job."""

    found: bool
    job_id: str
    state: PcdJobState | None = None

    backend: PcdBackend | None = None
    cuda_device: int = 0

    rgb_jpg_path: str = ""
    left_jpg_path: str = ""
    right_jpg_path: str = ""
    output_pcd_path: str = ""
    output_json_path: str | None = None

    # Nanosecond timestamps; 0 means the job has not reached that stage.
    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0

    cache_hit: bool | None = None
    point_count: int | None = None
    num_segments: int | None = None
    timing: PcdTiming = Field(default_factory=PcdTiming)
    error: str = ""


@binary_schema("npb-rpc.pcd.job.result.response", version=1)
class PcdJobResultResponse(BinaryModel):
    """Full result of one asynchronous PCD build job."""

    found: bool
    job_id: str
    state: PcdJobState | None = None
    result: PcdBuildResult | None = None  # Populated only when succeeded.
    error: str = ""


# Server status

@binary_schema("npb-rpc.pcd.status.response", version=2)
class PcdStatusResponse(BinaryModel):
    """PCD server, job queue, and disparity-backend cache status."""

    online: bool

    queued_jobs: int = 0
    running_jobs: int = 0
    succeeded_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    build_count: int = 0

    cache_hits: int = 0
    cache_misses: int = 0
    # Examples: cpu, cuda@cuda:0, dnn@cuda:0, vpi@cuda:0
    cached_backends: tuple[str, ...] = ()

    last_job_id: str = ""
    last_build_ns: int = 0
    last_build_ms: float = 0.0
    error: str = ""


class PcdInterface(Protocol):
    """Single source of truth for RPC, generated client, server, and FastAPI."""

    service = "pcd"

    @api("pcd.build", "POST", "build")
    def build(self, request: PcdBuildRequest) -> PcdBuildSubmitResponse: ...

    @api("pcd.job_status", "GET", "job_status")
    def job_status(self, request: PcdJobRequest) -> PcdJobStatusResponse: ...

    @api("pcd.job_result", "GET", "job_result")
    def job_result(self, request: PcdJobRequest) -> PcdJobResultResponse: ...

    @api("pcd.status", "GET", "status")
    def status(self, request: EmptyRequest) -> PcdStatusResponse: ...


PcdClient = build_client_class(PcdInterface, "PcdClient")

# Flow:
# client build() -> queued -> worker running -> cached backend -> calibration/images
# -> rectify -> disparity -> XYZ -> RGB projection -> full PCD
# -> optional YOLO segmentation/result JSON -> succeeded


# Client
#   │
#   │ build(request)
#   ▼
# PCD Server
#   │
#   ├─ create job_id
#   ├─ store request
#   ├─ state = queued
#   ├─ put into worker queue
#   │
#   └──────────────► return immediately
#                    {
#                      accepted: true,
#                      job_id: "...",
#                      state: "queued"
#                    }
#
#                          │
#                          ▼
#                     Worker Thread
#                          │
#                          ├─ state = running
#                          │
#                          ├─ acquire cached disparity backend
#                          │
#                          ├─ read calibration
#                          ├─ read RGB / left / right
#                          │
#                          ├─ rectify left/right
#                          ├─ disparity prediction
#                          │
#                          ├─ disparity -> XYZ
#                          ├─ rectified-left -> original-left
#                          ├─ project XYZ -> RGB pixels
#                          ├─ attach RGB colors
#                          │
#                          ├─ write full .pcd
#                          │
#                          ├─ optional YOLO JSON
#                          │      └─ split into per-object .pcd
#                          │
#                          ├─ optional result JSON
#                          │
#                          └─ state = succeeded
