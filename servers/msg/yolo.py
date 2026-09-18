from __future__ import annotations

from typing import Literal, Protocol

from npb import BinaryModel, binary_schema
from pydantic import BaseModel, Field, model_validator

from npb_rpc import RpcEvent
from npb_rpc.utils import api, build_client_class


# ============================================================
# Common
# ============================================================


YoloJobState = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


@binary_schema("npb-rpc.yolo.empty", version=1)
class EmptyRequest(BinaryModel):
    pass


# ============================================================
# Inference Request
# ============================================================


@binary_schema("npb-rpc.yolo.inference.request", version=3)
class YoloInferenceRequest(BinaryModel):
    """Submit YOLO inference for one server-local JPEG image."""

    # I/O
    input_jpg_path: str = Field(min_length=1)
    output_json_path: str = Field(min_length=1)

    # Model
    # Server cache key: (model_name, cuda_device)
    model_name: str = "yolo11l-seg.pt"
    cuda_device: int = Field(default=0, ge=-1)  # -1=CPU, 0+=CUDA

    # Inference
    size_mode: Literal["resize", "tiling"] = "tiling"
    imgsz: int = Field(default=1280, gt=0)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, gt=0)
    half: bool = True

    # Tiling
    stride: int = Field(default=32, gt=0)
    tile_overlap: int = Field(default=416, ge=0)
    tile_batch_size: int = Field(default=4, gt=0)

    # Masks
    include_masks: bool = True
    mask_format: Literal["polygon"] = "polygon"
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tiled_mask_output: Literal["full_image"] = "full_image"
    merge_tiled_masks: bool = True
    tile_merge_iom: float = Field(default=0.15, ge=0.0, le=1.0)

    # Polygon
    polygon_epsilon: float = Field(default=1.0, ge=0.0)
    polygon_min_area: float = Field(default=1.0, ge=0.0)

    # Optional ROI in absolute original-image XYXY coordinates.
    detection_bbox_xyxy: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )

    # Optional job dependencies / completion signal.
    wait_for: list[RpcEvent] = Field(default_factory=list) # not support yet
    done_event: RpcEvent | None = None

    @model_validator(mode="after")
    def validate_request(self):
        if self.size_mode == "tiling" and self.tile_overlap >= self.imgsz:
            raise ValueError("tile_overlap must be smaller than imgsz")

        if self.detection_bbox_xyxy is not None:
            x1, y1, x2, y2 = self.detection_bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "detection_bbox_xyxy must satisfy x2 > x1 and y2 > y1"
                )

        return self


# ============================================================
# Detection Result Models
# ============================================================


class YoloPolygon(BaseModel):
    """One polygon ring in absolute source-image XY coordinates."""

    points_xy: list[list[float]]
    is_hole: bool = False

    # Index into YoloInstancePolygon.polygons.
    # None means this ring is not a hole.
    parent_index: int | None = None


class YoloTile(BaseModel):
    """Original-image bounds of one tile contributing to a detection."""

    left: int
    top: int
    right: int
    bottom: int


class YoloInstancePolygon(BaseModel):
    """JSON-friendly polygon representation of one instance mask."""

    format: Literal["polygon"] = "polygon"

    # [image_height, image_width]
    size: list[int] = Field(min_length=2, max_length=2)

    # Absolute original-image coordinates.
    polygons: list[YoloPolygon] = Field(default_factory=list)

    area: int = Field(default=0, ge=0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class YoloDetection(BaseModel):
    """One final detection in original-image coordinates."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)

    bbox_xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
    )

    # Empty in resize mode.
    #
    # Multiple tiles are possible when duplicate detections/masks
    # are merged across overlapping tiles.
    tiles: list[YoloTile] = Field(default_factory=list)

    # None for detection-only output.
    mask: YoloInstancePolygon | None = None


class YoloTiming(BaseModel):
    """Inference timing in milliseconds."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


# ============================================================
# Complete Detection Result
# ============================================================


@binary_schema("npb-rpc.yolo.detect.result", version=3)
class YoloDetectResult(YoloInferenceRequest):
    """
    Complete YOLO result.

    The same model can be:
      1. returned by job_result()
      2. written directly to output_json_path
    """

    task: Literal["detect", "segment"] = "detect"

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)

    detections: list[YoloDetection] = Field(default_factory=list)

    has_masks: bool = False
    num_detections: int = Field(default=0, ge=0)

    # Effective tiling information.
    # None in resize mode.
    tile_size: int | None = Field(default=None, gt=0)
    effective_tile_overlap: int | None = Field(default=None, ge=0)
    tile_count: int | None = Field(default=None, ge=0)

    # Requested ROI remains available through inherited:
    #
    #   detection_bbox_xyxy
    #
    # This is the actual ROI after clipping against image bounds.
    # None means full-image inference.
    effective_detection_bbox_xyxy: list[int] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )

    timing: YoloTiming = Field(default_factory=YoloTiming)


# ============================================================
# Async Job Submission
# ============================================================


@binary_schema("npb-rpc.yolo.inference.submit.response", version=2)
class YoloInferenceSubmitResponse(BinaryModel):
    """Returned immediately after an inference job is accepted."""

    accepted: bool

    job_id: str = ""
    state: YoloJobState | None = None

    # Echo the completion event supplied by the caller, if any.
    done_event: RpcEvent | None = None

    input_jpg_path: str = ""
    output_json_path: str = ""

    error: str = ""


# ============================================================
# Async Job Request
# ============================================================


@binary_schema("npb-rpc.yolo.job.request", version=1)
class YoloJobRequest(BinaryModel):
    """Identify one asynchronous YOLO job."""

    job_id: str = Field(min_length=1)


# ============================================================
# Async Job Status
# ============================================================


@binary_schema("npb-rpc.yolo.job.status.response", version=1)
class YoloJobStatusResponse(BinaryModel):
    """Lightweight state of one asynchronous inference job."""

    found: bool
    job_id: str

    state: YoloJobState | None = None

    model_name: str = ""
    cuda_device: int = 0

    input_jpg_path: str = ""
    output_json_path: str = ""

    # Nanosecond timestamps.
    # 0 means the job has not reached that stage yet.
    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0

    # None until the worker selects/loads the model.
    cache_hit: bool | None = None

    # None until inference has completed.
    num_detections: int | None = None

    timing: YoloTiming = Field(default_factory=YoloTiming)

    error: str = ""


# ============================================================
# Async Job Result
# ============================================================


@binary_schema("npb-rpc.yolo.job.result.response", version=1)
class YoloJobResultResponse(BinaryModel):
    """Full result of one asynchronous inference job."""

    found: bool
    job_id: str

    state: YoloJobState | None = None

    # Populated only when state == "succeeded".
    result: YoloDetectResult | None = None

    error: str = ""


# ============================================================
# Server Status
# ============================================================


@binary_schema("npb-rpc.yolo.status.response", version=4)
class YoloStatusResponse(BinaryModel):
    """YOLO server, job queue, and model-cache status."""

    online: bool

    # Jobs
    queued_jobs: int = 0
    running_jobs: int = 0
    succeeded_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # Completed inference count.
    inference_count: int = 0

    # Model cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Examples:
    #   yolo11l-seg.pt@cuda:0
    #   custom.pt@cuda:1
    #   yolo11n.pt@cpu
    cached_models: tuple[str, ...] = ()

    # Last completed inference.
    last_job_id: str = ""
    last_inference_ns: int = 0
    last_inference_ms: float = 0.0

    error: str = ""


class YoloInterface(Protocol):
    """Single source of truth for RPC, generated client, server, and HTTP."""

    service = "yolo"

    @api("yolo.inference", "POST", "inference")
    def inference(self,request: YoloInferenceRequest)->YoloInferenceSubmitResponse:
        ...    
    @api("yolo.job_status", "GET", "job_status")
    def job_status(self,request: YoloJobRequest)->YoloJobStatusResponse:
        ...    
    @api("yolo.job_result", "GET", "job_result")
    def job_result(self,request: YoloJobRequest)->YoloJobResultResponse:
        ...    
    @api("yolo.status", "GET", "status")
    def status(self,request: EmptyRequest)->YoloStatusResponse:
        ...    


YoloClient = build_client_class(YoloInterface, "YoloClient")
# Client
#   │
#   │ inference(request)
#   ▼
# YOLO Server
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

#                          │
#                          ▼
#                     Worker Thread
#                          │
#                          ├─ state = running
#                          │
#                          ├─ get cached model
#                          │    or load model
#                          │
#                          ├─ read input JPEG
#                          ├─ resize / tiling
#                          ├─ YOLO inference
#                          ├─ merge results
#                          ├─ create YoloDetectResult
#                          ├─ write output JSON
#                          │
#                          └─ state = succeeded
