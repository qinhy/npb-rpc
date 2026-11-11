from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from servers.logger import logging
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import numpy as np
import torch
from ultralytics import YOLO

from servers.msg.yolo import (
    YoloDetectResult,
    YoloInferenceRequest,
    YoloTiming,
)
from servers.server_yolo.yolo_utils import (
    Candidate,
    cluster_tiled_candidates,
    class_name,
    effective_roi,
    extract_mask_patches,
    group_to_detection,
    make_divisible,
    make_tiles,
    normalize_root,
    precision_args,
    prepare_yolo_batch,
    read_jpeg_tensor,
    resolve_path,
    write_json_atomic,
    torch_device,
    yolo_device,
)

from servers.msg.worker import Worker

LOG = logging.getLogger(__name__.replace(".",":"))


# ============================================================
# Device-aware multi-model cache
# ============================================================


@dataclass(frozen=True)
class _ModelEntry:
    model: YOLO
    task: str


class YoloModelCache:
    """Cache YOLO models by (model_name, cuda_device), serializing work per device."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[tuple[str, int], _ModelEntry] = {}
        self._device_locks: dict[int, threading.Lock] = {}
        self._hits = 0
        self._misses = 0

    @contextmanager
    def acquire(self, model_name: str, cuda_device: int) -> Iterator[tuple[_ModelEntry, bool]]:
        with self._get_device_lock(cuda_device):
            yield self._get_or_load(model_name, cuda_device)

    def snapshot(self) -> tuple[tuple[str, ...], int, int]:
        with self._lock:
            models = tuple(
                f"{name}@{'cpu' if device < 0 else f'cuda:{device}'}"
                for name, device in self._models
            )
            return models, self._hits, self._misses

    def clear(self) -> None:
        with self._lock:
            self._models.clear()

        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_device_lock(self, cuda_device: int) -> threading.Lock:
        with self._lock:
            lock = self._device_locks.get(cuda_device)
            if lock is None:
                lock = threading.Lock()
                self._device_locks[cuda_device] = lock
            return lock

    def _get_or_load(self, model_name: str, cuda_device: int) -> tuple[_ModelEntry, bool]:
        key = (model_name, cuda_device)

        with self._lock:
            entry = self._models.get(key)
            if entry is not None:
                self._hits += 1
                return entry, True
            self._misses += 1

        LOG.info(
            "loading YOLO model %r on %s",
            model_name,
            "cpu" if cuda_device < 0 else f"cuda:{cuda_device}",
        )
        model = YOLO(model_name)
        entry = _ModelEntry(model=model, task=str(getattr(model, "task", "detect") or "detect"))

        with self._lock:
            existing = self._models.get(key)
            if existing is not None:
                return existing, True
            self._models[key] = entry
        return entry, False


# ============================================================
# Ultralytics detector
# ============================================================


@dataclass
class _InferenceOutput:
    """Everything detect needs after the GPU work; holds no device tensors."""

    image_width: int
    image_height: int
    effective_roi_box: list[int] | None
    tile_size: int | None
    tile_count: int | None
    effective_tile_overlap: int | None
    started: float
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    extract_ms: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)


class UltralyticsYoloDetector:
    """Ultralytics orchestration; reusable algorithms live in yolo_utils.py.

    Work is split in two so the caller can hold the device lock only for the
    first half:
      infer()        -- decode, tile, predict, pull candidates off the device
      build_result() -- CPU-only merging, polygons, result model
    """

    def detect(
        self,
        model: YOLO,
        model_task: str,
        request: YoloInferenceRequest,
        input_path: Path,
    ) -> YoloDetectResult:
        return self.build_result(self.infer(model, request, input_path), model_task, request)

    def infer(
        self,
        model: YOLO,
        request: YoloInferenceRequest,
        input_path: Path,
    ) -> _InferenceOutput:
        t_total = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        image = read_jpeg_tensor(input_path, torch_device(request))
        image_h, image_w = int(image.shape[-2]), int(image.shape[-1])

        roi_box = effective_roi(request.detection_bbox_xyxy, image_w, image_h)
        if roi_box is None:
            roi_left, roi_top, roi_right, roi_bottom = 0, 0, image_w, image_h
            effective_roi_box = None
        else:
            roi_left, roi_top, roi_right, roi_bottom = roi_box
            effective_roi_box = [roi_left, roi_top, roi_right, roi_bottom]

        roi = image[:, roi_top:roi_bottom, roi_left:roi_right]
        if roi.numel() == 0:
            raise ValueError("detection ROI is empty after clipping to image bounds")

        tile_size: int | None = None
        tile_count: int | None = None
        effective_tile_overlap: int | None = None
        tiles = []

        if request.size_mode == "tiling":
            tile_size = make_divisible(request.imgsz, request.stride)
            if tile_size != request.imgsz:
                raise ValueError(
                    f"imgsz={request.imgsz} must be divisible by stride={request.stride} "
                    "because tensor input is not resized or letterboxed"
                )
            effective_tile_overlap = request.tile_overlap
            tiles = make_tiles(
                roi,
                origin_x=roi_left,
                origin_y=roi_top,
                tile_size=tile_size,
                overlap=request.tile_overlap,
            )
            tile_count = len(tiles)

        output = _InferenceOutput(
            image_width=image_w,
            image_height=image_h,
            effective_roi_box=effective_roi_box,
            tile_size=tile_size,
            tile_count=tile_count,
            effective_tile_overlap=effective_tile_overlap,
            started=t_total,
            preprocess_ms=(time.perf_counter() - t0) * 1000.0,
        )

        # Inference. Every phase is timed: batch prep counts as preprocess,
        # model.predict as inference, candidate extraction as extract.
        if request.size_mode == "resize":
            t0 = time.perf_counter()
            source = prepare_yolo_batch([roi], imgsz=request.imgsz, half=request.half)
            t1 = time.perf_counter()
            results = self._predict(model, source, request, request.imgsz)
            t2 = time.perf_counter()
            output.preprocess_ms += (t1 - t0) * 1000.0
            output.inference_ms += (t2 - t1) * 1000.0

            if len(results) != 1:
                raise RuntimeError(f"expected one YOLO result, got {len(results)}")

            output.candidates.extend(
                self._extract_candidates(
                    results[0],
                    request=request,
                    origin_x=roi_left,
                    origin_y=roi_top,
                    source_shape=(int(roi.shape[-2]), int(roi.shape[-1])),
                    tile=None,
                )
            )
            output.extract_ms += (time.perf_counter() - t2) * 1000.0
        else:
            for start in range(0, len(tiles), request.tile_batch_size):
                batch = tiles[start:start + request.tile_batch_size]

                t0 = time.perf_counter()
                source = prepare_yolo_batch(
                    [item.image for item in batch],
                    imgsz=tile_size or request.imgsz,
                    half=request.half,
                )
                t1 = time.perf_counter()
                results = self._predict(model, source, request, tile_size or request.imgsz)
                t2 = time.perf_counter()
                output.preprocess_ms += (t1 - t0) * 1000.0
                output.inference_ms += (t2 - t1) * 1000.0

                if len(results) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} YOLO results, got {len(results)}")

                for result, item in zip(results, batch):
                    output.candidates.extend(
                        self._extract_candidates(
                            result,
                            request=request,
                            origin_x=item.tile.left,
                            origin_y=item.tile.top,
                            source_shape=(int(item.image.shape[-2]), int(item.image.shape[-1])),
                            tile=item.tile,
                        )
                    )
                output.extract_ms += (time.perf_counter() - t2) * 1000.0

        return output

    def build_result(
        self,
        output: _InferenceOutput,
        model_task: str,
        request: YoloInferenceRequest,
    ) -> YoloDetectResult:
        """CPU-only postprocess; safe to run without the device lock."""
        image_w, image_h = output.image_width, output.image_height

        t0 = time.perf_counter()
        groups = (
            cluster_tiled_candidates(
                output.candidates,
                iou_threshold=request.iou,
                iom_threshold=request.tile_merge_iom,
            )
            if request.size_mode == "tiling"
            else [[candidate] for candidate in output.candidates]
        )

        detections = [
            group_to_detection(
                group,
                request=request,
                image_width=image_w,
                image_height=image_h,
            )
            for group in groups
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        detections = detections[:request.max_detections]
        for i,d in enumerate(detections):d.detection_index=i

        has_masks = any(item.mask is not None for item in detections)
        task = "segment" if request.include_masks and model_task == "segment" else "detect"
        merge_ms = (time.perf_counter() - t0) * 1000.0

        # YoloTiming has no extract field, so extraction is reported as postprocess.
        postprocess_ms = output.extract_ms + merge_ms
        total_ms = (time.perf_counter() - output.started) * 1000.0

        timing = YoloTiming(
            preprocess_ms=output.preprocess_ms,
            inference_ms=output.inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
        )
        
        res = YoloDetectResult(
            **request.model_dump(),
            task=task,
            image_width=image_w,
            image_height=image_h,
            detections=detections,
            has_masks=has_masks,
            num_detections=len(detections),
            tile_size=output.tile_size,
            effective_tile_overlap=output.effective_tile_overlap,
            tile_count=output.tile_count,
            effective_detection_bbox_xyxy=output.effective_roi_box,
            timing=timing,
        )
        LOG.info(f"{dict(
            img_size=(image_w,image_h),
            crop=output.effective_roi_box,
            tile=f"{output.tile_size}x{output.tile_count}p",
            candidates=len(output.candidates),
            pre_ms=int(output.preprocess_ms),
            infer_ms=int(output.inference_ms),
            extract_ms=int(output.extract_ms),
            merge_ms=int(merge_ms),
            total_ms=int(total_ms),
        )}")
        return res

    def _predict(
        self,
        model: YOLO,
        source: torch.Tensor,
        request: YoloInferenceRequest,
        imgsz: int,
    ) -> list[Any]:
        kwargs: dict[str, Any] = {
            "source": source,
            "imgsz": imgsz,
            "conf": request.confidence,
            "iou": request.iou,
            "max_det": request.max_detections,
            "device": yolo_device(request),
            "rect": False,
            "retina_masks": request.include_masks,
            "save": False,
            "verbose": False,
        }
        kwargs.update(precision_args(request))
        results = list(model.predict(**kwargs))
        return results

        ### only for debug ###
        LOG.info(
            "YOLO request: half=%s cuda_device=%s device=%s precision=%s",
            request.half,request.cuda_device,kwargs["device"],
            precision_args(request),)
        # Actual inference predictor/backend
        predictor = model.predictor
        backend = predictor.model
        LOG.info(
            "YOLO actual backend: quantize=%s fp16=%s device=%s",
            getattr(predictor.args, "quantize", None),
            getattr(backend, "fp16", None),
            getattr(backend, "device", None),
        )

        # Actual inference model weight dtype
        torch_model = getattr(backend, "model", None)
        if torch_model is not None:
            try:
                p = next(torch_model.parameters())
                LOG.info(
                    "YOLO actual weight: dtype=%s device=%s",
                    p.dtype,
                    p.device,
                )
            except (StopIteration, AttributeError):
                pass

        return results

    def _extract_candidates(
        self,
        result: Any,
        *,
        request: YoloInferenceRequest,
        origin_x: int,
        origin_y: int,
        source_shape: tuple[int, int],
        tile: Any,
    ) -> list[Candidate]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        offset = np.array([origin_x, origin_y, origin_x, origin_y], dtype=np.float32)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False) + offset
        confidences = boxes.conf.detach().cpu().numpy().tolist()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False).tolist()

        # Masks are thresholded and cropped on the device; only tight boolean
        # patches are copied to the CPU (see extract_mask_patches).
        patches: list[Any] = []
        if request.include_masks and result.masks is not None:
            patches = extract_mask_patches(
                result.masks.data.detach(),
                threshold=request.mask_threshold,
                source_shape=source_shape,
                origin_x=origin_x,
                origin_y=origin_y,
            )

        names = result.names
        return [
            Candidate(
                class_id=class_ids[index],
                class_name=class_name(names, class_ids[index]),
                confidence=confidences[index],
                bbox_xyxy=xyxy[index],
                tile=tile,
                mask=patches[index] if index < len(patches) else None,
            )
            for index in range(len(xyxy))
        ]


# ============================================================
# Async worker façade
# ============================================================


class YoloWorker(
    Worker[YoloInferenceRequest, YoloDetectResult]
):
    def __init__(
        self,
        *,
        worker_count: int = 1,
        queue_size: int = 0,
        job_ttl_s: float = 3600.0,
        max_completed_jobs: int = 128,
        read_root: str | Path | None = None,
        write_root: str | Path | None = None,
    ) -> None:

        super().__init__(
            name="yolo",
            worker_count=worker_count,
            queue_size=queue_size,
            job_ttl_s=job_ttl_s,
            max_completed_jobs=max_completed_jobs,
        )

        self.models = YoloModelCache()
        self.detector = UltralyticsYoloDetector()

        self._read_root = normalize_root(read_root)
        self._write_root = normalize_root(write_root)

    def process(
        self,
        *,
        request: YoloInferenceRequest,
        job_id: str,
    ) -> YoloDetectResult:

        input_path = self._resolve_input_path(
            request.input_jpg_path
        )

        output_path = self._resolve_output_path(
            request.output_json_path
        )

        with self.models.acquire(
            request.model_name,
            request.cuda_device,
        ) as (entry, cache_hit):

            self.store.set_cache_hit(
                job_id,
                cache_hit,
            )

            # Only the GPU half runs under the device lock.
            output = self.detector.infer(
                entry.model,
                request,
                input_path,
            )

        # CPU-only merging/polygons run unlocked, so with worker_count > 1 the
        # next job can start inference on this device meanwhile.
        result = self.detector.build_result(
            output,
            entry.task,
            request,
        )

        t0 = time.perf_counter()
        write_json_atomic(
            output_path,
            result,
            job_id,
        )
        LOG.info(
            "job %s wrote %s in %d ms",
            job_id,
            output_path,
            int((time.perf_counter() - t0) * 1000.0),
        )

        return result

    def cleanup(self) -> None:
        self.models.clear()

    def _resolve_input_path(self, value: str) -> Path:
        path = resolve_path(value, self._read_root)

        if not path.is_file():
            raise FileNotFoundError(
                f"input JPEG not found: {path}"
            )

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(
                f"input path must end in .jpg or .jpeg: {path}"
            )

        return path

    def _resolve_output_path(self, value: str) -> Path:
        path = resolve_path(value, self._write_root)

        if path.suffix.lower() != ".json":
            raise ValueError(
                f"output path must end in .json: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

