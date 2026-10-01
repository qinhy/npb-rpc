from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from servers.logger import logging
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Any, Iterator
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

from servers.msg.yolo import (
    YoloDetectResult,
    YoloInferenceRequest,
    JobSubmitResponse,
    YoloJobResultResponse,
    YoloJobStatusResponse,
    YoloStatusResponse,
    YoloTiming,
)
from servers.server_yolo.yolo_utils import (
    Candidate,
    cluster_tiled_candidates,
    class_name,
    crop_mask,
    effective_roi,
    group_to_detection,
    make_divisible,
    make_tiles,
    normalize_root,
    precision_args,
    read_jpeg,
    resolve_path,
    write_json_atomic,
    yolo_device,
)

from servers.msg.worker import JobStore
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


class UltralyticsYoloDetector:
    """Ultralytics orchestration; reusable algorithms live in yolo_utils.py."""

    def detect(
        self,
        model: YOLO,
        model_task: str,
        request: YoloInferenceRequest,
        input_path: Path,
    ) -> YoloDetectResult:
        t_total = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        image = read_jpeg(input_path)
        image_h, image_w = image.shape[:2]

        roi_box = effective_roi(request.detection_bbox_xyxy, image_w, image_h)
        if roi_box is None:
            roi_left, roi_top, roi_right, roi_bottom = 0, 0, image_w, image_h
            effective_roi_box = None
        else:
            roi_left, roi_top, roi_right, roi_bottom = roi_box
            effective_roi_box = [roi_left, roi_top, roi_right, roi_bottom]

        roi = image[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            raise ValueError("detection ROI is empty after clipping to image bounds")

        tile_size: int | None = None
        tile_count: int | None = None
        effective_tile_overlap: int | None = None
        tiles = []

        if request.size_mode == "tiling":
            tile_size = make_divisible(request.imgsz, request.stride)
            effective_tile_overlap = request.tile_overlap
            tiles = make_tiles(
                roi,
                origin_x=roi_left,
                origin_y=roi_top,
                tile_size=tile_size,
                overlap=request.tile_overlap,
            )
            tile_count = len(tiles)

        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        # Inference
        inference_ms = 0.0
        candidates: list[Candidate] = []

        if request.size_mode == "resize":
            t0 = time.perf_counter()
            results = self._predict(model, np.ascontiguousarray(roi), request, request.imgsz)
            inference_ms += (time.perf_counter() - t0) * 1000.0

            if len(results) != 1:
                raise RuntimeError(f"expected one YOLO result, got {len(results)}")

            candidates.extend(
                self._extract_candidates(
                    results[0],
                    request=request,
                    origin_x=roi_left,
                    origin_y=roi_top,
                    source_shape=roi.shape[:2],
                    tile=None,
                )
            )
        else:
            for start in range(0, len(tiles), request.tile_batch_size):
                batch = tiles[start:start + request.tile_batch_size]
                sources = [np.ascontiguousarray(item.image) for item in batch]

                t0 = time.perf_counter()
                results = self._predict(model, sources, request, tile_size or request.imgsz)
                inference_ms += (time.perf_counter() - t0) * 1000.0

                if len(results) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} YOLO results, got {len(results)}")

                for result, item in zip(results, batch):
                    candidates.extend(
                        self._extract_candidates(
                            result,
                            request=request,
                            origin_x=item.tile.left,
                            origin_y=item.tile.top,
                            source_shape=item.image.shape[:2],
                            tile=item.tile,
                        )
                    )

        # Postprocess
        t0 = time.perf_counter()
        groups = (
            cluster_tiled_candidates(
                candidates,
                iou_threshold=request.iou,
                iom_threshold=request.tile_merge_iom,
            )
            if request.size_mode == "tiling"
            else [[candidate] for candidate in candidates]
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
        postprocess_ms = (time.perf_counter() - t0) * 1000.0

        timing = YoloTiming(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=(time.perf_counter() - t_total) * 1000.0,
        )
        
        res = YoloDetectResult(
            **request.model_dump(),
            task=task,
            image_width=image_w,
            image_height=image_h,
            detections=detections,
            has_masks=has_masks,
            num_detections=len(detections),
            tile_size=tile_size,
            effective_tile_overlap=effective_tile_overlap,
            tile_count=tile_count,
            effective_detection_bbox_xyxy=effective_roi_box,
            timing=timing,
        )
        LOG.info(f"{dict(
            image_width=image_w,
            image_height=image_h,
            tile_size=tile_size,
            tile_count=tile_count,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=(time.perf_counter() - t_total) * 1000.0,
        )}")
        return res

    def _predict(
        self,
        model: YOLO,
        source: np.ndarray | list[np.ndarray],
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
        return list(model.predict(**kwargs))

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

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)

        mask_data: np.ndarray | None = None
        if request.include_masks and result.masks is not None:
            mask_data = result.masks.data.detach().cpu().numpy()

        source_h, source_w = source_shape
        candidates: list[Candidate] = []

        for index in range(len(xyxy)):
            box = xyxy[index].astype(np.float32, copy=True)
            box[[0, 2]] += origin_x
            box[[1, 3]] += origin_y

            class_id = int(class_ids[index])
            mask = None
            if mask_data is not None and index < len(mask_data):
                raw_mask = mask_data[index]
                if raw_mask.shape != (source_h, source_w):
                    raw_mask = cv2.resize(
                        raw_mask,
                        (source_w, source_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = crop_mask(raw_mask >= request.mask_threshold, origin_x, origin_y)

            candidates.append(
                Candidate(
                    class_id=class_id,
                    class_name=class_name(result.names, class_id),
                    confidence=float(confidences[index]),
                    bbox_xyxy=box,
                    tile=tile,
                    mask=mask,
                )
            )
        return candidates


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

            result = self.detector.detect(
                entry.model,
                entry.task,
                request,
                input_path,
            )

        write_json_atomic(
            output_path,
            result,
            job_id,
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
