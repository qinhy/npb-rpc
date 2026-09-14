from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Any, Iterator
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .msg import (
        YoloDetectResult,
        YoloInferenceRequest,
        YoloInferenceSubmitResponse,
        YoloJobResultResponse,
        YoloJobState,
        YoloJobStatusResponse,
        YoloStatusResponse,
        YoloTiming,
    )
    from .yolo_utils import (
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
except ImportError:  # Support running files directly from one directory.
    from msg import (
        YoloDetectResult,
        YoloInferenceRequest,
        YoloInferenceSubmitResponse,
        YoloJobResultResponse,
        YoloJobState,
        YoloJobStatusResponse,
        YoloStatusResponse,
        YoloTiming,
    )
    from yolo_utils import (
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


LOG = logging.getLogger("npb_rpc_yolo")


# ============================================================
# Internal snapshots / records
# ============================================================


@dataclass
class _JobRecord:
    request: YoloInferenceRequest
    state: YoloJobState = "queued"
    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0
    cache_hit: bool | None = None
    num_detections: int | None = None
    timing: YoloTiming | None = None
    result: YoloDetectResult | None = None
    error: str = ""


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    request: YoloInferenceRequest
    state: YoloJobState
    created_ns: int
    started_ns: int
    finished_ns: int
    cache_hit: bool | None
    num_detections: int | None
    timing: YoloTiming
    result: YoloDetectResult | None
    error: str


@dataclass(frozen=True)
class JobStoreSummary:
    queued_jobs: int
    running_jobs: int
    succeeded_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    inference_count: int
    last_inference_ns: int
    last_inference_ms: float
    error: str


# ============================================================
# Thread-safe async job store
# ============================================================


class YoloJobStore:
    """Thread-safe async job state/results with bounded completed-job retention."""

    TERMINAL = {"succeeded", "failed", "cancelled"}

    def __init__(self, *, job_ttl_s: float = 3600.0, max_completed_jobs: int = 128) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobRecord] = {}
        self._job_ttl_ns = max(0, int(job_ttl_s * 1e9))
        self._max_completed_jobs = max(1, int(max_completed_jobs))
        self._succeeded_total = 0
        self._failed_total = 0
        self._cancelled_total = 0
        self._last_inference_ns = 0
        self._last_inference_ms = 0.0
        self._last_error = ""

    def create(self, request: YoloInferenceRequest) -> str:
        job_id = uuid.uuid4().hex
        now_ns = time.time_ns()
        with self._lock:
            self._prune_locked(now_ns)
            self._jobs[job_id] = _JobRecord(
                request=request.model_copy(deep=True),
                created_ns=now_ns,
            )
        return job_id

    def mark_running(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state != "queued":
                return False
            record.state = "running"
            record.started_ns = time.time_ns()
            return True

    def set_cache_hit(self, job_id: str, cache_hit: bool) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.cache_hit = bool(cache_hit)

    def succeed(self, job_id: str, result: YoloDetectResult) -> None:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in self.TERMINAL:
                return

            record.state = "succeeded"
            record.finished_ns = now_ns
            record.result = result
            record.num_detections = result.num_detections
            record.timing = result.timing
            record.error = ""

            self._succeeded_total += 1
            self._last_inference_ns = now_ns
            self._last_inference_ms = result.timing.total_ms
            self._last_error = ""
            self._prune_locked(now_ns)

    def fail(self, job_id: str, error: str) -> None:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in self.TERMINAL:
                return

            record.state = "failed"
            record.finished_ns = now_ns
            record.error = str(error)
            self._failed_total += 1
            self._last_error = str(error)
            self._prune_locked(now_ns)

    def cancel(self, job_id: str, error: str = "cancelled") -> bool:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state != "queued":
                return False

            record.state = "cancelled"
            record.finished_ns = now_ns
            record.error = error
            self._cancelled_total += 1
            self._prune_locked(now_ns)
            return True

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        now_ns = time.time_ns()
        with self._lock:
            self._prune_locked(now_ns)
            record = self._jobs.get(job_id)
            if record is None:
                return None

            return JobSnapshot(
                job_id=job_id,
                request=record.request,
                state=record.state,
                created_ns=record.created_ns,
                started_ns=record.started_ns,
                finished_ns=record.finished_ns,
                cache_hit=record.cache_hit,
                num_detections=record.num_detections,
                timing=record.timing or YoloTiming(),
                result=record.result,
                error=record.error,
            )

    def summary(self) -> JobStoreSummary:
        now_ns = time.time_ns()
        with self._lock:
            self._prune_locked(now_ns)
            return JobStoreSummary(
                queued_jobs=sum(r.state == "queued" for r in self._jobs.values()),
                running_jobs=sum(r.state == "running" for r in self._jobs.values()),
                succeeded_jobs=self._succeeded_total,
                failed_jobs=self._failed_total,
                cancelled_jobs=self._cancelled_total,
                inference_count=self._succeeded_total,
                last_inference_ns=self._last_inference_ns,
                last_inference_ms=self._last_inference_ms,
                error=self._last_error,
            )

    def _prune_locked(self, now_ns: int) -> None:
        terminal = [
            (job_id, record.finished_ns)
            for job_id, record in self._jobs.items()
            if record.state in self.TERMINAL and record.finished_ns
        ]

        if self._job_ttl_ns:
            cutoff = now_ns - self._job_ttl_ns
            for job_id, finished_ns in terminal:
                if finished_ns < cutoff:
                    self._jobs.pop(job_id, None)

        terminal = sorted(
            (
                (job_id, record.finished_ns)
                for job_id, record in self._jobs.items()
                if record.state in self.TERMINAL and record.finished_ns
            ),
            key=lambda item: item[1],
        )
        excess = len(terminal) - self._max_completed_jobs
        for job_id, _ in terminal[:max(0, excess)]:
            self._jobs.pop(job_id, None)


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

        has_masks = any(item.mask is not None for item in detections)
        task = "segment" if request.include_masks and model_task == "segment" else "detect"
        postprocess_ms = (time.perf_counter() - t0) * 1000.0

        timing = YoloTiming(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=(time.perf_counter() - t_total) * 1000.0,
        )

        return YoloDetectResult(
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


class YoloWorker:
    """
    Async YOLO worker pool.

    RPC threads only call submit()/job_status()/job_result()/status().
    Ultralytics model loading and inference happen only in worker threads.
    """

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
        self.worker_count = max(1, int(worker_count))
        self.store = YoloJobStore(job_ttl_s=job_ttl_s, max_completed_jobs=max_completed_jobs)
        self.models = YoloModelCache()
        self.detector = UltralyticsYoloDetector()

        self._queue: Queue[str | None] = Queue(maxsize=max(0, int(queue_size)))
        self._state_lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._read_root = normalize_root(read_root)
        self._write_root = normalize_root(write_root)

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("YOLO worker is closed")
            if self._started:
                return

            self._started = True
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"yolo-worker-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        LOG.info("started %d YOLO worker thread(s)", self.worker_count)

    def close(self, *, timeout_s: float = 10.0) -> None:
        """Gracefully stop after already queued jobs have been processed."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

            if not self._started:
                self.models.clear()
                return

            for _ in self._threads:
                self._queue.put(None)
            threads = list(self._threads)

        deadline = time.monotonic() + max(0.0, timeout_s)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            LOG.warning("YOLO workers still running after shutdown timeout: %s", alive)
        else:
            self.models.clear()

    def submit(self, request: YoloInferenceRequest) -> YoloInferenceSubmitResponse:
        with self._state_lock:
            if not self._started or self._closed:
                return YoloInferenceSubmitResponse(
                    accepted=False,
                    input_jpg_path=request.input_jpg_path,
                    output_json_path=request.output_json_path,
                    error="YOLO worker is not running",
                )

        job_id = self.store.create(request)
        try:
            self._queue.put_nowait(job_id)
        except Exception as exc:
            self.store.fail(job_id, f"failed to queue job: {exc}")
            return YoloInferenceSubmitResponse(
                accepted=False,
                input_jpg_path=request.input_jpg_path,
                output_json_path=request.output_json_path,
                error=f"failed to queue job: {exc}",
            )

        return YoloInferenceSubmitResponse(
            accepted=True,
            job_id=job_id,
            state="queued",
            input_jpg_path=request.input_jpg_path,
            output_json_path=request.output_json_path,
        )

    def job_status(self, job_id: str) -> YoloJobStatusResponse:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None:
            return YoloJobStatusResponse(found=False, job_id=job_id, error="job not found or expired")

        return YoloJobStatusResponse(
            found=True,
            job_id=job_id,
            state=snapshot.state,
            model_name=snapshot.request.model_name,
            cuda_device=snapshot.request.cuda_device,
            input_jpg_path=snapshot.request.input_jpg_path,
            output_json_path=snapshot.request.output_json_path,
            created_ns=snapshot.created_ns,
            started_ns=snapshot.started_ns,
            finished_ns=snapshot.finished_ns,
            cache_hit=snapshot.cache_hit,
            num_detections=snapshot.num_detections,
            timing=snapshot.timing,
            error=snapshot.error,
        )

    def job_result(self, job_id: str) -> YoloJobResultResponse:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None:
            return YoloJobResultResponse(found=False, job_id=job_id, error="job not found or expired")

        return YoloJobResultResponse(
            found=True,
            job_id=job_id,
            state=snapshot.state,
            result=snapshot.result if snapshot.state == "succeeded" else None,
            error=snapshot.error,
        )

    def status(self) -> YoloStatusResponse:
        jobs = self.store.summary()
        cached_models, cache_hits, cache_misses = self.models.snapshot()
        with self._state_lock:
            online = self._started and not self._closed

        return YoloStatusResponse(
            online=online,
            queued_jobs=jobs.queued_jobs,
            running_jobs=jobs.running_jobs,
            succeeded_jobs=jobs.succeeded_jobs,
            failed_jobs=jobs.failed_jobs,
            cancelled_jobs=jobs.cancelled_jobs,
            inference_count=jobs.inference_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cached_models=cached_models,
            last_inference_ns=jobs.last_inference_ns,
            last_inference_ms=jobs.last_inference_ms,
            error=jobs.error,
        )

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None or snapshot.state != "queued" or not self.store.mark_running(job_id):
            return

        request = snapshot.request
        try:
            input_path = self._resolve_input_path(request.input_jpg_path)
            output_path = self._resolve_output_path(request.output_json_path)

            with self.models.acquire(request.model_name, request.cuda_device) as (entry, cache_hit):
                self.store.set_cache_hit(job_id, cache_hit)
                result = self.detector.detect(entry.model, entry.task, request, input_path)

            write_json_atomic(output_path, result, job_id)
            self.store.succeed(job_id, result)
            LOG.info(
                "YOLO job %s succeeded: %d detections, %.1f ms",
                job_id,
                result.num_detections,
                result.timing.total_ms,
            )
        except BaseException as exc:
            LOG.exception("YOLO job %s failed", job_id)
            self.store.fail(job_id, str(exc))

    def _resolve_input_path(self, value: str) -> Path:
        path = resolve_path(value, self._read_root)
        if not path.is_file():
            raise FileNotFoundError(f"input JPEG not found: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"input path must end in .jpg or .jpeg: {path}")
        return path

    def _resolve_output_path(self, value: str) -> Path:
        path = resolve_path(value, self._write_root)
        if path.suffix.lower() != ".json":
            raise ValueError(f"output path must end in .json: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
