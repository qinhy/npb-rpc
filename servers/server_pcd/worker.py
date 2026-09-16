from __future__ import annotations
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
import json
from servers.logger import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Iterator, Mapping
import uuid
import numpy as np
import torch

from servers.msg.pcd import (
    PcdBackend,
    PcdBuildRequest,
    PcdBuildResult,
    PcdBuildSubmitResponse,
    PcdJobResultResponse,
    PcdJobState,
    PcdJobStatusResponse,
    PcdSegment,
    PcdStatusResponse,
    PcdTiming,
)
from servers.server_pcd.disparity_predictors import (
    DisparityPredictor,
    FastFoundationStereoDisparity,
    SGBMDisparityPredictor,
    SGBMDisparityPredictorCuda,
    VPIStereoDisparityGPU,
)
from servers.server_pcd.matops import MatDevice, MatOps, NumpyMatOps, TorchMatOps
from servers.server_pcd.pcd_calculation import (
    StereoRgbCalibration,
    StereoRectifier,
    project_points_to_rgb_pixels,
    read_image,
    rectified_left_to_original_left,
    rgb8,
    save_pcd,
    split_cloud_uv,
)


LOG = logging.getLogger(__name__.replace(".",":"))

# --- Path helpers ---

def normalize_root(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    return Path(root).expanduser().resolve()

def resolve_path(value: str | Path, root: Path | None) -> Path:
    raw = Path(value).expanduser()
    if root is not None and (not raw.is_absolute()):
        raw = root / raw
    path = raw.resolve()
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f'path escapes configured root {root}: {path}') from exc
    return path

# --- Atomic output helpers ---

def write_json_atomic(path: Path, model: PcdBuildResult, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{job_id}.tmp')
    try:
        temp.write_text(model.model_dump_json(indent=2), encoding='utf-8')
        os.replace(temp, path)
    finally:
        with suppress(FileNotFoundError):
            temp.unlink()

def save_pcd_atomic(
    path: Path,
    points: Any,
    colors: Any,
    *,
    ops: MatOps,
    binary: bool,
    job_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{job_id}.tmp')
    try:
        save_pcd(temp, points, colors, ops=ops, binary=binary)
        os.replace(temp, path)
    finally:
        with suppress(FileNotFoundError):
            temp.unlink()

# --- Internal job records ---

@dataclass
class _JobRecord:
    request: PcdBuildRequest
    state: PcdJobState = 'queued'
    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0
    cache_hit: bool | None = None
    point_count: int | None = None
    num_segments: int | None = None
    timing: PcdTiming | None = None
    result: PcdBuildResult | None = None
    error: str = ''

@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    request: PcdBuildRequest
    state: PcdJobState
    created_ns: int
    started_ns: int
    finished_ns: int
    cache_hit: bool | None
    point_count: int | None
    num_segments: int | None
    timing: PcdTiming
    result: PcdBuildResult | None
    error: str

@dataclass(frozen=True)
class JobStoreSummary:
    queued_jobs: int
    running_jobs: int
    succeeded_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    build_count: int
    last_job_id: str
    last_build_ns: int
    last_build_ms: float
    error: str

# --- Job store ---

class PcdJobStore:
    """
    Thread-safe asynchronous job state/results.

    Completed jobs are retained according to:

        job_ttl_s
        max_completed_jobs
    """
    TERMINAL = {'succeeded', 'failed', 'cancelled'}

    def __init__(self, *, job_ttl_s: float=3600.0, max_completed_jobs: int=128) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobRecord] = {}
        self._job_ttl_ns = max(0, int(job_ttl_s * 1000000000.0))
        self._max_completed_jobs = max(1, int(max_completed_jobs))
        self._succeeded_total = 0
        self._failed_total = 0
        self._cancelled_total = 0
        self._last_job_id = ''
        self._last_build_ns = 0
        self._last_build_ms = 0.0
        self._last_error = ''

    def create(self, request: PcdBuildRequest) -> str:
        job_id = uuid.uuid4().hex
        now_ns = time.time_ns()
        with self._lock:
            self._prune_locked(now_ns)
            self._jobs[job_id] = _JobRecord(
                request=request.model_copy(deep=True),
                created_ns=now_ns,
            )
        return job_id

    def discard_queued(self, job_id: str) -> bool:
        """
        Remove a job that failed admission into the worker queue.

        This is intentionally not counted as a failed execution.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state != 'queued':
                return False
            del self._jobs[job_id]
            return True

    def mark_running(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state != 'queued':
                return False
            record.state = 'running'
            record.started_ns = time.time_ns()
            return True

    def set_cache_hit(self, job_id: str, cache_hit: bool) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.cache_hit = bool(cache_hit)

    def succeed(self, job_id: str, result: PcdBuildResult) -> None:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in self.TERMINAL:
                return
            record.state = 'succeeded'
            record.finished_ns = now_ns
            record.result = result
            record.point_count = result.point_count
            record.num_segments = result.num_segments
            record.timing = result.timing
            record.error = ''
            self._succeeded_total += 1
            self._last_job_id = job_id
            self._last_build_ns = now_ns
            self._last_build_ms = result.timing.total_ms
            self._last_error = ''
            self._prune_locked(now_ns)

    def fail(self, job_id: str, error: str) -> None:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in self.TERMINAL:
                return
            message = str(error)
            record.state = 'failed'
            record.finished_ns = now_ns
            record.error = message
            self._failed_total += 1
            self._last_error = message
            self._prune_locked(now_ns)

    def cancel(self, job_id: str, error: str='cancelled') -> bool:
        now_ns = time.time_ns()
        with self._lock:
            record = self._jobs.get(job_id)
            # Only queued jobs can be cancelled.
            if record is None or record.state != 'queued':
                return False
            record.state = 'cancelled'
            record.finished_ns = now_ns
            record.error = str(error)
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
                point_count=record.point_count,
                num_segments=record.num_segments,
                timing=record.timing or PcdTiming(),
                result=record.result,
                error=record.error,
            )

    def summary(self) -> JobStoreSummary:
        now_ns = time.time_ns()
        with self._lock:
            self._prune_locked(now_ns)
            return JobStoreSummary(
                queued_jobs=sum((record.state == 'queued' for record in self._jobs.values())),
                running_jobs=sum((record.state == 'running' for record in self._jobs.values())),
                succeeded_jobs=self._succeeded_total,
                failed_jobs=self._failed_total,
                cancelled_jobs=self._cancelled_total,
                build_count=self._succeeded_total,
                last_job_id=self._last_job_id,
                last_build_ns=self._last_build_ns,
                last_build_ms=self._last_build_ms,
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

# --- Calibration cache ---

class PcdCalibrationCache:
    """
    Cache parsed calibration objects.

    Calibration tensors/arrays are cached separately for CPU and
    individual CUDA devices.
    """

    def __init__(self, *, source_translation_unit: str='cm') -> None:
        if source_translation_unit not in {'m', 'cm', 'mm'}:
            raise ValueError("source_translation_unit must be 'm', 'cm', or 'mm'")
        self.source_translation_unit = source_translation_unit
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str, int, int], StereoRgbCalibration] = {}

    def get(self, path: Path, *, ops: MatOps, device_label: str) -> StereoRgbCalibration:
        stat = path.stat()
        key = (str(path), device_label, int(stat.st_mtime_ns), int(stat.st_size))
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                return existing
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'calibration JSON must contain an object: {path}')
        calibration = StereoRgbCalibration.from_dict(
            data,
            source_translation_unit=self.source_translation_unit,
            ops=ops,
        )
        with self._lock:
            stale = [
                existing_key
                for existing_key in self._items
                if existing_key[0] == str(path)
                and existing_key[1] == device_label
                and existing_key != key
            ]
            for existing_key in stale:
                self._items.pop(existing_key, None)
            existing = self._items.get(key)
            if existing is not None:
                return existing
            self._items[key] = calibration
        return calibration

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

@dataclass(frozen=True)
class _BackendEntry:
    backend: PcdBackend
    predictor: DisparityPredictor
    ops: MatOps
    device_label: str
    cache_name: str

# --- Disparity backend cache ---

class PcdBackendCache:
    """
    Cache disparity predictors.

    Cache behavior:

        cpu
            one SGBM predictor

        dnn
            one FoundationStereo predictor per device

        vpi
            one VPI predictor per device

        cuda
            one libSGM predictor per device + image resolution

    libSGM must be constructed with its fixed width/height, hence
    image resolution is part of the CUDA/libSGM cache key.
    """
    DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
        'cpu': {},
        'cuda': {'dll': 'build/Release/sgm_py.dll', 'num_disparities': 256},
        'dnn': {
            'repo_dir': './fast-foundationstereo',
            'model_path': 'weights/23-36-37/model_best_bp2_serialize.pth',
        },
        'vpi': {'num_disparities': 256},
    }

    def __init__(self, *, backend_options: Mapping[str, Mapping[str, Any]] | None=None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str, int, int], _BackendEntry] = {}
        self._resource_locks: dict[str, threading.Lock] = {}
        self._hits = 0
        self._misses = 0
        supplied = backend_options or {}
        self._options: dict[str, dict[str, Any]] = {}
        for backend in ('cpu', 'cuda', 'dnn', 'vpi'):
            options = dict(self.DEFAULT_OPTIONS[backend])
            options.update(supplied.get(backend, {}))
            self._options[backend] = options

    @contextmanager
    def acquire(
        self,
        backend: PcdBackend,
        cuda_device: int,
        *,
        width: int,
        height: int,
    ) -> Iterator[tuple[_BackendEntry, bool, float]]:
        """
        Acquire a cached backend.

        Returns:

            entry
            cache_hit
            backend_load_ms

        Work on one CUDA device is serialized across backend types.
        """
        device_label = self._effective_device(backend, cuda_device)
        key = self._cache_key(backend, device_label, width, height)
        resource_key = device_label if device_label.startswith('cuda:') else f'cpu:{backend}'
        lock = self._get_resource_lock(resource_key)
        with lock:
            with self._device_context(device_label):
                t0 = time.perf_counter()
                entry, hit = self._get_or_load(key, backend, device_label, width, height)
                load_ms = (time.perf_counter() - t0) * 1000.0
                if hit:
                    load_ms = 0.0
                yield (entry, hit, load_ms)

    def snapshot(self) -> tuple[tuple[str, ...], int, int]:
        with self._lock:
            names = tuple((entry.cache_name for entry in self._entries.values()))
            return (names, self._hits, self._misses)

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            close = getattr(entry.predictor, 'close', None)
            if callable(close):
                with suppress(Exception):
                    close()
        with suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cache_key(
        self,
        backend: PcdBackend,
        device_label: str,
        width: int,
        height: int,
    ) -> tuple[str, str, int, int]:
        # libSGM is resolution-specific.
        if backend == 'cuda':
            return (backend, device_label, int(width), int(height))
        return (backend, device_label, 0, 0)

    def _effective_device(self, backend: PcdBackend, cuda_device: int) -> str:
        if backend == 'cpu':
            return 'cpu'
        # FoundationStereo can run on CPU when CUDA is unavailable.
        if backend == 'dnn' and (not torch.cuda.is_available()):
            return 'cpu'
        if not torch.cuda.is_available():
            raise RuntimeError(f'backend {backend!r} requires CUDA, but CUDA is unavailable')
        count = torch.cuda.device_count()
        if not 0 <= cuda_device < count:
            raise ValueError(
                f'CUDA device {cuda_device} does not exist; available device count is {count}'
            )
        return f'cuda:{cuda_device}'

    def _device_context(self, device_label: str):
        if not device_label.startswith('cuda:'):
            return nullcontext()
        device_index = int(device_label.split(':', 1)[1])
        return torch.cuda.device(device_index)

    def _get_resource_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._resource_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._resource_locks[key] = lock
            return lock

    def _get_or_load(
        self,
        key: tuple[str, str, int, int],
        backend: PcdBackend,
        device_label: str,
        width: int,
        height: int,
    ) -> tuple[_BackendEntry, bool]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._hits += 1
                return (entry, True)
            self._misses += 1
        LOG.info('loading PCD disparity backend %s on %s', backend, device_label)
        entry = self._create_entry(backend, device_label, width, height)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                # Another loader won the race; discard the duplicate backend.
                close = getattr(entry.predictor, 'close', None)
                if callable(close):
                    with suppress(Exception):
                        close()
                return (existing, True)
            self._entries[key] = entry
        return (entry, False)

    def _create_entry(
        self,
        backend: PcdBackend,
        device_label: str,
        width: int,
        height: int,
    ) -> _BackendEntry:
        if device_label == 'cpu':
            ops: MatOps = TorchMatOps(device=MatDevice.CPU)
        else:
            # MatDevice.CUDA means the current device selected by acquire().
            ops = TorchMatOps(device=MatDevice.CUDA)
        options = dict(self._options[backend])
        if backend == 'cpu':
            predictor = SGBMDisparityPredictor(**options)
            cache_name = 'cpu'
        elif backend == 'cuda':
            if width <= 0 or height <= 0:
                raise ValueError('libSGM requires a positive image width/height')
            options['width'] = int(width)
            options['height'] = int(height)
            options['device'] = device_label
            predictor = SGBMDisparityPredictorCuda(**options)
            cache_name = f'cuda@{device_label}[{width}x{height}]'
        elif backend == 'dnn':
            options['device'] = device_label
            predictor = FastFoundationStereoDisparity(**options)
            actual_device = str(getattr(predictor, 'device', device_label))
            if actual_device.startswith('cpu'):
                device_label = 'cpu'
                ops = TorchMatOps(device=MatDevice.CPU)
            cache_name = f'dnn@{device_label}'
        elif backend == 'vpi':
            options['device'] = device_label
            predictor = VPIStereoDisparityGPU(**options)
            cache_name = f'vpi@{device_label}'
        else:
            raise ValueError(f'unsupported PCD backend: {backend!r}')
        return _BackendEntry(
            backend=backend,
            predictor=predictor,
            ops=ops,
            device_label=device_label,
            cache_name=cache_name,
        )

# --- Input bundle ---

@dataclass(frozen=True)
class _PcdInputs:
    rgb: np.ndarray
    left: np.ndarray
    right: np.ndarray
    read_ms: float

# --- PCD calculator ---

class PcdCalculator:
    """
    Stereo -> disparity -> XYZ -> RGB projection -> PCD.

    Geometry/math stays in pcd_calculation.py.
    This class is only orchestration and timing.
    """

    def __init__(self, *, calibrations: PcdCalibrationCache) -> None:
        self.calibrations = calibrations
        self.numpy_ops = NumpyMatOps()

    def read_inputs(self, *, rgb_path: Path, left_path: Path, right_path: Path) -> _PcdInputs:
        t0 = time.perf_counter()
        # Decode on CPU first; backend selection may depend on image size.
        left = read_image(left_path, ops=self.numpy_ops, color='gray')
        right = read_image(right_path, ops=self.numpy_ops, color='gray')
        rgb = read_image(rgb_path, ops=self.numpy_ops, color='RGB')
        if left.ndim != 2:
            raise ValueError(f'left image must be HxW grayscale, got {left.shape}')
        if right.ndim != 2:
            raise ValueError(f'right image must be HxW grayscale, got {right.shape}')
        if left.shape != right.shape:
            raise ValueError(f'left/right image sizes differ: {left.shape} != {right.shape}')
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f'RGB image must be HxWx3, got {rgb.shape}')
        return _PcdInputs(
            rgb=np.ascontiguousarray(rgb),
            left=np.ascontiguousarray(left),
            right=np.ascontiguousarray(right),
            read_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def build(
        self,
        *,
        request: PcdBuildRequest,
        inputs: _PcdInputs,
        entry: _BackendEntry,
        calibration_path: Path,
        output_pcd_path: Path,
        detections_path: Path | None,
        segments_output_dir: Path | None,
        job_id: str,
        backend_ms: float,
    ) -> PcdBuildResult:
        ops = entry.ops
        timing = PcdTiming(backend_ms=backend_ms, read_ms=inputs.read_ms)
        t0 = time.perf_counter()
        rgb = ops.from_numpy(inputs.rgb)
        left = ops.from_numpy(inputs.left)
        right = ops.from_numpy(inputs.right)
        timing.read_ms += (time.perf_counter() - t0) * 1000.0
        rgb_h, rgb_w = inputs.rgb.shape[:2]
        stereo_h, stereo_w = inputs.left.shape[:2]
        t0 = time.perf_counter()
        calibration = self.calibrations.get(
            calibration_path,
            ops=ops,
            device_label=entry.device_label,
        )
        timing.calibration_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        rectifier = StereoRectifier(calibration, alpha=request.alpha, ops=ops)
        left_rectified, right_rectified, rectification = rectifier.rectify(left, right)
        timing.rectify_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        disparity = entry.predictor.predict(left_rectified, right_rectified)
        timing.disparity_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        points_rectified, _pixels = rectification.disparity_to_points_rectified(
            disparity,
            min_disparity=max(0.5, float(request.min_disparity)),
            min_depth_m=request.min_depth_m,
            max_depth_m=request.max_depth_m,
            stride=request.stride,
        )
        if ops.shape(points_rectified)[0] == 0:
            raise RuntimeError('No valid 3D points')
        points_left = rectified_left_to_original_left(points_rectified, rectification)
        timing.points_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        uv, points_left = project_points_to_rgb_pixels(
            points_left,
            rgb,
            calibration,
            rgb_image_is_undistorted=request.rgb_image_is_undistorted,
            only_inside=True,
        )
        point_count = int(ops.shape(points_left)[0])
        if point_count == 0:
            raise RuntimeError('No 3D points project inside the RGB image')
        u = ops.astype_int64(ops.round(uv[:, 0]))
        v = ops.astype_int64(ops.round(uv[:, 1]))
        colors_rgb = rgb8(rgb[v, u, :3], order='RGB', ops=ops)
        timing.projection_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        save_pcd_atomic(
            output_pcd_path,
            points_left,
            colors_rgb,
            ops=ops,
            binary=request.binary_pcd,
            job_id=job_id,
        )
        timing.write_ms = (time.perf_counter() - t0) * 1000.0
        segments: list[PcdSegment] = []
        if detections_path is not None:
            assert segments_output_dir is not None
            t0 = time.perf_counter()
            detections_json = json.loads(detections_path.read_text(encoding='utf-8'))
            if not isinstance(detections_json, dict):
                raise ValueError('detections JSON must contain an object')
            # Segmentation uses NumPy/OpenCV, so move this stage back to CPU.
            points_np = ops.to_numpy(points_left)
            uv_np = ops.to_numpy(uv)
            rgb_np = ops.to_numpy(rgb)
            manifest = split_cloud_uv(
                points_np,
                uv_np,
                rgb_np,
                detections_json,
                segments_output_dir,
                rgb_image_color_order='RGB',
                min_points=request.min_segment_points,
                erode_pixels=request.erode_pixels,
                exclusive=request.exclusive_segments,
                save_background=request.save_background,
                save_full_cloud=False,
                binary_pcd=request.binary_pcd,
                ops=self.numpy_ops,
            )
            segments = [PcdSegment(
                detection_index=int(item['detection_index']),
                class_id=int(item['class_id']),
                class_name=str(item['class_name']),
                confidence=float(item['confidence']),
                point_count=int(item['point_count']),
                pcd_path=str(segments_output_dir / item['pcd']),
            ) for item in manifest]
            timing.segmentation_ms = (time.perf_counter() - t0) * 1000.0
        timing.total_ms = (
            timing.backend_ms
            + timing.read_ms
            + timing.calibration_ms
            + timing.rectify_ms
            + timing.disparity_ms
            + timing.points_ms
            + timing.projection_ms
            + timing.segmentation_ms
            + timing.write_ms
        )
        return PcdBuildResult(
            **request.model_dump(),
            backend_used=entry.backend,
            device_used=entry.device_label,
            rgb_image_width=int(rgb_w),
            rgb_image_height=int(rgb_h),
            stereo_image_width=int(stereo_w),
            stereo_image_height=int(stereo_h),
            point_count=point_count,
            num_segments=len(segments),
            segments=segments,
            timing=timing,
        )

# --- Worker façade ---

class PcdWorker:
    """
    Asynchronous PCD worker pool.

    RPC threads only call:

        submit()
        job_status()
        job_result()
        status()

    Stereo/disparity/PCD work happens in worker threads.
    """

    def __init__(
        self,
        *,
        worker_count: int=1,
        queue_size: int=0,
        job_ttl_s: float=3600.0,
        max_completed_jobs: int=128,
        read_root: str | Path | None=None,
        write_root: str | Path | None=None,
        backend_options: Mapping[str, Mapping[str, Any]] | None=None,
        calibration_translation_unit: str='cm',
    ) -> None:
        self.worker_count = max(1, int(worker_count))
        self.store = PcdJobStore(job_ttl_s=job_ttl_s, max_completed_jobs=max_completed_jobs)
        self.backends = PcdBackendCache(backend_options=backend_options)
        self.calibrations = PcdCalibrationCache(
            source_translation_unit=calibration_translation_unit
        )
        self.calculator = PcdCalculator(calibrations=self.calibrations)
        self._queue: Queue[str] = Queue(maxsize=max(0, int(queue_size)))
        self._state_lock = threading.RLock()
        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._read_root = normalize_root(read_root)
        self._write_root = normalize_root(write_root)

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError('PCD worker is closed')
            if self._started:
                return
            self._shutdown.clear()
            self._started = True
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f'pcd-worker-{index}',
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        LOG.info('started %d PCD worker thread(s)', self.worker_count)

    def close(self, *, timeout_s: float=10.0) -> None:
        """
        Stop accepting new work, drain already queued jobs, and
        terminate worker threads.

        Unlike sentinel-based shutdown, this does not perform a
        blocking Queue.put() during close(), so timeout_s remains
        meaningful even for bounded queues.
        """
        with self._state_lock:
            self._closed = True
            self._shutdown.set()
            threads = list(self._threads)
            started = self._started
        if not started:
            self.backends.clear()
            self.calibrations.clear()
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            LOG.warning('PCD workers still running after shutdown timeout: %s', alive)
            return
        self.backends.clear()
        self.calibrations.clear()
        with self._state_lock:
            self._started = False

    def submit(self, request: PcdBuildRequest) -> PcdBuildSubmitResponse:
        """
        Queue one build.

        The worker-state check and queue insertion are protected by
        the same lock as close(), preventing a job from being queued
        after shutdown begins.
        """
        with self._state_lock:
            if not self._started or self._closed:
                return PcdBuildSubmitResponse(
                    accepted=False,
                    rgb_jpg_path=request.rgb_jpg_path,
                    left_jpg_path=request.left_jpg_path,
                    right_jpg_path=request.right_jpg_path,
                    output_pcd_path=request.output_pcd_path,
                    output_json_path=request.output_json_path,
                    error='PCD worker is not running',
                )
            job_id = self.store.create(request)
            try:
                self._queue.put_nowait(job_id)
            except Full:
                self.store.discard_queued(job_id)
                return PcdBuildSubmitResponse(
                    accepted=False,
                    rgb_jpg_path=request.rgb_jpg_path,
                    left_jpg_path=request.left_jpg_path,
                    right_jpg_path=request.right_jpg_path,
                    output_pcd_path=request.output_pcd_path,
                    output_json_path=request.output_json_path,
                    error='PCD worker queue is full',
                )
        return PcdBuildSubmitResponse(
            accepted=True,
            job_id=job_id,
            state='queued',
            rgb_jpg_path=request.rgb_jpg_path,
            left_jpg_path=request.left_jpg_path,
            right_jpg_path=request.right_jpg_path,
            output_pcd_path=request.output_pcd_path,
            output_json_path=request.output_json_path,
        )

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a queued job.

        A cancelled job may still physically remain in Queue; when
        a worker reaches it, mark_running() fails and the item is
        discarded safely.
        """
        return self.store.cancel(job_id)

    def job_status(self, job_id: str) -> PcdJobStatusResponse:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None:
            return PcdJobStatusResponse(
                found=False,
                job_id=job_id,
                error='job not found or expired',
            )
        request = snapshot.request
        return PcdJobStatusResponse(
            found=True,
            job_id=job_id,
            state=snapshot.state,
            backend=request.backend,
            cuda_device=request.cuda_device,
            rgb_jpg_path=request.rgb_jpg_path,
            left_jpg_path=request.left_jpg_path,
            right_jpg_path=request.right_jpg_path,
            output_pcd_path=request.output_pcd_path,
            output_json_path=request.output_json_path,
            created_ns=snapshot.created_ns,
            started_ns=snapshot.started_ns,
            finished_ns=snapshot.finished_ns,
            cache_hit=snapshot.cache_hit,
            point_count=snapshot.point_count,
            num_segments=snapshot.num_segments,
            timing=snapshot.timing,
            error=snapshot.error,
        )

    def job_result(self, job_id: str) -> PcdJobResultResponse:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None:
            return PcdJobResultResponse(
                found=False,
                job_id=job_id,
                error='job not found or expired',
            )
        return PcdJobResultResponse(
            found=True,
            job_id=job_id,
            state=snapshot.state,
            result=snapshot.result if snapshot.state == 'succeeded' else None,
            error=snapshot.error,
        )

    def status(self) -> PcdStatusResponse:
        jobs = self.store.summary()
        cached_backends, cache_hits, cache_misses = self.backends.snapshot()
        with self._state_lock:
            online = self._started and (not self._closed)
        return PcdStatusResponse(
            online=online,
            queued_jobs=jobs.queued_jobs,
            running_jobs=jobs.running_jobs,
            succeeded_jobs=jobs.succeeded_jobs,
            failed_jobs=jobs.failed_jobs,
            cancelled_jobs=jobs.cancelled_jobs,
            build_count=jobs.build_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cached_backends=cached_backends,
            last_job_id=jobs.last_job_id,
            last_build_ns=jobs.last_build_ns,
            last_build_ms=jobs.last_build_ms,
            error=jobs.error,
        )

    def _worker_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=0.1)
            except Empty:
                if self._shutdown.is_set():
                    return
                continue
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        snapshot = self.store.snapshot(job_id)
        if snapshot is None or snapshot.state != 'queued' or (not self.store.mark_running(job_id)):
            return
        request = snapshot.request
        job_started = time.perf_counter()
        try:
            rgb_path = self._resolve_image_path(request.rgb_jpg_path)
            left_path = self._resolve_image_path(request.left_jpg_path)
            right_path = self._resolve_image_path(request.right_jpg_path)
            calibration_path = self._resolve_input_json(request.calibration_json_path)
            output_pcd_path = self._resolve_output_pcd(request.output_pcd_path)
            output_json_path = self._resolve_optional_output_json(request.output_json_path)
            detections_path = self._resolve_optional_input_json(request.detections_json_path)
            segments_output_dir = self._resolve_optional_output_dir(request.segments_output_dir)
            if (
                output_json_path is not None
                and detections_path is not None
                and output_json_path == detections_path
            ):
                raise ValueError('output_json_path must not overwrite detections_json_path')
            if output_json_path is not None and output_json_path == calibration_path:
                raise ValueError('output_json_path must not overwrite calibration_json_path')
            inputs = self.calculator.read_inputs(
                rgb_path=rgb_path,
                left_path=left_path,
                right_path=right_path,
            )
            stereo_h, stereo_w = inputs.left.shape[:2]
            with self.backends.acquire(
                request.backend,
                request.cuda_device,
                width=int(stereo_w),
                height=int(stereo_h),
            ) as (entry, cache_hit, backend_ms):
                self.store.set_cache_hit(job_id, cache_hit)
                result = self.calculator.build(
                    request=request,
                    inputs=inputs,
                    entry=entry,
                    calibration_path=calibration_path,
                    output_pcd_path=output_pcd_path,
                    detections_path=detections_path,
                    segments_output_dir=segments_output_dir,
                    job_id=job_id,
                    backend_ms=backend_ms,
                )
            # Includes decode, backend acquisition, calculation, and PCD writes.
            result.timing.total_ms = (time.perf_counter() - job_started) * 1000.0
            if output_json_path is not None:
                write_json_atomic(output_json_path, result, job_id)
            self.store.succeed(job_id, result)
            LOG.info(
                'PCD job %s succeeded: %d points, %d segments, %.1f ms',
                job_id,
                result.point_count,
                result.num_segments,
                result.timing.total_ms,
            )
        except Exception as exc:
            LOG.exception('PCD job %s failed', job_id)
            self.store.fail(job_id, f'{type(exc).__name__}: {exc}')

    def _resolve_image_path(self, value: str) -> Path:
        path = resolve_path(value, self._read_root)
        if not path.is_file():
            raise FileNotFoundError(f'input image not found: {path}')
        if path.suffix.lower() not in {'.jpg', '.jpeg'}:
            raise ValueError(f'input image path must end in .jpg or .jpeg: {path}')
        return path

    def _resolve_input_json(self, value: str) -> Path:
        path = resolve_path(value, self._read_root)
        if not path.is_file():
            raise FileNotFoundError(f'input JSON not found: {path}')
        if path.suffix.lower() != '.json':
            raise ValueError(f'input path must end in .json: {path}')
        return path

    def _resolve_optional_input_json(self, value: str | None) -> Path | None:
        if value is None:
            return None
        return self._resolve_input_json(value)

    def _resolve_output_pcd(self, value: str) -> Path:
        path = resolve_path(value, self._write_root)
        if path.suffix.lower() != '.pcd':
            raise ValueError(f'output path must end in .pcd: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_optional_output_json(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = resolve_path(value, self._write_root)
        if path.suffix.lower() != '.json':
            raise ValueError(f'output path must end in .json: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_optional_output_dir(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = resolve_path(value, self._write_root)
        if path.exists() and (not path.is_dir()):
            raise ValueError(f'segments_output_dir exists but is not a directory: {path}')
        path.mkdir(parents=True, exist_ok=True)
        return path
