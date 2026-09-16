from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics.cfg import DEFAULT_CFG_DICT
except ImportError:  # Older Ultralytics.
    DEFAULT_CFG_DICT = {}

from servers.msg.yolo import (
    YoloDetection,
    YoloInferenceRequest,
    YoloInstancePolygon,
    YoloPolygon,
    YoloTile,
)


# ============================================================
# Internal inference representations
# ============================================================


@dataclass(frozen=True)
class MaskPatch:
    data: np.ndarray  # bool [H, W], tightly cropped
    left: int
    top: int


@dataclass(frozen=True)
class Candidate:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray  # float32 [4], global image coordinates
    tile: YoloTile | None
    mask: MaskPatch | None


@dataclass(frozen=True)
class TileInput:
    image: np.ndarray
    tile: YoloTile


# ============================================================
# Image / ROI / tiling
# ============================================================


def read_jpeg(path: Path) -> np.ndarray:
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"input image must be JPEG: {path}")

    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"input JPEG is empty: {path}")

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode JPEG: {path}")
    return image


def effective_roi(
    bbox: list[float] | None,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    left = max(0, min(image_width, math.floor(x1)))
    top = max(0, min(image_height, math.floor(y1)))
    right = max(0, min(image_width, math.ceil(x2)))
    bottom = max(0, min(image_height, math.ceil(y2)))

    if right <= left or bottom <= top:
        raise ValueError("detection_bbox_xyxy is empty after clipping to image bounds")
    return left, top, right, bottom


def make_divisible(value: int, stride: int) -> int:
    return int(math.ceil(value / stride) * stride)


def axis_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    if step <= 0:
        raise ValueError("tile overlap must be smaller than tile size")

    last = length - tile_size
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_tiles(
    image: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    tile_size: int,
    overlap: int,
) -> list[TileInput]:
    height, width = image.shape[:2]
    xs = axis_positions(width, tile_size, overlap)
    ys = axis_positions(height, tile_size, overlap)

    tiles: list[TileInput] = []
    for y in ys:
        for x in xs:
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            tiles.append(
                TileInput(
                    image=image[y:bottom, x:right],
                    tile=YoloTile(
                        left=origin_x + x,
                        top=origin_y + y,
                        right=origin_x + right,
                        bottom=origin_y + bottom,
                    ),
                )
            )
    return tiles


# ============================================================
# Ultralytics request helpers
# ============================================================


def yolo_device(request: YoloInferenceRequest) -> str:
    return "cpu" if request.cuda_device < 0 else f"cuda:{request.cuda_device}"


def precision_args(request: YoloInferenceRequest) -> dict[str, Any]:
    use_fp16 = request.half and request.cuda_device >= 0
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16} if use_fp16 else {}
    return {"half": use_fp16}


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


# ============================================================
# Masks
# ============================================================


def crop_mask(mask: np.ndarray, left: int, top: int) -> MaskPatch | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    cropped = np.ascontiguousarray(mask[y1:y2, x1:x2], dtype=np.bool_)
    return MaskPatch(cropped, left + x1, top + y1)


def union_mask_patches(patches: list[MaskPatch]) -> MaskPatch:
    left = min(patch.left for patch in patches)
    top = min(patch.top for patch in patches)
    right = max(patch.left + patch.data.shape[1] for patch in patches)
    bottom = max(patch.top + patch.data.shape[0] for patch in patches)

    merged = np.zeros((bottom - top, right - left), dtype=np.bool_)
    for patch in patches:
        y = patch.top - top
        x = patch.left - left
        h, w = patch.data.shape
        merged[y:y + h, x:x + w] |= patch.data
    return MaskPatch(merged, left, top)


def mask_to_polygon(
    patch: MaskPatch,
    *,
    image_width: int,
    image_height: int,
    threshold: float,
    epsilon: float,
    min_area: float,
) -> YoloInstancePolygon:
    mask_u8 = patch.data.astype(np.uint8, copy=False) * 255
    contours, hierarchy = cv2.findContours(
        mask_u8,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[YoloPolygon] = []
    if hierarchy is not None and contours:
        hierarchy = hierarchy[0]
        approximated: dict[int, np.ndarray] = {}
        valid: set[int] = set()

        for index, contour in enumerate(contours):
            area = abs(float(cv2.contourArea(contour)))
            if area < min_area:
                continue

            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue

            approximated[index] = points
            valid.add(index)

        # Do not emit an orphan hole if its parent was filtered out.
        final_indices = [
            index
            for index in range(len(contours))
            if index in valid
            and (hierarchy[index][3] < 0 or hierarchy[index][3] in valid)
        ]
        output_index = {contour_index: i for i, contour_index in enumerate(final_indices)}

        for contour_index in final_indices:
            points = approximated[contour_index]
            parent = int(hierarchy[contour_index][3])
            polygons.append(
                YoloPolygon(
                    points_xy=[
                        [float(x + patch.left), float(y + patch.top)]
                        for x, y in points
                    ],
                    is_hole=parent >= 0,
                    parent_index=output_index.get(parent) if parent >= 0 else None,
                )
            )

    return YoloInstancePolygon(
        size=[image_height, image_width],
        polygons=polygons,
        area=int(np.count_nonzero(patch.data)),
        threshold=threshold,
    )


# ============================================================
# Detection merging
# ============================================================


def box_iou_iom(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))

    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    iou = inter / union if union > 0.0 else 0.0
    iom = inter / smaller if smaller > 0.0 else 0.0
    return iou, iom


def cluster_tiled_candidates(
    candidates: list[Candidate],
    *,
    iou_threshold: float,
    iom_threshold: float,
) -> list[list[Candidate]]:
    count = len(candidates)
    if count <= 1:
        return [[item] for item in candidates]

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(count):
        a = candidates[i]
        for j in range(i + 1, count):
            b = candidates[j]
            if a.class_id != b.class_id:
                continue

            # Ultralytics already suppresses duplicates inside one tile.
            if a.tile is not None and b.tile is not None and a.tile == b.tile:
                continue

            iou, iom = box_iou_iom(a.bbox_xyxy, b.bbox_xyxy)
            if iou >= iou_threshold or iom >= iom_threshold:
                union(i, j)

    grouped: dict[int, list[Candidate]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def unique_tiles(group: list[Candidate]) -> list[YoloTile]:
    output: list[YoloTile] = []
    seen: set[tuple[int, int, int, int]] = set()

    for candidate in group:
        tile = candidate.tile
        if tile is None:
            continue

        key = (tile.left, tile.top, tile.right, tile.bottom)
        if key not in seen:
            seen.add(key)
            output.append(tile)
    return output


def group_to_detection(
    group: list[Candidate],
    *,
    request: YoloInferenceRequest,
    image_width: int,
    image_height: int,
) -> YoloDetection:
    best = max(group, key=lambda item: item.confidence)

    if len(group) == 1:
        bbox = best.bbox_xyxy
    else:
        stacked = np.stack([item.bbox_xyxy for item in group], axis=0)
        bbox = np.array(
            [
                stacked[:, 0].min(),
                stacked[:, 1].min(),
                stacked[:, 2].max(),
                stacked[:, 3].max(),
            ],
            dtype=np.float32,
        )

    instance_mask: YoloInstancePolygon | None = None
    if request.include_masks:
        patches = [item.mask for item in group if item.mask is not None]
        if patches:
            if request.merge_tiled_masks and len(patches) > 1:
                merged = union_mask_patches(patches)
            else:
                merged = best.mask or patches[0]

            instance_mask = mask_to_polygon(
                merged,
                image_width=image_width,
                image_height=image_height,
                threshold=request.mask_threshold,
                epsilon=request.polygon_epsilon,
                min_area=request.polygon_min_area,
            )

    return YoloDetection(
        class_id=best.class_id,
        class_name=best.class_name,
        confidence=best.confidence,
        bbox_xyxy=[float(v) for v in bbox],
        tiles=unique_tiles(group),
        mask=instance_mask,
    )


# ============================================================
# Paths / JSON
# ============================================================


def normalize_root(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    return Path(root).expanduser().resolve()


def resolve_path(value: str, root: Path | None) -> Path:
    raw = Path(value).expanduser()
    if root is not None and not raw.is_absolute():
        raw = root / raw

    path = raw.resolve()
    if root is not None and not path.is_relative_to(root):
        raise ValueError(f"path escapes configured root {root}: {value}")
    return path


def write_json_atomic(output_path: Path, result: Any, job_id: str) -> None:
    temporary = output_path.with_name(f".{output_path.name}.{job_id}.tmp")
    try:
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
