"""Shared helpers for the benchmark examples."""

from __future__ import annotations

import csv
import math
import socket
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

_MIB = 1024 * 1024
_SIZE_SUFFIXES = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
}


@dataclass(frozen=True)
class BenchmarkResult:
    layer: str
    backend: str
    transport: str
    mode: str
    payload_bytes: int
    count: int
    mean_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    requests_per_second: float
    payload_mib_per_second: float


def parse_size(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    if not text:
        raise ValueError("empty size")

    split = len(text)
    while split > 0 and text[split - 1].isalpha():
        split -= 1
    number, suffix = text[:split], text[split:]
    if suffix not in _SIZE_SUFFIXES:
        raise ValueError(f"unsupported size suffix: {suffix!r}")

    size = int(float(number) * _SIZE_SUFFIXES[suffix])
    if size < 0:
        raise ValueError("size must be non-negative")
    return size


def parse_sizes(value: str) -> list[int]:
    sizes = [parse_size(item) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("at least one payload size is required")
    return sizes


def human_size(size: int) -> str:
    if size >= 1024**3 and size % 1024**3 == 0:
        return f"{size // 1024**3} GiB"
    if size >= 1024**2 and size % 1024**2 == 0:
        return f"{size // 1024**2} MiB"
    if size >= 1024 and size % 1024 == 0:
        return f"{size // 1024} KiB"
    return f"{size} B"


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def make_result(
    *,
    layer: str,
    backend: str,
    transport: str,
    mode: str,
    payload_bytes: int,
    latencies_us: Sequence[float],
    payload_factor: int,
) -> BenchmarkResult:
    if not latencies_us:
        raise ValueError("no benchmark samples were collected")

    ordered = sorted(latencies_us)
    mean_us = statistics.fmean(latencies_us)
    requests_per_second = 1_000_000.0 / mean_us if mean_us else float("inf")
    payload_mib_per_second = (
        payload_bytes * payload_factor * requests_per_second / _MIB
    )
    return BenchmarkResult(
        layer=layer,
        backend=backend,
        transport=transport,
        mode=mode,
        payload_bytes=payload_bytes,
        count=len(latencies_us),
        mean_us=mean_us,
        p50_us=percentile(ordered, 0.50),
        p95_us=percentile(ordered, 0.95),
        p99_us=percentile(ordered, 0.99),
        min_us=ordered[0],
        max_us=ordered[-1],
        requests_per_second=requests_per_second,
        payload_mib_per_second=payload_mib_per_second,
    )


def iterations_for_size(
    size: int,
    *,
    count: int | None,
    target_bytes: int,
    max_count: int,
    min_count: int,
) -> int:
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        return count
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    effective_size = max(size, 1)
    return max(min_count, min(max_count, target_bytes // effective_size))


def warmup_for_count(count: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("warmup must be non-negative")
        return requested
    return min(100, max(5, count // 10))


def free_tcp_endpoint(host: str = "127.0.0.1") -> str:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    if family == socket.AF_INET6:
        return f"tcp://[{host}]:{port}"
    return f"tcp://{host}:{port}"


def print_results(results: Iterable[BenchmarkResult]) -> None:
    rows = list(results)
    if not rows:
        print("no benchmark results")
        return

    header = (
        f"{'layer':<10} {'backend':<9} {'trans':<5} {'mode':<7} "
        f"{'payload':>9} {'n':>7} {'mean us':>10} {'p50':>10} "
        f"{'p95':>10} {'p99':>10} {'req/s':>11} {'MiB/s':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.layer:<10} {row.backend:<9} {row.transport:<5} {row.mode:<7} "
            f"{human_size(row.payload_bytes):>9} {row.count:>7,d} "
            f"{row.mean_us:>10,.1f} {row.p50_us:>10,.1f} "
            f"{row.p95_us:>10,.1f} {row.p99_us:>10,.1f} "
            f"{row.requests_per_second:>11,.1f} "
            f"{row.payload_mib_per_second:>11,.1f}"
        )


def write_csv(path: str | Path, results: Iterable[BenchmarkResult]) -> None:
    rows = list(results)
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
