"""Large NumPy-array codec and end-to-end npb_rpc throughput benchmark.

The benchmark intentionally separates standalone NPB codec cost from complete
RPC cost. Multi-GiB sizes are supported when the backend, NPB format, Python
processes, and available RAM can handle them; they are not part of the safe
default sweep.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import time

import numpy as np
from npb import decode, encode

from _common import (
    BenchmarkResult,
    human_size,
    iterations_for_size,
    make_result,
    parse_size,
    parse_sizes,
    print_results,
    print_tail_note,
    write_csv,
)
from rpc_rtt import (
    PayloadAck,
    PayloadRequest,
    PayloadResponse,
    _backend_kwargs,
    _client_type,
    _endpoint_for,
    _server,
    _verify_echo,
    cases_for,
)

_MESSAGE_MARGIN = 16 * 1024 * 1024
_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


def _make_payload(size: int, dtype: np.dtype) -> np.ndarray:
    if size % dtype.itemsize:
        raise ValueError(
            f"payload size {human_size(size)} is not divisible by "
            f"{dtype.name} itemsize {dtype.itemsize}"
        )
    count = size // dtype.itemsize
    payload = np.empty(count, dtype=dtype)
    if count:
        if np.issubdtype(dtype, np.floating):
            payload.fill(1.25)
        else:
            payload.fill(0x5A)
    return payload


def _throughput_iterations(
    size: int,
    *,
    count: int | None,
    target_bytes: int,
    min_count: int,
    max_count: int,
) -> int:
    return iterations_for_size(
        size,
        count=count,
        target_bytes=target_bytes,
        min_count=min_count,
        max_count=max_count,
    )


def _throughput_warmups(size: int, iterations: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("warmup must be non-negative")
        return requested
    if size >= 64 * 1024 * 1024:
        return 1
    return min(5, max(1, iterations // 10))


def _codec_iterations(size: int, target_bytes: int, max_count: int) -> int:
    return iterations_for_size(
        size,
        count=None,
        target_bytes=target_bytes,
        min_count=1,
        max_count=max_count,
    )


def benchmark_codec(
    *,
    sizes: list[int],
    dtype: np.dtype,
    codec_max_size: int,
    codec_target_bytes: int,
    codec_max_count: int,
) -> tuple[list[BenchmarkResult], list[int]]:
    results: list[BenchmarkResult] = []
    skipped: list[int] = []

    for size in sizes:
        if size > codec_max_size:
            skipped.append(size)
            continue

        payload = _make_payload(size, dtype)
        request = PayloadRequest(payload=payload)
        iterations = _codec_iterations(size, codec_target_bytes, codec_max_count)

        encode_us: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            binary = encode(request)
            elapsed = time.perf_counter_ns() - start
            if binary.nbytes <= 0:
                raise RuntimeError("encoded NPB payload is empty")
            encode_us.append(elapsed / 1000.0)
            del binary

        encoded = encode(request)
        decode_us: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            decoded = decode(PayloadRequest, encoded)
            elapsed = time.perf_counter_ns() - start
            if decoded.payload.nbytes != size:
                raise RuntimeError("decoded ndarray size mismatch")
            if decoded.payload.dtype != dtype:
                raise RuntimeError("decoded ndarray dtype mismatch")
            decode_us.append(elapsed / 1000.0)
            del decoded

        results.append(
            make_result(
                layer="npb",
                backend="codec",
                transport="-",
                mode="encode",
                payload_bytes=size,
                latencies_us=encode_us,
                payload_factor=1,
            )
        )
        results.append(
            make_result(
                layer="npb",
                backend="codec",
                transport="-",
                mode="decode",
                payload_bytes=size,
                latencies_us=decode_us,
                payload_factor=1,
            )
        )

        del encoded, request, payload
        gc.collect()

    return results, skipped


def run_rpc_case(
    *,
    backend: str,
    transport: str,
    host: str,
    modes: list[str],
    sizes: list[int],
    dtype: np.dtype,
    count: int | None,
    warmup: int | None,
    target_bytes: int,
    min_count: int,
    max_count: int,
    max_message_bytes: int | None,
    iceoryx2_poll_interval: float,
    iceoryx2_wait_strategy: str,
    iceoryx2_spin_duration: float,
    call_timeout: float,
) -> list[BenchmarkResult]:
    required_message_bytes = max(sizes, default=0) + _MESSAGE_MARGIN
    effective_max_message_bytes = max_message_bytes or required_message_bytes
    if effective_max_message_bytes <= max(sizes, default=0):
        raise ValueError(
            "max_message_bytes must be larger than the largest application payload"
        )

    endpoint = _endpoint_for(backend, transport, host)
    ready = mp.Event()
    status = mp.Queue()
    process = mp.Process(
        target=_server,
        args=(
            backend,
            endpoint,
            ready,
            status,
            effective_max_message_bytes,
            iceoryx2_poll_interval,
            iceoryx2_wait_strategy,
            iceoryx2_spin_duration,
        ),
    )
    process.start()

    results: list[BenchmarkResult] = []
    try:
        deadline = time.monotonic() + 20.0
        while not ready.is_set():
            if process.exitcode is not None:
                detail = ""
                while not status.empty():
                    kind, _pid, message = status.get_nowait()
                    if kind == "error":
                        detail = f"\n{message}"
                raise RuntimeError(
                    f"{backend} benchmark server exited with {process.exitcode}{detail}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{backend} benchmark server did not start")
            time.sleep(0.05)

        def health_check() -> None:
            if process.exitcode is None:
                return
            detail = ""
            while not status.empty():
                kind, _pid, message = status.get_nowait()
                if kind == "error":
                    detail = f"\n{message}"
            raise RuntimeError(
                f"{backend} benchmark server exited with {process.exitcode}{detail}"
            )

        client_type = _client_type(backend)
        with client_type.connect(
            endpoint,
            **_backend_kwargs(
                backend,
                max_message_bytes=effective_max_message_bytes,
                iceoryx2_poll_interval=iceoryx2_poll_interval,
                iceoryx2_wait_strategy=iceoryx2_wait_strategy,
                iceoryx2_spin_duration=iceoryx2_spin_duration,
            ),
        ) as client:
            for size in sizes:
                payload = _make_payload(size, dtype)
                request = PayloadRequest(payload=payload)
                iterations = _throughput_iterations(
                    size,
                    count=count,
                    target_bytes=target_bytes,
                    min_count=min_count,
                    max_count=max_count,
                )
                warmups = _throughput_warmups(size, iterations, warmup)

                for mode in modes:
                    if mode == "echo":
                        method = "payload.echo"
                        response_type = PayloadResponse
                        payload_factor = 2

                        def verify(response) -> None:
                            _verify_echo(response, payload)

                    elif mode == "consume":
                        method = "payload.consume"
                        response_type = PayloadAck
                        payload_factor = 1

                        def verify(response) -> None:
                            if response.nbytes != size:
                                raise RuntimeError("consume response size mismatch")

                    else:
                        raise ValueError(mode)

                    for _ in range(warmups):
                        health_check()
                        verify(client.call(method, request, response_type, timeout=call_timeout))

                    latencies_us: list[float] = []
                    for _ in range(iterations):
                        health_check()
                        start = time.perf_counter_ns()
                        response = client.call(
                            method, request, response_type, timeout=call_timeout
                        )
                        elapsed = time.perf_counter_ns() - start
                        verify(response)
                        latencies_us.append(elapsed / 1000.0)
                        del response

                    results.append(
                        make_result(
                            layer="npb_rpc",
                            backend=backend,
                            transport=transport,
                            mode=mode,
                            payload_bytes=size,
                            latencies_us=latencies_us,
                            payload_factor=payload_factor,
                        )
                    )

                del request, payload
                gc.collect()
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        status.close()
        status.join_thread()

    return results


def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(
        description="Large ndarray NPB codec + npb_rpc throughput benchmark"
    )
    parser.add_argument(
        "--backend", choices=("nng", "zmq", "iceoryx2", "all"), default="all"
    )
    parser.add_argument("--transport", choices=("ipc", "tcp", "all"), default="ipc")
    parser.add_argument("--mode", choices=("echo", "consume", "both"), default="consume")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="uint8")
    parser.add_argument(
        "--sizes",
        default="1M,4M,16M,64M",
        help=(
            "comma-separated ndarray byte sizes; multi-GiB sizes such as 1G,2G "
            "are supported but require substantial RAM"
        ),
    )
    parser.add_argument("-n", "--count", type=int, help="fixed RPC iterations per size")
    parser.add_argument("--warmup", type=int, help="fixed RPC warmup iterations per size")
    parser.add_argument(
        "--target-bytes",
        type=parse_size,
        default=parse_size("2G"),
        help="adaptive request bytes per size when --count is omitted",
    )
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=100)
    parser.add_argument(
        "--max-message-bytes",
        type=parse_size,
        help="RPC payload limit; default is largest size + 16 MiB",
    )
    parser.add_argument(
        "--iceoryx2-wait-strategy",
        choices=("sleep", "yield", "spin", "hybrid"),
        default="hybrid",
        help="iceoryx2 receive wait strategy (default: hybrid)",
    )
    parser.add_argument(
        "--iceoryx2-poll-us",
        type=float,
        default=1000.0,
        help="sleep-strategy poll interval in microseconds (default: 1000)",
    )
    parser.add_argument(
        "--iceoryx2-spin-us",
        type=float,
        default=50.0,
        help="hybrid busy-spin window in microseconds (default: 50)",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=30.0,
        help="timeout for each large-array RPC call in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-codec",
        action="store_true",
        help="skip standalone npb encode/decode measurements",
    )
    parser.add_argument(
        "--codec-max-size",
        type=parse_size,
        default=parse_size("256M"),
        help="largest payload measured by standalone codec stage (default: 256M)",
    )
    parser.add_argument(
        "--codec-target-bytes",
        type=parse_size,
        default=parse_size("256M"),
        help="adaptive bytes processed per standalone codec stage",
    )
    parser.add_argument("--codec-max-count", type=int, default=20)
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args()

    if args.iceoryx2_poll_us <= 0:
        parser.error("--iceoryx2-poll-us must be > 0")
    if args.iceoryx2_spin_us < 0:
        parser.error("--iceoryx2-spin-us must be >= 0")
    if args.call_timeout <= 0:
        parser.error("--call-timeout must be > 0")
    if args.codec_max_count <= 0:
        parser.error("--codec-max-count must be > 0")

    dtype = _DTYPES[args.dtype]
    sizes = parse_sizes(args.sizes)
    for size in sizes:
        if size % dtype.itemsize:
            parser.error(
                f"size {human_size(size)} is not divisible by dtype {dtype.name} "
                f"itemsize {dtype.itemsize}"
            )

    modes = ["echo", "consume"] if args.mode == "both" else [args.mode]
    results: list[BenchmarkResult] = []
    failures: list[str] = []

    skipped_codec: list[int] = []
    if not args.no_codec:
        codec_results, skipped_codec = benchmark_codec(
            sizes=sizes,
            dtype=dtype,
            codec_max_size=args.codec_max_size,
            codec_target_bytes=args.codec_target_bytes,
            codec_max_count=args.codec_max_count,
        )
        results.extend(codec_results)

    for backend, transport in cases_for(args.backend, args.transport):
        try:
            results.extend(
                run_rpc_case(
                    backend=backend,
                    transport=transport,
                    host=args.host,
                    modes=modes,
                    sizes=sizes,
                    dtype=dtype,
                    count=args.count,
                    warmup=args.warmup,
                    target_bytes=args.target_bytes,
                    min_count=args.min_count,
                    max_count=args.max_count,
                    max_message_bytes=args.max_message_bytes,
                    iceoryx2_poll_interval=args.iceoryx2_poll_us / 1_000_000.0,
                    iceoryx2_wait_strategy=args.iceoryx2_wait_strategy,
                    iceoryx2_spin_duration=args.iceoryx2_spin_us / 1_000_000.0,
                    call_timeout=args.call_timeout,
                )
            )
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            if args.backend != "all" and args.transport != "all":
                raise
            failures.append(f"{backend}/{transport}: {exc}")

    print(f"dtype: {dtype.name}")
    if args.backend in ("iceoryx2", "all"):
        print(
            f"iceoryx2 wait strategy: {args.iceoryx2_wait_strategy}; "
            f"poll={args.iceoryx2_poll_us:g} us; "
            f"spin={args.iceoryx2_spin_us:g} us"
        )
    print_results(results)
    print_tail_note(results)

    if skipped_codec:
        values = ", ".join(human_size(size) for size in skipped_codec)
        print(
            f"\ncodec stage skipped above {human_size(args.codec_max_size)}: {values}. "
            "Raise --codec-max-size if enough RAM is available."
        )
    if failures:
        print("\nskipped/failed cases:")
        for failure in failures:
            print(f"  - {failure}")
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
