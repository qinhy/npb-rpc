"""Raw iceoryx2 request/response RTT benchmark without NPB or npb_rpc framing.

This benchmark intentionally uses the same dynamic byte-slice request/response
shape as npb_rpc's iceoryx2 backend, but talks to iceoryx2 directly.  It is the
baseline to compare against ``iceoryx2_rtt.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import time
import traceback
from typing import Literal

from _common import (
    BenchmarkResult,
    human_size,
    iterations_for_size,
    make_result,
    parse_size,
    parse_sizes,
    print_results,
    print_tail_note,
    warmup_for_count,
    write_csv,
)

WaitStrategy = Literal["sleep", "yield", "spin", "hybrid"]


def _open_port(endpoint: str, *, server: bool):
    import iceoryx2 as iox

    name = endpoint.removeprefix("iceoryx2://")
    node = (
        iox.NodeBuilder.new()
        .signal_handling_mode(iox.SignalHandlingMode.Disabled)
        .create(iox.ServiceType.Ipc)
    )
    service = (
        node.service_builder(iox.ServiceName.new(name))
        .request_response(iox.Slice[ctypes.c_uint8], iox.Slice[ctypes.c_uint8])
        .max_servers(1)
        .max_clients(32)
        .max_nodes(64)
        .max_active_requests_per_client(1)
        .max_loaned_requests(1)
        .max_response_buffer_size(1)
        .enable_safe_overflow_for_requests(False)
        .enable_safe_overflow_for_responses(False)
        .open_or_create()
    )
    builder = service.server_builder() if server else service.client_builder()
    port = (
        builder.initial_max_slice_len(4096)
        .allocation_strategy(iox.AllocationStrategy.PowerOfTwo)
        .backpressure_strategy(iox.BackpressureStrategy.DiscardData)
        .create()
    )
    return node, service, port


def _idle(strategy: WaitStrategy, *, poll_interval: float, spin_duration: float, started: float) -> float:
    if strategy == "spin":
        return started
    if strategy == "yield":
        time.sleep(0)
        return time.perf_counter()
    if strategy == "hybrid":
        now = time.perf_counter()
        if now - started < spin_duration:
            return started
        time.sleep(0)
        return time.perf_counter()
    time.sleep(poll_interval)
    return time.perf_counter()


def _server(
    endpoint: str,
    ready: mp.Event,
    status: mp.Queue,
    stop: mp.Event,
    wait_strategy: WaitStrategy,
    poll_interval: float,
    spin_duration: float,
) -> None:
    node = service = server = None
    try:
        node, service, server = _open_port(endpoint, server=True)
        status.put(("ready", os.getpid(), ""))
        ready.set()
        spin_started = time.perf_counter()
        while not stop.is_set():
            active = server.receive()
            if active is None:
                spin_started = _idle(
                    wait_strategy,
                    poll_interval=poll_interval,
                    spin_duration=spin_duration,
                    started=spin_started,
                )
                continue
            spin_started = time.perf_counter()
            try:
                request = active.payload()
                response = active.loan_slice_uninit(request.len())
                if request.len():
                    ctypes.memmove(
                        response.payload().as_ptr(),
                        request.as_ptr(),
                        request.len(),
                    )
                response.assume_init().send()
            finally:
                active.delete()
    except BaseException:
        status.put(("error", os.getpid(), traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.delete()


def _call_once(
    client,
    payload: bytes,
    *,
    wait_strategy: WaitStrategy,
    poll_interval: float,
    spin_duration: float,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        request = client.loan_slice_uninit(len(payload))
        if payload:
            ctypes.memmove(request.payload().as_ptr(), payload, len(payload))
        pending = request.assume_init().send()
        if pending.number_of_server_connections:
            break
        pending.delete()
        if time.monotonic() >= deadline:
            raise TimeoutError("iceoryx2 raw request could not find a server")
        time.sleep(0)

    spin_started = time.perf_counter()
    try:
        while True:
            response = pending.receive()
            if response is not None:
                try:
                    body = response.payload()
                    if body.len() != len(payload):
                        raise RuntimeError(
                            f"echo length mismatch: expected {len(payload)}, got {body.len()}"
                        )
                    if payload and ctypes.string_at(body.as_ptr(), 1)[0] != payload[0]:
                        raise RuntimeError("echo content mismatch")
                    return
                finally:
                    response.delete()
            if time.monotonic() >= deadline:
                raise TimeoutError("iceoryx2 raw response timed out")
            spin_started = _idle(
                wait_strategy,
                poll_interval=poll_interval,
                spin_duration=spin_duration,
                started=spin_started,
            )
    finally:
        pending.delete()


def run_case(
    *,
    sizes: list[int],
    count: int | None,
    warmup: int | None,
    target_bytes: int,
    min_count: int,
    max_count: int,
    wait_strategy: WaitStrategy,
    poll_interval: float,
    spin_duration: float,
    timeout: float,
    progress: bool,
) -> list[BenchmarkResult]:
    endpoint = f"iceoryx2://npb-rpc-raw-bench-{os.getpid()}"
    ready = mp.Event()
    stop = mp.Event()
    status = mp.Queue()
    process = mp.Process(
        target=_server,
        args=(endpoint, ready, status, stop, wait_strategy, poll_interval, spin_duration),
    )
    process.start()

    results: list[BenchmarkResult] = []
    try:
        if not ready.wait(15.0):
            detail = ""
            while not status.empty():
                kind, _pid, message = status.get_nowait()
                if kind == "error":
                    detail = f"\n{message}"
            raise RuntimeError(f"raw iceoryx2 server did not start{detail}")
        if process.exitcode is not None:
            raise RuntimeError(f"raw iceoryx2 server exited with {process.exitcode}")

        if progress:
            print(f"server ready: endpoint={endpoint} pid={process.pid}", flush=True)

        node, service, client = _open_port(endpoint, server=False)
        try:
            for size in sizes:
                iterations = iterations_for_size(
                    size,
                    count=count,
                    target_bytes=target_bytes,
                    min_count=min_count,
                    max_count=max_count,
                )
                warmups = warmup_for_count(iterations, warmup)
                payload = bytes([0x5A]) * size
                if progress:
                    print(
                        f"[raw/iceoryx2] echo {human_size(size)}: "
                        f"warmup={warmups}, measured={iterations}",
                        flush=True,
                    )
                for _ in range(warmups):
                    _call_once(
                        client,
                        payload,
                        wait_strategy=wait_strategy,
                        poll_interval=poll_interval,
                        spin_duration=spin_duration,
                        timeout=timeout,
                    )

                latencies_us: list[float] = []
                last_progress = time.monotonic()
                for index in range(iterations):
                    start = time.perf_counter_ns()
                    _call_once(
                        client,
                        payload,
                        wait_strategy=wait_strategy,
                        poll_interval=poll_interval,
                        spin_duration=spin_duration,
                        timeout=timeout,
                    )
                    latencies_us.append((time.perf_counter_ns() - start) / 1000.0)
                    now = time.monotonic()
                    if progress and now - last_progress >= 2.0:
                        done = index + 1
                        print(
                            f"  progress {done:,}/{iterations:,} ({done / iterations:.0%})",
                            flush=True,
                        )
                        last_progress = now

                results.append(
                    make_result(
                        layer="raw",
                        backend="iceoryx2",
                        transport="ipc",
                        mode="echo",
                        payload_bytes=size,
                        latencies_us=latencies_us,
                        payload_factor=2,
                    )
                )
        finally:
            client.delete()
    finally:
        stop.set()
        process.join(2.0)
        if process.is_alive():
            process.terminate()
            process.join()
        status.close()
        status.join_thread()

    return results


def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Raw iceoryx2 request/response RTT benchmark")
    parser.add_argument("--sizes", default="64,1K,16K,256K,1M,4M")
    parser.add_argument("-n", "--count", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--target-bytes", type=parse_size, default=parse_size("256M"))
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--max-count", type=int, default=2_000)
    parser.add_argument(
        "--wait-strategy",
        choices=("sleep", "yield", "spin", "hybrid"),
        default="spin",
        help="receive wait strategy (default: spin)",
    )
    parser.add_argument("--poll-us", type=float, default=100.0)
    parser.add_argument("--spin-us", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--csv")
    args = parser.parse_args()
    if args.poll_us <= 0:
        parser.error("--poll-us must be > 0")
    if args.spin_us < 0:
        parser.error("--spin-us must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")

    results = run_case(
        sizes=parse_sizes(args.sizes),
        count=args.count,
        warmup=args.warmup,
        target_bytes=args.target_bytes,
        min_count=args.min_count,
        max_count=args.max_count,
        wait_strategy=args.wait_strategy,
        poll_interval=args.poll_us / 1_000_000.0,
        spin_duration=args.spin_us / 1_000_000.0,
        timeout=args.timeout,
        progress=not args.quiet,
    )
    print(
        f"iceoryx2 raw wait strategy: {args.wait_strategy}; "
        f"poll={args.poll_us:g} us; spin={args.spin_us:g} us"
    )
    print_results(results)
    print_tail_note(results)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
