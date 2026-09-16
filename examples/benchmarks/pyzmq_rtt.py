"""Raw pyzmq REQ/REP RTT benchmark over IPC or TCP.

This intentionally bypasses npb_rpc. Use it as a transport-level baseline.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import zmq

from npb_rpc import portable_ipc

from _common import (
    BenchmarkResult,
    free_tcp_endpoint,
    iterations_for_size,
    make_result,
    parse_size,
    parse_sizes,
    print_results,
    print_tail_note,
    warmup_for_count,
    write_csv,
)


def _server(endpoint: str, ready: mp.Event) -> None:
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.RCVTIMEO, 10_000)
    server.setsockopt(zmq.SNDTIMEO, 10_000)
    try:
        server.bind(endpoint)
        ready.set()
        while True:
            server.send(server.recv())
    finally:
        server.close(linger=0)
        context.term()


def _endpoint_for(transport: str, host: str) -> str:
    if transport == "ipc":
        if not zmq.has("ipc"):
            raise RuntimeError(
                "this libzmq build does not support ipc://; use --transport tcp"
            )
        return portable_ipc(f"pyzmq-rtt-{os.getpid()}")
    if transport == "tcp":
        return free_tcp_endpoint(host)
    raise ValueError(transport)


def run_benchmark(
    *,
    transport: str,
    host: str,
    sizes: list[int],
    count: int | None,
    warmup: int | None,
    target_bytes: int,
    min_count: int,
    max_count: int,
) -> list[BenchmarkResult]:
    endpoint = _endpoint_for(transport, host)
    ready = mp.Event()
    process = mp.Process(target=_server, args=(endpoint, ready))
    process.start()

    results: list[BenchmarkResult] = []
    try:
        if not ready.wait(10):
            raise RuntimeError("pyzmq benchmark server did not start")
        if process.exitcode is not None:
            raise RuntimeError(f"pyzmq benchmark server exited with {process.exitcode}")

        context = zmq.Context()
        client = context.socket(zmq.REQ)
        client.setsockopt(zmq.RCVTIMEO, 10_000)
        client.setsockopt(zmq.SNDTIMEO, 10_000)
        try:
            client.connect(endpoint)
            for size in sizes:
                iterations = iterations_for_size(
                    size,
                    count=count,
                    target_bytes=target_bytes,
                    max_count=max_count,
                    min_count=min_count,
                )
                warmups = warmup_for_count(iterations, warmup)
                payload = b"x" * size

                for _ in range(warmups):
                    client.send(payload)
                    if client.recv() != payload:
                        raise RuntimeError("warmup response mismatch")

                latencies_us: list[float] = []
                for _ in range(iterations):
                    start = time.perf_counter_ns()
                    client.send(payload)
                    response = client.recv()
                    elapsed = time.perf_counter_ns() - start
                    if response != payload:
                        raise RuntimeError("benchmark response mismatch")
                    latencies_us.append(elapsed / 1000.0)

                results.append(
                    make_result(
                        layer="raw",
                        backend="pyzmq",
                        transport=transport,
                        mode="echo",
                        payload_bytes=size,
                        latencies_us=latencies_us,
                        payload_factor=2,
                    )
                )
        finally:
            client.close(linger=0)
            context.term()
    finally:
        if process.is_alive():
            process.terminate()
        process.join()

    return results


def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Raw pyzmq REQ/REP RTT benchmark")
    parser.add_argument("--transport", choices=("ipc", "tcp"), default="tcp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--sizes",
        default="64,1K,16K,256K,1M,4M",
        help="comma-separated payload sizes; K/M/G suffixes are supported",
    )
    parser.add_argument("-n", "--count", type=int, help="fixed iterations per size")
    parser.add_argument("--warmup", type=int, help="fixed warmup iterations per size")
    parser.add_argument(
        "--target-bytes",
        type=parse_size,
        default=parse_size("1G"),
        help="adaptive bytes per payload size when --count is omitted",
    )
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--max-count", type=int, default=10_000)
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args()

    results = run_benchmark(
        transport=args.transport,
        host=args.host,
        sizes=parse_sizes(args.sizes),
        count=args.count,
        warmup=args.warmup,
        target_bytes=args.target_bytes,
        min_count=args.min_count,
        max_count=args.max_count,
    )
    print_results(results)
    print_tail_note(results)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
