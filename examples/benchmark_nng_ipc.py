from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time

import pynng

from npb_rpc import portable_ipc

_NNG_PROTOCOL_OVERHEAD = 64 * 1024


def _server(endpoint: str, count: int, payload_size: int, ready: mp.Event) -> None:
    with pynng.Rep0(
        recv_timeout=10_000,
        send_timeout=10_000,
    ) as server:
        server.recv_max_size = payload_size + _NNG_PROTOCOL_OVERHEAD
        server.listen(endpoint)
        ready.set()
        for _ in range(count):
            server.send(server.recv())


def run_benchmark(count: int, size: int) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    if size < 0:
        raise ValueError("size must be non-negative")

    warmup_count = 50
    endpoint = portable_ipc(f"nng-bench-{os.getpid()}")
    ready = mp.Event()
    process = mp.Process(
        target=_server,
        args=(endpoint, count + warmup_count, size, ready),
    )
    process.start()

    try:
        if not ready.wait(10):
            raise RuntimeError("NNG benchmark server did not start")

        payload = b"x" * size
        latencies_us: list[float] = []
        with pynng.Req0(
            recv_timeout=10_000,
            send_timeout=10_000,
        ) as client:
            client.recv_max_size = size + _NNG_PROTOCOL_OVERHEAD
            client.dial(endpoint)
            for _ in range(warmup_count):
                client.send(payload)
                assert client.recv() == payload

            for _ in range(count):
                start = time.perf_counter_ns()
                client.send(payload)
                response = client.recv()
                elapsed = time.perf_counter_ns() - start
                if response != payload:
                    raise RuntimeError("benchmark response mismatch")
                latencies_us.append(elapsed / 1000.0)

        process.join(10)
        if process.exitcode != 0:
            raise RuntimeError(f"NNG benchmark server exited with {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()

    ordered = sorted(latencies_us)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]

    print(f"endpoint: {endpoint}")
    print(f"messages: {count:,}")
    print(f"payload:  {size:,} bytes")
    print(f"p50 RTT:  {p50:,.1f} us")
    print(f"p95 RTT:  {p95:,.1f} us")
    print(f"p99 RTT:  {p99:,.1f} us")
    print(f"mean RTT: {statistics.mean(latencies_us):,.1f} us")


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="NNG IPC request/reply RTT benchmark")
    parser.add_argument("-n", "--count", type=int, default=10_000)
    parser.add_argument("-s", "--size", type=int, default=64)
    args = parser.parse_args()

    run_benchmark(args.count, args.size)
