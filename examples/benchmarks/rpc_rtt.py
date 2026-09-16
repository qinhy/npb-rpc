"""End-to-end npb_rpc payload benchmark across supported backends.

Unlike the raw pynng/pyzmq benchmarks, this includes NPB encoding/decoding,
RPC framing/dispatch, transport, and response construction.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
from npb import BinaryModel, binary_schema

from npb_rpc import (
    NngRpcClient,
    NngRpcServer,
    RpcContext,
    ZmqRpcClient,
    ZmqRpcServer,
    portable_ipc,
)

from _common import (
    BenchmarkResult,
    free_tcp_endpoint,
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

_MESSAGE_MARGIN = 16 * 1024 * 1024


@binary_schema("npb-rpc.benchmark.payload.request", version=1)
class PayloadRequest(BinaryModel):
    payload: np.ndarray


@binary_schema("npb-rpc.benchmark.payload.response", version=1)
class PayloadResponse(BinaryModel):
    payload: np.ndarray


@binary_schema("npb-rpc.benchmark.payload.ack", version=1)
class PayloadAck(BinaryModel):
    nbytes: int


def _server_type(backend: str):
    if backend == "nng":
        return NngRpcServer
    if backend == "zmq":
        return ZmqRpcServer
    if backend == "iceoryx2":
        from npb_rpc import Iceoryx2RpcServer

        return Iceoryx2RpcServer
    raise ValueError(backend)


def _client_type(backend: str):
    if backend == "nng":
        return NngRpcClient
    if backend == "zmq":
        return ZmqRpcClient
    if backend == "iceoryx2":
        from npb_rpc import Iceoryx2RpcClient

        return Iceoryx2RpcClient
    raise ValueError(backend)


def _backend_kwargs(
    backend: str,
    *,
    max_message_bytes: int,
    iceoryx2_poll_interval: float,
    iceoryx2_wait_strategy: str,
    iceoryx2_spin_duration: float,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"max_message_bytes": max_message_bytes}
    if backend == "iceoryx2":
        kwargs.update(
            poll_interval=iceoryx2_poll_interval,
            wait_strategy=iceoryx2_wait_strategy,
            spin_duration=iceoryx2_spin_duration,
        )
    return kwargs


def _server(
    backend: str,
    endpoint: str,
    ready: mp.Event,
    status: mp.Queue,
    max_message_bytes: int,
    iceoryx2_poll_interval: float,
    iceoryx2_wait_strategy: str,
    iceoryx2_spin_duration: float,
) -> None:
    try:
        server = _server_type(backend).bind(
            endpoint,
            **_backend_kwargs(
                backend,
                max_message_bytes=max_message_bytes,
                iceoryx2_poll_interval=iceoryx2_poll_interval,
                iceoryx2_wait_strategy=iceoryx2_wait_strategy,
                iceoryx2_spin_duration=iceoryx2_spin_duration,
            ),
        )

        @server.method("payload.echo", request=PayloadRequest, response=PayloadResponse)
        def echo(request: PayloadRequest, context: RpcContext) -> PayloadResponse:
            return PayloadResponse(payload=request.payload)

        @server.method("payload.consume", request=PayloadRequest, response=PayloadAck)
        def consume(request: PayloadRequest, context: RpcContext) -> PayloadAck:
            return PayloadAck(nbytes=int(request.payload.nbytes))

        with server:
            status.put(("ready", os.getpid(), ""))
            ready.set()
            server.serve_forever()
    except BaseException:
        status.put(("error", os.getpid(), traceback.format_exc()))
        raise


def _endpoint_for(backend: str, transport: str, host: str) -> str:
    name = f"npb-rpc-bench-{backend}-{os.getpid()}"
    if backend == "iceoryx2":
        if transport != "ipc":
            raise ValueError("iceoryx2 only supports ipc in this benchmark")
        return f"iceoryx2://{name}"
    if transport == "ipc":
        if backend == "zmq":
            import zmq

            if not zmq.has("ipc"):
                raise RuntimeError(
                    "this libzmq build does not support ipc://; use ZeroMQ over TCP"
                )
        return portable_ipc(name)
    if transport == "tcp":
        return free_tcp_endpoint(host)
    raise ValueError(transport)


def _verify_echo(response: PayloadResponse, payload: np.ndarray) -> None:
    if response.payload.dtype != payload.dtype:
        raise RuntimeError("echo response dtype mismatch")
    if response.payload.shape != payload.shape:
        raise RuntimeError("echo response shape mismatch")
    if payload.size and (
        response.payload.flat[0] != payload.flat[0]
        or response.payload.flat[-1] != payload.flat[-1]
    ):
        raise RuntimeError("echo response content mismatch")


def _run_mode(
    client,
    *,
    backend: str,
    transport: str,
    mode: str,
    size: int,
    iterations: int,
    warmups: int,
    call_timeout: float,
    progress: bool,
    health_check=None,
) -> BenchmarkResult:
    payload = np.full(size, 0x5A, dtype=np.uint8)
    request = PayloadRequest(payload=payload)

    if mode == "echo":
        method = "payload.echo"
        response_type = PayloadResponse

        def verify(response) -> None:
            _verify_echo(response, payload)

        payload_factor = 2
    elif mode == "consume":
        method = "payload.consume"
        response_type = PayloadAck

        def verify(response) -> None:
            if response.nbytes != size:
                raise RuntimeError("consume response size mismatch")

        payload_factor = 1
    else:
        raise ValueError(mode)

    if progress:
        print(
            f"[{backend}/{transport}] {mode} {human_size(size)}: "
            f"warmup={warmups}, measured={iterations}",
            flush=True,
        )

    for index in range(warmups):
        if health_check is not None:
            health_check()
        verify(client.call(method, request, response_type, timeout=call_timeout))

    latencies_us: list[float] = []
    last_progress = time.monotonic()
    for index in range(iterations):
        if health_check is not None:
            health_check()
        start = time.perf_counter_ns()
        response = client.call(method, request, response_type, timeout=call_timeout)
        elapsed = time.perf_counter_ns() - start
        verify(response)
        latencies_us.append(elapsed / 1000.0)
        now = time.monotonic()
        if progress and now - last_progress >= 2.0:
            done = index + 1
            print(
                f"  progress {done:,}/{iterations:,} ({done / iterations:.0%})",
                flush=True,
            )
            last_progress = now

    if progress:
        print("  done", flush=True)

    return make_result(
        layer="npb_rpc",
        backend=backend,
        transport=transport,
        mode=mode,
        payload_bytes=size,
        latencies_us=latencies_us,
        payload_factor=payload_factor,
    )


def run_case(
    *,
    backend: str,
    transport: str,
    host: str,
    modes: list[str],
    sizes: list[int],
    count: int | None,
    warmup: int | None,
    target_bytes: int,
    min_count: int,
    max_count: int,
    max_message_bytes: int | None = None,
    iceoryx2_poll_interval: float = 0.001,
    iceoryx2_wait_strategy: str = "sleep",
    iceoryx2_spin_duration: float = 50e-6,
    call_timeout: float = 5.0,
    progress: bool = True,
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
        deadline = time.monotonic() + 15.0
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

        if progress:
            print(f"server ready: backend={backend} endpoint={endpoint} pid={process.pid}", flush=True)

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
            for mode in modes:
                for size in sizes:
                    iterations = iterations_for_size(
                        size,
                        count=count,
                        target_bytes=target_bytes,
                        max_count=max_count,
                        min_count=min_count,
                    )
                    warmups = warmup_for_count(iterations, warmup)
                    results.append(
                        _run_mode(
                            client,
                            backend=backend,
                            transport=transport,
                            mode=mode,
                            size=size,
                            iterations=iterations,
                            warmups=warmups,
                            call_timeout=call_timeout,
                            progress=progress,
                            health_check=health_check,
                        )
                    )
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        status.close()
        status.join_thread()

    return results


def cases_for(backend: str, transport: str) -> list[tuple[str, str]]:
    backends = ["nng", "zmq", "iceoryx2"] if backend == "all" else [backend]
    cases: list[tuple[str, str]] = []
    for selected_backend in backends:
        if selected_backend == "iceoryx2":
            if transport in ("ipc", "all"):
                cases.append((selected_backend, "ipc"))
            continue
        transports = ["ipc", "tcp"] if transport == "all" else [transport]
        cases.extend((selected_backend, item) for item in transports)
    return cases


def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="End-to-end npb_rpc RTT benchmark")
    parser.add_argument(
        "--backend", choices=("nng", "zmq", "iceoryx2", "all"), default="nng"
    )
    parser.add_argument("--transport", choices=("ipc", "tcp", "all"), default="ipc")
    parser.add_argument("--mode", choices=("echo", "consume", "both"), default="both")
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
        help="adaptive request bytes per payload size when --count is omitted",
    )
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--max-count", type=int, default=10_000)
    parser.add_argument(
        "--max-message-bytes",
        type=parse_size,
        help="RPC payload limit; default is largest size + 16 MiB",
    )
    parser.add_argument(
        "--iceoryx2-poll-us",
        type=float,
        default=1000.0,
        help="sleep-strategy poll interval in microseconds (default: 1000)",
    )
    parser.add_argument(
        "--iceoryx2-wait-strategy",
        choices=("sleep", "yield", "spin", "hybrid"),
        default="sleep",
        help="iceoryx2 receive wait strategy (default: sleep)",
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
        default=5.0,
        help="timeout for each RPC call in seconds (default: 5)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="disable per-case progress output"
    )
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args()
    if args.iceoryx2_poll_us <= 0:
        parser.error("--iceoryx2-poll-us must be > 0")
    if args.iceoryx2_spin_us < 0:
        parser.error("--iceoryx2-spin-us must be >= 0")
    if args.call_timeout <= 0:
        parser.error("--call-timeout must be > 0")

    modes = ["echo", "consume"] if args.mode == "both" else [args.mode]
    sizes = parse_sizes(args.sizes)
    results: list[BenchmarkResult] = []
    failures: list[str] = []

    for backend, transport in cases_for(args.backend, args.transport):
        try:
            results.extend(
                run_case(
                    backend=backend,
                    transport=transport,
                    host=args.host,
                    modes=modes,
                    sizes=sizes,
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
                    progress=not args.quiet,
                )
            )
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            if args.backend != "all" and args.transport != "all":
                raise
            failures.append(f"{backend}/{transport}: {exc}")

    print_results(results)
    print_tail_note(results)
    if failures:
        print("\nskipped/failed cases:")
        for failure in failures:
            print(f"  - {failure}")
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
