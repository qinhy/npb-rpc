"""End-to-end npb_rpc payload benchmark across supported backends.

Unlike the raw pynng/pyzmq benchmarks, this includes BinaryModel encoding,
RPC framing/dispatch, transport, decoding, and response construction.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

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
    iterations_for_size,
    make_result,
    parse_size,
    parse_sizes,
    print_results,
    warmup_for_count,
    write_csv,
)


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


def _server(backend: str, endpoint: str, ready: mp.Event) -> None:
    server = _server_type(backend).bind(endpoint)

    @server.method("payload.echo", request=PayloadRequest, response=PayloadResponse)
    def echo(request: PayloadRequest, context: RpcContext) -> PayloadResponse:
        return PayloadResponse(payload=request.payload)

    @server.method("payload.consume", request=PayloadRequest, response=PayloadAck)
    def consume(request: PayloadRequest, context: RpcContext) -> PayloadAck:
        return PayloadAck(nbytes=int(request.payload.nbytes))

    with server:
        ready.set()
        server.serve_forever()


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
        response.payload[0] != payload[0] or response.payload[-1] != payload[-1]
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

    for _ in range(warmups):
        verify(client.call(method, request, response_type))

    latencies_us: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        response = client.call(method, request, response_type)
        elapsed = time.perf_counter_ns() - start
        verify(response)
        latencies_us.append(elapsed / 1000.0)

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
) -> list[BenchmarkResult]:
    endpoint = _endpoint_for(backend, transport, host)
    ready = mp.Event()
    process = mp.Process(target=_server, args=(backend, endpoint, ready))
    process.start()

    results: list[BenchmarkResult] = []
    try:
        if not ready.wait(10):
            if process.exitcode is not None:
                raise RuntimeError(
                    f"{backend} benchmark server exited with {process.exitcode}"
                )
            raise RuntimeError(f"{backend} benchmark server did not start")

        client_type = _client_type(backend)
        with client_type.connect(endpoint) as client:
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
                        )
                    )
    finally:
        if process.is_alive():
            process.terminate()
        process.join()

    return results


def _cases(backend: str, transport: str) -> list[tuple[str, str]]:
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
        default=parse_size("256M"),
        help="adaptive bytes per payload size when --count is omitted",
    )
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--max-count", type=int, default=10_000)
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args()

    modes = ["echo", "consume"] if args.mode == "both" else [args.mode]
    sizes = parse_sizes(args.sizes)
    results: list[BenchmarkResult] = []
    failures: list[str] = []

    for backend, transport in _cases(args.backend, args.transport):
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
                )
            )
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            if args.backend != "all" and args.transport != "all":
                raise
            failures.append(f"{backend}/{transport}: {exc}")

    print_results(results)
    if failures:
        print("\nskipped/failed cases:")
        for failure in failures:
            print(f"  - {failure}")
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
