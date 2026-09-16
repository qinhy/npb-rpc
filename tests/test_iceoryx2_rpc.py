from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest

pytest.importorskip("iceoryx2")

from npb import BinaryModel, binary_schema  # noqa: E402

from npb_rpc import (  # noqa: E402
    Iceoryx2RpcClient,
    Iceoryx2RpcServer,
    RemoteRpcError,
    RpcContext,
    Status,
)


@binary_schema("npb-rpc.tests.iceoryx2-sum-request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.tests.iceoryx2-sum-response", version=1)
class SumResponse(BinaryModel):
    total: float
    trace_id: str | None = None


def make_pair():
    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint)

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, rpc: RpcContext) -> SumResponse:
        if request.values.size == 0:
            rpc.abort(Status.INVALID_ARGUMENT, "values cannot be empty")
        return SumResponse(
            total=float(request.values.sum()),
            trace_id=rpc.metadata.get("trace-id"),
        )

    client = Iceoryx2RpcClient.connect(endpoint, default_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, client, thread


def close_pair(server, client, thread) -> None:
    server.stop()
    thread.join(timeout=2.0)
    client.close()
    server.close()


def test_iceoryx2_typed_call_and_metadata() -> None:
    server, client, thread = make_pair()
    try:
        response = client.call(
            "array.sum",
            SumRequest(values=np.arange(10, dtype=np.float32)),
            SumResponse,
            metadata={"trace-id": "trace-1"},
        )

        assert response.total == 45.0
        assert response.trace_id == "trace-1"
    finally:
        close_pair(server, client, thread)


def test_iceoryx2_handler_abort_becomes_remote_error() -> None:
    server, client, thread = make_pair()
    try:
        with pytest.raises(RemoteRpcError) as caught:
            client.call(
                "array.sum",
                SumRequest(values=np.array([], dtype=np.float32)),
                SumResponse,
            )

        assert caught.value.status == Status.INVALID_ARGUMENT
        assert caught.value.message == "values cannot be empty"
    finally:
        close_pair(server, client, thread)


def test_iceoryx2_unknown_method_is_structured_error() -> None:
    server, client, thread = make_pair()
    try:
        with pytest.raises(RemoteRpcError) as caught:
            client.call(
                "missing.method",
                SumRequest(values=np.arange(3)),
                SumResponse,
            )

        assert caught.value.status == Status.UNIMPLEMENTED
    finally:
        close_pair(server, client, thread)


@binary_schema("npb-rpc.tests.iceoryx2-array", version=1)
class ArrayMessage(BinaryModel):
    values: np.ndarray


def echo(request: ArrayMessage, rpc: RpcContext) -> ArrayMessage:
    return request


def test_large_arrays_survive_loan_release_and_close() -> None:
    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint)
    server.register("echo", ArrayMessage, ArrayMessage, echo)
    client = Iceoryx2RpcClient.connect(endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    values = np.arange(250_000, dtype=np.float32)
    try:
        first = client.call("echo", ArrayMessage(values=values), ArrayMessage)
        for _ in range(4):
            client.call("echo", ArrayMessage(values=-values), ArrayMessage)
    finally:
        close_pair(server, client, thread)
    np.testing.assert_array_equal(first.values, values)


def test_timeout_then_reuse_client() -> None:
    import time

    from npb_rpc import RpcTimeoutError

    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint)
    entered = threading.Event()
    finished = threading.Event()

    @server.method("echo", request=ArrayMessage, response=ArrayMessage)
    def slow_echo(request, rpc):
        if not entered.is_set():
            entered.set()
            time.sleep(0.1)
            finished.set()
        return request

    client = Iceoryx2RpcClient.connect(endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = ArrayMessage(values=np.arange(3))
        with pytest.raises(RpcTimeoutError):
            client.call("echo", request, ArrayMessage, timeout=0.03)
        assert entered.wait(1)
        assert finished.wait(1)
        response = client.call("echo", request, ArrayMessage, timeout=1)
        np.testing.assert_array_equal(response.values, request.values)
    finally:
        close_pair(server, client, thread)


def test_missing_server_times_out_and_can_start_later() -> None:
    from npb_rpc import RpcTimeoutError

    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    with Iceoryx2RpcClient.connect(endpoint) as client:
        request = ArrayMessage(values=np.arange(3))
        with pytest.raises(RpcTimeoutError):
            client.call("echo", request, ArrayMessage, timeout=0.02)
        with Iceoryx2RpcServer.bind(endpoint) as server:
            server.register("echo", ArrayMessage, ArrayMessage, echo)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = client.call("echo", request, ArrayMessage, timeout=1)
                np.testing.assert_array_equal(response.values, request.values)
            finally:
                server.stop()
                thread.join(2)
                assert not thread.is_alive()


def test_duplicate_server_is_rejected() -> None:
    from npb_rpc import RpcTransportError

    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    with Iceoryx2RpcServer.bind(endpoint), pytest.raises(RpcTransportError):
        Iceoryx2RpcServer.bind(endpoint)


def test_message_limits_and_internal_errors() -> None:
    from npb_rpc import RpcProtocolError

    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint, max_message_bytes=140000)

    @server.method("large", request=SumRequest, response=ArrayMessage)
    def large(request, rpc):
        return ArrayMessage(values=np.arange(10000))

    @server.method("broken", request=SumRequest, response=SumResponse)
    def broken(request, rpc):
        raise RuntimeError("private error")

    client = Iceoryx2RpcClient.connect(endpoint, max_message_bytes=140000)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RpcProtocolError, match="request payload exceeds"):
            client.call("large", SumRequest(values=np.arange(10000)), ArrayMessage)
        with pytest.raises(RemoteRpcError) as caught:
            client.call("large", SumRequest(values=np.arange(3)), ArrayMessage)
        assert caught.value.status == Status.RESOURCE_EXHAUSTED
        with pytest.raises(RemoteRpcError) as caught:
            client.call("broken", SumRequest(values=np.arange(3)), SumResponse)
        assert caught.value.status == Status.INTERNAL
        assert caught.value.message == "RPC handler failed"
    finally:
        close_pair(server, client, thread)


def test_poll_stop_close_and_rebind() -> None:
    from npb_rpc import RpcTransportError

    endpoint = f"npb-rpc-tests-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint)
    assert server.endpoint == f"iceoryx2://{endpoint}"
    assert server.serve_once(timeout_ms=0) is False
    thread = threading.Thread(target=server.serve_once, daemon=True)
    thread.start()
    server.stop()
    thread.join(1)
    assert not thread.is_alive()
    server.close()
    server.close()
    assert server.closed
    with pytest.raises(RpcTransportError, match="closed"):
        server.serve_once(timeout_ms=0)
    with Iceoryx2RpcServer.bind(endpoint):
        pass


def test_close_interrupts_unbounded_client_call() -> None:
    from npb_rpc import RpcTransportError

    client = Iceoryx2RpcClient.connect(f"npb-rpc-tests-{uuid.uuid4().hex}", default_timeout=None)
    errors = []

    def call():
        try:
            client.call("echo", ArrayMessage(values=np.arange(3)), ArrayMessage)
        except RpcTransportError as exc:
            errors.append(exc)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    client.close()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1
    client.close()
    assert client.closed


def process_server(endpoint, ready, stop):
    with Iceoryx2RpcServer.bind(endpoint) as server:
        server.register("echo", ArrayMessage, ArrayMessage, echo)
        ready.set()
        while not stop.is_set():
            server.serve_once(timeout_ms=10)


def test_separate_processes() -> None:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    ready, stop = context.Event(), context.Event()
    endpoint = f"iceoryx2://npb-rpc-tests-{uuid.uuid4().hex}"
    process = context.Process(target=process_server, args=(endpoint, ready, stop))
    process.start()
    try:
        assert ready.wait(5)
        with Iceoryx2RpcClient.connect(endpoint) as client:
            values = np.arange(100_000, dtype=np.float32)
            response = client.call("echo", ArrayMessage(values=values), ArrayMessage)
            np.testing.assert_array_equal(response.values, values)
    finally:
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0


def test_discovery_and_generated_client(tmp_path) -> None:
    import time

    from npb_rpc import DiscoveredRpcClient, DiscoveredRpcServer, FilesystemDiscovery, api
    from npb_rpc.utils import build_client_class

    class EchoApi:
        service = "echo"

        @api("echo")
        def echo(self, request: ArrayMessage) -> ArrayMessage: ...

    discovery = FilesystemDiscovery(tmp_path)
    transport = Iceoryx2RpcServer.bind(f"npb-rpc-tests-{uuid.uuid4().hex}")
    transport.register("echo", ArrayMessage, ArrayMessage, echo)
    server = DiscoveredRpcServer("echo", transport, discovery, heartbeat_interval=0.01)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        expires = time.monotonic() + 2
        while not discovery.list_instances("echo") and time.monotonic() < expires:
            time.sleep(0.01)
        record = discovery.resolve("echo")
        assert record.backend == "iceoryx2"
        request = ArrayMessage(values=np.arange(3))
        client = DiscoveredRpcClient(
            discovery, backend_options={"iceoryx2": {"poll_interval": 0.002}}
        )
        result = client.call("echo", "echo", request, ArrayMessage)
        np.testing.assert_array_equal(result.values, request.values)
        generated = build_client_class(EchoApi)(endpoint=record.endpoint, backend="iceoryx2")
        np.testing.assert_array_equal(generated.echo(request).values, request.values)
    finally:
        server.stop()
        thread.join(2)
        server.close()
    assert discovery.list_instances("echo") == []


@pytest.mark.parametrize("endpoint", ["", " ", "iceoryx2://", "tcp://localhost:1234", "a\0b"])
def test_invalid_endpoints(endpoint) -> None:
    with pytest.raises(ValueError):
        Iceoryx2RpcClient.connect(endpoint)
    with pytest.raises(ValueError):
        Iceoryx2RpcServer.bind(endpoint)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_poll_interval(value) -> None:
    with pytest.raises(ValueError):
        Iceoryx2RpcClient.connect("test", poll_interval=value)
    with pytest.raises(ValueError):
        Iceoryx2RpcServer.bind("test", poll_interval=value)


def test_optional_import(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "iceoryx2", None)
    with pytest.raises(ImportError, match=r"npb-rpc\[iceoryx2\]"):
        Iceoryx2RpcClient.connect("test")
    with pytest.raises(ImportError, match=r"npb-rpc\[iceoryx2\]"):
        Iceoryx2RpcServer.bind("test")
