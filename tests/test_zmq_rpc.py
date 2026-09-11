from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from npb import BinaryModel, binary_schema  # noqa: E402

from npb_rpc import (  # noqa: E402
    RemoteRpcError,
    RpcContext,
    Status,
    ZmqRpcClient,
    ZmqRpcServer,
)


@binary_schema("npb-rpc.tests.sum-request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.tests.sum-response", version=1)
class SumResponse(BinaryModel):
    total: float
    trace_id: str | None = None


def make_pair():
    context = zmq.Context()
    endpoint = f"inproc://npb-rpc-{uuid.uuid4().hex}"
    server = ZmqRpcServer.bind(endpoint, context=context)

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, rpc: RpcContext) -> SumResponse:
        if request.values.size == 0:
            rpc.abort(Status.INVALID_ARGUMENT, "values cannot be empty")
        return SumResponse(
            total=float(request.values.sum()),
            trace_id=rpc.metadata.get("trace-id"),
        )

    client = ZmqRpcClient.connect(endpoint, context=context, default_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return context, server, client, thread


def close_pair(context, server, client, thread) -> None:
    server.stop()
    thread.join(timeout=2.0)
    client.close()
    server.close()
    context.term()


def test_typed_call_and_metadata() -> None:
    context, server, client, thread = make_pair()
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
        close_pair(context, server, client, thread)


def test_handler_abort_becomes_remote_error() -> None:
    context, server, client, thread = make_pair()
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
        close_pair(context, server, client, thread)


def test_unknown_method_is_structured_error() -> None:
    context, server, client, thread = make_pair()
    try:
        with pytest.raises(RemoteRpcError) as caught:
            client.call(
                "missing.method",
                SumRequest(values=np.arange(3)),
                SumResponse,
            )

        assert caught.value.status == Status.UNIMPLEMENTED
    finally:
        close_pair(context, server, client, thread)
