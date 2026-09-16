from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest

pytest.importorskip("iceoryx2")

from npb import BinaryModel, binary_schema  # noqa: E402
from npb_rpc import Iceoryx2RpcClient, Iceoryx2RpcServer, RpcContext  # noqa: E402


@binary_schema("npb-rpc.tests.iceoryx2-zero-copy-array", version=1)
class ArrayMessage(BinaryModel):
    values: np.ndarray


def _make_pair():
    endpoint = f"iceoryx2://npb-rpc-zero-copy-{uuid.uuid4().hex}"
    server = Iceoryx2RpcServer.bind(endpoint, wait_strategy="yield")

    @server.method("echo", request=ArrayMessage, response=ArrayMessage)
    def echo(request: ArrayMessage, rpc: RpcContext) -> ArrayMessage:
        del rpc
        return request

    client = Iceoryx2RpcClient.connect(endpoint, wait_strategy="yield")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, client, thread


def _close_pair(server, client, thread) -> None:
    server.stop()
    thread.join(timeout=2.0)
    client.close()
    server.close()
    assert not thread.is_alive()


def test_owned_call_survives_iceoryx2_sample_release() -> None:
    server, client, thread = _make_pair()
    source = np.arange(250_000, dtype=np.float32)
    try:
        first = client.call("echo", ArrayMessage(values=source), ArrayMessage)
        np.testing.assert_array_equal(first.values, source)

        # Reuse transport buffers several times. The first result must remain
        # independent because call() makes one final owned-payload copy.
        for index in range(4):
            current = source + (index + 1)
            response = client.call("echo", ArrayMessage(values=current), ArrayMessage)
            np.testing.assert_array_equal(response.values, current)

        np.testing.assert_array_equal(first.values, source)
    finally:
        _close_pair(server, client, thread)


def test_borrowed_call_releases_sample_and_allows_reuse() -> None:
    server, client, thread = _make_pair()
    source = np.arange(100_000, dtype=np.float32)
    try:
        borrowed = client.call_borrowed(
            "echo",
            ArrayMessage(values=source),
            ArrayMessage,
        )
        assert not borrowed.closed
        with borrowed as response:
            np.testing.assert_array_equal(response.values, source)
            assert not response.values.flags.owndata
        assert borrowed.closed

        # Exiting the context releases the pending response and the client lock.
        second = client.call(
            "echo",
            ArrayMessage(values=-source),
            ArrayMessage,
        )
        np.testing.assert_array_equal(second.values, -source)
    finally:
        _close_pair(server, client, thread)


def test_borrowed_context_returns_model_directly() -> None:
    server, client, thread = _make_pair()
    source = np.arange(32, dtype=np.uint8)
    try:
        with client.call_borrowed(
            "echo",
            ArrayMessage(values=source),
            ArrayMessage,
        ) as response:
            assert isinstance(response, ArrayMessage)
            np.testing.assert_array_equal(response.values, source)
    finally:
        _close_pair(server, client, thread)
