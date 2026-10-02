"""The shared contract must work consistently across concrete transports."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from npb_rpc import (
    Iceoryx2RpcClient,
    Iceoryx2RpcServer,
    NngRpcClient,
    NngRpcServer,
    RpcClient,
    RpcContext,
    RpcServer,
    ZmqRpcClient,
    ZmqRpcServer,
)


class Message(BaseModel):
    value: int


def echo(request: Message, context: RpcContext) -> Message:
    return request


@pytest.fixture(params=[
    ("pynng", NngRpcClient, NngRpcServer, "inproc://"),
    ("zmq", ZmqRpcClient, ZmqRpcServer, "inproc://"),
    ("iceoryx2", Iceoryx2RpcClient, Iceoryx2RpcServer, "iceoryx2://"),
])
def transport(request):
    module, client, server, scheme = request.param
    pytest.importorskip(module)
    return client, server, f"{scheme}interface-{uuid4().hex}"


@pytest.mark.parametrize("interface", [RpcClient, RpcServer])
def test_interfaces_require_transport_implementation(interface):
    with pytest.raises(TypeError, match="abstract"):
        interface()


def test_shared_registration_and_isolation(transport):
    _, server_type, endpoint = transport
    with server_type.bind(endpoint) as server, server_type.bind(endpoint + "-other") as other:
        assert isinstance(server, RpcServer)
        assert server.methods == other.methods == ()
        server.register("z", Message, Message, echo)
        assert server.method("a", request=Message, response=Message)(echo) is echo
        assert server.methods == ("a", "z")
        assert other.methods == ()
        with pytest.raises(ValueError, match="already registered"):
            server.register("a", Message, Message, echo)
        with pytest.raises(ValueError, match="non-empty"):
            server.register(" ", Message, Message, echo)
        with pytest.raises(TypeError, match="request_type"):
            server.register("invalid", int, Message, echo)
        with pytest.raises(TypeError, match="response_type"):
            server.register("invalid", Message, int, echo)
        with pytest.raises(TypeError, match="callable"):
            server.register("invalid", Message, Message, None)
        assert server.methods == ("a", "z")
    assert server.closed and other.closed


def test_context_managers_close_on_error(transport):
    client_type, server_type, endpoint = transport
    with (
        pytest.raises(RuntimeError, match="application failure"),
        server_type.bind(endpoint) as server,
        client_type.connect(endpoint) as client,
    ):
        assert isinstance(server, RpcServer)
        assert isinstance(client, RpcClient)
        assert not server.closed and not client.closed
        raise RuntimeError("application failure")
    assert server.closed and client.closed
    client.close()
    server.close()
