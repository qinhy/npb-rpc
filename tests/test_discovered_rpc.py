from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import zmq
from npb import BinaryModel, binary_schema

from npb_rpc import (
    DiscoveredRpcClient,
    DiscoveredRpcServer,
    FilesystemDiscovery,
    RpcContext,
    ZmqRpcServer,
)


@binary_schema("npb-rpc.tests.discovered-sum-request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.tests.discovered-sum-response", version=1)
class SumResponse(BinaryModel):
    total: float


def test_discovered_client_calls_registered_server(tmp_path) -> None:
    context = zmq.Context()
    endpoint = f"inproc://npb-rpc-discovery-{uuid.uuid4().hex}"
    discovery = FilesystemDiscovery(tmp_path, heartbeat_timeout=1.0)
    server = DiscoveredRpcServer(
        "sum",
        ZmqRpcServer.bind(endpoint, context=context),
        discovery,
        heartbeat_interval=0.1,
    )

    @server.method("array.sum", request=SumRequest, response=SumResponse)
    def array_sum(request: SumRequest, rpc: RpcContext) -> SumResponse:
        return SumResponse(total=float(request.values.sum()))

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while not discovery.list_instances("sum") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert discovery.list_instances("sum")

        client = DiscoveredRpcClient(
            discovery,
            backend_options={"zmq": {"context": context}},
        )
        response = client.call(
            "sum",
            "array.sum",
            SumRequest(values=np.arange(10, dtype=np.float32)),
            SumResponse,
        )

        assert response.total == 45.0
    finally:
        server.stop()
        thread.join(timeout=2.0)
        server.close()
        context.term()

    assert discovery.list_instances("sum") == []
