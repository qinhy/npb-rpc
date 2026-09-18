#!/usr/bin/env python3
"""Fault-tolerant DepthAI camera RPC server bootstrap."""

from __future__ import annotations

from servers.logger import logging
import threading

from npb_rpc import DiscoveredRpcServer, RedisDiscovery, NngRpcServer, ZmqRpcServer, Iceoryx2RpcServer

from npb_rpc.utils import add_rpc

from servers.server_dai.interface import CameraService
from servers.msg.dai import CameraInterface
from servers.server_dai.worker import CameraSupervisor, DaiStereoCameraStream


LOG = logging.getLogger(__name__.replace(".",":"))
STOP = threading.Event()


def make_server(endpoint: str, camera: CameraSupervisor, *, backend: str = "nng"):
    server_type = {"nng": NngRpcServer, "zmq": ZmqRpcServer, "iceoryx2": Iceoryx2RpcServer}.get(backend)
    if server_type is None:
        raise ValueError(f"unsupported RPC backend: {backend!r}")

    server = server_type.bind(endpoint)
    add_rpc(server, CameraInterface, CameraService(camera, LOG))
    return server


def run_server(
    endpoint: str,
    reconnect_delay: float,
    *,
    backend: str = "nng",
    discovery: RedisDiscovery | None = None,
    service: str = CameraInterface.service,
    instance_id: str | None = None,
    advertise_endpoint: str | None = None,
    device: str = "",
    auto_open: bool = True,
) -> None:
    STOP.clear()
    camera = CameraSupervisor(
        DaiStereoCameraStream(device_ip=device),
        reconnect_delay=reconnect_delay,
        auto_open=auto_open,
    )
    camera.start()

    try:
        while not STOP.is_set():
            try:
                raw_server = make_server(endpoint, camera, backend=backend)
                server = (
                    raw_server
                    if discovery is None
                    else DiscoveredRpcServer(
                        service,
                        raw_server,
                        discovery,
                        instance_id=instance_id or service,
                        advertise_endpoint=advertise_endpoint,
                    )
                )

                LOG.info(
                    "%s camera server listening on %s%s",
                    backend.upper(),
                    endpoint,
                    (
                        f" as {instance_id or service!r} for service {service!r}"
                        if discovery is not None
                        else ""
                    ),
                )

                with server:
                    server.serve_forever()
                if not STOP.is_set():
                    raise RuntimeError("RPC serve_forever returned unexpectedly")

            except KeyboardInterrupt:
                STOP.set()
            except Exception:
                if STOP.is_set():
                    break
                LOG.exception("RPC server failed; restarting")
                STOP.wait(1.0)
    finally:
        camera.close()
