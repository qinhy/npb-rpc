#!/usr/bin/env python3
"""Fault-tolerant DepthAI camera RPC server bootstrap.

The public RPC interface and all handler registration live in interface.py.
This module only owns transport/server lifetime and the camera worker lifetime.
"""

from __future__ import annotations

import logging
import threading

from npb_rpc import (
    DiscoveredRpcServer,
    FilesystemDiscovery,
    NngRpcServer,
    ZmqRpcServer,
)

try:
    from .interface import CAMERA_API, add_camera_rpc
    from .worker import CameraSupervisor, DaiStereoCameraStream
except ImportError:  # Support running files directly from this directory.
    from interface import CAMERA_API, add_camera_rpc
    from worker import CameraSupervisor, DaiStereoCameraStream


LOG = logging.getLogger("nng_dai_camera")
STOP = threading.Event()


def make_server(
    endpoint: str,
    camera: CameraSupervisor,
    *,
    backend: str = "nng",
):
    """Bind one transport server and install the complete camera RPC interface."""
    if backend == "nng":
        server_type = NngRpcServer
    elif backend == "zmq":
        server_type = ZmqRpcServer
    else:
        raise ValueError(f"unsupported RPC backend: {backend!r}")

    server = server_type.bind(endpoint)
    add_camera_rpc(server, camera, logger=LOG)
    return server


def run_server(
    endpoint: str,
    reconnect_delay: float,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = CAMERA_API.service,
    instance_id: str | None = None,
    advertise_endpoint: str | None = None,
    device: str = "",
    auto_open: bool = True,
) -> None:
    """Run the resilient camera service, optionally registered for discovery."""
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

                if discovery is None:
                    server = raw_server
                else:
                    server = DiscoveredRpcServer(
                        service,
                        raw_server,
                        discovery,
                        instance_id=instance_id or service,
                        advertise_endpoint=advertise_endpoint,
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
