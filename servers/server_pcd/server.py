#!/usr/bin/env python3
"""Fault-tolerant PCD RPC server bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Mapping

from npb_rpc import DiscoveredRpcServer, FilesystemDiscovery, NngRpcServer, ZmqRpcServer
from npb_rpc.utils import add_rpc

from servers.server_pcd.interface import PcdService
from servers.msg.pcd import PcdInterface
from servers.server_pcd.worker import PcdWorker


LOG = logging.getLogger("npb_rpc_pcd")
STOP = threading.Event()


def make_server(endpoint: str, pcd: PcdWorker, *, backend: str = "nng"):
    """Create one raw NNG/ZMQ RPC server and register the PCD API."""
    server_type = {"nng": NngRpcServer, "zmq": ZmqRpcServer}.get(backend)
    if server_type is None:
        raise ValueError(f"unsupported RPC backend: {backend!r}")

    server = server_type.bind(endpoint)
    add_rpc(server, PcdInterface, PcdService(pcd, LOG))
    return server


def run_server(
    endpoint: str,
    *,
    # RPC
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = PcdInterface.service,
    instance_id: str | None = None,
    advertise_endpoint: str | None = None,
    # Worker
    worker_count: int = 1,
    queue_size: int = 0,
    job_ttl_s: float = 3600.0,
    max_completed_jobs: int = 128,
    # Filesystem sandbox
    read_root: str | Path | None = None,
    write_root: str | Path | None = None,
    # PCD backends
    backend_options: Mapping[str, Mapping[str, Any]] | None = None,
    calibration_translation_unit: str = "cm",
    # Shutdown
    shutdown_timeout_s: float = 10.0,
) -> None:
    """
    Start the long-lived PCD worker and expose it over RPC.

    The PCD worker survives RPC-server failures/restarts. Only the
    transport/server wrapper is recreated inside the restart loop.
    """
    STOP.clear()

    pcd = PcdWorker(
        worker_count=worker_count,
        queue_size=queue_size,
        job_ttl_s=job_ttl_s,
        max_completed_jobs=max_completed_jobs,
        read_root=read_root,
        write_root=write_root,
        backend_options=backend_options,
        calibration_translation_unit=calibration_translation_unit,
    )
    pcd.start()

    try:
        while not STOP.is_set():
            try:
                raw_server = make_server(endpoint, pcd, backend=backend)
                server = raw_server if discovery is None else DiscoveredRpcServer(
                    service,
                    raw_server,
                    discovery,
                    instance_id=instance_id or service,
                    advertise_endpoint=advertise_endpoint,
                )

                advertised = (
                    f" as {instance_id or service!r} for service {service!r}"
                    if discovery is not None
                    else ""
                )
                LOG.info(
                    "%s PCD server listening on %s%s",
                    backend.upper(),
                    endpoint,
                    advertised,
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
                LOG.exception("PCD RPC server failed; restarting")
                STOP.wait(1.0)
    finally:
        pcd.close(timeout_s=shutdown_timeout_s)
