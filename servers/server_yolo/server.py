#!/usr/bin/env python3
"""Fault-tolerant yolo RPC server bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path
import threading

from npb_rpc import DiscoveredRpcServer, FilesystemDiscovery, NngRpcServer, ZmqRpcServer, Iceoryx2RpcServer

from npb_rpc.utils import add_rpc

from servers.server_yolo.interface import YoloService
from servers.msg.yolo import YoloInterface
from servers.server_yolo.worker import YoloWorker


LOG = logging.getLogger("nng_dai_yolo")
STOP = threading.Event()


def make_server(endpoint: str, yolo: YoloWorker, *, backend: str = "nng"):
    server_type = {"nng": NngRpcServer, "zmq": ZmqRpcServer, "iceoryx2": Iceoryx2RpcServer}.get(backend)
    if server_type is None:
        raise ValueError(f"unsupported RPC backend: {backend!r}")

    server = server_type.bind(endpoint)
    add_rpc(server, YoloInterface, YoloService(yolo, LOG))
    return server


def run_server(
    endpoint: str,
    *,
    backend: str = "nng",
    discovery: FilesystemDiscovery | None = None,
    service: str = YoloInterface.service,
    instance_id: str | None = None,
    advertise_endpoint: str | None = None,
    worker_count: int = 1,
    queue_size: int = 0,
    job_ttl_s: float = 3600,
    max_completed_jobs: int = 128,
    read_root: str | Path | None = None,
    write_root: str | Path | None = None
) -> None:
    STOP.clear()
    yolo = YoloWorker(
        worker_count=worker_count,
        queue_size=queue_size,
        job_ttl_s=job_ttl_s,
        max_completed_jobs=max_completed_jobs,
        read_root=read_root,
        write_root=write_root,
    )
    yolo.start()

    try:
        while not STOP.is_set():
            try:
                raw_server = make_server(endpoint, yolo, backend=backend)
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
                    "%s yolo server listening on %s%s",
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
        yolo.close()
