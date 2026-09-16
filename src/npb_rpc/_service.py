"""Discovery-aware wrappers for the concrete RPC transports."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from ._discovery import BackendName, DiscoveryBackend, ServiceRecord, validate_service_name
from ._iceoryx2 import Iceoryx2RpcClient, Iceoryx2RpcServer
from ._nng import NngRpcClient, NngRpcServer
from ._protocol import RpcContext
from ._zmq import ZmqRpcClient, ZmqRpcServer

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
RpcServer = ZmqRpcServer | NngRpcServer | Iceoryx2RpcServer

logger = logging.getLogger(__name__)


def _server_backend(server: RpcServer) -> BackendName:
    if isinstance(server, ZmqRpcServer):
        return "zmq"
    if isinstance(server, NngRpcServer):
        return "nng"
    if isinstance(server, Iceoryx2RpcServer):
        return "iceoryx2"
    raise TypeError("server must be a ZmqRpcServer, NngRpcServer, or Iceoryx2RpcServer")


class DiscoveredRpcServer:
    """Publish an RPC server instance and keep its discovery record alive."""

    def __init__(
        self,
        service: str,
        server: RpcServer,
        discovery: DiscoveryBackend,
        *,
        advertise_endpoint: str | None = None,
        instance_id: str | None = None,
        heartbeat_interval: float = 2.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validate_service_name(service)
        _server_backend(server)
        if not isinstance(discovery, DiscoveryBackend):
            raise TypeError("discovery must implement DiscoveryBackend")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be > 0")
        endpoint = advertise_endpoint or server.endpoint
        if not isinstance(endpoint, str) or "://" not in endpoint:
            raise ValueError("advertise_endpoint must be a transport URL")

        self.service = service
        self.server = server
        self.discovery = discovery
        self.endpoint = endpoint
        self.backend = _server_backend(server)
        self.instance_id = instance_id or uuid4().hex[:12]
        self.heartbeat_interval = heartbeat_interval
        self.metadata = dict(metadata or {})
        self.started_at = time.time()
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._registered = False

    def _record(self) -> ServiceRecord:
        return ServiceRecord(
            service=self.service,
            instance_id=self.instance_id,
            endpoint=self.endpoint,
            backend=self.backend,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            started_at=self.started_at,
            heartbeat_at=time.time(),
            metadata=self.metadata,
        )

    @property
    def methods(self) -> tuple[str, ...]:
        return self.server.methods

    @property
    def closed(self) -> bool:
        return self.server.closed

    def register(
        self,
        name: str,
        request_type: type[RequestT],
        response_type: type[ResponseT],
        handler: Callable[[RequestT, RpcContext], ResponseT],
    ) -> None:
        self.server.register(name, request_type, response_type, handler)

    def method(
        self,
        name: str,
        *,
        request: type[RequestT],
        response: type[ResponseT],
    ) -> Callable[
        [Callable[[RequestT, RpcContext], ResponseT]],
        Callable[[RequestT, RpcContext], ResponseT],
    ]:
        return self.server.method(name, request=request, response=response)

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self.heartbeat_interval):
            try:
                self.discovery.heartbeat(self._record())
            except Exception:
                logger.exception(
                    "discovery heartbeat failed for service=%s instance=%s",
                    self.service,
                    self.instance_id,
                )

    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        """Register the instance, serve requests, and unregister on shutdown."""
        if self._registered:
            raise RuntimeError("discovered RPC server is already running")

        self.discovery.register(self._record())
        self._registered = True
        self._stop_heartbeat.clear()
        try:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"npb-rpc-heartbeat-{self.service}-{self.instance_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()
            self.server.serve_forever(poll_interval_ms=poll_interval_ms)
        finally:
            self._stop_heartbeat.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join()
            self._heartbeat_thread = None
            try:
                self.discovery.unregister(self.service, self.instance_id)
            except Exception:
                logger.exception(
                    "discovery unregister failed for service=%s instance=%s",
                    self.service,
                    self.instance_id,
                )
            self._registered = False

    def stop(self) -> None:
        self.server.stop()

    def close(self) -> None:
        self.stop()
        self.server.close()

    def __enter__(self) -> DiscoveredRpcServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class DiscoveredRpcClient:
    """Resolve a service name and make a direct typed call to one instance."""

    def __init__(
        self,
        discovery: DiscoveryBackend,
        *,
        backend_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(discovery, DiscoveryBackend):
            raise TypeError("discovery must implement DiscoveryBackend")
        options = backend_options or {}
        unknown = set(options) - {"zmq", "nng", "iceoryx2"}
        if unknown:
            raise ValueError(f"unknown backend options: {', '.join(sorted(unknown))}")
        self.discovery = discovery
        self.backend_options = {
            name: dict(values) for name, values in options.items()
        }

    def call(
        self,
        service: str,
        method: str,
        request: RequestT,
        response_type: type[ResponseT],
        *,
        timeout: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ResponseT:
        """Resolve one healthy instance and make a direct RPC call."""
        validate_service_name(service)
        record = self.discovery.resolve(service)
        options = self.backend_options.get(record.backend, {})
        client_type = {
            "zmq": ZmqRpcClient,
            "nng": NngRpcClient,
            "iceoryx2": Iceoryx2RpcClient,
        }[record.backend]
        with client_type.connect(record.endpoint, **options) as client:
            return client.call(
                method,
                request,
                response_type,
                timeout=timeout,
                metadata=metadata,
            )

    def close(self) -> None:
        self.discovery.close()

    def __enter__(self) -> DiscoveredRpcClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
