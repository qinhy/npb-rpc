"""Synchronous unary RPC over iceoryx2 shared-memory request/response."""

from __future__ import annotations

import ctypes
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
from npb import BlobStore, NPBError, decode, encode
from pydantic import BaseModel, ValidationError

from ._errors import (
    RemoteRpcError,
    RpcAbort,
    RpcProtocolError,
    RpcTimeoutError,
    RpcTransportError,
)
from ._framing import _LENGTH, _pack_message, _unpack_message, _validate_limit
from ._protocol import (
    DEFAULT_MAX_ENVELOPE_BYTES,
    Envelope,
    RpcContext,
    Status,
    decode_envelope,
    request_envelope,
    response_envelope,
)

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
Handler = Callable[[BaseModel, RpcContext], BaseModel]


def _load_iceoryx2():
    try:
        import iceoryx2
    except ImportError as exc:
        raise ImportError(
            "iceoryx2 support is optional. Install it with "
            "`pip install 'npb-rpc[iceoryx2]'` or `uv add 'npb-rpc[iceoryx2]'`."
        ) from exc
    return iceoryx2


def _normalize_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")
    name = endpoint.removeprefix("iceoryx2://")
    if not name.strip() or "://" in name or "\0" in name:
        raise ValueError("expected an iceoryx2://service-name endpoint or a bare service name")
    return f"iceoryx2://{name}"


def _validate_timeout(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and > 0 or None")


def _open_port(endpoint: str, *, server: bool):
    iox = _load_iceoryx2()
    try:
        node = (
            iox.NodeBuilder.new()
            .signal_handling_mode(iox.SignalHandlingMode.Disabled)
            .create(iox.ServiceType.Ipc)
        )
        service = (
            node.service_builder(iox.ServiceName.new(endpoint.removeprefix("iceoryx2://")))
            .request_response(iox.Slice[ctypes.c_uint8], iox.Slice[ctypes.c_uint8])
            # Native requests fan out to all servers. Unary RPC needs exactly one.
            .max_servers(1)
            .max_clients(32)
            .max_nodes(64)
            .max_active_requests_per_client(1)
            .max_loaned_requests(1)
            .max_response_buffer_size(1)
            .enable_safe_overflow_for_requests(False)
            .enable_safe_overflow_for_responses(False)
            .open_or_create()
        )
        if service.static_config.max_servers != 1:
            raise ValueError("RPC service must allow exactly one server")
        builder = service.server_builder() if server else service.client_builder()
        port = (
            builder.initial_max_slice_len(4096)
            .allocation_strategy(iox.AllocationStrategy.PowerOfTwo)
            .backpressure_strategy(iox.BackpressureStrategy.DiscardData)
            .create()
        )
        return node, service, port
    except Exception as exc:
        raise RpcTransportError(
            f"failed to open iceoryx2 RPC endpoint {endpoint!r}: {exc}"
        ) from exc


def _send_bytes(port: Any, message: bytes):
    loan = port.loan_slice_uninit(len(message))
    ctypes.memmove(loan.payload().as_ptr(), message, len(message))
    return loan.assume_init().send()


def _receive_bytes(sample: Any, *, max_envelope_bytes: int, max_message_bytes: int) -> bytes:
    payload = sample.payload()
    # Check the size before copying from shared memory into Python-owned storage.
    if payload.len() > _LENGTH.size + max_envelope_bytes + max_message_bytes:
        raise RpcProtocolError("iceoryx2 RPC message exceeds the configured size limit")
    return ctypes.string_at(payload.as_ptr(), payload.len())


class Iceoryx2RpcClient:
    """A thread-safe synchronous typed RPC client using iceoryx2 shared memory."""

    def __init__(
        self,
        endpoint: str,
        *,
        default_timeout: float | None = 10.0,
        poll_interval: float = 0.001,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
    ) -> None:
        endpoint = _normalize_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        _validate_timeout("default_timeout", default_timeout)
        _validate_timeout("poll_interval", poll_interval)
        if poll_interval is None:
            raise ValueError("poll_interval must be > 0")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        self.endpoint = endpoint
        self.default_timeout = default_timeout
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.poll_interval = poll_interval
        self._node, self._service, self._port = _open_port(endpoint, server=False)
        self._lock = threading.Lock()
        self._closed = False
        self._closing = threading.Event()

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> Iceoryx2RpcClient:
        return cls(endpoint, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closing.set()
        if self._closed:
            return
        # Do not wait for the call lock: close() must wake a call waiting for a
        # server even when that call has no deadline.
        port = self._port
        self._port = self._service = self._node = None
        self._closed = True
        if port is not None:
            port.delete()

    def __enter__(self) -> Iceoryx2RpcClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def call(
        self,
        method: str,
        request: RequestT,
        response_type: type[ResponseT],
        *,
        timeout: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ResponseT:
        """Call one method and decode its typed NPB response."""
        if self._closed:
            raise RpcTransportError("RPC client is closed")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(request, BaseModel):
            raise TypeError("request must be a Pydantic model instance")
        if not isinstance(response_type, type) or not issubclass(response_type, BaseModel):
            raise TypeError("response_type must be a Pydantic model class")
        effective_timeout = self.default_timeout if timeout is None else timeout
        _validate_timeout("timeout", effective_timeout)

        expires = None if effective_timeout is None else time.monotonic() + effective_timeout
        request_id = uuid4().hex
        deadline_ns = (
            None
            if effective_timeout is None
            else time.time_ns() + int(effective_timeout * 1_000_000_000)
        )
        request_payload = encode(
            request,
            blob_store=self.blob_store,
            externalize_min_bytes=self.externalize_min_bytes,
        )
        if request_payload.nbytes > self.max_message_bytes:
            raise RpcProtocolError(
                f"request payload exceeds the {self.max_message_bytes:,}-byte limit"
            )
        message = _pack_message(
            request_envelope(
                request_id,
                method,
                deadline_unix_ns=deadline_ns,
                metadata=metadata,
            ),
            request_payload,
            max_envelope_bytes=self.max_envelope_bytes,
        )

        acquired = self._lock.acquire(
            timeout=-1 if expires is None else max(0, expires - time.monotonic())
        )
        if not acquired:
            raise RpcTimeoutError(f"RPC call to {method!r} timed out")
        pending = None
        try:
            while True:
                if self._closing.is_set():
                    raise RpcTransportError("RPC client is closed")
                if expires is not None and time.monotonic() >= expires:
                    raise RpcTimeoutError(f"RPC call to {method!r} timed out")
                if pending is None:
                    pending = _send_bytes(self._port, message)
                    if pending.number_of_server_connections == 0:
                        # Nothing received the request, so waiting and retrying is safe.
                        pending.delete()
                        pending = None
                if pending is not None:
                    response = pending.receive()
                    if response is not None:
                        try:
                            response_message = _receive_bytes(
                                response,
                                max_envelope_bytes=self.max_envelope_bytes,
                                max_message_bytes=self.max_message_bytes,
                            )
                        finally:
                            response.delete()
                        break
                delay = self.poll_interval
                if expires is not None:
                    delay = min(delay, max(0, expires - time.monotonic()))
                self._closing.wait(delay)
        except (RpcTimeoutError, RpcTransportError, RpcProtocolError):
            raise
        except Exception as exc:
            raise RpcTransportError(f"iceoryx2 RPC call failed: {exc}") from exc
        finally:
            if pending is not None:
                pending.delete()
            self._lock.release()

        response_meta, response_payload = _unpack_message(
            response_message,
            max_envelope_bytes=self.max_envelope_bytes,
            max_message_bytes=self.max_message_bytes,
        )
        if response_meta.kind != "response":
            raise RpcProtocolError("received an RPC request on an iceoryx2 client")
        if response_meta.request_id != request_id:
            raise RpcProtocolError(
                "received an iceoryx2 RPC response with an unexpected request ID"
            )
        if response_meta.status != Status.OK:
            raise RemoteRpcError(
                response_meta.status or Status.UNKNOWN,
                response_meta.error_message or "remote RPC failed",
                details=response_meta.error_details,
            )
        if not response_payload:
            raise RpcProtocolError("successful RPC response has no NPB payload")
        binary = np.frombuffer(response_payload, dtype=np.uint8)
        try:
            return decode(response_type, binary, blob_store=self.blob_store)
        except (NPBError, ValidationError, TypeError, ValueError) as exc:
            raise RpcProtocolError(f"invalid response payload: {exc}") from exc


@dataclass(frozen=True, slots=True)
class _Method:
    request_type: type[BaseModel]
    response_type: type[BaseModel]
    handler: Handler


class Iceoryx2RpcServer:
    """A synchronous typed RPC server using iceoryx2 shared memory."""

    def __init__(
        self,
        endpoint: str,
        *,
        poll_interval: float = 0.001,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
        debug_errors: bool = False,
    ) -> None:
        endpoint = _normalize_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        _validate_timeout("poll_interval", poll_interval)
        if poll_interval is None:
            raise ValueError("poll_interval must be > 0")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        self.endpoint = endpoint
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.debug_errors = debug_errors
        self.poll_interval = poll_interval
        self._node, self._service, self._port = _open_port(endpoint, server=True)
        self._lock = threading.RLock()
        self._methods: dict[str, _Method] = {}
        self._stopping = threading.Event()
        self._closed = False

    @classmethod
    def bind(cls, endpoint: str, **kwargs: Any) -> Iceoryx2RpcServer:
        return cls(endpoint, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._methods))

    def close(self) -> None:
        self._stopping.set()
        with self._lock:
            if self._closed:
                return
            self._port.delete()
            self._port = self._service = self._node = None
            self._closed = True

    def stop(self) -> None:
        self._stopping.set()

    def __enter__(self) -> Iceoryx2RpcServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def register(
        self,
        name: str,
        request_type: type[RequestT],
        response_type: type[ResponseT],
        handler: Callable[[RequestT, RpcContext], ResponseT],
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("method name must be a non-empty string")
        if name in self._methods:
            raise ValueError(f"RPC method {name!r} is already registered")
        for label, model_type in (
            ("request_type", request_type),
            ("response_type", response_type),
        ):
            if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
                raise TypeError(f"{label} must be a Pydantic model class")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._methods[name] = _Method(  # type: ignore[arg-type]
            request_type,
            response_type,
            handler,
        )

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
        def decorate(
            handler: Callable[[RequestT, RpcContext], ResponseT],
        ) -> Callable[[RequestT, RpcContext], ResponseT]:
            self.register(name, request, response, handler)
            return handler

        return decorate

    def _send(self, active: Any, envelope: Envelope, payload: np.ndarray | bytes = b"") -> None:
        message = _pack_message(envelope, payload, max_envelope_bytes=self.max_envelope_bytes)
        try:
            if active.is_connected:
                _send_bytes(active, message)
        except Exception as exc:
            # A timed-out caller may disconnect while its handler is still running.
            if active.is_connected:
                raise RpcTransportError(f"failed to send iceoryx2 RPC response: {exc}") from exc

    def _send_error(
        self,
        active: Any,
        request_id: str,
        status: Status | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._send(
            active,
            response_envelope(
                request_id,
                status=status,
                error_message=message,
                error_details=details,
            ),
        )

    def serve_once(self, *, timeout_ms: int | None = None) -> bool:
        """Serve at most one request; return False on timeout or stop()."""
        if timeout_ms is not None and (not math.isfinite(timeout_ms) or timeout_ms < 0):
            raise ValueError("timeout_ms must be finite and non-negative or None")
        expires = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000
        with self._lock:
            if self._closed:
                raise RpcTransportError("RPC server is closed")
            while not self._stopping.is_set():
                try:
                    active = self._port.receive()
                except Exception as exc:
                    raise RpcTransportError(
                        f"failed to receive iceoryx2 RPC request: {exc}"
                    ) from exc
                if active is not None:
                    try:
                        self._dispatch(active)
                    finally:
                        active.delete()
                    return True
                delay = self.poll_interval
                if expires is not None:
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        return False
                    delay = min(delay, remaining)
                self._stopping.wait(delay)
            return False

    def _dispatch(self, active: Any) -> None:
        payload = active.payload()
        # Read only the bounded envelope first so oversized requests can receive
        # a correlated error without copying their body from shared memory.
        try:
            if payload.len() < _LENGTH.size:
                return
            (envelope_size,) = _LENGTH.unpack(ctypes.string_at(payload.as_ptr(), _LENGTH.size))
            offset = _LENGTH.size + envelope_size
            if envelope_size > self.max_envelope_bytes or offset > payload.len():
                return
            envelope = decode_envelope(
                ctypes.string_at(payload.as_ptr() + _LENGTH.size, envelope_size),
                max_bytes=self.max_envelope_bytes,
            )
        except RpcProtocolError:
            return
        if envelope.kind != "request":
            self._send_error(
                active,
                envelope.request_id,
                Status.INVALID_ARGUMENT,
                "server expected an RPC request envelope",
            )
            return

        method_name = envelope.method or ""
        method = self._methods.get(method_name)
        if method is None:
            self._send_error(
                active,
                envelope.request_id,
                Status.UNIMPLEMENTED,
                f"unknown RPC method: {method_name}",
            )
            return
        if envelope.deadline_unix_ns is not None and time.time_ns() >= envelope.deadline_unix_ns:
            self._send_error(
                active,
                envelope.request_id,
                Status.DEADLINE_EXCEEDED,
                "RPC deadline expired before dispatch",
            )
            return

        if payload.len() - offset > self.max_message_bytes:
            self._send_error(
                active,
                envelope.request_id,
                Status.RESOURCE_EXHAUSTED,
                f"request payload exceeds the {self.max_message_bytes:,}-byte limit",
            )
            return

        raw_payload = ctypes.string_at(payload.as_ptr() + offset, payload.len() - offset)
        try:
            request_binary = np.frombuffer(raw_payload, dtype=np.uint8)
            request = decode(
                method.request_type,
                request_binary,
                blob_store=self.blob_store,
            )
        except (NPBError, ValidationError, TypeError, ValueError) as exc:
            self._send_error(
                active,
                envelope.request_id,
                Status.INVALID_ARGUMENT,
                f"invalid request payload: {exc}",
            )
            return

        context = RpcContext(
            request_id=envelope.request_id,
            method=method_name,
            deadline_unix_ns=envelope.deadline_unix_ns,
            metadata=envelope.metadata or {},
            peer=b"",
        )
        try:
            response = method.handler(request, context)
            if not isinstance(response, method.response_type):
                response = method.response_type.model_validate(response)
            response_payload = encode(
                response,
                blob_store=self.blob_store,
                externalize_min_bytes=self.externalize_min_bytes,
            )
            if response_payload.nbytes > self.max_message_bytes:
                self._send_error(
                    active,
                    envelope.request_id,
                    Status.RESOURCE_EXHAUSTED,
                    f"response payload exceeds the {self.max_message_bytes:,}-byte limit",
                )
                return
        except RpcAbort as exc:
            self._send_error(
                active,
                envelope.request_id,
                exc.status,
                exc.message,
                details=exc.details,
            )
            return
        except Exception as exc:
            message = str(exc) if self.debug_errors else "RPC handler failed"
            self._send_error(active, envelope.request_id, Status.INTERNAL, message)
            return

        self._send(active, response_envelope(envelope.request_id), response_payload)

    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be > 0")
        if self._closed:
            raise RpcTransportError("RPC server is closed")
        self._stopping.clear()
        while not self._stopping.is_set():
            self.serve_once(timeout_ms=poll_interval_ms)
