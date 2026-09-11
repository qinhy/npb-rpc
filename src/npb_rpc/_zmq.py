"""Synchronous unary RPC over ZeroMQ DEALER/ROUTER sockets."""

from __future__ import annotations

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
from ._protocol import (
    DEFAULT_MAX_ENVELOPE_BYTES,
    Envelope,
    RpcContext,
    Status,
    decode_envelope,
    encode_envelope,
    request_envelope,
    response_envelope,
)

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
Handler = Callable[[BaseModel, RpcContext], BaseModel]


def _load_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise ImportError(
            "ZeroMQ support is optional. Install it with "
            "`pip install 'npb-rpc[zmq]'` or `uv add 'npb-rpc[zmq]'`."
        ) from exc
    return zmq


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")


def _validate_limit(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _remaining_poll_ms(deadline_ns: int | None) -> int | None:
    if deadline_ns is None:
        return None
    remaining_ns = deadline_ns - time.time_ns()
    if remaining_ns <= 0:
        return 0
    return max(1, math.ceil(remaining_ns / 1_000_000))


class ZmqRpcClient:
    """A thread-safe synchronous unary RPC client.

    Calls on one client are serialized because ZeroMQ sockets are not thread-safe.
    Create one client per calling thread when parallel in-flight calls are required.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        context: Any | None = None,
        default_timeout: float | None = 10.0,
        send_timeout_ms: int = 10_000,
        linger_ms: int = 0,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
    ) -> None:
        _validate_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        if default_timeout is not None and default_timeout <= 0:
            raise ValueError("default_timeout must be > 0 or None")
        if send_timeout_ms < 0 or linger_ms < 0:
            raise ValueError("socket timeouts must be non-negative")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        zmq = _load_zmq()
        self.endpoint = endpoint
        self.default_timeout = default_timeout
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self._zmq = zmq
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, linger_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, send_timeout_ms)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> ZmqRpcClient:
        return cls(endpoint, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close()
        self._closed = True

    def __enter__(self) -> ZmqRpcClient:
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
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("timeout must be > 0 or None")

        request_id = uuid4().hex
        deadline_ns = (
            None
            if effective_timeout is None
            else time.time_ns() + int(effective_timeout * 1_000_000_000)
        )
        envelope = encode_envelope(
            request_envelope(
                request_id,
                method,
                deadline_unix_ns=deadline_ns,
                metadata=metadata,
            ),
            max_bytes=self.max_envelope_bytes,
        )
        payload = encode(
            request,
            blob_store=self.blob_store,
            externalize_min_bytes=self.externalize_min_bytes,
        )
        if payload.nbytes > self.max_message_bytes:
            raise RpcProtocolError(
                f"request payload exceeds the {self.max_message_bytes:,}-byte limit"
            )

        with self._lock:
            try:
                self._socket.send_multipart([envelope, payload], copy=True)
            except self._zmq.Again as exc:
                raise RpcTimeoutError("timed out while sending RPC request") from exc
            except self._zmq.ZMQError as exc:
                raise RpcTransportError(f"failed to send RPC request: {exc}") from exc

            while True:
                poll_ms = _remaining_poll_ms(deadline_ns)
                if poll_ms == 0:
                    raise RpcTimeoutError(f"RPC call to {method!r} timed out")
                try:
                    if self._socket.poll(poll_ms, self._zmq.POLLIN) == 0:
                        raise RpcTimeoutError(f"RPC call to {method!r} timed out")
                    frames = self._socket.recv_multipart(copy=True)
                except self._zmq.Again as exc:
                    raise RpcTimeoutError(f"RPC call to {method!r} timed out") from exc
                except self._zmq.ZMQError as exc:
                    raise RpcTransportError(f"failed to receive RPC response: {exc}") from exc
                if len(frames) != 2:
                    raise RpcProtocolError("RPC response must contain two frames")
                response_meta = decode_envelope(
                    frames[0],
                    max_bytes=self.max_envelope_bytes,
                )
                if response_meta.kind != "response":
                    raise RpcProtocolError("received an RPC request on a client socket")
                # A timed-out call may leave a late response queued. Discard it.
                if response_meta.request_id != request_id:
                    continue
                break

        if response_meta.status != Status.OK:
            raise RemoteRpcError(
                response_meta.status or Status.UNKNOWN,
                response_meta.error_message or "remote RPC failed",
                details=response_meta.error_details,
            )
        response_payload = frames[1]
        if not response_payload:
            raise RpcProtocolError("successful RPC response has no NPB payload")
        if len(response_payload) > self.max_message_bytes:
            raise RpcProtocolError(
                f"response payload exceeds the {self.max_message_bytes:,}-byte limit"
            )
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


class ZmqRpcServer:
    """A synchronous unary RPC server using a ZeroMQ ROUTER socket."""

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool = True,
        context: Any | None = None,
        linger_ms: int = 0,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
        debug_errors: bool = False,
    ) -> None:
        _validate_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        if linger_ms < 0:
            raise ValueError("linger_ms must be non-negative")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        zmq = _load_zmq()
        self.endpoint = endpoint
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.debug_errors = debug_errors
        self._zmq = zmq
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, linger_ms)
        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)
        self._methods: dict[str, _Method] = {}
        self._stopping = threading.Event()
        self._closed = False

    @classmethod
    def bind(cls, endpoint: str, **kwargs: Any) -> ZmqRpcServer:
        return cls(endpoint, bind=True, **kwargs)

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> ZmqRpcServer:
        return cls(endpoint, bind=False, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._methods))

    def close(self) -> None:
        if self._closed:
            return
        self._stopping.set()
        self._socket.close()
        self._closed = True

    def stop(self) -> None:
        """Ask serve_forever() to return after its current handler finishes."""
        self._stopping.set()

    def __enter__(self) -> ZmqRpcServer:
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
        """Register a typed unary method."""
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
        """Decorator form of register()."""

        def decorate(
            handler: Callable[[RequestT, RpcContext], ResponseT],
        ) -> Callable[[RequestT, RpcContext], ResponseT]:
            self.register(name, request, response, handler)
            return handler

        return decorate

    def _send(
        self,
        peer: bytes,
        envelope: Envelope,
        payload: np.ndarray | bytes = b"",
    ) -> None:
        encoded_envelope = encode_envelope(
            envelope,
            max_bytes=self.max_envelope_bytes,
        )
        self._socket.send_multipart([peer, encoded_envelope, payload], copy=True)

    def _send_error(
        self,
        peer: bytes,
        request_id: str,
        status: Status | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._send(
            peer,
            response_envelope(
                request_id,
                status=status,
                error_message=message,
                error_details=details,
            ),
        )

    def serve_once(self, *, timeout_ms: int | None = None) -> bool:
        """Serve at most one request; return False when polling times out."""
        if self._closed:
            raise RpcTransportError("RPC server is closed")
        if timeout_ms is not None and timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative or None")
        try:
            if self._socket.poll(timeout_ms, self._zmq.POLLIN) == 0:
                return False
            frames = self._socket.recv_multipart(copy=True)
        except self._zmq.ZMQError as exc:
            if self._stopping.is_set():
                return False
            raise RpcTransportError(f"failed to receive RPC request: {exc}") from exc
        if len(frames) != 3:
            # Without a valid routing frame and request ID there is no safe reply.
            return True

        peer, raw_envelope, raw_payload = frames
        try:
            envelope = decode_envelope(
                raw_envelope,
                max_bytes=self.max_envelope_bytes,
            )
        except RpcProtocolError:
            # An invalid envelope may not contain a trustworthy correlation ID.
            return True
        if envelope.kind != "request":
            self._send_error(
                peer,
                envelope.request_id,
                Status.INVALID_ARGUMENT,
                "server expected an RPC request envelope",
            )
            return True

        method_name = envelope.method or ""
        method = self._methods.get(method_name)
        if method is None:
            self._send_error(
                peer,
                envelope.request_id,
                Status.UNIMPLEMENTED,
                f"unknown RPC method: {method_name}",
            )
            return True
        if envelope.deadline_unix_ns is not None and time.time_ns() >= envelope.deadline_unix_ns:
            self._send_error(
                peer,
                envelope.request_id,
                Status.DEADLINE_EXCEEDED,
                "RPC deadline expired before dispatch",
            )
            return True
        if len(raw_payload) > self.max_message_bytes:
            self._send_error(
                peer,
                envelope.request_id,
                Status.RESOURCE_EXHAUSTED,
                f"request payload exceeds the {self.max_message_bytes:,}-byte limit",
            )
            return True

        try:
            request_binary = np.frombuffer(raw_payload, dtype=np.uint8)
            request = decode(
                method.request_type,
                request_binary,
                blob_store=self.blob_store,
            )
        except (NPBError, ValidationError, TypeError, ValueError) as exc:
            self._send_error(
                peer,
                envelope.request_id,
                Status.INVALID_ARGUMENT,
                f"invalid request payload: {exc}",
            )
            return True

        context = RpcContext(
            request_id=envelope.request_id,
            method=method_name,
            deadline_unix_ns=envelope.deadline_unix_ns,
            metadata=envelope.metadata or {},
            peer=peer,
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
                    peer,
                    envelope.request_id,
                    Status.RESOURCE_EXHAUSTED,
                    f"response payload exceeds the {self.max_message_bytes:,}-byte limit",
                )
                return True
        except RpcAbort as exc:
            self._send_error(
                peer,
                envelope.request_id,
                exc.status,
                exc.message,
                details=exc.details,
            )
            return True
        except Exception as exc:
            message = str(exc) if self.debug_errors else "RPC handler failed"
            self._send_error(
                peer,
                envelope.request_id,
                Status.INTERNAL,
                message,
            )
            return True

        self._send(peer, response_envelope(envelope.request_id), response_payload)
        return True

    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        """Serve requests until stop() is called."""
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be > 0")
        self._stopping.clear()
        while not self._stopping.is_set():
            self.serve_once(timeout_ms=poll_interval_ms)
