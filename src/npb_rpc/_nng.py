"""Synchronous unary RPC over NNG REQ/REP sockets."""

from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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

_LENGTH = struct.Struct("!I")
# NNG's REQ/REP protocol adds a routing backtrace outside the application body.
# Leave room for transport/protocol metadata while enforcing the exact payload
# limit after receipt in _unpack_message().
_NNG_PROTOCOL_OVERHEAD = 64 * 1024


def _load_pynng():
    try:
        import pynng
    except ImportError as exc:
        raise ImportError(
            "NNG support is optional. Install it with "
            "`pip install 'npb-rpc[nng]'` or `uv add 'npb-rpc[nng]'`."
        ) from exc
    return pynng


def portable_ipc(name: str, directory: str | os.PathLike[str] | None = None) -> str:
    """Return an NNG IPC endpoint suitable for the current platform.

    NNG maps the short form to a Windows Named Pipe on Windows. On POSIX,
    an absolute path in the temporary directory avoids depending on the two
    processes having the same working directory.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("IPC name must be a non-empty string")
    if name.startswith("ipc://"):
        return name
    if "://" in name:
        raise ValueError(f"expected an IPC name, got transport URL: {name!r}")
    if os.name == "nt":
        return f"ipc://{name}"
    base = directory or os.environ.get("NNG_IPC_DIR") or tempfile.gettempdir()
    path = Path(base).expanduser().resolve() / name
    return f"ipc://{path}"


def portable_tcp(
    name: str,
    host: str = "127.0.0.1",
) -> str:
    """Return a deterministic NNG TCP endpoint derived from a server name.

    The same server name always maps to the same TCP port. This allows
    clients and servers to independently derive the endpoint without a
    registry or hard-coded port table.

    Note:
        Different names can theoretically map to the same port because
        the available TCP port space is finite.
    """
    TCP_PORT_MIN = 15000
    TCP_PORT_MAX = 29999
    if not isinstance(name, str) or not name.strip():
        raise ValueError("TCP name must be a non-empty string")

    if name.startswith("tcp://"):
        return name

    if "://" in name:
        raise ValueError(
            f"expected a TCP name, got transport URL: {name!r}"
        )

    if not isinstance(host, str) or not host.strip():
        raise ValueError("TCP host must be a non-empty string")

    digest = hashlib.blake2s(
        f"npb-rpc:{name}".encode("utf-8"),
        digest_size=4,
    ).digest()

    value = int.from_bytes(digest, byteorder="big")

    port_count = TCP_PORT_MAX - TCP_PORT_MIN + 1
    port = TCP_PORT_MIN + (value % port_count)

    return f"tcp://{host}:{port}"


def _normalize_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")
    if endpoint.startswith("nng+"):
        endpoint = endpoint[4:]
    if "://" not in endpoint:
        return portable_ipc(endpoint)
    return endpoint


def _validate_limit(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _pack_message(
    envelope: Envelope,
    payload: np.ndarray | bytes = b"",
    *,
    max_envelope_bytes: int,
) -> bytes:
    encoded_envelope = encode_envelope(envelope, max_bytes=max_envelope_bytes)
    return _LENGTH.pack(len(encoded_envelope)) + encoded_envelope + bytes(payload)


def _unpack_message(
    message: bytes,
    *,
    max_envelope_bytes: int,
    max_message_bytes: int,
) -> tuple[Envelope, bytes]:
    if len(message) < _LENGTH.size:
        raise RpcProtocolError("NNG RPC message is missing its envelope length")
    (envelope_size,) = _LENGTH.unpack_from(message)
    if envelope_size > max_envelope_bytes:
        raise RpcProtocolError(f"RPC envelope exceeds the {max_envelope_bytes:,}-byte limit")
    payload_offset = _LENGTH.size + envelope_size
    if payload_offset > len(message):
        raise RpcProtocolError("NNG RPC message contains a truncated envelope")
    payload = message[payload_offset:]
    if len(payload) > max_message_bytes:
        raise RpcProtocolError(
            f"RPC payload exceeds the {max_message_bytes:,}-byte limit"
        )
    envelope = decode_envelope(
        message[_LENGTH.size:payload_offset],
        max_bytes=max_envelope_bytes,
    )
    return envelope, payload


class NngRpcClient:
    """A thread-safe synchronous typed RPC client using NNG REQ sockets."""

    def __init__(
        self,
        endpoint: str,
        *,
        default_timeout: float | None = 10.0,
        send_timeout_ms: int = 10_000,
        connect_block: bool = False,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
    ) -> None:
        endpoint = _normalize_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        if default_timeout is not None and default_timeout <= 0:
            raise ValueError("default_timeout must be > 0 or None")
        if send_timeout_ms < 0:
            raise ValueError("send_timeout_ms must be non-negative")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        pynng = _load_pynng()
        self.endpoint = endpoint
        self.default_timeout = default_timeout
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self._pynng = pynng
        self._socket = pynng.Req0(send_timeout=send_timeout_ms)
        self._socket.recv_max_size = (
            _LENGTH.size
            + max_envelope_bytes
            + max_message_bytes
            + _NNG_PROTOCOL_OVERHEAD
        )
        try:
            self._socket.dial(endpoint, block=connect_block)
        except pynng.NNGException as exc:
            self._socket.close()
            raise RpcTransportError(f"failed to connect NNG RPC client: {exc}") from exc
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> NngRpcClient:
        return cls(endpoint, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close()
        self._closed = True

    def __enter__(self) -> NngRpcClient:
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

        with self._lock:
            previous_timeout = self._socket.recv_timeout
            self._socket.recv_timeout = (
                -1
                if effective_timeout is None
                else max(1, math.ceil(effective_timeout * 1000))
            )
            try:
                self._socket.send(message)
                response_message = self._socket.recv()
            except self._pynng.Timeout as exc:
                raise RpcTimeoutError(f"RPC call to {method!r} timed out") from exc
            except self._pynng.NNGException as exc:
                raise RpcTransportError(f"NNG RPC call failed: {exc}") from exc
            finally:
                self._socket.recv_timeout = previous_timeout

        response_meta, response_payload = _unpack_message(
            response_message,
            max_envelope_bytes=self.max_envelope_bytes,
            max_message_bytes=self.max_message_bytes,
        )
        if response_meta.kind != "response":
            raise RpcProtocolError("received an RPC request on a client socket")
        if response_meta.request_id != request_id:
            raise RpcProtocolError("received an NNG RPC response with an unexpected request ID")
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


class NngRpcServer:
    """A synchronous typed RPC server using an NNG REP socket."""

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool = True,
        send_timeout_ms: int = 10_000,
        max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
        max_message_bytes: int = 256 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
        debug_errors: bool = False,
    ) -> None:
        endpoint = _normalize_endpoint(endpoint)
        _validate_limit("max_envelope_bytes", max_envelope_bytes)
        _validate_limit("max_message_bytes", max_message_bytes)
        if send_timeout_ms < 0:
            raise ValueError("send_timeout_ms must be non-negative")
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        pynng = _load_pynng()
        self.endpoint = endpoint
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.debug_errors = debug_errors
        self._pynng = pynng
        self._socket = pynng.Rep0(send_timeout=send_timeout_ms)
        self._socket.recv_max_size = (
            _LENGTH.size
            + max_envelope_bytes
            + max_message_bytes
            + _NNG_PROTOCOL_OVERHEAD
        )
        try:
            if bind:
                self._socket.listen(endpoint)
            else:
                self._socket.dial(endpoint, block=False)
        except pynng.NNGException as exc:
            self._socket.close()
            raise RpcTransportError(f"failed to start NNG RPC server: {exc}") from exc
        self._methods: dict[str, _Method] = {}
        self._stopping = threading.Event()
        self._closed = False

    @classmethod
    def bind(cls, endpoint: str, **kwargs: Any) -> NngRpcServer:
        return cls(endpoint, bind=True, **kwargs)

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> NngRpcServer:
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
        self._stopping.set()

    def __enter__(self) -> NngRpcServer:
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

    def _send(self, envelope: Envelope, payload: np.ndarray | bytes = b"") -> None:
        message = _pack_message(
            envelope,
            payload,
            max_envelope_bytes=self.max_envelope_bytes,
        )
        try:
            self._socket.send(message)
        except self._pynng.Timeout as exc:
            raise RpcTimeoutError("timed out while sending NNG RPC response") from exc
        except self._pynng.NNGException as exc:
            raise RpcTransportError(f"failed to send NNG RPC response: {exc}") from exc

    def _send_error(
        self,
        request_id: str,
        status: Status | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._send(
            response_envelope(
                request_id,
                status=status,
                error_message=message,
                error_details=details,
            )
        )

    def serve_once(self, *, timeout_ms: int | None = None) -> bool:
        if self._closed:
            raise RpcTransportError("RPC server is closed")
        if timeout_ms is not None and timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative or None")
        previous_timeout = self._socket.recv_timeout
        self._socket.recv_timeout = -1 if timeout_ms is None else timeout_ms
        try:
            message = self._socket.recv()
        except self._pynng.Timeout:
            return False
        except self._pynng.NNGException as exc:
            if self._stopping.is_set():
                return False
            raise RpcTransportError(f"failed to receive NNG RPC request: {exc}") from exc
        finally:
            self._socket.recv_timeout = previous_timeout

        try:
            envelope, raw_payload = _unpack_message(
                message,
                max_envelope_bytes=self.max_envelope_bytes,
                max_message_bytes=self.max_message_bytes,
            )
        except RpcProtocolError as exc:
            self._send_error(uuid4().hex, Status.INVALID_ARGUMENT, str(exc))
            return True
        if envelope.kind != "request":
            self._send_error(
                envelope.request_id,
                Status.INVALID_ARGUMENT,
                "server expected an RPC request envelope",
            )
            return True

        method_name = envelope.method or ""
        method = self._methods.get(method_name)
        if method is None:
            self._send_error(
                envelope.request_id,
                Status.UNIMPLEMENTED,
                f"unknown RPC method: {method_name}",
            )
            return True
        if envelope.deadline_unix_ns is not None and time.time_ns() >= envelope.deadline_unix_ns:
            self._send_error(
                envelope.request_id,
                Status.DEADLINE_EXCEEDED,
                "RPC deadline expired before dispatch",
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
                    envelope.request_id,
                    Status.RESOURCE_EXHAUSTED,
                    f"response payload exceeds the {self.max_message_bytes:,}-byte limit",
                )
                return True
        except RpcAbort as exc:
            self._send_error(
                envelope.request_id,
                exc.status,
                exc.message,
                details=exc.details,
            )
            return True
        except Exception as exc:
            message = str(exc) if self.debug_errors else "RPC handler failed"
            self._send_error(envelope.request_id, Status.INTERNAL, message)
            return True

        self._send(response_envelope(envelope.request_id), response_payload)
        return True

    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be > 0")
        self._stopping.clear()
        while not self._stopping.is_set():
            self.serve_once(timeout_ms=poll_interval_ms)
