"""Synchronous unary RPC over iceoryx2 shared-memory request/response."""

from __future__ import annotations

import ctypes
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

import numpy as np
from npb import BlobStore, NPBError, decode, encode, encoded_size
from pydantic import BaseModel, ValidationError

from ._errors import (
    RemoteRpcError,
    RpcAbort,
    RpcProtocolError,
    RpcTimeoutError,
    RpcTransportError,
)
from ._framing import _LENGTH, _validate_limit
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
WaitStrategy = Literal["sleep", "yield", "spin", "hybrid"]

_WAIT_STRATEGIES = frozenset({"sleep", "yield", "spin", "hybrid"})


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


def _validate_wait_strategy(value: str) -> WaitStrategy:
    if value not in _WAIT_STRATEGIES:
        choices = ", ".join(sorted(_WAIT_STRATEGIES))
        raise ValueError(f"wait_strategy must be one of: {choices}")
    return value  # type: ignore[return-value]


def _validate_spin_duration(value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("spin_duration must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("spin_duration must be finite and >= 0")


def _idle_wait(
    *,
    strategy: WaitStrategy,
    event: threading.Event,
    poll_interval: float,
    spin_duration: float,
    spin_started: float,
    deadline: float | None = None,
) -> float:
    """Apply one idle step and return the next spin-cycle start time.

    ``sleep`` preserves the original low-CPU behavior. ``yield`` cooperatively
    yields the current thread with ``time.sleep(0)``. ``spin`` returns
    immediately for the lowest polling latency and highest CPU usage.
    ``hybrid`` spins for ``spin_duration`` then yields once before beginning a
    new spin window.  The latter avoids the millisecond-scale timer wait that
    dominated the original iceoryx2 RPC latency while still giving peer
    threads/processes regular scheduling opportunities.
    """
    if strategy == "spin":
        return spin_started

    if strategy == "yield":
        time.sleep(0)
        return time.perf_counter()

    if strategy == "hybrid":
        now = time.perf_counter()
        if now - spin_started < spin_duration:
            return spin_started
        time.sleep(0)
        return time.perf_counter()

    delay = poll_interval
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return spin_started
        delay = min(delay, remaining)
    event.wait(delay)
    return time.perf_counter()


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


class _PayloadTooLarge(ValueError):
    def __init__(self, nbytes: int, limit: int) -> None:
        self.nbytes = nbytes
        self.limit = limit
        super().__init__(f"payload exceeds the {limit:,}-byte limit: {nbytes:,} bytes")


def _slice_as_numpy(payload: Any) -> np.ndarray:
    """Return a zero-copy 1-D uint8 NumPy view over an iceoryx2 Slice."""
    length = payload.len()
    ptr = ctypes.cast(payload.as_ptr(), ctypes.POINTER(ctypes.c_uint8))
    return np.ctypeslib.as_array(ptr, shape=(length,))


def _payload_view(payload: np.ndarray | bytes) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        if payload.dtype != np.uint8:
            raise TypeError("RPC payload ndarray must have dtype uint8")
        if not payload.flags.c_contiguous:
            payload = np.ascontiguousarray(payload)
        return payload.reshape(-1)
    return np.frombuffer(payload, dtype=np.uint8)


def _prepare_loan(
    port: Any,
    envelope: Envelope,
    payload_size: int,
    *,
    max_envelope_bytes: int,
) -> tuple[Any, np.ndarray]:
    encoded_envelope = encode_envelope(envelope, max_bytes=max_envelope_bytes)
    payload_offset = _LENGTH.size + len(encoded_envelope)
    loan = port.loan_slice_uninit(payload_offset + payload_size)
    shared = _slice_as_numpy(loan.payload())
    _LENGTH.pack_into(shared, 0, len(encoded_envelope))
    shared[_LENGTH.size:payload_offset] = np.frombuffer(encoded_envelope, dtype=np.uint8)
    return loan, shared[payload_offset:]


def _send_payload(
    port: Any,
    envelope: Envelope,
    payload: np.ndarray | bytes = b"",
    *,
    max_envelope_bytes: int,
    max_message_bytes: int,
):
    view = _payload_view(payload)
    if view.nbytes > max_message_bytes:
        raise _PayloadTooLarge(view.nbytes, max_message_bytes)
    loan, output = _prepare_loan(
        port,
        envelope,
        view.nbytes,
        max_envelope_bytes=max_envelope_bytes,
    )
    if view.nbytes:
        output[:] = view
    return loan.assume_init().send()


def _send_model(
    port: Any,
    envelope: Envelope,
    model: BaseModel,
    *,
    max_envelope_bytes: int,
    max_message_bytes: int,
    blob_store: BlobStore | None,
    externalize_min_bytes: int | None,
):
    # encoded_size() currently sizes inline NPB encoding. When externalization
    # is requested, encode first and copy the (normally tiny) reference frame
    # into SHM. The common inline path writes NPB directly into the loan.
    if externalize_min_bytes is not None:
        payload = encode(
            model,
            blob_store=blob_store,
            externalize_min_bytes=externalize_min_bytes,
        )
        return _send_payload(
            port,
            envelope,
            payload,
            max_envelope_bytes=max_envelope_bytes,
            max_message_bytes=max_message_bytes,
        )

    payload_size = encoded_size(
        model=model,
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
    )
    if payload_size > max_message_bytes:
        raise _PayloadTooLarge(payload_size, max_message_bytes)

    loan, output = _prepare_loan(
        port,
        envelope,
        payload_size,
        max_envelope_bytes=max_envelope_bytes,
    )
    encoded = encode(model, out=output, blob_store=blob_store)
    if encoded.nbytes != payload_size or not np.shares_memory(encoded, output):
        raise RpcProtocolError("NPB did not encode directly into the iceoryx2 loan")
    return loan.assume_init().send()


def _view_message(
    sample: Any,
    *,
    max_envelope_bytes: int,
) -> tuple[Envelope, np.ndarray]:
    """Parse an RPC sample while leaving its NPB body as a zero-copy SHM view."""
    shared = _slice_as_numpy(sample.payload())
    if shared.nbytes < _LENGTH.size:
        raise RpcProtocolError("RPC message is missing its envelope length")
    (envelope_size,) = _LENGTH.unpack_from(shared, 0)
    if envelope_size > max_envelope_bytes:
        raise RpcProtocolError(f"RPC envelope exceeds the {max_envelope_bytes:,}-byte limit")
    payload_offset = _LENGTH.size + envelope_size
    if payload_offset > shared.nbytes:
        raise RpcProtocolError("RPC message contains a truncated envelope")
    # Control envelopes are intentionally small JSON, so this is a tiny copy.
    envelope = decode_envelope(
        shared[_LENGTH.size:payload_offset].tobytes(),
        max_bytes=max_envelope_bytes,
    )
    return envelope, shared[payload_offset:]


class Iceoryx2BorrowedResponse(Generic[ResponseT]):
    """Context-managed RPC result whose ndarray leaves borrow iceoryx2 SHM.

    The value is valid only until :meth:`close` / context-manager exit. While a
    borrowed response is open, the originating client is intentionally locked
    so its single-active-request iceoryx2 port cannot be reused prematurely.
    """

    __slots__ = ("value", "_sample", "_pending", "_lock", "_closed")

    def __init__(
        self,
        value: ResponseT,
        *,
        sample: Any,
        pending: Any,
        lock: threading.Lock,
    ) -> None:
        self.value = value
        self._sample = sample
        self._pending = pending
        self._lock = lock
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._sample.delete()
        finally:
            try:
                self._pending.delete()
            finally:
                self._closed = True
                self._lock.release()

    def __enter__(self) -> ResponseT:
        if self._closed:
            raise RuntimeError("borrowed response is closed")
        return self.value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class Iceoryx2RpcClient:
    """A thread-safe synchronous typed RPC client using iceoryx2 shared memory."""

    def __init__(
        self,
        endpoint: str,
        *,
        default_timeout: float | None = 10.0,
        poll_interval: float = 0.001,
        wait_strategy: WaitStrategy = "sleep",
        spin_duration: float = 50e-6,
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
        wait_strategy = _validate_wait_strategy(wait_strategy)
        _validate_spin_duration(spin_duration)
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        self.endpoint = endpoint
        self.default_timeout = default_timeout
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.poll_interval = poll_interval
        self.wait_strategy = wait_strategy
        self.spin_duration = float(spin_duration)
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

    def _start_call(
        self,
        method: str,
        request: RequestT,
        *,
        timeout: float | None,
        metadata: dict[str, str] | None,
    ) -> tuple[str, Any, Any]:
        if self._closed:
            raise RpcTransportError("RPC client is closed")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(request, BaseModel):
            raise TypeError("request must be a Pydantic model instance")

        effective_timeout = self.default_timeout if timeout is None else timeout
        _validate_timeout("timeout", effective_timeout)
        expires = None if effective_timeout is None else time.monotonic() + effective_timeout
        request_id = uuid4().hex
        deadline_ns = (
            None
            if effective_timeout is None
            else time.time_ns() + int(effective_timeout * 1_000_000_000)
        )
        envelope = request_envelope(
            request_id,
            method,
            deadline_unix_ns=deadline_ns,
            metadata=metadata,
        )

        acquired = self._lock.acquire(
            timeout=-1 if expires is None else max(0, expires - time.monotonic())
        )
        if not acquired:
            raise RpcTimeoutError(f"RPC call to {method!r} timed out")

        pending = None
        spin_started = time.perf_counter()
        try:
            while True:
                if self._closing.is_set():
                    raise RpcTransportError("RPC client is closed")
                if expires is not None and time.monotonic() >= expires:
                    raise RpcTimeoutError(f"RPC call to {method!r} timed out")
                if pending is None:
                    try:
                        pending = _send_model(
                            self._port,
                            envelope,
                            request,
                            max_envelope_bytes=self.max_envelope_bytes,
                            max_message_bytes=self.max_message_bytes,
                            blob_store=self.blob_store,
                            externalize_min_bytes=self.externalize_min_bytes,
                        )
                    except _PayloadTooLarge as exc:
                        raise RpcProtocolError(
                            f"request payload exceeds the {self.max_message_bytes:,}-byte limit"
                        ) from exc
                    if pending.number_of_server_connections == 0:
                        # Nothing received the request, so waiting and retrying is safe.
                        pending.delete()
                        pending = None
                if pending is not None:
                    response = pending.receive()
                    if response is not None:
                        return request_id, pending, response
                spin_started = _idle_wait(
                    strategy=self.wait_strategy,
                    event=self._closing,
                    poll_interval=self.poll_interval,
                    spin_duration=self.spin_duration,
                    spin_started=spin_started,
                    deadline=expires,
                )
        except (RpcTimeoutError, RpcTransportError, RpcProtocolError):
            if pending is not None:
                pending.delete()
            self._lock.release()
            raise
        except Exception as exc:
            if pending is not None:
                pending.delete()
            self._lock.release()
            raise RpcTransportError(f"iceoryx2 RPC call failed: {exc}") from exc

    def _validate_response(
        self,
        request_id: str,
        sample: Any,
    ) -> np.ndarray:
        response_meta, response_payload = _view_message(
            sample,
            max_envelope_bytes=self.max_envelope_bytes,
        )
        if response_payload.nbytes > self.max_message_bytes:
            raise RpcProtocolError(
                f"response payload exceeds the {self.max_message_bytes:,}-byte limit"
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
        if response_payload.nbytes == 0:
            raise RpcProtocolError("successful RPC response has no NPB payload")
        return response_payload

    def call(
        self,
        method: str,
        request: RequestT,
        response_type: type[ResponseT],
        *,
        timeout: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ResponseT:
        """Call one method and return an owned response.

        The transport parses directly from iceoryx2 shared memory, then makes
        exactly one payload copy before releasing the response loan. ndarray
        leaves in the returned model therefore remain valid independently of
        the iceoryx2 sample lifetime. Use :meth:`call_borrowed` to avoid this
        final copy when the caller can keep work inside a context manager.
        """
        if not isinstance(response_type, type) or not issubclass(response_type, BaseModel):
            raise TypeError("response_type must be a Pydantic model class")

        request_id, pending, response = self._start_call(
            method, request, timeout=timeout, metadata=metadata
        )
        try:
            borrowed_payload = self._validate_response(request_id, response)
            # One deliberate large copy: returned ndarray leaves must outlive
            # the iceoryx2 response sample. The old implementation copied the
            # whole message plus sliced bytes before decode.
            owned_payload = borrowed_payload.copy()
        finally:
            try:
                response.delete()
            finally:
                try:
                    pending.delete()
                finally:
                    self._lock.release()

        try:
            return decode(response_type, owned_payload, blob_store=self.blob_store)
        except (NPBError, ValidationError, TypeError, ValueError) as exc:
            raise RpcProtocolError(f"invalid response payload: {exc}") from exc

    def call_borrowed(
        self,
        method: str,
        request: RequestT,
        response_type: type[ResponseT],
        *,
        timeout: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Iceoryx2BorrowedResponse[ResponseT]:
        """Call one method and borrow the response directly from shared memory.

        Use only as a context manager. ndarray leaves in the returned value are
        zero-copy views into iceoryx2 SHM and become invalid when the borrowed
        response is closed. The client remains locked until that point.
        """
        if not isinstance(response_type, type) or not issubclass(response_type, BaseModel):
            raise TypeError("response_type must be a Pydantic model class")

        request_id, pending, response = self._start_call(
            method, request, timeout=timeout, metadata=metadata
        )
        try:
            borrowed_payload = self._validate_response(request_id, response)
            value = decode(response_type, borrowed_payload, blob_store=self.blob_store)
        except (NPBError, ValidationError, TypeError, ValueError) as exc:
            try:
                response.delete()
            finally:
                try:
                    pending.delete()
                finally:
                    self._lock.release()
            if isinstance(exc, (RpcProtocolError, RemoteRpcError)):
                raise
            raise RpcProtocolError(f"invalid response payload: {exc}") from exc
        except Exception:
            try:
                response.delete()
            finally:
                try:
                    pending.delete()
                finally:
                    self._lock.release()
            raise

        return Iceoryx2BorrowedResponse(
            value,
            sample=response,
            pending=pending,
            lock=self._lock,
        )


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
        wait_strategy: WaitStrategy = "sleep",
        spin_duration: float = 50e-6,
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
        wait_strategy = _validate_wait_strategy(wait_strategy)
        _validate_spin_duration(spin_duration)
        if externalize_min_bytes is not None and blob_store is None:
            raise ValueError("blob_store is required when externalize_min_bytes is set")

        self.endpoint = endpoint
        self.max_envelope_bytes = max_envelope_bytes
        self.max_message_bytes = max_message_bytes
        self.blob_store = blob_store
        self.externalize_min_bytes = externalize_min_bytes
        self.debug_errors = debug_errors
        self.poll_interval = poll_interval
        self.wait_strategy = wait_strategy
        self.spin_duration = float(spin_duration)
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

    def _send_empty(self, active: Any, envelope: Envelope) -> None:
        try:
            if active.is_connected:
                _send_payload(
                    active,
                    envelope,
                    max_envelope_bytes=self.max_envelope_bytes,
                    max_message_bytes=self.max_message_bytes,
                )
        except Exception as exc:
            # A timed-out caller may disconnect while its handler is still running.
            if active.is_connected:
                raise RpcTransportError(f"failed to send iceoryx2 RPC response: {exc}") from exc

    def _send_model_response(
        self,
        active: Any,
        envelope: Envelope,
        response: BaseModel,
    ) -> None:
        try:
            if active.is_connected:
                _send_model(
                    active,
                    envelope,
                    response,
                    max_envelope_bytes=self.max_envelope_bytes,
                    max_message_bytes=self.max_message_bytes,
                    blob_store=self.blob_store,
                    externalize_min_bytes=self.externalize_min_bytes,
                )
        except _PayloadTooLarge:
            raise
        except Exception as exc:
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
        self._send_empty(
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
            spin_started = time.perf_counter()
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
                if expires is not None and time.monotonic() >= expires:
                    return False
                spin_started = _idle_wait(
                    strategy=self.wait_strategy,
                    event=self._stopping,
                    poll_interval=self.poll_interval,
                    spin_duration=self.spin_duration,
                    spin_started=spin_started,
                    deadline=expires,
                )
            return False

    def _dispatch(self, active: Any) -> None:
        try:
            envelope, request_binary = _view_message(
                active,
                max_envelope_bytes=self.max_envelope_bytes,
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

        if request_binary.nbytes > self.max_message_bytes:
            self._send_error(
                active,
                envelope.request_id,
                Status.RESOURCE_EXHAUSTED,
                f"request payload exceeds the {self.max_message_bytes:,}-byte limit",
            )
            return

        # NPB ndarray leaves are zero-copy views into the active request sample.
        # They are valid for the duration of this dispatch/handler call only.
        try:
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
            self._send_model_response(
                active,
                response_envelope(envelope.request_id),
                response,
            )
        except _PayloadTooLarge:
            self._send_error(
                active,
                envelope.request_id,
                Status.RESOURCE_EXHAUSTED,
                f"response payload exceeds the {self.max_message_bytes:,}-byte limit",
            )
        except RpcAbort as exc:
            self._send_error(
                active,
                envelope.request_id,
                exc.status,
                exc.message,
                details=exc.details,
            )
        except RpcTransportError:
            raise
        except Exception as exc:
            message = str(exc) if self.debug_errors else "RPC handler failed"
            self._send_error(active, envelope.request_id, Status.INTERNAL, message)

    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be > 0")
        if self._closed:
            raise RpcTransportError("RPC server is closed")
        self._stopping.clear()
        while not self._stopping.is_set():
            self.serve_once(timeout_ms=poll_interval_ms)
