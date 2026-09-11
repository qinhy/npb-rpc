"""Small JSON control envelopes used beside NPB payload frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from ._errors import RpcAbort, RpcProtocolError

PROTOCOL_VERSION = 1
DEFAULT_MAX_ENVELOPE_BYTES = 64 * 1024


class Status(StrEnum):
    """Portable RPC statuses, following the familiar gRPC status vocabulary."""

    OK = "ok"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    INVALID_ARGUMENT = "invalid_argument"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    FAILED_PRECONDITION = "failed_precondition"
    ABORTED = "aborted"
    OUT_OF_RANGE = "out_of_range"
    UNIMPLEMENTED = "unimplemented"
    INTERNAL = "internal"
    UNAVAILABLE = "unavailable"
    DATA_LOSS = "data_loss"
    UNAUTHENTICATED = "unauthenticated"


@dataclass(frozen=True, slots=True)
class RpcContext:
    """Metadata made available to a server method handler."""

    request_id: str
    method: str
    deadline_unix_ns: int | None
    metadata: dict[str, str]
    peer: bytes

    def abort(
        self,
        status: Status | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Stop the handler and return a structured error to the caller."""
        status_value = status.value if isinstance(status, Status) else str(status)
        try:
            parsed_status = Status(status_value)
        except ValueError as exc:
            raise ValueError(f"unknown RPC status: {status_value!r}") from exc
        if parsed_status == Status.OK:
            raise ValueError("cannot abort with status 'ok'")
        raise RpcAbort(parsed_status.value, message, details=details)


@dataclass(frozen=True, slots=True)
class Envelope:
    kind: Literal["request", "response"]
    request_id: str
    method: str | None = None
    deadline_unix_ns: int | None = None
    metadata: dict[str, str] | None = None
    status: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None


def request_envelope(
    request_id: str,
    method: str,
    *,
    deadline_unix_ns: int | None,
    metadata: dict[str, str] | None,
) -> Envelope:
    return Envelope(
        kind="request",
        request_id=request_id,
        method=method,
        deadline_unix_ns=deadline_unix_ns,
        metadata=dict(metadata or {}),
    )


def response_envelope(
    request_id: str,
    *,
    status: Status | str = Status.OK,
    error_message: str | None = None,
    error_details: dict[str, Any] | None = None,
) -> Envelope:
    status_value = status.value if isinstance(status, Status) else str(status)
    try:
        parsed_status = Status(status_value)
    except ValueError as exc:
        raise ValueError(f"unknown RPC status: {status_value!r}") from exc
    if parsed_status == Status.OK and error_message is not None:
        raise ValueError("successful RPC response cannot contain an error")
    if parsed_status != Status.OK and error_message is None:
        raise ValueError("failed RPC response must contain an error message")
    return Envelope(
        kind="response",
        request_id=request_id,
        status=parsed_status.value,
        error_message=error_message,
        error_details=error_details,
    )


def encode_envelope(
    envelope: Envelope,
    *,
    max_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
) -> bytes:
    value: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "kind": envelope.kind,
        "id": envelope.request_id,
    }
    if envelope.kind == "request":
        value["method"] = envelope.method
        if envelope.deadline_unix_ns is not None:
            value["deadline_unix_ns"] = envelope.deadline_unix_ns
        if envelope.metadata:
            value["metadata"] = envelope.metadata
    else:
        value["status"] = envelope.status
        if envelope.error_message is not None:
            value["error"] = {
                "message": envelope.error_message,
                "details": envelope.error_details or {},
            }

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RpcProtocolError("RPC envelope is not JSON serializable") from exc

    if len(encoded) > max_bytes:
        raise RpcProtocolError(f"RPC envelope exceeds the {max_bytes:,}-byte limit")
    return encoded


def decode_envelope(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
) -> Envelope:
    if len(payload) > max_bytes:
        raise RpcProtocolError(f"RPC envelope exceeds the {max_bytes:,}-byte limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RpcProtocolError("invalid RPC envelope JSON") from exc
    if not isinstance(value, dict):
        raise RpcProtocolError("RPC envelope must be an object")
    if value.get("v") != PROTOCOL_VERSION:
        raise RpcProtocolError(f"unsupported RPC protocol version: {value.get('v')!r}")

    kind = value.get("kind")
    if kind not in {"request", "response"}:
        raise RpcProtocolError("RPC envelope has an invalid kind")

    request_id = value.get("id")
    if not isinstance(request_id, str):
        raise RpcProtocolError("RPC envelope is missing its request ID")
    try:
        UUID(hex=request_id)
    except ValueError as exc:
        raise RpcProtocolError("RPC request ID is not a UUID") from exc

    if kind == "request":
        method = value.get("method")
        if not isinstance(method, str) or not method.strip():
            raise RpcProtocolError("RPC request is missing its method")
        deadline = value.get("deadline_unix_ns")
        if deadline is not None and (not isinstance(deadline, int) or isinstance(deadline, bool)):
            raise RpcProtocolError("RPC deadline must be an integer")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in raw_metadata.items()
        ):
            raise RpcProtocolError("RPC metadata must map strings to strings")
        return Envelope(
            kind="request",
            request_id=request_id,
            method=method,
            deadline_unix_ns=deadline,
            metadata=dict(raw_metadata),
        )

    status = value.get("status")
    if not isinstance(status, str) or not status:
        raise RpcProtocolError("RPC response is missing its status")
    try:
        parsed_status = Status(status)
    except ValueError as exc:
        raise RpcProtocolError(f"unknown RPC response status: {status!r}") from exc
    raw_error = value.get("error")
    error_message = None
    error_details = None
    if raw_error is not None:
        if not isinstance(raw_error, dict):
            raise RpcProtocolError("RPC response error must be an object")
        error_message = raw_error.get("message")
        error_details = raw_error.get("details", {})
        if not isinstance(error_message, str) or not isinstance(error_details, dict):
            raise RpcProtocolError("RPC response contains an invalid error")
    if parsed_status == Status.OK and raw_error is not None:
        raise RpcProtocolError("successful RPC response cannot contain an error")
    if parsed_status != Status.OK and error_message is None:
        raise RpcProtocolError("failed RPC response must contain an error")
    return Envelope(
        kind="response",
        request_id=request_id,
        status=parsed_status.value,
        error_message=error_message,
        error_details=error_details,
    )
