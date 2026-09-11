"""Public npb-rpc exception hierarchy."""

from __future__ import annotations

from typing import Any


class RpcError(Exception):
    """Base class for npb-rpc failures."""


class RpcProtocolError(RpcError, ValueError):
    """The peer sent an invalid or unsupported RPC envelope."""


class RpcTransportError(RpcError, RuntimeError):
    """The underlying transport failed."""


class RpcTimeoutError(RpcTransportError, TimeoutError):
    """A call could not complete before its local deadline."""


class RemoteRpcError(RpcError):
    """The remote server returned a non-OK status."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.details = details or {}
        super().__init__(f"{status}: {message}")


class RpcAbort(RpcError):
    """A deliberate handler abort that is converted into a remote status."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.details = details or {}
        super().__init__(f"{status}: {message}")
