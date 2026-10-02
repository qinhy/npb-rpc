"""Transport-independent interfaces for synchronous typed unary RPC.

Concrete transports own their I/O, synchronization, and payload lifetimes.
The base classes provide lifecycle helpers and typed method registration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self, TypeVar, cast

from pydantic import BaseModel

from ._protocol import RpcContext

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
Handler = Callable[[BaseModel, RpcContext], BaseModel]


class _RpcResource(ABC):
    endpoint: str
    _closed: bool

    @property
    def closed(self) -> bool:
        """Whether this resource has been closed."""
        return self._closed

    @abstractmethod
    def close(self) -> None:
        """Release transport resources; repeated calls must be harmless."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class RpcClient(_RpcResource):
    """Common interface implemented by NNG, ZeroMQ, and iceoryx2 clients.

    Constructor options remain transport-specific. ``call`` returns an owned
    response; borrowed shared-memory responses are a transport extension.
    """

    @classmethod
    @abstractmethod
    def connect(cls, endpoint: str, **kwargs: Any) -> Self:
        """Create a client using transport-specific connection options."""

    @abstractmethod
    def call(
        self,
        method: str,
        request: RequestT,
        response_type: type[ResponseT],
        *,
        timeout: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ResponseT:
        """Call a typed unary method; None selects the client's default timeout."""


@dataclass(frozen=True, slots=True)
class _Method:
    request_type: type[BaseModel]
    response_type: type[BaseModel]
    handler: Handler


class RpcServer(_RpcResource):
    """Common server interface and typed method registry.

    Subclasses must call ``super().__init__()`` to create their own registry,
    initialize ``endpoint`` and ``_closed``, and implement transport operations.
    """

    def __init__(self) -> None:
        self._methods: dict[str, _Method] = {}

    @classmethod
    @abstractmethod
    def bind(cls, endpoint: str, **kwargs: Any) -> Self:
        """Create a server using transport-specific binding options."""

    @staticmethod
    @abstractmethod
    def has_protocol(protocol: Literal["tcp", "ipc"]) -> bool:
        """Whether the transport supports the requested protocol."""

    @abstractmethod
    def stop(self) -> None:
        """Request that the serving loop stop."""

    @abstractmethod
    def serve_once(self, *, timeout_ms: int | None = None) -> bool:
        """Serve at most one request; return False if no request was served."""

    @abstractmethod
    def serve_forever(self, *, poll_interval_ms: int = 100) -> None:
        """Serve requests until stop() is called."""

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._methods))

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
        self._methods[name] = _Method(
            request_type,
            response_type,
            cast(Handler, handler),
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
