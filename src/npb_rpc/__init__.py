"""Typed RPC for Pydantic and NumPy payloads using NPB."""

from ._discovery import (
    BackendName,
    DiscoveryBackend,
    FilesystemDiscovery,
    ServiceRecord,
    validate_service_name,
)
from ._errors import (
    DiscoveryError,
    RemoteRpcError,
    RpcAbort,
    RpcError,
    RpcProtocolError,
    RpcTimeoutError,
    RpcTransportError,
    ServiceNotFoundError,
)
from ._nng import NngRpcClient, NngRpcServer, portable_ipc
from ._protocol import PROTOCOL_VERSION, RpcContext, Status
from ._service import DiscoveredRpcClient, DiscoveredRpcServer
from ._zmq import ZmqRpcClient, ZmqRpcServer

__all__ = [
    "BackendName",
    "PROTOCOL_VERSION",
    "DiscoveredRpcClient",
    "DiscoveredRpcServer",
    "DiscoveryBackend",
    "DiscoveryError",
    "FilesystemDiscovery",
    "NngRpcClient",
    "NngRpcServer",
    "RemoteRpcError",
    "RpcAbort",
    "RpcContext",
    "RpcError",
    "RpcProtocolError",
    "RpcTimeoutError",
    "RpcTransportError",
    "ServiceNotFoundError",
    "ServiceRecord",
    "Status",
    "ZmqRpcClient",
    "ZmqRpcServer",
    "portable_ipc",
    "validate_service_name",
]
