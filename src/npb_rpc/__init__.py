"""Typed RPC for Pydantic and NumPy payloads using NPB."""

from ._discovery import (
    BackendName,
    DiscoveryBackend,
    FilesystemDiscovery,
    RedisDiscovery,
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
from ._iceoryx2 import (
    Iceoryx2BorrowedResponse,
    Iceoryx2RpcClient,
    Iceoryx2RpcServer,
    WaitStrategy,
)
from ._nng import NngRpcClient, NngRpcServer, portable_ipc, portable_tcp
from ._protocol import PROTOCOL_VERSION, RpcContext, Status
from ._service import DiscoveredRpcClient, DiscoveredRpcServer
from ._zmq import ZmqRpcClient, ZmqRpcServer
from .utils import RpcSpec, api, api_methods
