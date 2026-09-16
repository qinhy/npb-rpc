# npb-rpc

`npb-rpc` adds typed unary RPC and pluggable service discovery to
[NPB](https://github.com/qinhy/npb), preserving NPB's Pydantic schema
validation, zero-copy NumPy decode, preallocated-buffer support, and optional
external blob stores.

ZeroMQ, NNG, and iceoryx2 transports are optional:

```bash
pip install "npb-rpc[zmq]"
pip install "npb-rpc[nng]"
pip install "npb-rpc[iceoryx2]"
```

Or, with uv:

```bash
uv add "npb-rpc[zmq]"
uv add "npb-rpc[nng]"
uv add "npb-rpc[iceoryx2]"
```

Each call is a two-frame ZeroMQ message:

```text
DEALER client                         ROUTER server
     |                                     |
     |  compact JSON RPC envelope          |
     |  NPB request payload                |
     | ----------------------------------> |
     |                                     |
     |  compact JSON RPC envelope          |
     |  NPB response payload               |
     | <---------------------------------- |
```

The envelope carries method, request ID, deadline, metadata, and status. The
NPB frame remains a transport-independent typed body.

## Example

Shared models:

```python
import numpy as np
from npb import BinaryModel, binary_schema


@binary_schema("example.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("example.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float
```

Server:

```python
from npb_rpc import RpcContext, ZmqRpcServer

server = ZmqRpcServer.bind("tcp://127.0.0.1:5555")


@server.method("array.sum", request=SumRequest, response=SumResponse)
def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
    return SumResponse(total=float(request.values.sum()))


server.serve_forever()
```

Client:

```python
import numpy as np
from npb_rpc import ZmqRpcClient

with ZmqRpcClient.connect("tcp://127.0.0.1:5555") as client:
    result = client.call(
        "array.sum",
        SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
        SumResponse,
        timeout=5.0,
        metadata={"trace-id": "demo-1"},
    )

print(result.total)
```

Runnable TCP and IPC versions of this example are included. Start each server
in one terminal and its matching client in another:

```bash
# ZeroMQ TCP
uv run python examples/zmq_sum.py server --transport tcp
uv run python examples/zmq_sum.py client --transport tcp

# ZeroMQ IPC (systems whose libzmq provides Unix-domain sockets)
uv run python examples/zmq_sum.py server --transport ipc
uv run python examples/zmq_sum.py client --transport ipc

# NNG IPC (Windows, macOS, and Linux)
uv run python examples/nng_sum.py server --transport ipc
uv run python examples/nng_sum.py client --transport ipc

# NNG TCP
uv run python examples/nng_sum.py server --transport tcp
uv run python examples/nng_sum.py client --transport tcp
```

The ZeroMQ IPC example creates `npb-rpc-zmq-sum.sock` in the system temporary
directory. Change `IPC_ENDPOINT` in `examples/zmq_sum.py` if the socket should
live elsewhere.
Native Windows builds do not support ZeroMQ's `ipc://` transport; use the
loopback TCP example on Windows, run the ZeroMQ IPC example under WSL, or use
the portable NNG IPC example. The ZeroMQ example checks `zmq.has("ipc")` and
reports this limitation before binding or connecting.

For native Windows IPC, use the NNG examples. `portable_ipc("npb-rpc-sum")`
selects `ipc://npb-rpc-sum` on Windows, where NNG maps it to a Named Pipe, and
an absolute temporary-directory socket path on macOS and Linux:

```python
from npb_rpc import NngRpcClient, NngRpcServer, portable_ipc

endpoint = portable_ipc("npb-rpc-sum")
server = NngRpcServer.bind(endpoint)
client = NngRpcClient.connect(endpoint)
```

`NngRpcClient` and `NngRpcServer` have the same typed `call()` and `method()`
interfaces shown above. NNG uses a length-prefixed envelope and payload in one
REQ/REP message; it is not wire-compatible with the ZeroMQ backend, so both
peers must use NNG.

To measure raw NNG IPC request/reply latency on the current platform:

```bash
uv run python examples/benchmark_nng_ipc.py --count 10000 --size 64
```

## iceoryx2 shared-memory backend

Install `npb-rpc[iceoryx2]` to use the native iceoryx2 request/response transport
on one machine. It uses the same typed `method()`, `register()`, and `call()`
interfaces as the other backends:

```python
from npb_rpc import Iceoryx2RpcClient, Iceoryx2RpcServer

server = Iceoryx2RpcServer.bind("iceoryx2://array-sum")
# Register handlers, then run server.serve_forever() in the server process.
client = Iceoryx2RpcClient.connect("iceoryx2://array-sum")
```

A bare name such as `array-sum` is also accepted. Names identify local
shared-memory services, not TCP addresses or filesystem socket paths. Both
processes must use compatible iceoryx2 configuration and have access to the same
shared-memory resources. No broker is needed. The optional dependency targets
iceoryx2 0.9.3; consult its [Python package](https://pypi.org/project/iceoryx2/)
for available platform wheels.

Run the complete example in two terminals:

```bash
uv run --extra iceoryx2 python examples/iceoryx2_sum.py server
uv run --extra iceoryx2 python examples/iceoryx2_sum.py client
```

Each endpoint allows one server and up to 32 clients. Give each server instance
its own endpoint when using discovery. A client serializes its calls; separate
clients can call concurrently, while handlers execute serially. A client may
start before its server and waits up to the call timeout. Once a request reaches
a server it is never automatically retried. Timing out releases the client's
pending response but does not cancel a running handler.

`poll_interval` (seconds, default `0.001`) controls receive polling on both the
client and server. `default_timeout`, `max_envelope_bytes`, `max_message_bytes`,
`blob_store`, and `externalize_min_bytes` work as with the other transports.
Use context managers or `close()` to release shared-memory resources; `stop()`
wakes a waiting server loop.

The transport uses shared memory, but this adapter copies encoded bytes into
iceoryx2 loans and copies received bytes into Python-owned buffers before NPB
decode. Returned arrays therefore remain valid after the next call or after
closing the client. This is not end-to-end zero-copy RPC.

## Service discovery

Discovery is a control plane only. A server publishes its endpoint and refreshes
a heartbeat; a client resolves a logical service name and then calls the chosen
instance directly:

```text
server instance ---> FilesystemDiscovery <--- client
       ^                                      |
       +------------ direct RPC --------------+
```

Wrap any concrete server with `DiscoveredRpcServer`:

```python
from npb_rpc import (
    DiscoveredRpcServer,
    FilesystemDiscovery,
    RpcContext,
    ZmqRpcServer,
)

discovery = FilesystemDiscovery()
server = DiscoveredRpcServer(
    "array-service",
    ZmqRpcServer.bind("tcp://0.0.0.0:7001"),
    discovery,
    advertise_endpoint="tcp://192.168.1.10:7001",
)


@server.method("array.sum", request=SumRequest, response=SumResponse)
def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
    return SumResponse(total=float(request.values.sum()))


server.serve_forever()
```

The client only needs the logical service name. Discovery records include the
endpoint and the `zmq`, `nng`, or `iceoryx2` backend, so all backends can share
one registry:

```python
from npb_rpc import DiscoveredRpcClient, FilesystemDiscovery

with DiscoveredRpcClient(FilesystemDiscovery()) as client:
    result = client.call(
        "array-service",
        "array.sum",
        SumRequest(values=np.arange(10, dtype=np.float32)),
        SumResponse,
    )
```

List every service that currently has at least one healthy instance, then
inspect its instances:

```python
for service in discovery.list_services():
    print(service)
    for instance in discovery.list_instances(service):
        print(instance.to_dict())
```

`FilesystemDiscovery` stores atomic JSON records under the system temporary
directory at `npb-rpc/registry` by default. Records older than the heartbeat
timeout are ignored and pruned. When multiple healthy instances use the same
service name, `resolve()` selects them round-robin. Supply the same custom
registry path to clients and servers when overriding the default. The default
registry is intended for processes on one machine; use a shared registry path
or implement `DiscoveryBackend` for discovery across hosts. Keep a server's
`heartbeat_interval` shorter than the registry's `heartbeat_timeout`.

The combined example supports ZeroMQ and NNG over TCP or IPC, and iceoryx2
over shared memory:

```bash
uv run python examples/discovery_sum.py server --backend zmq --transport tcp
uv run python examples/discovery_sum.py client
uv run python examples/discovery_sum.py list

uv run python examples/discovery_sum.py server --backend nng --transport ipc
uv run python examples/discovery_sum.py client

uv run --extra iceoryx2 python examples/discovery_sum.py server --backend iceoryx2 --transport ipc
uv run --extra iceoryx2 python examples/discovery_sum.py client
```

Handlers may return structured failures:

```python
from npb_rpc import Status

if request.values.size == 0:
    context.abort(Status.INVALID_ARGUMENT, "values cannot be empty")
```

The client receives this as `RemoteRpcError` with `status`, `message`, and
`details` attributes.

## Large local arrays

Pass the same compatible blob-store backend to the client and server, and set
`externalize_min_bytes`. Large ndarray leaves then travel as small references
while the RPC control plane remains compact:

```python
client = ZmqRpcClient.connect(
    endpoint,
    blob_store=store,
    externalize_min_bytes=16 << 20,
)
```

For remote hosts, the blob store must itself make referenced objects available
to both hosts.

## Current scope

Version 0.1 intentionally implements synchronous unary RPC. One client object
serializes its calls; create one client per calling thread for parallel calls. Server
handlers are also dispatched serially in this first version.

Planned follow-ups include an asyncio API, bounded concurrent server dispatch,
streaming, cancellation, interceptors, observability hooks, and authentication
integration.

## Development with uv

The checked-in `uv.lock` pins the development environment. From the repository
root, install the project and its development dependencies, then run the checks:

```bash
uv sync --locked
uv run pytest
uv run ruff check .

# Include the optional iceoryx2 integration tests
uv run --extra iceoryx2 pytest
```

After changing dependencies in `pyproject.toml`, refresh the lockfile with
`uv lock`.
