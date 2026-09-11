# npb-rpc

`npb-rpc` adds typed unary RPC to [NPB](https://github.com/qinhy/npb), preserving NPB's Pydantic
schema validation, zero-copy NumPy decode, preallocated-buffer support, and
optional external blob stores.

ZeroMQ and NNG transports are optional:

```bash
pip install "npb-rpc[zmq]"
pip install "npb-rpc[nng]"
```

Or, with uv:

```bash
uv add "npb-rpc[zmq]"
uv add "npb-rpc[nng]"
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
serializes its calls because the underlying ZeroMQ and NNG sockets are
stateful; create one client per calling thread for parallel calls. Server
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
```

After changing dependencies in `pyproject.toml`, refresh the lockfile with
`uv lock`.
