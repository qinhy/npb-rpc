# npb-rpc examples

The examples are grouped by what they demonstrate rather than by filename history.

```text
examples/
├── basic/
│   ├── nng_sum.py
│   ├── zmq_sum.py
│   └── iceoryx2_sum.py
├── integrations/
│   ├── nng_fastapi_sum.py
│   └── nng_typed_api_sum.py
├── discovery/
│   └── sum_service.py
└── benchmarks/
    ├── README.md
    ├── pynng_rtt.py
    ├── pyzmq_rtt.py
    ├── rpc_rtt.py
    ├── iceoryx2_rtt.py
    ├── ndarray_throughput.py
    └── pynng_ipc_rtt.py
```

## Basic RPC

All three basic examples implement the same `array.sum` request/response flow so the transport backends are easy to compare.

### NNG

Terminal 1:

```bash
uv run python examples/basic/nng_sum.py server --transport ipc
```

Terminal 2:

```bash
uv run python examples/basic/nng_sum.py client --transport ipc
```

Use `--transport tcp` to use TCP instead. `--name` changes the logical endpoint name, and `--endpoint` can override the generated endpoint entirely.

### ZeroMQ

Terminal 1:

```bash
uv run python examples/basic/zmq_sum.py server --transport tcp
```

Terminal 2:

```bash
uv run python examples/basic/zmq_sum.py client --transport tcp
```

IPC is also available when the local libzmq build supports it:

```bash
uv run python examples/basic/zmq_sum.py server --transport ipc
```

On native Windows, some libzmq builds do not provide `ipc://`; use TCP or the NNG IPC example in that case.

### iceoryx2

Terminal 1:

```bash
uv run --extra iceoryx2 python examples/basic/iceoryx2_sum.py server
```

Terminal 2:

```bash
uv run --extra iceoryx2 python examples/basic/iceoryx2_sum.py client
```

## Integrations

### NNG + FastAPI

This example keeps the RPC implementation explicit and adds a small HTTP gateway.

```bash
uv run python examples/integrations/nng_fastapi_sum.py api --transport ipc
```

The `api` role starts the NNG service in a child process and runs FastAPI in the parent process. The RPC server and client roles can also be run independently:

```bash
uv run python examples/integrations/nng_fastapi_sum.py server --transport ipc
uv run python examples/integrations/nng_fastapi_sum.py client --transport ipc
```

### Typed API definition

This example defines the RPC and HTTP metadata once on `SumInterface`, then builds the NNG server, typed client, and FastAPI routes from that definition.

```bash
uv run python examples/integrations/nng_typed_api_sum.py api --transport ipc
```

Or run the RPC endpoints separately:

```bash
uv run python examples/integrations/nng_typed_api_sum.py server --transport ipc
uv run python examples/integrations/nng_typed_api_sum.py client --transport ipc
```

## Service discovery

The discovery example can advertise and resolve NNG, ZeroMQ, or iceoryx2 service instances through `FilesystemDiscovery`.

Start an NNG instance:

```bash
uv run python examples/discovery/sum_service.py server \
  --backend nng \
  --transport ipc \
  --server-name sum-a
```

Call any healthy instance of the `sum` service:

```bash
uv run python examples/discovery/sum_service.py client --service sum
```

Call a particular instance:

```bash
uv run python examples/discovery/sum_service.py client \
  --service sum \
  --server-name sum-a
```

Inspect the registry:

```bash
uv run python examples/discovery/sum_service.py list
```

## Benchmarks

Benchmarks are separated into raw transport baselines and end-to-end `npb_rpc` measurements. See [`benchmarks/README.md`](benchmarks/README.md) for methodology and all options.

Raw pynng IPC/TCP RTT and payload sweep:

```bash
uv run python examples/benchmarks/pynng_rtt.py --transport ipc
```

Raw pyzmq IPC/TCP RTT and payload sweep:

```bash
uv run python examples/benchmarks/pyzmq_rtt.py --transport tcp
```

End-to-end RPC benchmark, including serialization and dispatch:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc
```

Compare all available RPC backends and transports:

```bash
uv run --extra iceoryx2 python examples/benchmarks/rpc_rtt.py \
  --backend all \
  --transport all \
  --csv benchmark_results.csv
```

The RPC benchmark supports `echo` (large request + large response) and `consume` (large request + tiny acknowledgement), and reports mean/stddev/min/p50/p90/p95/p99/p99.9/max latency, requests/s, and application payload MiB/s.

Focused iceoryx2 RTT, including explicit polling-interval control:

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py --poll-us 100
```

Large NumPy-array codec and end-to-end throughput:

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py \
  --backend all \
  --transport ipc \
  --sizes 1M,4M,16M,64M,256M
```

`ndarray_throughput.py` also accepts GiB payloads such as `--sizes 256M,1G,2G`; see the benchmark README for memory-safe usage guidance.

The old `pynng_ipc_rtt.py` path remains as a compatibility wrapper.

## File migration

| Old path | New path |
| --- | --- |
| `examples/nng_sum.py` | `examples/integrations/nng_fastapi_sum.py` |
| `examples/nng_sum_OOP.py` | `examples/integrations/nng_typed_api_sum.py` |
| new extracted example | `examples/basic/nng_sum.py` |
| `examples/zmq_sum.py` | `examples/basic/zmq_sum.py` |
| `examples/iceoryx2_sum.py` | `examples/basic/iceoryx2_sum.py` |
| `examples/discovery_sum.py` | `examples/discovery/sum_service.py` |
| `examples/benchmark_nng_ipc.py` | `examples/benchmarks/pynng_ipc_rtt.py` |
