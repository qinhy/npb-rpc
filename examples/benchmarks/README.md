# Benchmarks

The benchmark examples are split into transport baselines, end-to-end RPC RTT,
and large-array throughput so transport cost, codec cost, and RPC-stack cost are
not confused.

```text
benchmarks/
├── _common.py             # statistics, size parsing, table/CSV output
├── pynng_rtt.py           # raw pynng REQ/REP baseline
├── pyzmq_rtt.py           # raw pyzmq REQ/REP baseline
├── rpc_rtt.py             # end-to-end npb_rpc: NNG / ZeroMQ / iceoryx2
├── iceoryx2_rtt.py        # focused npb_rpc iceoryx2 shared-memory RTT
├── ndarray_throughput.py  # NPB codec + large ndarray RPC throughput
└── pynng_ipc_rtt.py       # compatibility wrapper for old benchmark name
```

## Latency statistics

RTT benchmarks use `time.perf_counter_ns()` after warmup and report:

- `mean`: mean latency in microseconds.
- `sd`: population standard deviation.
- `min`, `p50`, `p90`, `p95`, `p99`, `p99.9`, `max`: latency distribution and tails.
- `max`: slowest observed operation.
- `req/s`: synchronous operations per second derived from mean latency.
- `MiB/s`: application payload goodput.

`p99.9` needs a large sample. When fewer than 1,000 measurements are present,
treat it as descriptive only and pay more attention to `max` plus repeated runs.

The generic RTT scripts target roughly **1 GiB of request payload per size** when
`--count` is omitted. The focused iceoryx2 benchmark uses a lighter default
(256 MiB target and at most 2,000 calls per size) because its Python backend is
poll based. Both scripts print live per-case progress unless `--quiet` is used.

## Raw pynng

```bash
uv run examples/benchmarks/pynng_rtt.py --transport ipc
uv run examples/benchmarks/pynng_rtt.py --transport tcp
```

Focused small-message latency:

```bash
uv run examples/benchmarks/pynng_rtt.py \
  --transport ipc \
  --sizes 0,16,64,256,1K \
  --count 20000
```

## Raw pyzmq

```bash
uv run examples/benchmarks/pyzmq_rtt.py --transport tcp
```

If the installed libzmq supports IPC:

```bash
uv run examples/benchmarks/pyzmq_rtt.py --transport ipc
```

## End-to-end npb_rpc RTT

NNG over IPC:

```bash
uv run examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc
```

ZeroMQ over TCP:

```bash
uv run examples/benchmarks/rpc_rtt.py \
  --backend zmq \
  --transport tcp
```

iceoryx2 shared memory:

```bash
uv run --extra iceoryx2 examples/benchmarks/rpc_rtt.py \
  --backend iceoryx2 \
  --transport ipc
```

Compare all available backend/transport combinations:

```bash
uv run --extra iceoryx2 examples/benchmarks/rpc_rtt.py \
  --backend all \
  --transport all
```

### `echo` vs `consume`

`rpc_rtt.py` provides two request shapes:

- `echo`: full ndarray goes to the server and comes back. `MiB/s` counts both
  directions.
- `consume`: full ndarray goes to the server and only a tiny acknowledgement
  comes back. `MiB/s` counts request bytes only.

```bash
uv run examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc \
  --mode consume
```

## Focused iceoryx2 RTT

`iceoryx2_rtt.py` uses the public `Iceoryx2RpcClient` / `Iceoryx2RpcServer`
backend. It is intentionally named as an `npb_rpc` benchmark rather than a raw
native iceoryx2 benchmark.

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py
```

The Python iceoryx2 RPC backend polls for requests/responses. The focused
benchmark defaults to 100 us polling so small-message sweeps complete promptly;
measure the latency/CPU trade-off explicitly:

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py --poll-us 1000
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py --poll-us 100
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py --poll-us 10
```

Keep the selected polling value with the result when comparing iceoryx2 with
blocking NNG/ZeroMQ transports.

A fast smoke test is recommended before a full sweep:

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py \
  --sizes 64,1K \
  --mode echo \
  --count 100 \
  --warmup 5 \
  --poll-us 100
```

The benchmark prints `server ready`, the current case, and periodic progress.
Each RPC has a finite 5-second timeout by default (`--call-timeout` can change
it), and a child server crash is reported instead of leaving a silent wait.

## Large ndarray throughput

`ndarray_throughput.py` measures two layers:

1. Standalone `npb.encode()` and `npb.decode()` throughput.
2. Complete `npb_rpc` calls over NNG, ZeroMQ, and iceoryx2.

The default workload is request-heavy `consume`, which resembles a producer
sending an image/tensor to a service and receiving a tiny ACK.

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py
```

Compare IPC/shared-memory paths only:

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py \
  --backend all \
  --transport ipc \
  --sizes 1M,4M,16M,64M,256M
```

Compare NNG TCP and IPC:

```bash
uv run examples/benchmarks/ndarray_throughput.py \
  --backend nng \
  --transport all \
  --sizes 1M,4M,16M,64M,256M
```

Use float arrays while keeping the requested values as **byte sizes**:

```bash
uv run examples/benchmarks/ndarray_throughput.py \
  --backend nng \
  --transport ipc \
  --dtype float32 \
  --sizes 4M,16M,64M
```

### Multi-GiB arrays

GiB sizes are supported, but are not part of the default sweep because the
current RPC path may temporarily hold several copies while NPB encodes and the
transport frames a message. Make sure the machine has substantially more free
RAM than the application payload itself.

A conservative first large-object run:

```bash
uv run examples/benchmarks/ndarray_throughput.py \
  --backend nng \
  --transport ipc \
  --mode consume \
  --sizes 256M,1G \
  --count 3 \
  --warmup 1 \
  --no-codec
```

Then, if memory headroom is sufficient:

```bash
uv run examples/benchmarks/ndarray_throughput.py \
  --backend nng \
  --transport ipc \
  --mode consume \
  --sizes 1G,2G \
  --count 3 \
  --warmup 1 \
  --no-codec
```

For iceoryx2:

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py \
  --backend iceoryx2 \
  --transport ipc \
  --mode consume \
  --sizes 64M,256M,1G \
  --count 3 \
  --warmup 1 \
  --no-codec \
  --iceoryx2-poll-us 100
```

Standalone codec measurements default to payloads <= 256 MiB to reduce peak
memory usage. Raise the limit deliberately when desired:

```bash
uv run examples/benchmarks/ndarray_throughput.py \
  --backend nng \
  --sizes 256M,1G \
  --codec-max-size 1G
```

## Message-size limit

The RPC backends normally default to a 256 MiB message limit. These benchmark
scripts automatically raise the configured limit to the largest requested
application payload plus 16 MiB of headroom. You can override it explicitly:

```bash
uv run examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --sizes 256M,1G \
  --max-message-bytes 1100M
```

## CSV output

All benchmark rows use the same schema and can be written to CSV:

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py \
  --backend all \
  --transport ipc \
  --csv benchmark_results.csv
```

## Fair comparison notes

1. Run compared backends on the same machine and OS.
2. Keep payload sizes, count policy, and warmup policy identical.
3. Compare raw transport results separately from full `npb_rpc` results.
4. Record the iceoryx2 poll interval with each result.
5. For large payloads, watch system memory and avoid paging; paging measures the
   OS memory-pressure path more than the transport.
6. Run multiple repetitions for tail-latency conclusions.
7. `MiB/s` is application goodput, not physical wire bandwidth. `echo` counts
   payload bytes in both directions while `consume` counts the request only.

## iceoryx2 wait-strategy comparison

The iceoryx2 RPC backend supports four receive wait strategies after applying
this patch: `sleep`, `yield`, `hybrid`, and `spin`.

For a transport baseline without NPB/RPC framing:

```bash
uv run --extra iceoryx2 examples/benchmarks/raw_iceoryx2_rtt.py \
  --sizes 64,1K,16K --wait-strategy spin
```

For the full RPC stack using the same low-latency wait policy:

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py \
  --sizes 64,1K,16K --mode echo --wait-strategy spin
```

`spin` intentionally consumes a CPU core while waiting and is meant for
latency measurement. `hybrid` is the better first production experiment.
The source API keeps `sleep` as its default for backward compatibility.

## iceoryx2 owned vs borrowed responses

The optimized iceoryx2 backend supports two response ownership modes.

`owned` is the normal `client.call()` path. It makes one final copy of the NPB
response payload before releasing the iceoryx2 sample, so returned ndarray
fields have normal independent lifetime.

`borrowed` uses `client.call_borrowed()` and decodes directly from the response
shared-memory sample. It avoids that final copy, but ndarray fields are valid
only inside the borrowed-response context.

Compare them with:

```bash
uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py \
  --wait-strategy spin --response-ownership owned

uv run --extra iceoryx2 examples/benchmarks/iceoryx2_rtt.py \
  --wait-strategy spin --response-ownership borrowed
```

For large arrays:

```bash
uv run --extra iceoryx2 examples/benchmarks/ndarray_throughput.py \
  --backend iceoryx2 --mode echo --sizes 1M,4M,16M,64M \
  --iceoryx2-wait-strategy spin \
  --iceoryx2-response-ownership borrowed --no-codec
```
