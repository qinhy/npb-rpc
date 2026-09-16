# Benchmarks

The benchmark examples are split into two layers so transport cost and RPC-stack cost are not confused.

```text
benchmarks/
├── _common.py          # statistics, payload parsing, table/CSV output
├── pynng_rtt.py        # raw pynng REQ/REP baseline
├── pyzmq_rtt.py        # raw pyzmq REQ/REP baseline
├── rpc_rtt.py          # end-to-end npb_rpc benchmark
└── pynng_ipc_rtt.py    # compatibility wrapper for the old benchmark name
```

## What is measured

All timings are synchronous request/reply round-trip times measured with `time.perf_counter_ns()` after warmup.

The table reports:

- `mean us`: mean RTT in microseconds.
- `p50`, `p95`, `p99`: RTT latency percentiles.
- `req/s`: synchronous request/reply operations per second, calculated from mean RTT.
- `MiB/s`: application payload goodput. `echo` counts payload bytes in both directions; `consume` counts request payload bytes only.

`MiB/s` intentionally excludes protocol headers and metadata. It is therefore application-level goodput, not physical wire bandwidth.

When `--count` is omitted, iteration count is adaptive. By default each size transfers roughly 256 MiB of request payload, capped between 25 and 10,000 measured iterations. This prevents a 4 MiB benchmark from running 10,000 times while still giving small messages enough samples.

## Raw pynng

IPC:

```bash
uv run python examples/benchmarks/pynng_rtt.py --transport ipc
```

TCP loopback:

```bash
uv run python examples/benchmarks/pynng_rtt.py --transport tcp
```

A focused small-message latency run:

```bash
uv run python examples/benchmarks/pynng_rtt.py \
  --transport ipc \
  --sizes 0,16,64,256,1K \
  --count 20000
```

## Raw pyzmq

```bash
uv run python examples/benchmarks/pyzmq_rtt.py --transport tcp
```

When the installed libzmq supports IPC:

```bash
uv run python examples/benchmarks/pyzmq_rtt.py --transport ipc
```

## End-to-end npb_rpc

NNG over IPC:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc
```

ZeroMQ over TCP:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend zmq \
  --transport tcp
```

iceoryx2 shared memory:

```bash
uv run --extra iceoryx2 python examples/benchmarks/rpc_rtt.py \
  --backend iceoryx2 \
  --transport ipc
```

Compare all available backend/transport combinations:

```bash
uv run --extra iceoryx2 python examples/benchmarks/rpc_rtt.py \
  --backend all \
  --transport all
```

Unavailable combinations are reported as skipped when `all` is used. iceoryx2 is IPC-only in this benchmark.

### `echo` vs `consume`

`rpc_rtt.py` benchmarks two RPC shapes:

- `echo`: the full NumPy payload is returned. This stresses serialization and transport in both directions.
- `consume`: the request contains the full payload but the response only contains `nbytes`. This isolates request-heavy transfer and avoids sending the large payload back.

Run only one mode when desired:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc \
  --mode consume
```

## Payload sweep

Sizes accept `K`, `M`, `G`, `KiB`, `MiB`, and `GiB` suffixes:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc \
  --sizes 64,1K,16K,256K,1M,4M,16M
```

For an exactly repeatable iteration count instead of adaptive counts:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend nng \
  --transport ipc \
  --sizes 64,1K,16K \
  --count 10000 \
  --warmup 500
```

## CSV output

Every benchmark can write the same CSV schema:

```bash
uv run python examples/benchmarks/rpc_rtt.py \
  --backend all \
  --transport all \
  --csv benchmark_results.csv
```

This makes it straightforward to plot latency versus payload size or compare backend goodput later.

## Fair comparison notes

For meaningful comparisons:

1. Run all compared backends on the same machine and OS.
2. Keep payload sizes, iteration policy, and warmup policy identical.
3. Compare raw transport results separately from `npb_rpc` results.
4. Do not compare IPC/shared-memory results directly with remote-host TCP and call it a backend comparison; the transport path is different.
5. Run several repetitions if tail latency (`p95`/`p99`) matters, because scheduling and background load can influence it.
