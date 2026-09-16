"""Focused npb_rpc RTT benchmark for the iceoryx2 shared-memory backend."""

from __future__ import annotations

import argparse
import multiprocessing as mp

from _common import parse_size, parse_sizes, print_results, print_tail_note, write_csv
from rpc_rtt import run_case


def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(
        description="npb_rpc iceoryx2 shared-memory RTT benchmark"
    )
    parser.add_argument(
        "--sizes",
        default="64,1K,16K,256K,1M,4M",
        help="comma-separated payload sizes; K/M/G suffixes are supported",
    )
    parser.add_argument("--mode", choices=("echo", "consume", "both"), default="both")
    parser.add_argument("-n", "--count", type=int, help="fixed iterations per size")
    parser.add_argument("--warmup", type=int, help="fixed warmup iterations per size")
    parser.add_argument(
        "--target-bytes",
        type=parse_size,
        default=parse_size("256M"),
        help="adaptive request bytes per payload size when --count is omitted",
    )
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--max-count", type=int, default=2_000)
    parser.add_argument(
        "--max-message-bytes",
        type=parse_size,
        help="RPC payload limit; default is largest size + 16 MiB",
    )
    parser.add_argument(
        "--wait-strategy",
        choices=("sleep", "yield", "spin", "hybrid"),
        default="spin",
        help="receive wait strategy; spin is best for latency benchmarking (default: spin)",
    )
    parser.add_argument(
        "--poll-us",
        type=float,
        default=100.0,
        help="sleep-strategy poll interval in microseconds (default: 100)",
    )
    parser.add_argument(
        "--spin-us",
        type=float,
        default=50.0,
        help="hybrid busy-spin window in microseconds (default: 50)",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=5.0,
        help="timeout for each RPC call in seconds (default: 5)",
    )
    parser.add_argument(
        "--response-ownership",
        choices=("owned", "borrowed"),
        default="owned",
        help=(
            "owned makes one final SHM-to-owned payload copy; borrowed keeps "
            "response ndarrays as zero-copy SHM views (default: owned)"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="disable live progress output"
    )
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args()
    if args.poll_us <= 0:
        parser.error("--poll-us must be > 0")
    if args.spin_us < 0:
        parser.error("--spin-us must be >= 0")
    if args.call_timeout <= 0:
        parser.error("--call-timeout must be > 0")

    modes = ["echo", "consume"] if args.mode == "both" else [args.mode]
    results = run_case(
        backend="iceoryx2",
        transport="ipc",
        host="127.0.0.1",
        modes=modes,
        sizes=parse_sizes(args.sizes),
        count=args.count,
        warmup=args.warmup,
        target_bytes=args.target_bytes,
        min_count=args.min_count,
        max_count=args.max_count,
        max_message_bytes=args.max_message_bytes,
        iceoryx2_poll_interval=args.poll_us / 1_000_000.0,
        iceoryx2_wait_strategy=args.wait_strategy,
        iceoryx2_spin_duration=args.spin_us / 1_000_000.0,
        call_timeout=args.call_timeout,
        progress=not args.quiet,
        borrowed_response=args.response_ownership == "borrowed",
    )
    print(
        f"iceoryx2 wait strategy: {args.wait_strategy}; "
        f"poll={args.poll_us:g} us; spin={args.spin_us:g} us; "
        f"response={args.response_ownership}"
    )
    print_results(results)
    print_tail_note(results)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
