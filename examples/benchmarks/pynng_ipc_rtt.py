"""Compatibility entry point for the original IPC-only benchmark.

Prefer ``pynng_rtt.py`` for new runs because it supports payload sweeps,
TCP/IPC selection, adaptive iteration counts, and CSV output.
"""

from __future__ import annotations

import sys

from pynng_rtt import main


if __name__ == "__main__":
    if not any(
        arg == "--transport" or arg.startswith("--transport=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(("--transport", "ipc"))
    main()
