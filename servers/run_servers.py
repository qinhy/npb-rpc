from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Service:
    name: str
    command: list[str]

# if sys.platform == "win32":
#     python = ("uv", "run")
# elif sys.platform.startswith("linux"):
#     python = ("python3")

backend="nng"   # [--backend {nng,zmq,iceoryx2}]
transport="ipc" # [--transport {tcp,ipc}]
SERVICES = [
    Service(
        "rgbd_left",
        [
            "uv", "run", "-m", "servers.server_dai.cli",
            "server",
            "--backend", backend,
            "--transport", transport,
            "--server-name", "rgbd_left",
            "--no-auto-open",
        ],
    ),
    Service(
        "rgbd_right",
        [
            "uv", "run", "-m", "servers.server_dai.cli",
            "server",
            "--backend", backend,
            "--transport", transport,
            "--server-name", "rgbd_right",
            "--no-auto-open",
        ],
    ),
    Service(
        "rgbd_hand",
        [
            "uv", "run", "-m", "servers.server_dai.cli",
            "server",
            "--backend", backend,
            "--transport", transport,
            "--server-name", "rgbd_hand",
            "--no-auto-open",
        ],
    ),
    Service(
        "yolo",
        [
            "uv", "run", "-m", "servers.server_yolo.cli",
            "server",
            "--backend", backend,
            "--transport", transport,
        ],
    ),
    Service(
        "pcd",
        [
            "uv", "run", "-m", "servers.server_pcd.cli",
            "server",
            "--backend", backend,
            "--transport", transport,
        ],
    ),
    Service(
        "web",
        [
            "uv", "run", "-m", "servers.server_web",
        ],
    ),
]


def main() -> None:
    env = os.environ.copy()
    env["LOG_REDIS_URL"] = "redis://127.0.0.1:6379/0"

    processes: dict[str, subprocess.Popen] = {}

    try:
        for service in SERVICES:
            print(f"[START] {service.name}")

            process = subprocess.Popen(
                service.command,
                env=env,
            )

            processes[service.name] = process

        print()
        print("All servers started.")
        print("Press Ctrl+C to stop all servers.")
        print()

        while True:
            for name, process in processes.items():
                returncode = process.poll()

                if returncode is not None:
                    print(
                        f"[EXIT] {name}: "
                        f"returncode={returncode}"
                    )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping servers...")

    finally:
        # First ask processes to terminate normally.
        for name, process in processes.items():
            if process.poll() is None:
                print(f"[STOP] {name}")
                process.terminate()

        # Give them a moment to exit gracefully.
        deadline = time.monotonic() + 5

        for process in processes.values():
            if process.poll() is None:
                timeout = max(0, deadline - time.monotonic())

                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass

        # Force kill anything still alive.
        for name, process in processes.items():
            if process.poll() is None:
                print(f"[KILL] {name}")
                process.kill()


if __name__ == "__main__":
    main()