from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from npb_rpc.utils import resolve_service_instance

try:
    import zmq
except ImportError:  # Only needed when checking ZeroMQ IPC capability.
    zmq = None

from npb_rpc import FilesystemDiscovery, portable_ipc, portable_tcp

from servers.msg.pcd import PcdClient, PcdInterface
from servers.msg.pcd import (
    EmptyRequest,
    PcdBuildRequest,
    PcdJobRequest,
    PcdJobResultResponse,
    PcdJobStatusResponse,
    PcdStatusResponse,
)
from servers.server_pcd.server import run_server


LOG = logging.getLogger("npb_rpc_pcd")
DEFAULT_SERVICE = PcdInterface.service
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def endpoint_for(backend: str, transport: str, name: str, host: str = "127.0.0.1") -> str:
    if backend == "iceoryx2":
        if transport != "ipc":
            raise SystemExit("iceoryx2 requires --transport ipc")
        return f"iceoryx2://{name}"
    if transport == "tcp":
        return portable_tcp(name, host=host)
    if backend == "zmq" and not zmq.has("ipc"):
            raise SystemExit("This libzmq build does not support ipc://; use TCP or NNG,IOX2")
    return portable_ipc(name)


def make_client(args: argparse.Namespace, discovery: FilesystemDiscovery) -> PcdInterface:
    if args.endpoint:
        return PcdClient(endpoint=args.endpoint, backend=args.backend, service=args.service)
    return PcdClient(discovery=discovery, service=args.service, server_name=args.server_name)


def print_connection(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    if args.endpoint:
        print(f"connecting directly via {args.backend} at {args.endpoint}")
    elif args.server_name:
        instance = resolve_service_instance(discovery, args.service, args.server_name)
        print(f"connecting to {instance.instance_id!r} via {instance.backend} at {instance.endpoint}")
    else:
        print(f"discovering a healthy instance of {args.service!r}; registry: {discovery.root}")


# Output

def print_status(status: PcdStatusResponse) -> None:
    print(
        "status:",
        f"online={status.online}",
        f"queued={status.queued_jobs}",
        f"running={status.running_jobs}",
        f"succeeded={status.succeeded_jobs}",
        f"failed={status.failed_jobs}",
        f"cancelled={status.cancelled_jobs}",
        f"builds={status.build_count}",
        f"cache_hits={status.cache_hits}",
        f"cache_misses={status.cache_misses}",
        f"cached_backends={list(status.cached_backends)!r}",
        f"last_ms={status.last_build_ms:.3f}",
        f"error={status.error!r}",
    )


def print_job_status(status: PcdJobStatusResponse) -> None:
    print(
        "job:",
        f"found={status.found}",
        f"id={status.job_id!r}",
        f"state={status.state!r}",
        f"backend={status.backend!r}",
        f"cuda_device={status.cuda_device}",
        f"cache_hit={status.cache_hit}",
        f"points={status.point_count}",
        f"segments={status.num_segments}",
        f"total_ms={status.timing.total_ms:.3f}",
        f"output={status.output_pcd_path!r}",
        f"error={status.error!r}",
    )


def print_job_result(response: PcdJobResultResponse, *, full: bool = False) -> None:
    if not response.found:
        print("job not found:", f"id={response.job_id!r}", f"error={response.error!r}")
        return

    if response.result is None:
        print(
            "result:",
            f"id={response.job_id!r}",
            f"state={response.state!r}",
            f"error={response.error!r}",
        )
        return

    result = response.result
    if full:
        print(result.model_dump_json(indent=2))
        return

    print(
        "result:",
        f"id={response.job_id!r}",
        f"state={response.state!r}",
        f"backend={result.backend_used!r}",
        f"device={result.device_used!r}",
        f"rgb={result.rgb_image_width}x{result.rgb_image_height}",
        f"stereo={result.stereo_image_width}x{result.stereo_image_height}",
        f"points={result.point_count}",
        f"segments={result.num_segments}",
        f"total_ms={result.timing.total_ms:.3f}",
        f"pcd={result.output_pcd_path!r}",
        f"json={result.output_json_path!r}",
        f"error={response.error!r}",
    )


# Build request / async job

def make_build_request(args: argparse.Namespace) -> PcdBuildRequest:
    required = {
        "--rgb-jpg-path": args.rgb_jpg_path,
        "--left-jpg-path": args.left_jpg_path,
        "--right-jpg-path": args.right_jpg_path,
        "--calibration-json-path": args.calibration_json_path,
        "--output-pcd-path": args.output_pcd_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("required for --build: " + ", ".join(missing))

    return PcdBuildRequest(
        rgb_jpg_path=args.rgb_jpg_path,
        left_jpg_path=args.left_jpg_path,
        right_jpg_path=args.right_jpg_path,
        calibration_json_path=args.calibration_json_path,
        output_pcd_path=args.output_pcd_path,
        output_json_path=args.output_json_path,
        backend=args.pcd_backend,
        cuda_device=args.cuda_device,
        min_disparity=args.min_disparity,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        stride=args.stride,
        alpha=args.alpha,
        rgb_image_is_undistorted=args.rgb_image_is_undistorted,
        binary_pcd=args.binary_pcd,
        detections_json_path=args.detections_json_path,
        segments_output_dir=args.segments_output_dir,
        min_segment_points=args.min_segment_points,
        erode_pixels=args.erode_pixels,
        exclusive_segments=args.exclusive_segments,
        save_background=args.save_background,
    )


def wait_for_job(
    client: PcdInterface,
    job_id: str,
    poll_interval: float,
    *,
    full_result: bool,
) -> None:
    last_state = None
    while True:
        status = client.job_status(PcdJobRequest(job_id=job_id))
        if not status.found:
            print_job_status(status)
            return

        if status.state != last_state:
            print_job_status(status)
            last_state = status.state

        if status.state in TERMINAL_STATES:
            break
        time.sleep(max(0.01, poll_interval))

    if status.state == "succeeded":
        print_job_result(client.job_result(PcdJobRequest(job_id=job_id)), full=full_result)


def run_client(args: argparse.Namespace, discovery: FilesystemDiscovery) -> None:
    client = make_client(args, discovery)
    print_connection(args, discovery)

    if args.job_status:
        print_job_status(client.job_status(PcdJobRequest(job_id=args.job_status)))
        return

    if args.job_result:
        print_job_result(
            client.job_result(PcdJobRequest(job_id=args.job_result)),
            full=args.print_result,
        )
        return

    if args.build:
        response = client.build(make_build_request(args))
        print(
            "submit:",
            f"accepted={response.accepted}",
            f"job_id={response.job_id!r}",
            f"state={response.state!r}",
            f"rgb={response.rgb_jpg_path!r}",
            f"left={response.left_jpg_path!r}",
            f"right={response.right_jpg_path!r}",
            f"pcd={response.output_pcd_path!r}",
            f"json={response.output_json_path!r}",
            f"error={response.error!r}",
        )
        if response.accepted and args.wait:
            wait_for_job(client, response.job_id, args.poll_interval, full_result=args.print_result)
        return

    print_status(client.status(EmptyRequest()))


# Discovery / server configuration

def show_services(discovery: FilesystemDiscovery) -> None:
    services = discovery.list_services()
    if not services:
        print("no healthy services")
        return

    for service in services:
        print(f"{service}:")
        for instance in discovery.list_instances(service):
            print(
                f"  {instance.instance_id}  {instance.backend}  {instance.endpoint}  "
                f"{instance.hostname} pid={instance.pid}"
            )


def make_backend_options(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Deployment-side configuration; intentionally not part of PcdBuildRequest."""
    return {
        "cpu": {"num_disparities": args.cpu_num_disparities},
        "cuda": {"dll": args.libsgm_dll, "num_disparities": args.libsgm_num_disparities},
        "dnn": {
            "repo_dir": args.foundation_repo_dir,
            "model_path": args.foundation_model_path,
            "valid_iters": args.foundation_valid_iters,
            "max_disp": args.foundation_max_disp,
            "compile_model": args.foundation_compile_model,
        },
        "vpi": {"num_disparities": args.vpi_num_disparities},
    }


# CLI parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Asynchronous discovered RGB-stereo point-cloud RPC service"
    )
    add = parser.add_argument
    bool_action = argparse.BooleanOptionalAction

    add("role", choices=("server", "client", "list"))

    # RPC / discovery
    add("--service", default=DEFAULT_SERVICE)
    add("--server-name", help="instance id to advertise (server) or select (client)")
    add("--backend", choices=("nng", "zmq", "iceoryx2"), default="iceoryx2", help="RPC backend")
    add("--transport", choices=("tcp", "ipc"), default="ipc")
    add("--host", default="127.0.0.1")
    add("--endpoint", default=None)
    add("--advertise-endpoint")
    add("--registry", type=Path)

    # Server worker / filesystem
    add("--worker-count", type=int, default=1)
    add("--queue-size", type=int, default=0, help="0 = unbounded")
    add("--job-ttl", type=float, default=3600.0)
    add("--max-completed-jobs", type=int, default=128)
    add("--read-root", type=Path)
    add("--write-root", type=Path)
    add("--shutdown-timeout", type=float, default=10.0)
    add("--calibration-translation-unit", choices=("m", "cm", "mm"), default="cm")

    # Server-side stereo backends
    add("--cpu-num-disparities", type=int, default=256)
    add("--libsgm-dll", type=Path, default=Path("build/Release/sgm_py.dll"))
    add("--libsgm-num-disparities", type=int, choices=(64, 128, 256), default=256)
    add("--foundation-repo-dir", type=Path, default=Path("./fast-foundationstereo"))
    add(
        "--foundation-model-path",
        type=Path,
        default=Path("weights/23-36-37/model_best_bp2_serialize.pth"),
    )
    add("--foundation-valid-iters", type=int, default=8)
    add("--foundation-max-disp", type=int, default=192)
    add("--foundation-compile-model", action=bool_action, default=False)
    add("--vpi-num-disparities", type=int, choices=(64, 128, 256), default=256)

    # Client operation
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true", help="submit a new PCD build job")
    action.add_argument("--job-status", metavar="JOB_ID")
    action.add_argument("--job-result", metavar="JOB_ID")
    add("--wait", action="store_true", help="wait for a submitted build job to finish")
    add("--poll-interval", type=float, default=0.2)
    add("--print-result", action="store_true", help="print the full PcdBuildResult JSON")

    # Build I/O
    add("--rgb-jpg-path")
    add("--left-jpg-path")
    add("--right-jpg-path")
    add("--calibration-json-path")
    add("--output-pcd-path")
    add("--output-json-path")

    # Point-cloud backend / geometry
    add(
        "--pcd-backend",
        choices=("cpu", "cuda", "dnn", "vpi"),
        default="cuda",
        help="stereo disparity backend; different from --backend, which selects RPC transport",
    )
    add("--cuda-device", type=int, default=0)
    add("--min-disparity", type=float, default=0.5)
    add("--min-depth-m", type=float, default=0.01)
    add("--max-depth-m", type=float, default=5.0)
    add("--stride", type=int, default=1)
    add("--alpha", type=float, default=0.0)
    add("--rgb-image-is-undistorted", action=bool_action, default=False)
    add("--binary-pcd", action=bool_action, default=True)

    # Optional YOLO segmentation
    add("--detections-json-path")
    add("--segments-output-dir")
    add("--min-segment-points", type=int, default=30)
    add("--erode-pixels", type=int, default=0)
    add("--exclusive-segments", action=bool_action, default=False)
    add("--save-background", action=bool_action, default=False)

    add("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )
    discovery = FilesystemDiscovery(args.registry)

    if args.role == "server":
        server_name = args.server_name or args.service
        endpoint = args.endpoint or endpoint_for(
            args.backend, args.transport, server_name, args.host
        )
        print(
            f"serving {args.service!r} as {server_name!r} via {args.backend} "
            f"at {endpoint}; registry: {discovery.root}"
        )
        run_server(
            endpoint,
            backend=args.backend,
            discovery=discovery,
            service=args.service,
            instance_id=server_name,
            advertise_endpoint=args.advertise_endpoint,
            worker_count=args.worker_count,
            queue_size=args.queue_size,
            job_ttl_s=args.job_ttl,
            max_completed_jobs=args.max_completed_jobs,
            read_root=args.read_root,
            write_root=args.write_root,
            backend_options=make_backend_options(args),
            calibration_translation_unit=args.calibration_translation_unit,
            shutdown_timeout_s=args.shutdown_timeout,
        )
    elif args.role == "client":
        run_client(args, discovery)
    else:
        show_services(discovery)


if __name__ == "__main__":
    main()
