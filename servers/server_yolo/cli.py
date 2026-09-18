from __future__ import annotations

import argparse
from servers.logger import logging
import time
from pathlib import Path

from npb_rpc.utils import resolve_service_instance

try:
    import zmq
except ImportError:  # Only needed for ZeroMQ IPC capability detection.
    zmq = None

from npb_rpc import RedisDiscovery, portable_ipc, portable_tcp

from servers.msg.yolo import YoloClient, YoloInterface
from servers.msg.yolo import EmptyRequest, YoloInferenceRequest, YoloJobRequest, YoloStatusResponse, YoloJobStatusResponse, YoloJobResultResponse
from servers.server_yolo.server import run_server


LOG = logging.getLogger(__name__.replace(".",":"))
DEFAULT_SERVICE = YoloInterface.service
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


def make_client(args: argparse.Namespace, discovery: RedisDiscovery) -> YoloInterface:
    if args.endpoint:
        return YoloClient(endpoint=args.endpoint, backend=args.backend, service=args.service)
    return YoloClient(discovery=discovery, service=args.service, server_name=args.server_name)


def print_connection(args: argparse.Namespace, discovery: RedisDiscovery) -> None:
    if args.endpoint:
        print(f"connecting directly via {args.backend} at {args.endpoint}")
    elif args.server_name:
        instance = resolve_service_instance(discovery, args.service, args.server_name)
        print(f"connecting to {instance.instance_id!r} via {instance.backend} at {instance.endpoint}")
    else:
        print(f"discovering a healthy instance of {args.service!r}; registry: {discovery.root}")


def print_status(status:YoloStatusResponse) -> None:
    print(
        "status:",
        f"online={status.online}",
        f"queued={status.queued_jobs}",
        f"running={status.running_jobs}",
        f"succeeded={status.succeeded_jobs}",
        f"failed={status.failed_jobs}",
        f"cancelled={status.cancelled_jobs}",
        f"inferences={status.inference_count}",
        f"cache_hits={status.cache_hits}",
        f"cache_misses={status.cache_misses}",
        f"cached_models={list(status.cached_models)!r}",
        f"last_job_id={status.last_job_id!r}",
        f"last_ms={status.last_inference_ms:.3f}",
        f"error={status.error!r}",
    )


def print_job_status(status:YoloJobStatusResponse) -> None:
    timing = status.timing
    print(
        "job:",
        f"found={status.found}",
        f"id={status.job_id!r}",
        f"state={status.state!r}",
        f"model={status.model_name!r}",
        f"device={status.cuda_device}",
        f"cache_hit={status.cache_hit}",
        f"detections={status.num_detections}",
        f"total_ms={timing.total_ms:.3f}",
        f"output={status.output_json_path!r}",
        f"error={status.error!r}",
    )


def print_job_result(response:YoloJobResultResponse, *, full: bool = False) -> None:
    if not response.found:
        print(f"job not found: {response.job_id!r}; error={response.error!r}")
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
        f"task={result.task!r}",
        f"image={result.image_width}x{result.image_height}",
        f"detections={result.num_detections}",
        f"has_masks={result.has_masks}",
        f"total_ms={result.timing.total_ms:.3f}",
        f"output={result.output_json_path!r}",
        f"error={response.error!r}",
    )


def make_inference_request(args: argparse.Namespace) -> YoloInferenceRequest:
    if not args.input_jpg_path:
        raise SystemExit("--input-jpg-path is required for inference")
    if not args.output_json_path:
        raise SystemExit("--output-json-path is required for inference")

    return YoloInferenceRequest(
        input_jpg_path=args.input_jpg_path,
        output_json_path=args.output_json_path,
        model_name=args.model_name,
        cuda_device=args.cuda_device,
        size_mode=args.size_mode,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
        half=args.half,
        stride=args.stride,
        tile_overlap=args.tile_overlap,
        tile_batch_size=args.tile_batch_size,
        include_masks=args.include_masks,
        mask_format="polygon",
        mask_threshold=args.mask_threshold,
        tiled_mask_output="full_image",
        merge_tiled_masks=args.merge_tiled_masks,
        tile_merge_iom=args.tile_merge_iom,
        polygon_epsilon=args.polygon_epsilon,
        polygon_min_area=args.polygon_min_area,
        detection_bbox_xyxy=args.detection_bbox_xyxy,
    )


def wait_for_job(client: YoloInterface, job_id: str, poll_interval: float, *, full_result: bool) -> None:
    last_state = None

    while True:
        status = client.job_status(YoloJobRequest(job_id=job_id))
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
        print_job_result(
            client.job_result(YoloJobRequest(job_id=job_id)),
            full=full_result,
        )


def run_client(args: argparse.Namespace, discovery: RedisDiscovery) -> None:
    client = make_client(args, discovery)
    print_connection(args, discovery)

    if args.job_status:
        print_job_status(client.job_status(YoloJobRequest(job_id=args.job_status)))
        return

    if args.job_result:
        print_job_result(
            client.job_result(YoloJobRequest(job_id=args.job_result)),
            full=args.print_result,
        )
        return

    if args.inference:
        request = make_inference_request(args)
        response = client.inference(request)
        print(
            "submit:",
            f"accepted={response.accepted}",
            f"job_id={response.job_id!r}",
            f"state={response.state!r}",
            f"input={response.input_jpg_path!r}",
            f"output={response.output_json_path!r}",
            f"error={response.error!r}",
        )

        if response.accepted and args.wait:
            wait_for_job(
                client,
                response.job_id,
                args.poll_interval,
                full_result=args.print_result,
            )
        return

    print_status(client.status(EmptyRequest()))


def show_services(discovery: RedisDiscovery) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Asynchronous discovered Ultralytics YOLO RPC service"
    )
    parser.add_argument("role", choices=("server", "client", "list"))

    # RPC / discovery
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--server-name", help="instance id to advertise (server) or select (client)")
    parser.add_argument("--backend", choices=("nng", "zmq", "iceoryx2"), default="nng")
    parser.add_argument("--transport", choices=("tcp", "ipc"), default="ipc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--advertise-endpoint")
    parser.add_argument("--registry", type=Path)

    # Server worker
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--queue-size", type=int, default=0)
    parser.add_argument("--job-ttl", type=float, default=3600.0)
    parser.add_argument("--max-completed-jobs", type=int, default=128)
    parser.add_argument("--read-root", type=Path)
    parser.add_argument("--write-root", type=Path)

    # Client operation
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--inference", action="store_true", help="submit a new inference job")
    action.add_argument("--job-status", metavar="JOB_ID")
    action.add_argument("--job-result", metavar="JOB_ID")

    parser.add_argument("--wait", action="store_true", help="wait for a submitted job to finish")
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--print-result", action="store_true", help="print the full YoloDetectResult JSON")

    # Inference I/O
    parser.add_argument("--input-jpg-path")
    parser.add_argument("--output-json-path")

    # Model
    parser.add_argument("--model-name", default="yolo11l-seg.pt")
    parser.add_argument("--cuda-device", type=int, default=0, help="-1=CPU, 0+=CUDA device")

    # Inference
    parser.add_argument("--size-mode", choices=("resize", "tiling"), default="tiling")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)

    # Tiling
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--tile-overlap", type=int, default=416)
    parser.add_argument("--tile-batch-size", type=int, default=4)

    # Masks / polygons
    parser.add_argument("--include-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--merge-tiled-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-merge-iom", type=float, default=0.15)
    parser.add_argument("--polygon-epsilon", type=float, default=1.0)
    parser.add_argument("--polygon-min-area", type=float, default=1.0)

    # Optional inference ROI: x1 y1 x2 y2
    parser.add_argument(
        "--detection-bbox-xyxy",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
    )

    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )

    discovery = RedisDiscovery(args.registry) if args.registry is not None else RedisDiscovery()

    if args.role == "server":
        server_name = args.server_name or args.service
        endpoint = args.endpoint or endpoint_for(
            args.backend,
            args.transport,
            server_name,
            args.host,
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
        )
    elif args.role == "client":
        run_client(args, discovery)
    else:
        show_services(discovery)


if __name__ == "__main__":
    main()
