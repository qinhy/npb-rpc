from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Literal

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from npb_rpc import RedisDiscovery

from npb_rpc._event import RpcEvent
from servers.msg.dai import CameraFrameSetRequest, CameraInterface
from servers.server_dai.interface import add_camera_routes
from servers.msg.yolo import YoloClient, YoloInferenceRequest, YoloInterface, JobRequest, EmptyRequest
from servers.server_rpc import (STORE, CameraPipelineConfig, RGBD_hand, RGBD_left, RGBD_right, console,
                                close_cams, close_rgbd_hand, close_rgbd_left, close_rgbd_right, get_db_root,
                                open_dual_rgb, open_hand, open_rgbd_hand, open_rgbd_left, open_rgbd_right, rprint,
                                status_rgbd_hand, status_rgbd_left, status_rgbd_right)
from servers.server_yolo.interface import add_yolo_routes
from servers.msg.pcd import PcdBackend, PcdBuildRequest, PcdInterface
from servers.server_pcd.interface import add_pcd_routes
from servers.server_yolo.yolo_utils import is_official_model_name
from servers.store.custom_record_store import CustomStore, PCDRecord
from servers.store.fs_nosql import FileSystemDB

app = FastAPI(title="Discovered RPC Web API")
DISCOVERY = RedisDiscovery()
DYNAMIC_PREFIX = "dynamic:"


def add_camera_service_routes(service: str, name: str) -> None:
    add_camera_routes(
        app,
        discovery=DISCOVERY,
        service=service,
        server_name=name,
        route_name_prefix=DYNAMIC_PREFIX,
    )


def add_yolo_service_routes(service: str, name: str) -> None:
    add_yolo_routes(
        app,
        discovery=DISCOVERY,
        service=service,
        server_name=name,
        route_name_prefix=DYNAMIC_PREFIX,
    )
    

def add_pcd_service_routes(service: str, name: str) -> None:
    add_pcd_routes(
        app,
        discovery=DISCOVERY,
        service=service,
        server_name=name,
        route_name_prefix=DYNAMIC_PREFIX,
    )


SERVICE_BUILDERS = {
    CameraInterface.service: add_camera_service_routes,
    YoloInterface.service: add_yolo_service_routes,
    PcdInterface.service: add_pcd_service_routes,
}


def refresh_routes():
    app.router.routes[:] = [
        route for route in app.router.routes
        if not getattr(route, "name", "").startswith(DYNAMIC_PREFIX)
    ]

    loaded: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}
    for service in DISCOVERY.list_services():
        names = [instance.instance_id for instance in DISCOVERY.list_instances(service)]
        builder = SERVICE_BUILDERS.get(service)
        if builder is None:
            skipped[service] = names
            continue
        for name in names:
            builder(service, name)
        loaded[service] = names

    app.openapi_schema = None
    return {"loaded": loaded, "unsupported": skipped}


@app.get("/refresh")
def refresh():
    return refresh_routes()


def last_ai_record()->List[Dict]:
    yolo:YoloInterface = YoloClient(discovery=RedisDiscovery(),server_name="yolo")
    yolo_st = yolo.status(EmptyRequest())
    yolo_rec = yolo.job_result(JobRequest(job_id=yolo_st.last_job_id))
    return [json.loads(yolo_rec.result.model_dump_json())]


def debug_get_file(path: str):
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}",
        )
    return FileResponse(file_path)


def debug_yolo():
    return FileResponse("yolo_debug.html",media_type="text/html")

def debug_cams():
    return FileResponse("cams_debug.html",media_type="text/html")

GLOBAL_yolo_config = YoloInferenceRequest(
    model_name="yolo11l-seg.pt",
    tile_batch_size=6,
    tile_overlap=416,
    
    imgsz=1280, confidence=0.25, iou=0.45,
    max_detections=100,
    input_jpg_path="null",
    output_json_path="null",
)
def yolo_set_config(config:dict=GLOBAL_yolo_config.model_dump()):
    if "model_name" in config:
        GLOBAL_yolo_config.model_name=config["model_name"]
    if "tile_batch_size" in config:
        GLOBAL_yolo_config.tile_batch_size=config["tile_batch_size"]
    if "confidence" in config:
        GLOBAL_yolo_config.confidence=config["confidence"]
    if "tile_overlap" in config:
        GLOBAL_yolo_config.tile_overlap=config["tile_overlap"]
    return GLOBAL_yolo_config

GLOBAL_pcd_config = PcdBuildRequest(
    backend="dnn",
    max_depth_m=2.0,
    
    rgb_jpg_path="null",
    left_jpg_path="null",
    right_jpg_path="null",
    calibration_json_path="null",
    output_pcd_path="null",
)
def pcd_set_config(config:dict=GLOBAL_pcd_config.model_dump()):
    if "backend" in config:
        if config["backend"]=="sgbm":
            config["backend"]="cpu"
        GLOBAL_pcd_config.backend=PcdBackend.from_str(config["backend"])
    if "max_depth_m" in config:
        GLOBAL_pcd_config.max_depth_m=config["max_depth_m"]
    return GLOBAL_pcd_config

def capture_cams(store:CustomStore=STORE,
        cams:list[CameraPipelineConfig]=[
            RGBD_left,RGBD_right,
            RGBD_hand
    ],
    params={}):
    # store = CustomStore(root_path=Path("../recordings/").absolute())
    mode="dual_rgb" if len(cams)>1 else "rgbd_hand"
    timestamp_ns_utc=time.time_ns()
    record = store.add_record(mode=mode,timestamp_ns_utc=timestamp_ns_utc)
    rprint("CAP", f"{mode} | {', '.join(cam.camera_id for cam in cams)}", "cyan")
    for cam in cams:
        t0 = time.perf_counter()
        cam.fs = cam.cli.frames(CameraFrameSetRequest())
        cam.cam_cap_ms = (time.perf_counter() - t0) * 1000.0
        
    if "meta" in params:
        if "gnss" in params["meta"]:
            record.add_gnss(params["meta"]["gnss"])
        if "arm" in params["meta"]:
            arm = record.get_arm()
            arm.add_result(run_id=params["meta"]["arm"]["run_id"],
                        data=params["meta"]["arm"]["data"])

    yolo_jobs = []
    for cam in cams:
        t0 = time.perf_counter()

        camera_id,fs = cam.camera_id,cam.fs
        record.add_mjpeg_image(camera_id=camera_id,stream="rgb",image_bytes=fs.rgb.tobytes())
        record.add_mjpeg_image(camera_id=camera_id,stream="left",image_bytes=fs.left.tobytes())
        record.add_mjpeg_image(camera_id=camera_id,stream="right",image_bytes=fs.right.tobytes())
        if cam.calib is None:
            cam.calib = cam.cli.get_calib(EmptyRequest())
        calib_path = record.add_calibration(camera_id=camera_id,data=json.loads(cam.calib.model_dump_json()))
        cam_rec = record.get_camera(camera_id)

        cam.cam_save_ms = (time.perf_counter() - t0) * 1000.0

        cam_rec.add_meta({"cam_open_ms":cam.cam_open_ms,
                            "cam_cap_ms":cam.cam_cap_ms,
                            "cam_save_ms":cam.cam_save_ms})

        if cam.need_yolo:
            stream = "rgb"
            yolo_rec = record.add_yolo(camera_id=camera_id,stream=stream,data={})
            yolo_job = cam.yolo.inference(YoloInferenceRequest(
                model_name=GLOBAL_yolo_config.model_name,
                confidence=GLOBAL_yolo_config.confidence,
                size_mode="tiling",
                imgsz=1280,
                iou=0.45,
                max_detections=100,
                tile_overlap=416,
                tile_batch_size=6,
                detection_bbox_xyxy=cam.detection_bbox_xyxy,
                
                input_jpg_path=str(cam_rec.expected_image_path(stream)),
                output_json_path=str(yolo_rec.expected_data_path()),
                done_event=RpcEvent.create(),
            ))
            yolo_jobs.append({"done_event":yolo_job.done_event.model_dump()})
        
        if cam.need_pcd and cam.need_yolo:
            pcd_rec = PCDRecord(parent=record, source_name=camera_id, kind="folder")
            
            with console.status(f"[cyan]YOLO[/] {camera_id}", spinner="dots"):
                yolo_job.done_event.wait()
            yolo_job.done_event.delete()
            yolo_job = cam.yolo.job_status(JobRequest(job_id=yolo_job.job_id))
            rprint("YOLO", f"{camera_id} | {yolo_job.state}",
                    "green" if yolo_job.state == "succeeded" else "red")

            pcd_res = cam.pcd.build(PcdBuildRequest(
                backend=GLOBAL_pcd_config.backend,
                max_depth_m=cam.pcd_max_depth_m,

                rgb_jpg_path=str(cam_rec.expected_rgb_path()),
                left_jpg_path=str(cam_rec.expected_left_path()),
                right_jpg_path=str(cam_rec.expected_right_path()),
                calibration_json_path=str(calib_path),
                output_pcd_path=str(pcd_rec.expected_full_pcd_path()),
                detections_json_path=str(yolo_rec.expected_data_path()),
                segments_output_dir=str(pcd_rec.expected_full_pcd_path().parent),
                done_event=RpcEvent.create(),
            ))
            with console.status(f"[cyan]PCD[/]  {camera_id}", spinner="dots"):
                pcd_res.done_event.wait()
            pcd_res.done_event.delete()
            pcd_res = cam.pcd.job_status(JobRequest(job_id=pcd_res.job_id))
            rprint("PCD", f"{camera_id} | {pcd_res.state}",
                    "green" if pcd_res.state == "succeeded" else "red")
        else:            
            rprint("YOLO", f"{camera_id} | {yolo_job.state}",
                    "green" if yolo_job.state == "succeeded" else "red")

    id = f"{record.date_jst}:{record.field_id}:{record.record_id}"
    return {"db_name":record.mode,"_id":id,"yolo_jobs":yolo_jobs}
    
def capture_hand(params:dict={"meta": {
                        # "gnss":{"the_data":"xxxxxxxxx"},
                        # "arm":{"run_id":"UUIDXXXX","data":"xxxxxxxxx"}
                }}):return capture_cams(cams=[RGBD_hand],params=params)
def capture_dual_rgb(params:dict={"meta": {
                        # "gnss":{"the_data":"xxxxxxxxx"},
                        # "arm":{"run_id":"UUIDXXXX","data":"xxxxxxxxx"}
                }}):return capture_cams(cams=[RGBD_left,RGBD_right,],params=params)

app.add_api_route("/debug/last_ai_record",
    last_ai_record,methods=["GET"], name="debug", tags=["debug"],)

app.add_api_route("/debug/get_file",
    debug_get_file, methods=["GET"], name="debug", tags=["debug"])

app.add_api_route("/debug/yolo",
    debug_yolo,methods=["GET"],tags=["debug"],)
app.add_api_route("/debug/cams",
    debug_cams,methods=["GET"],tags=["debug"],)

app.add_api_route("/close_cams",
    close_cams,methods=["GET"],tags=["release"],)

app.add_api_route("/open_hand",
    open_hand,methods=["POST"],tags=["release"],)

app.add_api_route("/open_dual_rgb",
    open_dual_rgb,methods=["POST"],tags=["release"],)

app.add_api_route("/capture_hand",
    capture_hand,methods=["POST"],tags=["release"],)

app.add_api_route("/capture_dual_rgb",
    capture_dual_rgb,methods=["POST"],tags=["release"],)


DBName = Literal["dual_rgb", "rgbd_hand"]

def get_db(db_name: DBName) -> FileSystemDB:
    root = get_db_root()
    return FileSystemDB(root=root / db_name)

def not_found(reason="missing"):
    return JSONResponse(status_code=404, content={"error": "not_found", "reason": reason})

class FindRequest(BaseModel):
    selector: dict[str, Any] = {
                # "_id": "2026-09-29:field_all:090000.000000000JST:yolo:camera_front:rgb"
                "_id": {
                    "$gte": "2026-09-29:field_all:090000.000000000JST",
                    "$lt":  "2026-09-29:field_all:170000.000000000JST",
                    "$regex": ":yolo:",
                },
                "detections": {
                    "$elemMatch": {
                        "class_name": "tie",
                        "confidence": {"$gte": 0.01}
                    }
                }
            }
    fields: list[str] = ["_id","_attachments"]    
    # limit: int = 25
    # skip: int = 0
    section:Literal["null","gnss"]="null"

def db_find(db_name: DBName, req: FindRequest):
    db = get_db(db_name)
    docs = list(db.find(req.selector, fields=req.fields))
    to_section_id = None
    if req.section == "gnss":
        to_section_id = lambda id:id.split("JST:")[0]+"JST:gnss:baselink"
        neighbors = set([to_section_id(r["_id"]) for r in docs])
        neighbors = sorted(list(neighbors))
        neighbors = [db.get(sct_id) for sct_id in neighbors]
        docs = [r for r in neighbors if r is not None]
    return docs
    # return {"docs": docs[req.skip:req.skip + req.limit]}

def db_get(db_name: DBName, doc_id: str):
    doc = get_db(db_name).get(doc_id)
    return doc if doc is not None else not_found()

def db_get_attachment(db_name: DBName, doc_id: str, attachment_name: str):
    db = get_db(db_name)
    doc = db.get(doc_id)
    if doc is None: return not_found()
    att = doc.get("_attachments", {}).get(attachment_name)
    if att is None: return not_found()
    path = db.root / att["path"]
    return FileResponse(path) if path.is_file() else not_found()

app.add_api_route("/db/{db_name}/_find", db_find, methods=["POST"], tags=["db"])
app.add_api_route("/db/{db_name}/{doc_id}", db_get, methods=["GET"], tags=["db"])
app.add_api_route("/db/{db_name}/{doc_id}/{attachment_name}",
                  db_get_attachment, methods=["GET"], tags=["db"])

def db_add_arm(db_name: DBName, doc_id: str, run_id:str, data:dict, kind:str="arm_result"):
    db = get_db(db_name)
    doc = db.get(doc_id)
    if doc is None:return not_found()    
    path = (Path(db.root)/Path(doc_id.replace(":","/"))/"arm"
            / run_id # legacy
            /f"{run_id}_{kind}.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))
    return path

app.add_api_route("/db_record/{db_name}/{doc_id}/add_arm",
                  db_add_arm, methods=["POST"], tags=["db"])



def job_wait(event:RpcEvent):
    try:
        if event.exists():
            event.wait()
            event.delete()
    except Exception as e:
        print("warning",e)

app.add_api_route("/job/wait",
                  job_wait, methods=["POST"], tags=["db"])

# legacy supports
def do_nothing():
    return None

app.add_api_route("/controllers/rgbd_left/open", # Open left RGB-D camera |
    open_rgbd_left,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_left/close", # Close left camera |
    close_rgbd_left,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_left/status", # Check left camera status |
    status_rgbd_left,methods=["POST"],tags=["release"],)

app.add_api_route("/controllers/rgbd_right/open", # Open right RGB-D camera |
    open_rgbd_right,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_right/close", # Close right camera |
    close_rgbd_right,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_right/status", # Check right camera status |
    status_rgbd_right,methods=["POST"],tags=["release"],)

app.add_api_route("/controllers/rgbd_hand/open", # Open hand RGB-D camera |
    open_rgbd_hand,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_hand/close", # Close hand camera |
    close_rgbd_hand,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/rgbd_hand/status", # Check hand camera status |
    status_rgbd_hand,methods=["POST"],tags=["release"],)

app.add_api_route("/controllers/store_dual/capture", # Capture dual-camera record |
    capture_dual_rgb,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/store_hand/capture", # Capture hand-camera record |
    capture_hand,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/store_dual/watch", # Watch dual-camera streams |
    do_nothing,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/store_hand/watch", # Watch hand-camera stream |
    do_nothing,methods=["POST"],tags=["release"],)

app.add_api_route("/controllers/yolo/set_model", # Configure YOLO model |
    yolo_set_config,methods=["POST"],tags=["release"],)
app.add_api_route("/controllers/pcd/set_backend", # Select SGBM/DNN PCD backend |
    pcd_set_config,methods=["POST"],tags=["release"],)


def warmup(env: dict[str, str]) -> None:
    """Prime library caches before launching servers; CUDA contexts remain per-process."""
    print("[WARMUP] Loading heavy libraries...", flush=True)
    # Use the servers' environment and release GPU allocations when this child exits.
    code = """
import importlib
import time

for name in ("numpy", "cv2", "torch", "torchvision", "cupy", "ultralytics", "depthai"):
    started = time.monotonic()
    try:
        module = importlib.import_module(name)
        if name == "torch":
            device = "cuda" if module.cuda.is_available() else "cpu"
            x = module.ones((64, 64), device=device)
            result = x @ x
            if device == "cuda":
                module.cuda.synchronize()
            del x, result
        elif name == "cupy":
            x = module.ones((64, 64), dtype=module.float32)
            result = x @ x
            module.cuda.get_current_stream().synchronize()
            del x, result
        print(f"[WARMUP] {name}: {time.monotonic() - started:.2f}s", flush=True)
    except Exception as exc:
        print(f"[WARMUP] {name}: skipped ({exc})", flush=True)
"""
    try:
        subprocess.run(["uv", "run", "python", "-c", code], env=env, check=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[WARMUP] Continuing without completed warmup: {exc}", flush=True)

if __name__ == "__main__":
    try:
        import vpi # for jetson
        GLOBAL_pcd_config.backend = PcdBackend.from_str("vpi")
        rprint("PCD","use vpi")
    except Exception as e:
        GLOBAL_pcd_config.backend = PcdBackend.from_str("cuda")
        rprint("PCD","use cuda")
        pass

    all_pt = set([f.name for f in list(Path("./").rglob("*.pt"))])
    official_pt = set([pt for pt in all_pt if is_official_model_name(pt)])
    unofficial_pt = all_pt-official_pt
    try:
        GLOBAL_yolo_config.model_name = list(official_pt)[0]
    except Exception as e:
        GLOBAL_yolo_config.model_name = "yolo26n.pt"
    GLOBAL_yolo_config.confidence = 0.25
    if len(unofficial_pt)>0:
        GLOBAL_yolo_config.model_name = list(unofficial_pt)[0]
    rprint("YOLO",f"use {GLOBAL_yolo_config.model_name}")

    env = os.environ.copy()
    env["LOG_REDIS_URL"] = "redis://127.0.0.1:6379/0"
    warmup(env)
    
    refresh_ok = False
    while not refresh_ok:
        try:
            refresh_routes()
            refresh_ok = True
        except Exception as e:
            refresh_ok = False

    uvicorn.run(app, host="0.0.0.0", port=8000)
