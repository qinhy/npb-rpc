from dataclasses import dataclass
import json
from pathlib import Path
import time

from rich.console import Console

from npb_rpc import RedisDiscovery
from servers.msg.dai import (
    CameraCalibrationResponse,
    CameraFrameSetResponse,
    EmptyRequest,
    CameraOpenRequest,
    CameraCloseRequest,
    CameraFrameSetRequest,
    CameraInterface, CameraClient
)
from servers.msg.yolo import YoloInferenceRequest, YoloJobRequest, YoloInterface, YoloClient
from servers.msg.pcd import PcdBuildRequest, PcdJobRequest, PcdInterface, PcdClient, PcdBackend

from servers.store.custom_record_store import CustomStore, PCDRecord
from servers.logger import logging

LOG = logging.getLogger(__name__.replace(".",":"))
console = Console()


def rprint(tag: str, msg: str, style: str = "cyan") -> None:
    console.print(f"[{style}]{tag:<6}[/] {msg}")


@dataclass(slots=True)
class CameraPipelineConfig:
    camera_id: str
    camera_ip: str

    need_yolo: bool = False
    need_pcd: bool = False

    yolo_stream: str = "rgb"
    pcd_backend: PcdBackend = "cpu"
    pcd_max_depth_m: float = 2.0

    calib: CameraCalibrationResponse | None = None
    
    cli: CameraInterface | None = None
    yolo: YoloInterface | None = None
    pcd: PcdInterface | None = None

    fs: CameraFrameSetResponse | None = None

    def __post_init__(self) -> None:
        discovery=RedisDiscovery()
        self.cli:CameraInterface = CameraClient(
            discovery=discovery,server_name=self.camera_id)
        self.yolo:YoloInterface = YoloClient(discovery=discovery,server_name="yolo")
        self.pcd:PcdInterface = PcdClient(discovery=discovery,server_name="pcd")


RGBD_left = CameraPipelineConfig(
    camera_id="rgbd_left",
    camera_ip="169.254.1.221",
    need_yolo=True,
)
RGBD_right = CameraPipelineConfig(
    camera_id="rgbd_right",
    camera_ip="169.254.1.222",
    need_yolo=True,
)
RGBD_hand = CameraPipelineConfig(
    camera_id="rgbd_hand",
    camera_ip="169.254.1.222",
    need_yolo=True,
    need_pcd=True,
)

def close_cams(cams:list[CameraPipelineConfig]=[
    RGBD_left,RGBD_right,
    RGBD_hand
]):
    results = [cam.cli.close(CameraCloseRequest()) for cam in cams]
    rprint("CLOSE", ", ".join(cam.camera_id for cam in cams), "green")
    return results

def open_cams(cams:list[CameraPipelineConfig]=[RGBD_left,RGBD_right]):
    for cam in cams:
        res = cam.cli.status(EmptyRequest())
        while not res.online:
            try:
                res = cam.cli.open(CameraOpenRequest(device=cam.camera_ip))
            except Exception as e:
                console.print("[red]OPEN[/]", cam.camera_id, e)
        cam.calib = cam.cli.get_calib(EmptyRequest())
        rprint("OPEN", f"{cam.camera_id} @ {cam.camera_ip}", "green")
    time.sleep(1)

def open_dual_rgb():open_cams(cams=[RGBD_left,RGBD_right])
def open_hand():open_cams(cams=[RGBD_hand])

def capture_cams(store:CustomStore,
        cams:list[CameraPipelineConfig]=[
            RGBD_left,RGBD_right,
            RGBD_hand
    ]):
    # store = CustomStore(root_path=Path("../recordings/").absolute())
    mode="dual_rgb" if len(cams)>1 else "rgbd_hand"
    timestamp_ns_utc=time.time_ns()
    record = store.add_record(mode=mode,timestamp_ns_utc=timestamp_ns_utc)
    rprint("CAP", f"{mode} | {', '.join(cam.camera_id for cam in cams)}", "cyan")
    for cam in cams:
        cam.fs = cam.cli.frames(CameraFrameSetRequest())

    for cam in cams:
        camera_id = cam.camera_id
        fs = cam.fs
        record.add_mjpeg_image(camera_id=camera_id,stream="rgb",image_bytes=fs.rgb.tobytes())
        record.add_mjpeg_image(camera_id=camera_id,stream="left",image_bytes=fs.left.tobytes())
        record.add_mjpeg_image(camera_id=camera_id,stream="right",image_bytes=fs.right.tobytes())
        if cam.calib is None:
            cam.calib = cam.cli.get_calib(EmptyRequest())
        calib_path = record.add_calibration(camera_id=camera_id,data=json.loads(cam.calib.model_dump_json()))
        cam_rec = record.get_camera(camera_id)

        if cam.need_yolo:
            stream = "rgb"
            yolo_rec = record.add_yolo(camera_id=camera_id,stream=stream,data={})
            yolo_res = cam.yolo.inference(YoloInferenceRequest(
                input_jpg_path=str(cam_rec.expected_image_path(stream)),
                output_json_path=str(yolo_rec.expected_data_path()),
            ))
        
        if cam.need_pcd:
            if cam.need_yolo:
                pcd_rec = PCDRecord(parent=record, source_name=camera_id, kind="folder")
                # time.sleep(1)
                with console.status(f"[cyan]YOLO[/] {camera_id}", spinner="dots"):
                    while yolo_res.state not in ["succeeded","failed","cancelled"]:
                        yolo_res = cam.yolo.job_status(YoloJobRequest(
                            job_id=yolo_res.job_id
                        ))
                        time.sleep(0.01)
                rprint("YOLO", f"{camera_id} | {yolo_res.state}",
                       "green" if yolo_res.state == "succeeded" else "red")
                
                pcd_res = cam.pcd.build(PcdBuildRequest(
                    backend=cam.pcd_backend,
                    rgb_jpg_path=str(cam_rec.expected_rgb_path()),
                    left_jpg_path=str(cam_rec.expected_left_path()),
                    right_jpg_path=str(cam_rec.expected_right_path()),
                    calibration_json_path=str(calib_path),
                    output_pcd_path=str(pcd_rec.expected_full_pcd_path()),
                    max_depth_m=cam.pcd_max_depth_m,
                    detections_json_path=str(yolo_rec.expected_data_path()),
                    segments_output_dir=str(pcd_rec.expected_full_pcd_path().parent)
                ))
                
                with console.status(f"[cyan]PCD[/]  {camera_id}", spinner="dots"):
                    while pcd_res.state not in ["succeeded","failed","cancelled"]:
                        pcd_res = cam.pcd.job_status(PcdJobRequest(
                            job_id=pcd_res.job_id
                        ))
                        time.sleep(0.01)
                rprint("PCD", f"{camera_id} | {pcd_res.state}",
                       "green" if pcd_res.state == "succeeded" else "red")
            else:
                pass
    

    
if __name__=="__main__":
    # root="D:/github/resultkit/recording/dual_rgb/2026-09-07/field_all/122825.590133625JST/"
    # yolo_args = YoloInferenceRequest(
    #     input_jpg_path=root+"imgs/rgbd_left/rgb.jpg",
    #     output_json_path=root+"imgs/rgbd_left/rgb.json",
    #     cuda_device=0,
    #     tile_batch_size=4,
    #     detection_bbox_xyxy=[864,864,3008,3008],
    # )
    # res = method_post("yolo","yolo","inference",json.loads(yolo_args.model_dump_json()))

    # pcd_args = PcdBuildRequest(
    #     rgb_jpg_path=root+"imgs/rgbd_left/rgb.jpg",
    #     left_jpg_path=root+"imgs/rgbd_left/left.jpg",
    #     right_jpg_path=root+"imgs/rgbd_left/right.jpg",
    #     calibration_json_path=root+"calib/rgbd_left.json",
    #     output_pcd_path=root+"imgs/rgbd_left/rgb.pcd",
    # )
    # res = method_post("pcd","pcd","build",json.loads(pcd_args.model_dump_json()))
    store = CustomStore(root_path=Path("./recordings/").absolute())
    for i in range(10):
        open_hand()
        for i in range(10):
            capture_cams(store,[RGBD_hand])
        close_cams()
        time.sleep(20)
        open_dual_rgb()
        for i in range(10):
            capture_cams(store,[RGBD_left,RGBD_right,])
        close_cams()
        time.sleep(20)
    pass




















