from __future__ import annotations

from npb_rpc import NngRpcClient

try:
    from msg import *
except Exception:    
    from .msg import *
    

def client_camera_status(endpoint:str) -> CameraStatusResponse:
    with NngRpcClient.connect(endpoint) as client:
        return client.call("camera.status", EmptyRequest(), CameraStatusResponse)


def client_camera_frame(request: CameraFrameRequest, endpoint:str) -> CameraFrameResponse:
    with NngRpcClient.connect(endpoint) as client:
        return client.call("camera.get_frame", request, CameraFrameResponse)


def client_camera_frame_set(request: CameraFrameSetRequest, endpoint:str) -> CameraFrameSetResponse:
    with NngRpcClient.connect(endpoint) as client:
        return client.call("camera.frame", request, CameraFrameSetResponse)
