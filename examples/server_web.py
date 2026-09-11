from server_dai import (EmptyRequest,
CameraStatusResponse,
CameraFrameRequest,
CameraFrameResponse,
CameraFrameSetRequest,
CameraFrameSetResponse,

client_camera_status,
client_camera_frame,
client_camera_frame_set)

import uvicorn
from fastapi import FastAPI
app = FastAPI()

@app.post("/camera_status")
def camera_status(endpoint:str) -> CameraStatusResponse:
    return client_camera_status(endpoint)

@app.post("/camera_frame")
def camera_frame(request: CameraFrameRequest) -> CameraStatusResponse:
    return client_camera_frame(request, endpoint)

uvicorn.run(app)