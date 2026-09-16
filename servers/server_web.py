from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from npb_rpc import FilesystemDiscovery

from servers.msg.dai import CameraInterface
from servers.server_dai.interface import add_camera_routes
from servers.msg.yolo import YoloInterface
from servers.server_yolo.interface import add_yolo_routes
from servers.msg.pcd import PcdInterface
from servers.server_pcd.interface import add_pcd_routes

app = FastAPI(title="Discovered RPC Web API")
_registry = os.getenv("RPC_REGISTRY") or os.getenv("CAMERA_REGISTRY")
DISCOVERY = FilesystemDiscovery(Path(_registry) if _registry else None)
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
    pass


def debug_get_file(path: str):
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}",
        )
    return FileResponse(file_path)


def debug_yolo():
    return FileResponse(
        "yolo_debug.html",
        media_type="text/html",
    )

if __name__ == "__main__":
    refresh_routes()
    uvicorn.run(app, host="0.0.0.0", port=8000)
