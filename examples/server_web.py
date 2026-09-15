from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from discovery_sum import SumRequest, SumResponse
from npb_rpc import FilesystemDiscovery
from server_dai.interface  import CameraInterface, RpcTarget, add_camera_routes
from server_yolo.interface import YoloInterface, add_yolo_routes
from server_pcd.interface  import PcdInterface, add_pcd_routes

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


def add_sum_routes(service: str, name: str) -> None:
    """Legacy sum service until discovery_sum exposes the same unified interface."""
    target = RpcTarget(discovery=DISCOVERY, service=service, server_name=name)
    base = f"/{service}/{name}"

    def array_sum(body: SumRequest):
        try:
            result = target.call_raw(
                "array.sum",
                SumRequest(values=np.asarray(body.values, dtype=np.float32)),
                SumResponse,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "was not found" in message:
                raise HTTPException(404, message) from exc
            if "ambiguous" in message:
                raise HTTPException(409, message) from exc
            raise HTTPException(502, f"RPC failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(502, f"RPC failed: {exc}") from exc
        return {"total": result.total}

    app.add_api_route(
        f"{base}/sum",
        array_sum,
        methods=["POST"],
        tags=[f"{service}:{name}"],
        name=f"{DYNAMIC_PREFIX}{service}:{name}:sum",
    )


SERVICE_BUILDERS = {
    CameraInterface.service: add_camera_service_routes,
    YoloInterface.service: add_yolo_service_routes,
    PcdInterface.service: add_pcd_service_routes,
    "sum": add_sum_routes,
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


refresh_routes()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
