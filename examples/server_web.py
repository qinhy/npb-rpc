from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response

from discovery_sum import SumRequest, SumResponse
from npb_rpc import FilesystemDiscovery

from server_dai.interface import ApiMethod, CAMERA_API
from server_dai.msg import (
    CameraCloseRequest,
    CameraFrameRequest,
    CameraFrameSetRequest,
    CameraOpenRequest,
    EmptyRequest,
)
from server_dai.rpcapi import RpcTarget


app = FastAPI(title="Discovered RPC Web API")

_registry = os.getenv("RPC_REGISTRY") or os.getenv("CAMERA_REGISTRY")
DISCOVERY = FilesystemDiscovery(Path(_registry) if _registry else None)

DYNAMIC_PREFIX = "dynamic:"


def rpc_target(service: str, name: str) -> RpcTarget:
    """Create one reusable discovered RPC destination."""
    return RpcTarget(
        discovery=DISCOVERY,
        service=service,
        server_name=name,
    )


def add_camera_routes(service: str, name: str) -> None:
    """Expose the central camera API contract through FastAPI."""
    target = rpc_target(service, name)
    base = f"/{service}/{name}"
    tag = f"{service}:{name}"

    def expose(api_method: ApiMethod):
        """Bind a Python HTTP adapter to the HTTP metadata in interface.py."""
        web = api_method.web
        if web is None:
            raise ValueError(f"{api_method.rpc!r} has no HTTP exposure")

        responses = None
        if web.response == "jpeg":
            responses = {200: {"content": {"image/jpeg": {}}}}
        elif web.response == "zip":
            responses = {200: {"content": {"application/zip": {}}}}

        def decorator(func: Callable):
            app.add_api_route(
                f"{base}/{web.path}",
                func,
                methods=[web.method],
                tags=[tag],
                name=f"{DYNAMIC_PREFIX}{service}:{name}:{web.path}",
                responses=responses,
            )
            return func

        return decorator

    def call(api_method: ApiMethod, request):
        """Translate RPC/discovery failures into HTTP gateway errors."""
        try:
            return target.call(api_method, request)
        except RuntimeError as exc:
            message = str(exc)
            if "was not found" in message:
                raise HTTPException(404, message) from exc
            if "ambiguous" in message:
                raise HTTPException(409, message) from exc
            raise HTTPException(502, f"RPC failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(502, f"RPC failed: {exc}") from exc

    @expose(CAMERA_API.open)
    def open_camera(device: str = "169.254.1.222"):
        return call(
            CAMERA_API.open,
            CameraOpenRequest(device=device),
        )

    @expose(CAMERA_API.close)
    def close_camera():
        return call(
            CAMERA_API.close,
            CameraCloseRequest(),
        )

    @expose(CAMERA_API.status)
    def status():
        return call(
            CAMERA_API.status,
            EmptyRequest(),
        )

    @expose(CAMERA_API.get_frame)
    def frame(stream: str = "rgb", thumbnail: bool = False):
        result = call(
            CAMERA_API.get_frame,
            CameraFrameRequest(
                stream=stream,
                thumbnail=thumbnail,
            ),
        )
        if not result.ok:
            raise HTTPException(503, result.error)

        return Response(
            result.jpeg.tobytes(),
            media_type="image/jpeg",
        )

    @expose(CAMERA_API.frame)
    def frames():
        result = call(
            CAMERA_API.frame,
            CameraFrameSetRequest(),
        )

        images = {
            "rgb.jpg": result.rgb,
            "left.jpg": result.left,
            "right.jpg": result.right,
            "rgb_thumbnail.jpg": result.rgb_thumbnail,
            "left_thumbnail.jpg": result.left_thumbnail,
            "right_thumbnail.jpg": result.right_thumbnail,
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for filename, image in images.items():
                if image.size:
                    archive.writestr(filename, image.tobytes())

        return Response(
            buf.getvalue(),
            media_type="application/zip",
        )

    @expose(CAMERA_API.get_calib)
    def get_calib():
        return call(
            CAMERA_API.get_calib,
            EmptyRequest(),
        )


def add_sum_routes(service: str, name: str) -> None:
    """Existing sum demo, now reusing the common RpcTarget transport logic."""
    target = rpc_target(service, name)
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


# Discovery service name -> HTTP route builder.
SERVICE_BUILDERS = {
    CAMERA_API.service: add_camera_routes,
    "sum": add_sum_routes,
}


def refresh_routes():
    # Remove only dynamically generated service-instance routes.
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not getattr(route, "name", "").startswith(DYNAMIC_PREFIX)
    ]

    loaded: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}

    for service in DISCOVERY.list_services():
        names = [
            instance.instance_id
            for instance in DISCOVERY.list_instances(service)
        ]
        builder = SERVICE_BUILDERS.get(service)

        if builder is None:
            skipped[service] = names
            continue

        for name in names:
            builder(service, name)
        loaded[service] = names

    # Dynamic routes changed, so FastAPI must regenerate OpenAPI.
    app.openapi_schema = None
    return {
        "loaded": loaded,
        "unsupported": skipped,
    }


@app.get("/refresh")
def refresh():
    return refresh_routes()


refresh_routes()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
