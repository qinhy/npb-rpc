from pathlib import Path
import io, os, zipfile

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from discovery_sum import SumRequest, SumResponse
from npb_rpc import FilesystemDiscovery, NngRpcClient, ZmqRpcClient
from server_dai import *

app = FastAPI(title="Discovered RPC Web API")
_registry = os.getenv("RPC_REGISTRY") or os.getenv("CAMERA_REGISTRY")
DISCOVERY = FilesystemDiscovery(Path(_registry) if _registry else None)
DYNAMIC_PREFIX = "dynamic:"


def rpc(service: str, name: str):
    return dict(discovery=DISCOVERY, service=service, server_name=name)


def rpc_call(service: str, name: str, method: str, request, response):
    matches = [x for x in DISCOVERY.list_instances(service) if x.instance_id == name]
    if not matches:
        raise HTTPException(404, f"server {service}/{name} not found")
    if len(matches) > 1:
        raise HTTPException(409, f"server {service}/{name} is ambiguous")

    instance = matches[0]
    client_type = ZmqRpcClient if instance.backend == "zmq" else NngRpcClient
    try:
        with client_type.connect(instance.endpoint) as client:
            return client.call(method, request, response)
    except Exception as e:
        raise HTTPException(502, f"RPC failed: {e}") from e


def add_camera_routes(service: str, name: str):
    target = rpc(service, name)
    base = f"/{service}/{name}"

    def open_camera(device: str = "169.254.1.222"):
        return client_camera_open(CameraOpenRequest(device=device), **target)

    def close_camera():
        return client_camera_close(**target)

    def status():
        return client_camera_status(**target)

    def frame(stream: str = "rgb", thumbnail: bool = False):
        r = client_camera_frame(
            CameraFrameRequest(stream=stream, thumbnail=thumbnail), **target
        )
        if not r.ok:
            raise HTTPException(503, r.error)
        return Response(r.jpeg.tobytes(), media_type="image/jpeg")

    def frames():
        r = client_camera_frame_set(CameraFrameSetRequest(), **target)
        images = {
            "rgb.jpg": r.rgb, "left.jpg": r.left, "right.jpg": r.right,
            "rgb_thumbnail.jpg": r.rgb_thumbnail,
            "left_thumbnail.jpg": r.left_thumbnail,
            "right_thumbnail.jpg": r.right_thumbnail,
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for filename, image in images.items():
                if image.size:
                    z.writestr(filename, image.tobytes())
        return Response(buf.getvalue(), media_type="application/zip")

    for method, action, func in (
        ("POST", "open", open_camera),
        ("POST", "close", close_camera),
        ("GET", "status", status),
        ("GET", "frame", frame),
        ("GET", "frames", frames),
    ):
        app.add_api_route(
            f"{base}/{action}", func, methods=[method], tags=[f"{service}:{name}"],
            name=f"{DYNAMIC_PREFIX}{service}:{name}:{action}",
        )


def add_sum_routes(service: str, name: str):
    base = f"/{service}/{name}"

    def array_sum(body: SumRequest):
        r = rpc_call(
            service, name, "array.sum",
            SumRequest(values=np.asarray(body.values, dtype=np.float32)),
            SumResponse,
        )
        return {"total": r.total}

    app.add_api_route(
        f"{base}/sum", array_sum, methods=["POST"], tags=[f"{service}:{name}"],
        name=f"{DYNAMIC_PREFIX}{service}:{name}:sum",
    )


# service name -> HTTP route builder
SERVICE_BUILDERS = {
    "camera": add_camera_routes,
    "sum": add_sum_routes,
}


def refresh_routes():
    app.router.routes[:] = [
        r for r in app.router.routes
        if not getattr(r, "name", "").startswith(DYNAMIC_PREFIX)
    ]

    loaded, skipped = {}, {}
    for service in DISCOVERY.list_services():
        names = [x.instance_id for x in DISCOVERY.list_instances(service)]
        builder = SERVICE_BUILDERS.get(service)
        if builder:
            for name in names:
                builder(service, name)
            loaded[service] = names
        else:
            skipped[service] = names

    app.openapi_schema = None
    return {"loaded": loaded, "unsupported": skipped}


@app.get("/refresh")
def refresh():
    return refresh_routes()


refresh_routes()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
