# DepthAI camera RPC with discovery + runtime device control

The camera service advertises itself through `FilesystemDiscovery`, supports NNG
or ZeroMQ, and can now open, close, or switch the DepthAI device while the RPC
service keeps running.

A device may be selected by DeviceID, PoE IP address, or USB path. An empty
device string uses DepthAI automatic device selection.

## Start the RPC service with the camera closed

```bash
python cli.py server --service camera --server-name camera-a --no-auto-open
```

Open a specific PoE camera later:

```bash
python cli.py client --service camera --server-name camera-a \
  --open-camera --device 169.254.1.222
```

Close that camera without stopping the RPC server:

```bash
python cli.py client --service camera --server-name camera-a --close-camera
```

Open another device; this also supports switching directly while a camera is
already open:

```bash
python cli.py client --service camera --server-name camera-a \
  --open-camera --device 169.254.1.223
```

## Auto-open a specific device when the service starts

Backward-compatible auto-open remains the default. To bind the camera service to
a specific PoE device immediately:

```bash
python cli.py server --service camera --server-name camera-a \
  --device 169.254.1.222
```

Omit `--device` to let DepthAI select a device automatically.

## Python RPC API

```python
from npb_rpc import FilesystemDiscovery
from your_camera_package import (
    CameraCloseRequest,
    CameraOpenRequest,
    client_camera_close,
    client_camera_open,
)

discovery = FilesystemDiscovery()

opened = client_camera_open(
    CameraOpenRequest(device="169.254.1.222", timeout_s=10.0),
    discovery=discovery,
    service="camera",
    server_name="camera-a",
)
print(opened)

closed = client_camera_close(
    CameraCloseRequest(timeout_s=5.0),
    discovery=discovery,
    service="camera",
    server_name="camera-a",
)
print(closed)
```

The RPC methods are `camera.open` and `camera.close`. `camera.open` waits up to
`timeout_s` for the selected device to become online. `camera.close` waits up to
`timeout_s` for the active DepthAI session to finish teardown. A close request
clears cached images; a switch to a different device also clears cached images.


## Camera status

`camera.status` now returns the configured device target and separates the desired
state from the current connection state:

```text
requested_open=True
online=True
device='169.254.1.222'
generation=1
restart_count=0
frames_published=...
last_frame_ns=...
error=''
```

`requested_open=True` means the supervisor should keep the camera open/reconnecting.
`online=True` means a DepthAI session is connected right now. The supervisor's
`status()` method returns `CameraStatusResponse` directly rather than an positional
tuple.

## Existing discovery/client examples

List healthy services/instances:

```bash
python cli.py list
```

Fetch the six-image frame set from any discovered instance:

```bash
python cli.py client --service camera
```

Use the legacy one-frame path:

```bash
python cli.py client --service camera --server-name camera-a \
  --single --stream rgb --thumbnail
```

Bypass discovery and connect directly:

```bash
python cli.py client --endpoint ipc:///path/to/socket --backend nng
```

Use a custom registry path by adding `--registry /path/to/registry`. When a
server binds to a wildcard/private RPC address but clients need a different RPC
address, use `--advertise-endpoint` as before.
