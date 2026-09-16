# PCD RPC Server

An asynchronous RGB-stereo point-cloud service built on `npb_rpc`.

The service accepts server-local RGB, left, and right camera images plus stereo/RGB calibration, generates an RGB-colored point cloud, writes the result as a `.pcd` file, and optionally splits the cloud into per-object PCD files using an existing YOLO detection/segmentation JSON result.

## Architecture

```text
PCD Client
    │
    │ PcdBuildRequest
    ▼
RPC
NNG / ZeroMQ
    │
    ▼
PcdService
    │
    │ submit()
    ▼
PcdWorker
    │
    ├── PcdJobStore
    ├── PcdBackendCache
    ├── PcdCalibrationCache
    └── PcdCalculator
          │
          ├── read RGB / left / right
          ├── load calibration
          ├── rectify stereo
          ├── disparity prediction
          ├── disparity -> XYZ
          ├── rectified-left -> original-left
          ├── project XYZ -> RGB
          ├── attach RGB colors
          ├── write full .pcd
          │
          └── optional YOLO segmentation
                ├── object_0.pcd
                ├── object_1.pcd
                └── ...
```

The RPC call is asynchronous. `build()` returns a job ID immediately. The client can then call `job_status()` and `job_result()`.

Large point-cloud arrays are never transported through RPC. PCD data is written to server-local storage, while RPC responses contain paths, counts, status, backend information, and timing metadata.

## Project layout

```text
pcd/
├── __init__.py
├── cli.py
├── interface.py
├── msg.py
├── server.py
├── worker.py
├── pcd_calculation.py
├── disparity_predictors.py
└── matops.py
```

## PCD pipeline

```text
left.jpg ──┐
           ├─> stereo rectification
right.jpg ─┘
                 │
                 ▼
             disparity
                 │
                 ▼
          rectified XYZ
                 │
                 ▼
      original-left coordinates
                 │
                 ├───────────────┐
                 │               │
                 ▼               ▼
            RGB projection     rgb.jpg
                 │               │
                 └───────┬───────┘
                         ▼
                     XYZ + RGB
                         │
                         ▼
                      full.pcd
```

Calibration translations are converted to meters when the calibration object is loaded. The default source translation unit is `cm`, which matches the calibration data currently used by the project.

## Disparity backends

Four disparity backends are supported.

```text
cpu
    OpenCV StereoSGBM
    SGBMDisparityPredictor

cuda
    libSGM CUDA
    SGBMDisparityPredictorCuda

dnn
    Fast-FoundationStereo
    FastFoundationStereoDisparity

vpi
    NVIDIA VPI CUDA Stereo
    VPIStereoDisparityGPU
```

Select the backend per build request with:

```bash
--pcd-backend cpu
--pcd-backend cuda
--pcd-backend dnn
--pcd-backend vpi
```

`--backend` has a different meaning: it selects the RPC transport and is either `nng` or `zmq`.

### Backend caching

Disparity backends are long-lived and cached.

```text
cpu
    cache key: cpu

dnn
    cache key: backend + CUDA device

vpi
    cache key: backend + CUDA device

cuda / libSGM
    cache key: backend + CUDA device + stereo resolution
```

libSGM requires width and height when its native handle is created, so different stereo image resolutions require different cached libSGM instances.

## Requirements

The common service requires Python plus the project dependencies providing:

```text
npb
npb_rpc
pydantic
numpy
opencv-python
torch
```

Optional dependencies depend on the selected backend.

For ZeroMQ RPC:

```text
pyzmq
```

For Fast-FoundationStereo, the Fast-FoundationStereo repository and model checkpoint must be available on the PCD server.

For libSGM, the native `sgm_py` library must be built and available on the PCD server.

For the VPI backend, NVIDIA VPI Python bindings and CuPy must be available.

FastAPI is only required when `add_pcd_routes()` is used.

## Important source prerequisite

`pcd_calculation.py` should begin with:

```python
from __future__ import annotations
```

This is required because `build_rgb_indexed_cloud()` contains a `DisparityPredictor` type annotation while `DisparityPredictor` is not imported at module scope in the current source.

It is also recommended to consolidate the two existing `read_image()` definitions in `pcd_calculation.py`. The server currently uses the later definition, which supports:

```python
color="RGB"
color="BGR"
color="gray"
```

## Start the server

A normal local NNG/IPC server can be started with:

```bash
python -m pcd.cli server \
    --backend nng \
    --transport ipc \
    --server-name pcd
```

For a filesystem-rooted deployment:

```bash
python -m pcd.cli server \
    --backend nng \
    --transport ipc \
    --server-name pcd \
    --read-root ./records \
    --write-root ./records
```

`read_root` limits input images, calibration JSON, and YOLO detection JSON to the configured input tree.

`write_root` limits generated `.pcd`, result JSON, and segmented PCD output to the configured output tree.

For a remotely reachable TCP endpoint:

```bash
python -m pcd.cli server \
    --backend nng \
    --transport tcp \
    --host 0.0.0.0 \
    --server-name pcd
```

When exposing the service over a network, configuring `--read-root` and `--write-root` is strongly recommended.

## CPU build

Start the server:

```bash
python -m pcd.cli server \
    --server-name pcd \
    --read-root ./records \
    --write-root ./records
```

Submit a build:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend cpu \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --output-json-path sample/pcd.json \
    --wait \
    --print-result
```

The input and output paths are interpreted by the server, not by the client.

When roots are configured, relative paths are resolved underneath those roots.

## CUDA libSGM build

Configure the server-side native library:

```bash
python -m pcd.cli server \
    --server-name pcd \
    --libsgm-dll ./build/Release/sgm_py.dll \
    --libsgm-num-disparities 256
```

Then submit:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend cuda \
    --cuda-device 0 \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --wait
```

The libSGM backend automatically creates a cached predictor using the actual width and height of the left/right stereo images.

## Fast-FoundationStereo build

Configure the DNN backend on the server:

```bash
python -m pcd.cli server \
    --server-name pcd \
    --foundation-repo-dir ./fast-foundationstereo \
    --foundation-model-path weights/23-36-37/model_best_bp2_serialize.pth \
    --foundation-valid-iters 8 \
    --foundation-max-disp 192
```

Submit a DNN build:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend dnn \
    --cuda-device 0 \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --wait
```

The FoundationStereo model is loaded once and reused by later jobs on the same device.

## VPI build

Start a server with VPI configuration:

```bash
python -m pcd.cli server \
    --server-name pcd \
    --vpi-num-disparities 256
```

Submit:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend vpi \
    --cuda-device 0 \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --wait
```

## Point-cloud controls

The main geometry options are:

```text
--min-disparity
--min-depth-m
--max-depth-m
--stride
--alpha
--rgb-image-is-undistorted
--binary-pcd / --no-binary-pcd
```

For example:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend dnn \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --min-depth-m 0.2 \
    --max-depth-m 10.0 \
    --stride 2 \
    --wait
```

`stride=2` samples every second stereo pixel in both directions and can substantially reduce point count and output size.

## YOLO-based PCD segmentation

The PCD service can consume an existing YOLO result JSON generated for the RGB image.

```text
YOLO server
    │
    ▼
detections.json
    │
    ▼
PCD server
    │
    ├── full.pcd
    │
    └── segments/
          ├── 000_class0_person_0.934_18573pts.pcd
          ├── 001_class2_car_0.881_29401pts.pcd
          └── ...
```

Submit:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend dnn \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd \
    --output-json-path sample/pcd.json \
    --detections-json-path sample/yolo.json \
    --segments-output-dir sample/segments \
    --min-segment-points 30 \
    --wait \
    --print-result
```

`detections_json_path` and `segments_output_dir` must be supplied together.

The segmentation stage supports polygon masks from the YOLO server. When a polygon mask is unavailable, the current helper falls back to the detection bounding box.

### Exclusive segmentation

Normally, points may belong to multiple overlapping detections.

Use:

```bash
--exclusive-segments
```

to make high-confidence detections claim their points first.

### Mask erosion

To reduce boundary contamination between an object and its surroundings:

```bash
--erode-pixels 2
```

### Background cloud

To additionally save points that belong to no detection:

```bash
--save-background
```

## Async jobs

Submitting without `--wait` returns immediately:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --build \
    --pcd-backend cpu \
    --rgb-jpg-path sample/rgb.jpg \
    --left-jpg-path sample/left.jpg \
    --right-jpg-path sample/right.jpg \
    --calibration-json-path sample/calibration.json \
    --output-pcd-path sample/full.pcd
```

Example:

```text
submit:
accepted=True
job_id='2bc91...'
state='queued'
...
```

Check the job:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --job-status 2bc91...
```

Retrieve the result:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --job-result 2bc91...
```

Retrieve the full JSON:

```bash
python -m pcd.cli client \
    --server-name pcd \
    --job-result 2bc91... \
    --print-result
```

Job states are:

```text
queued
running
succeeded
failed
cancelled
```

## Server status

With no client action, the CLI queries server status:

```bash
python -m pcd.cli client \
    --server-name pcd
```

Example fields include:

```text
online
queued
running
succeeded
failed
cancelled
builds
cache_hits
cache_misses
cached_backends
last_ms
error
```

## Discovery

List healthy registered services:

```bash
python -m pcd.cli list
```

A named PCD instance can be selected with:

```bash
--server-name pcd
```

Without `--server-name`, the discovered client can select a healthy instance of the `pcd` service.

## Direct endpoints

Discovery is optional.

A client can connect directly:

```bash
python -m pcd.cli client \
    --endpoint ipc://... \
    --backend nng
```

or over TCP:

```bash
python -m pcd.cli client \
    --endpoint tcp://127.0.0.1:PORT \
    --backend nng
```

## RPC methods

The service exposes:

```text
pcd.build
pcd.job_status
pcd.job_result
pcd.status
```

The generated HTTP equivalents are:

```text
POST /pcd/build
GET  /pcd/job_status
GET  /pcd/job_result
GET  /pcd/status
```

`add_pcd_routes()` can be used to register these routes with FastAPI.

## Output

A simple build may produce:

```text
record/
├── rgb.jpg
├── left.jpg
├── right.jpg
├── calibration.json
├── full.pcd
└── pcd.json
```

With YOLO segmentation:

```text
record/
├── rgb.jpg
├── left.jpg
├── right.jpg
├── calibration.json
├── yolo.json
├── full.pcd
├── pcd.json
└── segments/
    ├── 000_class0_person_0.934_18573pts.pcd
    ├── 001_class2_car_0.881_29401pts.pcd
    └── background_45612pts.pcd
```

This structure is intended to fit directly into the broader record-based RGB-D dataset layout.

## Current service flow

```text
client.build()
      │
      ▼
PcdService.build()
      │
      ▼
PcdWorker.submit()
      │
      ├─ create job
      ├─ state = queued
      └─ return job_id
             │
             ▼
       worker thread
             │
             ├─ state = running
             ├─ validate paths
             ├─ decode images
             ├─ acquire disparity backend
             ├─ load calibration
             ├─ rectify
             ├─ disparity
             ├─ XYZ
             ├─ RGB projection
             ├─ full PCD
             ├─ optional segments
             ├─ optional result JSON
             │
             └─ state = succeeded
```

The RPC server can restart independently of the PCD worker. Cached disparity models, queued jobs, and running work therefore remain owned by the long-lived worker rather than by the transport layer.
