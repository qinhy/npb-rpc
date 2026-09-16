from __future__ import annotations

"""Unified RPC + HTTP interface for the YOLO service."""

import logging
from typing import Any

from npb_rpc.utils import add_fastapi_routes

from servers.msg.yolo import (
    YoloInterface,
    EmptyRequest,
    YoloInferenceRequest,
    YoloInferenceSubmitResponse,
    YoloJobRequest,
    YoloJobResultResponse,
    YoloJobStatusResponse,
    YoloStatusResponse,
)
from servers.server_yolo.worker import YoloWorker


LOG = logging.getLogger("yolo.interface")


class YoloService(YoloInterface):
    """Typed RPC/HTTP façade over the long-lived asynchronous YoloWorker."""

    def __init__(
        self,
        worker: YoloWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.worker = worker
        self.log = logger or LOG

    def inference(
        self,
        request: YoloInferenceRequest,
    ) -> YoloInferenceSubmitResponse:
        """Queue inference and return immediately with a job id."""
        try:
            return self.worker.submit(request)
        except Exception as exc:
            self.log.exception("yolo.inference failed")
            return YoloInferenceSubmitResponse(
                accepted=False,
                input_jpg_path=request.input_jpg_path,
                output_json_path=request.output_json_path,
                error=f"inference submit error: {type(exc).__name__}: {exc}",
            )

    def job_status(
        self,
        request: YoloJobRequest,
    ) -> YoloJobStatusResponse:
        """Return lightweight state for one asynchronous inference job."""
        try:
            return self.worker.job_status(request.job_id)
        except Exception as exc:
            self.log.exception("yolo.job_status failed")
            return YoloJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )

    def job_result(
        self,
        request: YoloJobRequest,
    ) -> YoloJobResultResponse:
        """Return the full result when a job has succeeded."""
        try:
            return self.worker.job_result(request.job_id)
        except Exception as exc:
            self.log.exception("yolo.job_result failed")
            return YoloJobResultResponse(
                found=False,
                job_id=request.job_id,
                error=f"job result error: {type(exc).__name__}: {exc}",
            )

    def status(
        self,
        request: EmptyRequest,
    ) -> YoloStatusResponse:
        """Return worker, queue, and model-cache status."""
        del request

        try:
            return self.worker.status()
        except Exception as exc:
            self.log.exception("yolo.status failed")
            return YoloStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )


def add_yolo_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_fastapi_routes(app, YoloInterface, **kwargs)
