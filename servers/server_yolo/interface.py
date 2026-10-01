from __future__ import annotations

from servers.msg.job import JobResultResponse

"""Unified RPC + HTTP interface for the YOLO service."""

from servers.logger import logging
from typing import Any

from npb_rpc.utils import add_fastapi_routes

from servers.msg.yolo import (
    YoloDetectResult,
    YoloInterface,
    EmptyRequest,
    YoloInferenceRequest,
    JobSubmitResponse,
    JobRequest,
    YoloJobResultResponse,
    YoloJobStatusResponse,
    YoloStatusResponse,
)
from servers.server_yolo.worker import YoloWorker


LOG = logging.getLogger(__name__.replace(".",":"))


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
    ) -> JobSubmitResponse:
        """Queue inference and return immediately with a job id."""
        try:
            res = self.worker.submit(request)
            self.log.info(str(res))
        except Exception as exc:
            self.log.exception("yolo.inference failed")
            return JobSubmitResponse(
                accepted=False,
                error=f"inference submit error: {type(exc).__name__}: {exc}",
            )

    def job_status(
        self,
        request: JobRequest,
    ) -> YoloJobStatusResponse:
        """Return lightweight state for one asynchronous inference job."""
        try:
            snapshot = self.worker.job_status(request.job_id)
            res = YoloJobStatusResponse(
                found=True,
                job_id=request.job_id,
                state=snapshot.state,
                model_name=snapshot.request.model_name,
                cuda_device=snapshot.request.cuda_device,
                input_jpg_path=snapshot.request.input_jpg_path,
                output_json_path=snapshot.request.output_json_path,
                created_ns=snapshot.created_ns,
                started_ns=snapshot.started_ns,
                finished_ns=snapshot.finished_ns,
                cache_hit=snapshot.cache_hit,
                num_detections=snapshot.result.num_detections,
                timing=snapshot.result.timing,
                error=snapshot.error,
            )
        except Exception as exc:
            self.log.exception("yolo.job_status failed")
            return YoloJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )
        return res
    
    def job_result(
        self,
        request: JobRequest,
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
