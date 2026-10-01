from __future__ import annotations

"""Unified RPC + HTTP interface for the YOLO service."""

from servers.logger import logging
from typing import Any, TYPE_CHECKING

from npb_rpc.utils import add_fastapi_routes

from servers.msg.yolo import (
    YoloInterface,
    EmptyRequest,
    YoloInferenceRequest,
    JobSubmitResponse,
    JobRequest,
    YoloJobResultResponse,
    YoloJobStatusResponse,
    YoloStatusResponse,
)
if TYPE_CHECKING:
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
            return res
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
            if snapshot is None:
                return YoloJobStatusResponse(job_id=request.job_id, error="job not found or expired")
            return YoloJobStatusResponse(**snapshot.model_dump())
        except Exception as exc:
            self.log.exception("yolo.job_status failed")
            return YoloJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )
    
    def job_result(
        self,
        request: JobRequest,
    ) -> YoloJobResultResponse:
        """Return the full result when a job has succeeded."""
        try:
            snapshot = self.worker.job_status(request.job_id)
            if snapshot is None:
                return YoloJobResultResponse(found=False, job_id=request.job_id, error="job not found or expired")
            return YoloJobResultResponse(
                found=True, job_id=request.job_id, state=snapshot.state,
                result=snapshot.result if snapshot.state == "succeeded" else None,
                error=snapshot.error,
            )
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
            jobs = self.worker.status()
            models, hits, misses = self.worker.models.snapshot()
            return YoloStatusResponse(
                online=self.worker.online,
                **jobs.model_dump(exclude={"last_error", "last_finished_ns"}),
                inference_count=jobs.succeeded_jobs,
                cached_models=models, cache_hits=hits, cache_misses=misses,
                last_inference_ns=jobs.last_finished_ns,
                error=jobs.last_error,
            )
        except Exception as exc:
            self.log.exception("yolo.status failed")
            return YoloStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )


def add_yolo_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_fastapi_routes(app, YoloInterface, **kwargs)
