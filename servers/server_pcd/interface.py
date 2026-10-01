from __future__ import annotations

"""Unified RPC + HTTP interface for the PCD service."""

from servers.logger import logging
from typing import Any, TYPE_CHECKING

from npb_rpc.utils import add_fastapi_routes

from servers.msg.pcd import (
    PcdJobStatusResponse,
    PcdStatusResponse,
    PcdInterface,
    EmptyRequest,
    PcdBuildRequest,
    JobSubmitResponse,
    JobRequest,
    PcdJobResultResponse
)
if TYPE_CHECKING:
    from servers.server_pcd.worker import PcdWorker


LOG = logging.getLogger(__name__.replace(".",":"))


class PcdService(PcdInterface):
    """Typed RPC/HTTP façade over the long-lived asynchronous PcdWorker."""

    def __init__(
        self,
        worker: PcdWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.worker = worker
        self.log = logger or LOG

    def build(self, request: PcdBuildRequest) -> JobSubmitResponse:
        try:
            res = self.worker.submit(request)
        except Exception as exc:
            self.log.exception("pcd.build failed")
            res = JobSubmitResponse(
                accepted=False,
                error=f"build submit error: {type(exc).__name__}: {exc}",
            )
        return res

    def job_status(self, request: JobRequest) -> PcdJobStatusResponse:
        try:
            snapshot = self.worker.job_status(request.job_id)
            if snapshot is None:
                return PcdJobStatusResponse(job_id=request.job_id, error="job not found or expired")
            return PcdJobStatusResponse(**snapshot.model_dump())
        except Exception as exc:
            self.log.exception("pcd.job_status failed")
            return PcdJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )

    def job_result(self, request: JobRequest) -> PcdJobResultResponse:
        try:
            snapshot = self.worker.job_status(request.job_id)
            if snapshot is None:
                return PcdJobResultResponse(found=False, job_id=request.job_id, error="job not found or expired")
            return PcdJobResultResponse(
                found=True, job_id=request.job_id, state=snapshot.state,
                result=snapshot.result if snapshot.state == "succeeded" else None,
                error=snapshot.error,
            )
        except Exception as exc:
            self.log.exception("pcd.job_result failed")
            return PcdJobResultResponse(
                found=False,
                job_id=request.job_id,
                error=f"job result error: {type(exc).__name__}: {exc}",
            )

    def status(self, request: EmptyRequest) -> PcdStatusResponse:
        del request
        try:
            jobs = self.worker.status()
            backends, hits, misses = self.worker.backends.snapshot()
            return PcdStatusResponse(
                online=self.worker.online,
                **jobs.model_dump(exclude={"last_error", "last_finished_ns"}),
                build_count=jobs.succeeded_jobs,
                cached_backends=backends, cache_hits=hits, cache_misses=misses,
                last_build_ns=jobs.last_finished_ns,
                error=jobs.last_error,
            )
        except Exception as exc:
            self.log.exception("pcd.status failed")
            return PcdStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )


def add_pcd_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_fastapi_routes(app, PcdInterface, **kwargs)
