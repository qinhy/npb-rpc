from __future__ import annotations

"""Unified RPC + HTTP interface for the PCD service."""

import logging
from typing import Any

from npb_rpc.utils import add_fastapi_routes

try:
    from .msg import (
        PcdInterface,
        EmptyRequest,
        PcdBuildRequest,
        PcdBuildSubmitResponse,
        PcdJobRequest,
        PcdJobResultResponse,
        PcdJobStatusResponse,
        PcdStatusResponse,
    )
    from .worker import PcdWorker
except ImportError:  # Support running files directly from this directory.
    from msg import (
        PcdInterface,
        EmptyRequest,
        PcdBuildRequest,
        PcdBuildSubmitResponse,
        PcdJobRequest,
        PcdJobResultResponse,
        PcdJobStatusResponse,
        PcdStatusResponse,
    )
    from worker import PcdWorker


LOG = logging.getLogger("pcd.interface")


class PcdService(PcdInterface):
    """Typed RPC/HTTP façade over the long-lived asynchronous PcdWorker."""

    def __init__(
        self,
        worker: PcdWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.worker = worker
        self.log = logger or LOG

    def build(self, request: PcdBuildRequest) -> PcdBuildSubmitResponse:
        try:
            return self.worker.submit(request)
        except Exception as exc:
            self.log.exception("pcd.build failed")
            return PcdBuildSubmitResponse(
                accepted=False,
                rgb_jpg_path=request.rgb_jpg_path,
                left_jpg_path=request.left_jpg_path,
                right_jpg_path=request.right_jpg_path,
                output_pcd_path=request.output_pcd_path,
                output_json_path=request.output_json_path,
                error=f"build submit error: {type(exc).__name__}: {exc}",
            )

    def job_status(self, request: PcdJobRequest) -> PcdJobStatusResponse:
        try:
            return self.worker.job_status(request.job_id)
        except Exception as exc:
            self.log.exception("pcd.job_status failed")
            return PcdJobStatusResponse(
                found=False,
                job_id=request.job_id,
                error=f"job status error: {type(exc).__name__}: {exc}",
            )

    def job_result(self, request: PcdJobRequest) -> PcdJobResultResponse:
        try:
            return self.worker.job_result(request.job_id)
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
            return self.worker.status()
        except Exception as exc:
            self.log.exception("pcd.status failed")
            return PcdStatusResponse(
                online=False,
                error=f"status error: {type(exc).__name__}: {exc}",
            )


def add_pcd_routes(app: Any, **kwargs: Any):
    """Small compatibility/convenience wrapper."""
    return add_fastapi_routes(app, PcdInterface, **kwargs)
