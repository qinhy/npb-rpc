from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from npb import BinaryModel
from npb_rpc import RpcEvent


ResultT = TypeVar("ResultT")


class JobState(StrEnum):
    """Common lifecycle shared by finite asynchronous RPC jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobSubmitResponseBase(BinaryModel):
    """Common wire prefix for asynchronous job submission responses.

    Service-specific subclasses append their own useful echo fields and keep
    ``error`` as the final field so their existing fixed wire ordering can be
    preserved.
    """

    accepted: bool
    job_id: str = ""
    state: JobState | None = None
    done_event: RpcEvent | None = None


class JobResultResponse(BinaryModel, Generic[ResultT]):
    """Common wire shape for the result of a finite asynchronous job.

    Do not decorate this generic base with ``@binary_schema``. Each service
    should define a concrete specialized subclass and keep its own stable
    schema id/version, for example::

        @binary_schema("npb-rpc.yolo.job.result.response", version=1)
        class YoloJobResultResponse(JobResultResponse[YoloDetectResult]):
            pass

    This keeps the service-specific RPC contract/name while defining the
    shared field layout once.
    """

    found: bool
    job_id: str
    state: JobState | None = None
    result: ResultT | None = None
    error: str = ""
