from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from npb import BinaryModel, binary_schema
from pydantic import Field
from npb_rpc import RpcEvent


RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class JobState(StrEnum):
    """Common lifecycle shared by finite asynchronous RPC jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@binary_schema("npb-rpc.job.request", version=1)
class JobRequest(BinaryModel):
    """Identify one asynchronous job."""
    job_id: str = Field(min_length=1)


@binary_schema("npb-rpc.job.submit.response", version=1)
class JobSubmitResponse(BinaryModel):
    """Common wire prefix for asynchronous job submission responses.

    Service-specific subclasses append their own useful echo fields and keep
    ``error`` as the final field so their existing fixed wire ordering can be
    preserved.
    """

    accepted: bool
    job_id: str = ""
    state: JobState | None = None
    done_event: RpcEvent | None = None
    error: str | None = None


class JobResultResponse(BinaryModel, Generic[ResultT]):
    """Common wire shape for the result of a finite asynchronous job.

    Do not decorate this generic base with ``@binary_schema``. Each service
    should define a concrete specialized subclass and keep its own stable
    schema id/version, for example::

        @binary_schema("npb-rpc.yolo.job.result.response", version=1)
        class JobResultResponse[YoloDetectResult](JobResultResponse[YoloDetectResult]):
            pass

    This keeps the service-specific RPC contract/name while defining the
    shared field layout once.
    """

    found: bool
    job_id: str
    state: JobState | None = None
    result: ResultT | None = None
    error: str = ""


class JobRecord(BinaryModel, Generic[RequestT, ResultT]):
    request: RequestT

    state: JobState = "queued"

    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0

    cache_hit: bool = False

    result: ResultT | None = None
    error: str = ""


class JobSnapshot(BinaryModel, Generic[RequestT, ResultT]):
    job_id: str = ""
    request: RequestT | None

    state: JobState = JobState.FAILED

    created_ns: int = -1
    started_ns: int = -1
    finished_ns: int = -1

    cache_hit: bool = False

    result: ResultT | None
    error: str = ""


@binary_schema("npb-rpc.job.store.summary", version=1)
class JobStoreSummary(BinaryModel):
    queued_jobs: int = -1
    running_jobs: int = -1

    succeeded_jobs: int = -1
    failed_jobs: int = -1
    cancelled_jobs: int = -1

    last_job_id: str = ""
    last_finished_ns: int = -1
    last_error: str = ""
