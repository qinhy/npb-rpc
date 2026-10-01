from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from servers.msg.job import JobState, JobResultResponse, JobSubmitResponseBase

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass
class JobRecord(Generic[RequestT, ResultT]):
    request: RequestT

    state: JobState = "queued"

    created_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0

    cache_hit: bool = False

    result: ResultT | None = None
    error: str = ""


@dataclass(frozen=True)
class JobSnapshot(Generic[RequestT, ResultT]):
    job_id: str
    request: RequestT

    state: JobState

    created_ns: int
    started_ns: int
    finished_ns: int

    cache_hit: bool

    result: ResultT | None
    error: str


@dataclass(frozen=True)
class JobStoreSummary:
    queued_jobs: int
    running_jobs: int

    succeeded_jobs: int
    failed_jobs: int
    cancelled_jobs: int

    last_job_id: str
    last_finished_ns: int
    last_error: str


class JobStore(Generic[RequestT, ResultT]):

    TERMINAL = {"succeeded", "failed", "cancelled"}

    def __init__(
        self,
        *,
        job_ttl_s: float = 3600.0,
        max_completed_jobs: int = 128,
    ) -> None:
        self._lock = threading.RLock()

        self._jobs: dict[str, JobRecord[RequestT, ResultT]] = {}

        self._job_ttl_ns = max(0, int(job_ttl_s * 1e9))
        self._max_completed_jobs = max(1, int(max_completed_jobs))

        self._succeeded_total = 0
        self._failed_total = 0
        self._cancelled_total = 0

        self._last_job_id = ""
        self._last_finished_ns = 0
        self._last_error = ""

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    def create(self, request: RequestT) -> str:
        job_id = uuid.uuid4().hex
        now_ns = time.time_ns()

        with self._lock:
            self._prune_locked(now_ns)

            self._jobs[job_id] = JobRecord(
                request=request,
                created_ns=now_ns,
            )

        return job_id

    def discard_queued(self, job_id: str) -> bool:
        """Remove a queued job without treating it as execution failure."""

        with self._lock:
            record = self._jobs.get(job_id)

            if record is None or record.state != "queued":
                return False

            del self._jobs[job_id]
            return True

    def mark_running(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)

            if record is None or record.state != "queued":
                return False

            record.state = "running"
            record.started_ns = time.time_ns()

            return True

    def succeed(self, job_id: str, result: ResultT) -> bool:
        now_ns = time.time_ns()

        with self._lock:
            record = self._jobs.get(job_id)

            if record is None or record.state in self.TERMINAL:
                return False

            record.state = "succeeded"
            record.finished_ns = now_ns
            record.result = result
            record.error = ""

            self._succeeded_total += 1

            self._last_job_id = job_id
            self._last_finished_ns = now_ns
            self._last_error = ""

            self._prune_locked(now_ns)

            return True

    def fail(self, job_id: str, error: str) -> bool:
        now_ns = time.time_ns()
        message = str(error)

        with self._lock:
            record = self._jobs.get(job_id)

            if record is None or record.state in self.TERMINAL:
                return False

            record.state = "failed"
            record.finished_ns = now_ns
            record.error = message

            self._failed_total += 1

            self._last_job_id = job_id
            self._last_finished_ns = now_ns
            self._last_error = message

            self._prune_locked(now_ns)

            return True

    def cancel(self, job_id: str, error: str = "cancelled") -> bool:
        now_ns = time.time_ns()

        with self._lock:
            record = self._jobs.get(job_id)

            # Keep same semantics as your current stores:
            # running jobs cannot be cancelled.
            if record is None or record.state != "queued":
                return False

            record.state = "cancelled"
            record.finished_ns = now_ns
            record.error = str(error)

            self._cancelled_total += 1

            self._last_job_id = job_id
            self._last_finished_ns = now_ns

            self._prune_locked(now_ns)

            return True

    # ---------------------------------------------------------
    # optional metadata
    # ---------------------------------------------------------

    def set_cache_hit(self, job_id: str, value: bool = True) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)

            if record is None:
                return False

            record.cache_hit = bool(value)
            return True

    # ---------------------------------------------------------
    # query
    # ---------------------------------------------------------

    def snapshot(
        self,
        job_id: str,
    ) -> JobSnapshot[RequestT, ResultT] | None:

        now_ns = time.time_ns()

        with self._lock:
            self._prune_locked(now_ns)

            record = self._jobs.get(job_id)

            if record is None:
                return None

            return JobSnapshot(
                job_id=job_id,
                request=record.request,
                state=record.state,
                created_ns=record.created_ns,
                started_ns=record.started_ns,
                finished_ns=record.finished_ns,
                cache_hit=record.cache_hit,
                result=record.result,
                error=record.error,
            )

    def summary(self) -> JobStoreSummary:
        now_ns = time.time_ns()

        with self._lock:
            self._prune_locked(now_ns)

            return JobStoreSummary(
                queued_jobs=sum(
                    r.state == "queued"
                    for r in self._jobs.values()
                ),
                running_jobs=sum(
                    r.state == "running"
                    for r in self._jobs.values()
                ),
                succeeded_jobs=self._succeeded_total,
                failed_jobs=self._failed_total,
                cancelled_jobs=self._cancelled_total,
                last_job_id=self._last_job_id,
                last_finished_ns=self._last_finished_ns,
                last_error=self._last_error,
            )

    # ---------------------------------------------------------
    # retention
    # ---------------------------------------------------------

    def _prune_locked(self, now_ns: int) -> None:
        terminal = [
            (job_id, record.finished_ns)
            for job_id, record in self._jobs.items()
            if (
                record.state in self.TERMINAL
                and record.finished_ns
            )
        ]

        # TTL
        if self._job_ttl_ns:
            cutoff = now_ns - self._job_ttl_ns

            for job_id, finished_ns in terminal:
                if finished_ns < cutoff:
                    self._jobs.pop(job_id, None)

        # Max completed jobs
        terminal = sorted(
            (
                (job_id, record.finished_ns)
                for job_id, record in self._jobs.items()
                if (
                    record.state in self.TERMINAL
                    and record.finished_ns
                )
            ),
            key=lambda item: item[1],
        )

        excess = len(terminal) - self._max_completed_jobs

        for job_id, _ in terminal[:max(0, excess)]:
            self._jobs.pop(job_id, None)