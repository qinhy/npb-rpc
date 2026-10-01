from __future__ import annotations

import logging
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generic, TypeVar

from servers.msg.job import JobSubmitResponse

import threading
import time
import uuid

from typing import Generic
from servers.msg.job import JobSnapshot, JobStoreSummary, RequestT, ResultT, JobRecord


LOG = logging.getLogger(__name__)


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
    ) -> JobSnapshot[RequestT, ResultT]:

        now_ns = time.time_ns()

        with self._lock:
            self._prune_locked(now_ns)

            record = self._jobs.get(job_id)

            if record is None:
                return JobSnapshot()

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


class Worker(ABC, Generic[RequestT, ResultT]):
    """
    Generic asynchronous worker pool.

    Owns:
        - worker threads
        - queue
        - lifecycle
        - JobStore state transitions

    Subclass implements:
        - process()
        - optionally cleanup()
        - optionally on_finished()
    """

    def __init__(
        self,
        *,
        name: str,
        worker_count: int = 1,
        queue_size: int = 0,
        job_ttl_s: float = 3600.0,
        max_completed_jobs: int = 128,
    ) -> None:
        self.name = name
        self.worker_count = max(1, int(worker_count))

        self.store = JobStore[RequestT, ResultT](
            job_ttl_s=job_ttl_s,
            max_completed_jobs=max_completed_jobs,
        )

        self._queue: Queue[str] = Queue(
            maxsize=max(0, int(queue_size))
        )

        self._state_lock = threading.RLock()
        self._shutdown = threading.Event()

        self._threads: list[threading.Thread] = []

        self._started = False
        self._closed = False

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} worker is closed")

            if self._started:
                return

            self._shutdown.clear()
            self._started = True

            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.name}-worker-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

        LOG.info(
            "started %d %s worker thread(s)",
            self.worker_count,
            self.name,
        )

    def close(self, *, timeout_s: float = 10.0) -> None:
        """
        Stop accepting new jobs, drain queued jobs,
        then terminate worker threads.
        """

        with self._state_lock:
            if self._closed:
                return

            self._closed = True
            self._shutdown.set()

            started = self._started
            threads = list(self._threads)

        if not started:
            self.cleanup()
            return

        deadline = time.monotonic() + max(0.0, timeout_s)

        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

        alive = [
            thread.name
            for thread in threads
            if thread.is_alive()
        ]

        if alive:
            LOG.warning(
                "%s workers still running after shutdown timeout: %s",
                self.name,
                alive,
            )
            return

        self.cleanup()

        with self._state_lock:
            self._started = False

    # ------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------

    def submit(self, request: RequestT) -> JobSubmitResponse:
        with self._state_lock:
            if not self._started or self._closed:
                return JobSubmitResponse(
                    accepted=False,
                    error=f"{self.name} worker is not running",
                )

            job_id = self.store.create(request)

            try:
                self._queue.put_nowait(job_id)

            except Full:
                self.store.discard_queued(job_id)

                return JobSubmitResponse(
                    accepted=False,
                    error=f"{self.name} worker queue is full",
                )

        return JobSubmitResponse(
            accepted=True,
            job_id=job_id,
            state="queued",
        )

    def cancel(self, job_id: str) -> bool:
        return self.store.cancel(job_id)

    def job_status(self, job_id: str):
        return self.store.snapshot(job_id)

    def job_result(self, job_id: str) -> ResultT | None:
        snapshot = self.store.snapshot(job_id)

        if snapshot is None:
            return None

        if snapshot.state != "succeeded":
            return None

        return snapshot.result

    def status(self):
        return self.store.summary()

    # ------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=0.1)

            except Empty:
                if self._shutdown.is_set():
                    return

                continue

            try:
                self._run_job(job_id)

            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        snapshot = self.store.snapshot(job_id)

        if snapshot is None:
            return

        if snapshot.state != "queued":
            return

        if not self.store.mark_running(job_id):
            return

        request = snapshot.request

        try:
            result = self.process(
                request=request,
                job_id=job_id,
            )

            self.store.succeed(
                job_id,
                result,
            )

        except Exception as exc:
            LOG.exception(
                "%s job %s failed",
                self.name,
                job_id,
            )

            self.store.fail(
                job_id,
                f"{type(exc).__name__}: {exc}",
            )

        finally:
            try:
                self.on_finished(
                    request=request,
                    job_id=job_id,
                )
            except Exception:
                LOG.exception(
                    "%s job %s finish hook failed",
                    self.name,
                    job_id,
                )

    # ------------------------------------------------------------
    # Service-specific hooks
    # ------------------------------------------------------------

    @abstractmethod
    def process(
        self,
        *,
        request: RequestT,
        job_id: str,
    ) -> ResultT:
        """
        Execute one job.

        Called inside a worker thread.
        """
        raise NotImplementedError

    def on_finished(
        self,
        *,
        request: RequestT,
        job_id: str,
    ) -> None:
        pass

    def cleanup(self) -> None:
        pass
