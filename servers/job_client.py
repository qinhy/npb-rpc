"""Bounded waiting for the shared asynchronous job RPC contract."""

from __future__ import annotations

import time

from servers.msg.job import JobRequest, JobSubmitResponse


def wait_for_submission(client, submission: JobSubmitResponse, *, timeout_s: float = 300.0):
    if not submission.accepted:
        raise RuntimeError(submission.error or "job submission rejected")
    deadline = time.monotonic() + timeout_s
    while True:
        status = client.job_status(JobRequest(job_id=submission.job_id))
        if not status.found:
            raise RuntimeError(status.error or "job not found or expired")
        if status.state in {"succeeded", "failed", "cancelled"}:
            if status.state != "succeeded":
                raise RuntimeError(f"job {submission.job_id} {status.state}: {status.error}")
            return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"job {submission.job_id} did not finish within {timeout_s:g}s")
        time.sleep(min(0.1, remaining))
