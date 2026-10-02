from __future__ import annotations

import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from npb import decode, encode

from npb_rpc import NngRpcServer, RpcEvent, ZmqRpcServer, portable_ipc
from npb_rpc.utils import add_rpc, api_methods
from servers.msg import pcd, yolo
from servers.msg.job import JobRequest, JobSubmitResponse
from servers.msg.worker import Worker
from servers.server_pcd.interface import PcdService
from servers.server_yolo.interface import YoloService


class ControlledWorker(Worker):
    """Exercise the real shared lifecycle without loading inference libraries."""

    def __init__(self, kind, outcome="succeeded", **kwargs):
        super().__init__(name=kind, **kwargs)
        self.kind = kind
        self.outcome = outcome
        self.entered = threading.Event()
        self.release = threading.Event()
        self.models = self.backends = SimpleNamespace(snapshot=lambda: (("test-cpu",), 2, 1))

    def process(self, *, request, job_id):
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError("test did not release worker")
        if self.outcome == "failed":
            raise ValueError("invalid input image")
        if self.outcome == "empty":
            return None
        if self.kind == "pcd":
            return pcd.PcdBuildResult(**request.model_dump(), point_count=42)
        return yolo.YoloDetectResult(**request.model_dump(), image_width=64, image_height=32)


def contract(kind):
    if kind == "pcd":
        return pcd.PcdInterface, pcd.PcdClient, PcdService, pcd.EmptyRequest, "build"
    return yolo.YoloInterface, yolo.YoloClient, YoloService, yolo.EmptyRequest, "inference"


def request_for(kind, **kwargs):
    if kind == "pcd":
        return pcd.PcdBuildRequest(
            rgb_jpg_path="rgb.jpg", left_jpg_path="left.jpg", right_jpg_path="right.jpg",
            calibration_json_path="calibration.json", output_pcd_path="cloud.pcd", **kwargs,
        )
    return yolo.YoloInferenceRequest(input_jpg_path="rgb.jpg", output_json_path="yolo.json", **kwargs)


@pytest.mark.parametrize("kind", ["pcd", "yolo"])
@pytest.mark.parametrize("backend", ["nng", "zmq"])
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "empty"])
def test_service_rpc_lifecycle(kind, backend, outcome, monkeypatch):
    pytest.importorskip("pynng" if backend == "nng" else "zmq")
    interface, client_type, service_type, empty, submit_method = contract(kind)
    worker = ControlledWorker(kind, outcome)
    signals = []
    signalled = threading.Event()

    def signal(event, *, state, error):
        # Terminal state must be committed before notifying a waiting caller.
        snapshot = worker.job_status(worker.status().last_job_id)
        signals.append((event.key, state, error, snapshot.state))
        signalled.set()

    monkeypatch.setattr(RpcEvent, "set", signal)
    server_type = NngRpcServer if backend == "nng" else ZmqRpcServer
    endpoint = portable_ipc(f"job-tests-{uuid4().hex}") if backend == "nng" else "tcp://127.0.0.1:*"
    server = server_type.bind(endpoint)
    add_rpc(server, interface, service_type(worker))
    client = client_type(endpoint=server.endpoint, backend=backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    thread.start()
    try:
        status = client.status(empty())
        assert status.online
        assert status.cache_hits == 2
        missing = JobRequest(job_id="missing")
        assert not client.job_status(missing).found
        assert not client.job_result(missing).found

        done = RpcEvent.create()
        submission = getattr(client, submit_method)(request_for(kind, done_event=done))
        assert submission.accepted and submission.job_id
        assert submission.done_event == done
        assert worker.entered.wait(2)
        job = JobRequest(job_id=submission.job_id)
        running = client.job_status(job)
        assert running.found and running.state == "running"
        assert running.request.done_event == done
        assert running.result is None
        assert client.job_result(job).result is None

        worker.release.set()
        assert signalled.wait(2)
        status = client.job_status(job)
        result = client.job_result(job)
        expected = "succeeded" if outcome == "succeeded" else "failed"
        assert status.state == result.state == expected
        assert signals[0][1] == signals[0][3] == expected
        if outcome == "succeeded":
            assert result.result == status.result
            assert result.result is not None
            assert client.status(empty()).succeeded_jobs == 1
        else:
            assert result.result is None
            assert status.error == result.error == signals[0][2]
            assert "no result" in status.error if outcome == "empty" else "invalid input" in status.error
            assert client.status(empty()).failed_jobs == 1
    finally:
        worker.release.set()
        worker.close()
        server.stop()
        thread.join(2)
        server.close()


@pytest.mark.parametrize("kind", ["pcd", "yolo"])
def test_all_error_fallbacks_are_serializable(kind):
    interface, _, service_type, empty, submit_method = contract(kind)

    def fail(*args):
        raise RuntimeError("worker unavailable")

    service = service_type(SimpleNamespace(submit=fail, job_status=fail, status=fail))
    for method in api_methods(interface):
        request = (request_for(kind) if method.name == submit_method else
                   empty() if method.name == "status" else JobRequest(job_id="missing"))
        response = getattr(service, method.name)(request)
        assert isinstance(response, method.response)
        decoded = decode(method.response, encode(response))
        assert "worker unavailable" in decoded.error


@pytest.mark.parametrize("kind", ["pcd", "yolo"])
def test_queue_rejection_cancellation_and_expired_jobs(kind, monkeypatch):
    worker = ControlledWorker(kind, queue_size=1)
    signals = []
    monkeypatch.setattr(RpcEvent, "set", lambda self, **kw: signals.append(kw))
    assert not worker.submit(request_for(kind)).accepted
    worker.start()
    try:
        worker.submit(request_for(kind))
        assert worker.entered.wait(2)
        queued = worker.submit(request_for(kind, done_event=RpcEvent.create()))
        assert worker.job_status(queued.job_id).state == "queued"
        rejected = worker.submit(request_for(kind))
        assert not rejected.accepted and "full" in rejected.error
        assert worker.cancel(queued.job_id)
        assert signals == [{"state": "cancelled", "error": "cancelled"}]
        assert worker.job_status(queued.job_id).state == "cancelled"
        assert not worker.cancel(queued.job_id)
        # Force expiry deterministically without sleeping.
        worker.store._job_ttl_ns = 1
        assert worker.job_status(queued.job_id) is None
    finally:
        worker.release.set()
        worker.close()
    assert not worker.online
    assert not worker.submit(request_for(kind)).accepted
