"""Scheduling, paging and stream contracts without a GPU or external services."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_api import workspace as workspace_fixture

from clsweb.app import create_app
from clsweb.jobs import JobQueue
from clsweb.settings import WebSettings


@pytest.fixture
def workspace(tmp_path):
    return workspace_fixture.__wrapped__(tmp_path)


def until(check):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("Scheduler condition timed out")


def test_resource_wait_then_priority(workspace, monkeypatch):
    queue = JobQueue(workspace)
    queue.settings.resources.enabled = True
    low = queue.submit("report", "low", {"run_id": "unused", "priority": 1})
    high = queue.submit("report", "high", {"run_id": "unused", "priority": 8})
    available = {"memory_percent": 96, "available_gb": 1, "gpus": []}
    monkeypatch.setattr("clsweb.jobs.snapshot", lambda: available)
    launched = []

    def launch(job, category, gpu=None):
        launched.append((job["id"], gpu))
        queue.stopping.set()

    monkeypatch.setattr(queue, "launch", launch)
    queue.start()
    try:
        until(lambda: high["id"] in queue.blocked)
        assert "内存" in queue.blocked[high["id"]]
        assert queue.get(low["id"])["status"] == "queued" and not launched
        available.update(memory_percent=30, available_gb=32)
        until(lambda: bool(launched))
        assert launched[0][0] == high["id"]
    finally:
        queue.stop()


def test_gpu_reservation_uses_environment_not_core_schema(workspace, monkeypatch):
    import yaml

    queue = JobQueue(workspace)
    cfg = yaml.safe_load((workspace / "configs/flower/flower_tiny.yaml").read_text())
    cfg["trainer"]["accelerator"] = "gpu"
    captured = {}
    # Use a hashable process stand-in for resident membership checks.
    class Process:
        pid = 987654

        def poll(self):
            return None

    monkeypatch.setattr("clsweb.jobs.subprocess.Popen", lambda *a, **kw: (captured.update(kw), Process())[1])
    job = queue.submit("train", "gpu binding", {"config": cfg})
    queue.launch(job, "train", [2])
    try:
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert captured["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert queue.specification(job["id"])["config"]["trainer"]["devices"] == 1
        assert queue.reservations[job["id"]] == [2]
    finally:
        queue.active[job["id"]][1].close()


def test_pagination_filter_and_summary(workspace):
    app = create_app(workspace)
    with TestClient(app) as client:
        queue = app.state.queue
        queue.stopping.set()
        queue.thread.join(3)
        for i in range(25):
            job = queue.submit("report", f"result-{i:02}", {"run_id": "unused"})
            job.update(status="succeeded", result={"predictions": [{"path": "x"}] * 100})
            queue.save(job)
        page = client.get("/api/jobs/page?limit=12&offset=12&order=oldest").json()
        assert page["total"] == 25 and len(page["items"]) == 12
        assert page["items"][0]["title"] == "result-12"
        assert "result" not in page["items"][0]
        assert page["items"][0]["prediction_count"] == 100
        assert client.get("/api/jobs/page?query=result-24&status=succeeded").json()["total"] == 1
        assert client.get("/api/jobs/page?limit=201").status_code == 422
        assert client.get("/api/jobs/page?order=invalid").status_code == 422


def test_sse_incremental_logs_reconnect_and_delete(workspace):
    app = create_app(workspace)
    queue = app.state.queue
    queue.settings.refresh_interval_seconds = 1
    job = queue.submit("report", "stream", {"run_id": "unused"})
    path = workspace / "logs/web" / f"{job['id']}.log"
    path.write_bytes(b"first\n")
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/events")

    class Request:
        async def is_disconnected(self):
            return False

    async def run():
        response = await endpoint(Request(), job["id"])
        frames = response.body_iterator

        def parse(frame):
            return json.loads(frame.split("data: ")[1])

        first = parse(await anext(frames))
        assert first["log"] == {"reset": True, "text": "first\n"}
        with path.open("a", encoding="utf-8", newline="\n") as out:
            out.write("next\n")
        second = parse(await anext(frames))
        assert second["log"] == {"reset": False, "text": "next\n"}
        await frames.aclose()
        response = await endpoint(Request(), job["id"])
        frame = parse(await anext(response.body_iterator))
        assert frame["log"]["reset"] and "first\nnext\n" == frame["log"]["text"]
        queue.delete([job["id"]])
        frame = parse(await anext(response.body_iterator))
        assert frame["detail"]["deleted"] and frame["revision"] > first["revision"]
        await response.body_iterator.aclose()

    asyncio.run(run())


def test_timeout_stops_only_expired_job(workspace, monkeypatch):
    queue = JobQueue(workspace)
    monkeypatch.setattr("clsweb.jobs.terminate", lambda process: setattr(process, "stopped", True))

    class Process:
        stopped = False

        def poll(self):
            return None

    jobs = []
    for title, minutes in (("expired", 5), ("other", 0)):
        job = queue.submit("report", title, {"run_id": "unused", "timeout_minutes": 1})
        job.update(status="running", started_at=(datetime.now(UTC) - timedelta(minutes=minutes)).isoformat())
        queue.save(job)
        process = Process()
        queue.active[job["id"]] = (process, (workspace / f"{title}.log").open("wb"), "auxiliary")
        jobs.append((job, process))
    # start() recovers stale records, so launch just the dispatcher for this unit test.
    import threading

    thread = threading.Thread(target=queue.run)
    thread.start()
    try:
        until(lambda: queue.get(jobs[0][0]["id"])["status"] == "failed")
        assert jobs[0][1].stopped and not jobs[1][1].stopped
    finally:
        queue.stopping.set()
        thread.join(3)
        queue.active[jobs[1][0]["id"]][1].close()


def test_reject_resource_and_device_configuration(workspace):
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebSettings.model_validate({"resources": {"max_memory_percent": 100}})
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/api/train",
            json={"model_id": "flower_tiny", "dataset_id": "demo", "device": "cpu", "gpu_index": 3},
        )
        assert response.status_code == 422
