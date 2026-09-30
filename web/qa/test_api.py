"""Real subprocess lifecycle and local HTTP boundary checks (CPU synthetic data)."""

import io
import json
import random
import time

import pytest
import yaml
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from clsframework.config import Config  # noqa: E402
from clsweb.app import create_app  # noqa: E402
from clsweb.jobs import JobQueue  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    rng = random.Random(12)
    for split in ("train", "val", "test"):
        for category in ("red", "blue"):
            directory = tmp_path / "data/demo" / split / category
            directory.mkdir(parents=True)
            for index in range(3):
                Image.frombytes("RGB", (32, 32), rng.randbytes(3072)).save(directory / f"{index}.png")
    cfg = Config.model_validate(
        {
            "experiment": {"name": "web_test", "output_root": str(tmp_path / "runs")},
            "dataset": {"root": str(tmp_path / "data/demo")},
            "model": {"provider": "builtin", "name": "tiny_cnn", "weights": {"source": "none"}},
            "preprocessing": {"source": "explicit", "image_size": 32},
            "runtime": {"offline": True},
        }
    )
    directory = tmp_path / "configs/flower"
    directory.mkdir(parents=True)
    (directory / "flower_tiny.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    (tmp_path / "configs/web.yaml").write_text("resources:\n  enabled: false\n", encoding="utf-8")
    return tmp_path


def wait_job(client, job_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            assert job["status"] == "succeeded", job
            return job
        time.sleep(0.3)
    pytest.fail("Worker did not finish within 120 seconds")


def wait_report(client, job_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job.get("report_url"):
            return job
        assert job.get("result", {}).get("report_status") not in {"failed", "interrupted"}, job
        time.sleep(0.3)
    pytest.fail("Background report did not finish within 120 seconds")


def test_train_predict_evaluate_report(workspace):
    with TestClient(create_app(workspace)) as client:
        assert len(client.get("/api/catalog").json()["models"]) == 1
        response = client.post(
            "/api/train",
            json={
                "model_id": "flower_tiny",
                "dataset_id": "demo",
                "epochs": 1,
                "initialization": "official",
                "offline": True,
            },
        )
        assert response.status_code == 202, response.text
        train = wait_job(client, response.json()["id"])
        assert train["progress"] == {"completed": 1, "total": 1}
        run_id = train["run_id"]
        runs = client.get("/api/runs").json()
        assert runs[0]["id"] == run_id and runs[0]["reports"]
        assert client.get(runs[0]["reports"][0]).status_code == 200
        source = workspace / "data/demo/test/red/0.png"
        upload = client.post("/api/uploads", files={"file": ("flower.png", source.read_bytes(), "image/png")})
        assert upload.status_code == 201
        job = client.post(
            "/api/predict", json={"run_id": run_id, "uploads": [upload.json()["id"]], "save_images": True}
        )
        prediction = wait_job(client, job.json()["id"])
        assert len(prediction["result"]["predictions"]) == 1
        assert prediction["inputs"][0]["name"] == "flower.png"
        assert prediction["result"]["predictions"][0]["input_id"] == upload.json()["id"]
        assert client.get(prediction["inputs"][0]["url"]).status_code == 200
        assert prediction["metrics"] == [] and "progress" not in prediction
        assert prediction["input_count"] == 1 and prediction["elapsed_seconds"] > 0
        assert client.get(wait_report(client, prediction["id"])["report_url"]).status_code == 200
        temporary = client.post("/api/uploads", files={"file": ("temporary.png", source.read_bytes())}).json()
        response = client.post("/api/predict", json={"run_id": run_id, "uploads": [temporary["id"]]})
        transient = wait_job(client, response.json()["id"])
        assert transient["save_images"] is False
        assert transient["result"]["report_status"] == "skipped" and "report_url" not in transient
        assert (
            transient["result"]["predictions"][0]["probabilities"]
            == prediction["result"]["predictions"][0]["probabilities"]
        )
        assert transient["inputs"][0]["name"] == "temporary.png" and transient["inputs"][0]["url"] is None
        assert transient["inputs"][0]["path"] == str(
            workspace / "runs/web/jobs" / transient["id"] / "inputs" / f"{temporary['id']}.png"
        )
        assert not (workspace / "runs/web/uploads" / f"{temporary['id']}.png").exists()
        assert not list((workspace / "runs/web/jobs" / transient["id"]).rglob("*.png"))
        assert client.post(f"/api/jobs/{transient['id']}/clone").status_code == 409
        for endpoint, extra in (("evaluate", {"split": "test"}), ("report", {})):
            result = client.post(f"/api/{endpoint}", json={"run_id": run_id, **extra})
            finished = wait_job(client, result.json()["id"])
            assert client.get(finished["report_url"]).status_code == 200
        refreshed = client.get("/api/runs").json()[0]
        assert refreshed["reports"] == [finished["report_url"]]
        assert len(refreshed["images"]) >= 3
    with TestClient(create_app(workspace)) as client:
        assert len(client.get("/api/jobs").json()) == 5


def test_reject_invalid_inputs(workspace):
    with TestClient(create_app(workspace)) as client:
        assert (
            client.post("/api/predict", json={"run_id": "unused", "uploads": ["same", "same"]}).status_code
            == 422
        )
        assert client.post("/api/train", json={"model_id": "no", "dataset_id": "demo"}).status_code == 400
        assert (
            client.post(
                "/api/train", json={"model_id": "flower_tiny", "dataset_id": "demo", "unknown": True}
            ).status_code
            == 422
        )
        assert (
            client.post("/api/train", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
        )
        assert client.get("/api/catalog", headers={"Host": "evil.example"}).status_code == 400
        assert client.get("/api/artifacts/%2e%2e/pyproject.toml").status_code == 400
        # Windows 将反斜线解析为目录分隔符；Linux 将其视为不存在的文件名。
        assert client.get("/api/artifacts/..%5c..%5cpyproject.toml").status_code in {400, 404}
        assert client.get("/api/artifacts/web/jobs.sqlite3").status_code == 404
        assert (
            client.post("/api/uploads", files={"file": ("bad.png", io.BytesIO(b"no image"))}).status_code
            == 400
        )
        assert client.get("/api/jobs/missing").status_code == 404


def test_queue_cancel_and_restart(workspace):
    queue = JobQueue(workspace)
    job = queue.submit("report", "queued", {"run_id": "unused"})
    assert queue.cancel(job["id"])["status"] == "cancelled"
    job = queue.submit("report", "interrupted", {"run_id": "unused"})
    job["status"] = "running"
    queue.save(job)
    queue.start()
    try:
        assert queue.get(job["id"])["status"] == "interrupted"
        assert (
            json.loads((queue.directory / "jobs" / job["id"] / "spec.json").read_text())["kind"] == "report"
        )
    finally:
        queue.stop()


def test_cancel_running(workspace):
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/api/train",
            json={
                "model_id": "flower_tiny",
                "dataset_id": "demo",
                "epochs": 1000,
                "initialization": "official",
                "offline": True,
            },
        )
        job_id = response.json()["id"]
        for _ in range(100):
            if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
                break
            time.sleep(0.1)
        process = client.app.state.queue.active[job_id][0]
        assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        assert process.poll() is not None


def test_worker_failure_is_reported(workspace):
    app = create_app(workspace)
    with TestClient(app) as client:
        job = app.state.queue.submit("report", "missing bundle", {"run_id": "missing/bundle"})
        for _ in range(100):
            result = client.get(f"/api/jobs/{job['id']}").json()
            if result["status"] == "failed":
                assert result["error"] and "Traceback" in result["log"]
                break
            time.sleep(0.1)
        else:
            pytest.fail("Failed worker was not recorded")


def wait_running(client, ids):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        jobs = [client.get(f"/api/jobs/{job_id}").json() for job_id in ids]
        assert not any(j["status"] in {"failed", "cancelled"} for j in jobs), jobs
        if all(j["status"] == "running" for j in jobs):
            return jobs
        time.sleep(0.1)
    pytest.fail(f"Jobs did not overlap: {jobs}")


def test_parallel_training_and_inference_with_independent_limits(workspace):
    app = create_app(workspace)
    processes = []
    with TestClient(app) as client:
        client.post("/api/scheduler", json={"train": 2, "predict": 2, "auxiliary": 1})
        payload = {
            "model_id": "flower_tiny",
            "dataset_id": "demo",
            "epochs": 1,
            "initialization": "official",
            "offline": True,
        }
        base = wait_job(client, client.post("/api/train", json=payload).json()["id"])
        source = workspace / "data/demo/test/red/0.png"
        upload = client.post("/api/uploads", files={"file": ("flower.png", source.read_bytes())}).json()
        first = client.post(
            "/api/train", json={**payload, "epochs": 1000, "experiment_name": "parallel_a"}
        ).json()["id"]
        duplicate = client.post(
            "/api/train", json={**payload, "epochs": 1000, "experiment_name": "parallel_a"}
        ).json()["id"]
        ids = [
            first,
            *[
                client.post(
                    "/api/train", json={**payload, "epochs": 1000, "experiment_name": f"parallel_{name}"}
                ).json()["id"]
                for name in ("b", "c")
            ],
        ]
        running = wait_running(client, ids[:2])
        assert len({job["pid"] for job in running}) == 2
        assert client.get(f"/api/jobs/{ids[2]}").json()["status"] == "queued"
        assert client.get(f"/api/jobs/{duplicate}").json()["status"] == "queued"
        client.post(f"/api/jobs/{duplicate}/cancel")
        processes = [app.state.queue.active[job_id][0] for job_id in ids[:2]]
        for _ in range(300):
            if all(client.get(f"/api/jobs/{job_id}").json()["metrics"] for job_id in ids[:2]):
                break
            time.sleep(0.1)
        else:
            pytest.fail("Parallel training did not complete validation epochs")
        predictions = [
            client.post("/api/predict", json={"run_id": base["run_id"], "uploads": [upload["id"]]}).json()[
                "id"
            ]
            for _ in range(2)
        ]
        overlapping = wait_running(client, [*ids[:2], *predictions])
        assert len({job["pid"] for job in overlapping}) == 4
        for job_id in predictions:
            assert len(wait_job(client, job_id)["result"]["predictions"]) == 1
        # Lowering capacity is non-destructive. Cancelling one leaves its sibling running.
        response = client.post("/api/scheduler", json={"train": 1, "predict": 2, "auxiliary": 1})
        assert response.status_code == 200
        assert all(process.poll() is None for process in processes)
        client.post(f"/api/jobs/{ids[0]}/cancel")
        assert processes[0].poll() is not None and processes[1].poll() is None
        time.sleep(0.5)
        assert client.get(f"/api/jobs/{ids[2]}").json()["status"] == "queued"
        client.post(f"/api/jobs/{ids[1]}/cancel")
        wait_running(client, [ids[2]])
        processes.append(app.state.queue.active[ids[2]][0])
    assert all(process.poll() is not None for process in processes)
    assert app.state.queue.get(ids[2])["status"] == "cancelled"


def test_scheduler_configuration_persists_and_rejects_invalid(workspace):
    with TestClient(create_app(workspace)) as client:
        assert client.get("/api/scheduler").json()["limits"] == {"train": 0, "predict": 1, "auxiliary": 1}
        for body in ({"train": -1}, {"predict": 9}, {"auxiliary": True}, {"unknown": 3}):
            assert client.post("/api/scheduler", json=body).status_code == 422
        limits = {"train": 3, "predict": 1, "auxiliary": 2}
        assert client.post("/api/scheduler", json=limits).json()["limits"] == limits
    with TestClient(create_app(workspace)) as client:
        assert client.get("/api/scheduler").json()["limits"] == limits
    (workspace / "configs/web.yaml").write_text("train: -1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        create_app(workspace)


def test_parallel_initialization_does_not_read_running_checkpoints(workspace):
    cfg = yaml.safe_load((workspace / "configs/flower/flower_tiny.yaml").read_text(encoding="utf-8"))
    cfg["checkpoint"]["finetune_from"] = "last"
    directory = workspace / "runs/web_test"
    for name, state in (("finished", "SUCCEEDED"), ("active", "RUNNING")):
        run = directory / name
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints/last.ckpt").write_bytes(b"fixture")
        (run / "status.json").write_text(json.dumps({"state": state}))
    queue = JobQueue(workspace)
    job = queue.submit("train", "pinned initialization", {"config": cfg})
    spec = json.loads((queue.directory / "jobs" / job["id"] / "spec.json").read_text(encoding="utf-8"))
    assert spec["config"]["checkpoint"]["finetune_from"] == str(
        (directory / "finished/checkpoints/last.ckpt").resolve()
    )
    assert cfg["checkpoint"]["finetune_from"] == "last"


def test_global_settings_and_status(workspace):
    with TestClient(create_app(workspace)) as client:
        defaults = client.get("/api/settings").json()
        assert defaults["concurrency"] == {"train": 0, "predict": 1, "auxiliary": 1}
        settings = {
            **defaults,
            "appearance": {"theme": "ocean", "mode": "dark"},
            "refresh_interval_seconds": 5,
        }
        assert client.post("/api/settings", json=settings).json() == settings
        # Legacy scheduler clients must preserve appearance and refresh preferences.
        client.post("/api/scheduler", json={"train": 2, "predict": 3, "auxiliary": 1})
        settings["concurrency"] = {"train": 2, "predict": 3, "auxiliary": 1}
        status = client.get("/api/status").json()
        assert status["settings"] == settings
        assert status["scheduler"]["limits"] == settings["concurrency"]
        assert 0 <= status["system"]["cpu_percent"] <= 100
        assert 0 <= status["system"]["memory_percent"] <= 100
        for invalid in (
            {**settings, "appearance": {"theme": "invalid"}},
            {**settings, "refresh_interval_seconds": 0},
            {**settings, "refresh_interval_seconds": True},
            {**settings, "unknown": 2},
        ):
            assert client.post("/api/settings", json=invalid).status_code == 422
        assert client.get("/api/settings").json() == settings
    with TestClient(create_app(workspace)) as client:
        assert client.get("/api/settings").json() == settings


def test_legacy_settings_migrate_without_losing_limits(workspace):
    (workspace / "configs/web.yaml").write_text("train: 5\npredict: 2\nauxiliary: 1\n", encoding="utf-8")
    with TestClient(create_app(workspace)) as client:
        settings = client.get("/api/settings").json()
        assert settings["concurrency"] == {"train": 5, "predict": 2, "auxiliary": 1}
        assert settings["appearance"] == {"theme": "cyber", "mode": "dark"}
        assert client.post("/api/settings", json=settings).status_code == 200
        assert yaml.safe_load((workspace / "configs/web.yaml").read_text(encoding="utf-8")) == settings


def test_task_crud_and_atomic_batch_validation(workspace):
    app = create_app(workspace)
    with TestClient(app) as client:
        queue = app.state.queue
        queue.stopping.set()
        queue.thread.join(timeout=3)
        payload = dict(
            model_id="flower_tiny", dataset_id="demo", epochs=1, initialization="official", offline=True
        )
        first = client.post("/api/train", json=payload).json()
        second = client.post("/api/train", json=payload).json()
        legacy_spec = queue.specification(first["id"])
        legacy_spec["request"].pop("cpu_threads")
        legacy_spec["config"]["trainer"]["cpu_threads"] = 6
        (queue.directory / "jobs" / first["id"] / "spec.json").write_text(
            json.dumps(legacy_spec), encoding="utf-8"
        )
        url = f"/api/jobs/{first['id']}"
        assert client.get(url).json()["parameters"]["epochs"] == 1
        assert client.get(url).json()["parameters"]["cpu_threads"] == 6
        edited = client.patch(url, json={"title": "实验 A", "parameters": {**payload, "epochs": 4}})
        assert edited.status_code == 200, edited.text
        assert queue.specification(first["id"])["config"]["trainer"]["max_epochs"] == 4
        assert [job["id"] for job in queue.list()] == [second["id"], first["id"]]
        assert client.patch(url, json={"title": "  "}).status_code == 422
        assert client.patch(url, json={"parameters": {**payload, "unknown": True}}).status_code == 422
        job = queue.get(first["id"])
        job["status"] = "succeeded"
        queue.save(job)
        assert client.patch(url, json={"parameters": payload}).status_code == 409
        assert client.patch(url, json={"title": "已完成实验"}).status_code == 200
        clone = client.post(url + "/clone").json()
        assert clone["id"] != first["id"] and clone["run_id"] != first["run_id"]
        assert clone["status"] == "queued"
        assert queue.specification(clone["id"])["config"]["trainer"]["max_epochs"] == 4
        assert (
            client.post("/api/jobs/batch-delete", json={"job_ids": [first["id"], "missing"]}).status_code
            == 404
        )
        assert client.get(url).status_code == 200
        result = client.post(
            "/api/jobs/batch-delete", json={"job_ids": [first["id"], first["id"], clone["id"]]}
        )
        assert result.status_code == 200 and len(result.json()["deleted"]) == 2
        assert result.json()["artifacts_preserved"]
        assert (queue.directory / "jobs" / first["id"] / "spec.json").is_file()
        assert client.get(url).status_code == 404
        assert client.delete(f"/api/jobs/{second['id']}").status_code == 200
        assert queue.list() == []
    assert JobQueue(workspace).list() == []


def test_delete_running_job_preserves_other_process_and_files(workspace):
    app = create_app(workspace)
    with TestClient(app) as client:
        ids = [
            client.post(
                "/api/train",
                json={
                    "model_id": "flower_tiny",
                    "dataset_id": "demo",
                    "epochs": 1000,
                    "experiment_name": name,
                    "initialization": "official",
                    "offline": True,
                },
            ).json()["id"]
            for name in ("delete_a", "delete_b")
        ]
        wait_running(client, ids)
        processes = [app.state.queue.active[job_id][0] for job_id in ids]
        assert client.delete(f"/api/jobs/{ids[0]}").status_code == 409
        assert all(process.poll() is None for process in processes)
        result = client.post("/api/jobs/batch-delete", json={"job_ids": [ids[0]], "stop_running": True})
        assert result.status_code == 200
        assert processes[0].poll() is not None and processes[1].poll() is None
        time.sleep(0.6)
        assert client.get(f"/api/jobs/{ids[0]}").status_code == 404
        assert (workspace / "logs/web" / f"{ids[0]}.log").exists()
        assert client.get(f"/api/jobs/{ids[1]}").json()["status"] == "running"


def test_prediction_images_match_ids_not_result_order_and_persist(workspace):
    app = create_app(workspace)
    with TestClient(app) as client:
        queue = app.state.queue
        queue.stopping.set()
        queue.thread.join(timeout=3)
        source = workspace / "data/demo/test/red/0.png"
        uploads = [
            client.post("/api/uploads", files={"file": ("原图 花朵.png", source.read_bytes())}).json()
            for _ in range(2)
        ]
        ids = [item["id"] for item in uploads]
        job = queue.submit("predict", "multi-image mapping", {"run_id": "unused", "uploads": ids, "top_k": 2})
        job.update(
            status="succeeded",
            result={
                "predictions": [
                    {"source": f"C:\\inputs\\{upload_id}.png", "label": label, "probabilities": {label: 1.0}}
                    for upload_id, label in zip(reversed(ids), ("blue", "red"), strict=True)
                ]
            },
        )
        queue.save(job)
        detail = client.get(f"/api/jobs/{job['id']}").json()
        assert [item["id"] for item in detail["inputs"]] == ids
        assert [item["input_id"] for item in detail["result"]["predictions"]] == list(reversed(ids))
        for item in detail["inputs"]:
            assert item["name"] == "原图 花朵.png"
            assert item["path"] == str(workspace / "runs/web/uploads" / f"{item['id']}.png")
            assert client.get(item["url"]).status_code == 200
        # Legacy uploads without filename metadata remain viewable.
        (workspace / "runs/web/uploads" / f"{ids[1]}.json").unlink()
        spec_path = workspace / "runs/web/jobs" / job["id"] / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["upload_names"].pop(ids[1])
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with TestClient(create_app(workspace)) as client:
        detail = client.get(f"/api/jobs/{job['id']}").json()
        assert detail["inputs"][0]["name"] == "原图 花朵.png"
        assert detail["inputs"][1]["name"] == f"{ids[1]}.png"
        assert client.get(detail["inputs"][1]["url"]).status_code == 200


def test_resident_reuse_invalidation_async_report_and_cancel(workspace, monkeypatch):
    app = create_app(workspace)
    with TestClient(app) as client:
        settings = client.get("/api/settings").json()
        settings["resources"]["enabled"] = False
        client.post("/api/settings", json=settings)
        trained = wait_job(
            client,
            client.post(
                "/api/train",
                json={
                    "model_id": "flower_tiny",
                    "dataset_id": "demo",
                    "epochs": 1,
                    "initialization": "official",
                    "offline": True,
                    "cpu_threads": 2,
                },
            ).json()["id"],
        )
        assert app.state.queue.specification(trained["id"])["config"]["trainer"]["cpu_threads"] == 2
        source = workspace / "data/demo/test/red/0.png"
        upload = client.post("/api/uploads", files={"file": ("flower.png", source.read_bytes())}).json()
        payload = {"run_id": trained["run_id"], "uploads": [upload["id"]], "save_images": True}
        queue = app.state.queue
        launch = queue.launch

        def defer_reports(job, category, gpu=None):
            if not job.get("hidden"):
                return launch(job, category, gpu)

        monkeypatch.setattr(queue, "launch", defer_reports)
        first = wait_job(client, client.post("/api/predict", json=payload).json()["id"])
        second = wait_job(client, client.post("/api/predict", json=payload).json()["id"])
        assert first["pid"] == second["pid"]
        assert not first["result"]["cache_hit"] and second["result"]["cache_hit"]
        assert second["result"]["cache_models"] == 1
        assert (
            first["result"]["predictions"][0]["probabilities"]
            == second["result"]["predictions"][0]["probabilities"]
        )
        assert second["result"]["report_status"] == "queued" and "report_url" not in second
        assert all(value >= 0 for value in second["result"]["timings"].values())
        print(
            "CPU 合成数据推理耗时：",
            json.dumps(
                {"cold": first["result"]["timings"], "warm": second["result"]["timings"]}, ensure_ascii=False
            ),
        )
        assert len(client.get("/api/jobs").json()) == 3  # Internal reports are not duplicate tasks.
        assert client.get("/api/jobs/page").json()["total"] == 3
        checkpoint = workspace / "runs" / trained["run_id"] / "checkpoints/best.pt"
        replacement = checkpoint.with_suffix(".tmp")
        replacement.write_bytes(checkpoint.read_bytes())
        replacement.replace(checkpoint)
        third = wait_job(client, client.post("/api/predict", json=payload).json()["id"])
        assert third["pid"] == first["pid"] and not third["result"]["cache_hit"]
        monkeypatch.setattr(queue, "launch", launch)
        assert client.get(wait_report(client, second["id"])["report_url"]).status_code == 200
        wait_report(client, first["id"])
        wait_report(client, third["id"])
        # Cancellation kills only the selected inference slot, including a warm one.
        with queue.lock:
            cancelled = queue.submit("predict", "cancel resident", payload)
            queue.launch(cancelled, "predict")
            process = queue.active[cancelled["id"]][0]
            queue.cancel(cancelled["id"])
        assert process.poll() is not None
        fourth = wait_job(client, client.post("/api/predict", json=payload).json()["id"])
        assert fourth["pid"] != first["pid"] and not fourth["result"]["cache_hit"]
        resident = next(iter(queue.residents))
        resident.kill()
        resident.wait(timeout=10)
        fifth = wait_job(client, client.post("/api/predict", json=payload).json()["id"])
        assert fifth["pid"] != fourth["pid"] and not fifth["result"]["cache_hit"]
        residents = list(queue.residents)
    assert all(process.poll() is not None for process in residents)
