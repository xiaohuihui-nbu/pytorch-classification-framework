import json

import pytest
import torch
from PIL import Image

pytest.importorskip("fastapi")

from clsweb.jobs import JobQueue, now  # noqa: E402
from clsweb.predictor import PredictionSession  # noqa: E402
from clsweb.settings import Inference  # noqa: E402


def test_lru_budget_and_content_changes_with_preserved_timestamp(tmp_path, monkeypatch):
    from clsframework import inference

    for name in ("a", "b", "large"):
        path = tmp_path / "runs" / name / "checkpoints/best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"original")
    uploaded = tmp_path / "runs/web/uploads/image.png"
    uploaded.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "blue").save(uploaded)
    loaded = []

    def load(path):
        loaded.append(path)
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
        if path.parent.parent.name == "large":
            model.register_buffer("padding", torch.zeros(5_000_000))
        return (model.eval(), {"task": "multiclass"}, ["blue", "red"], {
            "image_size": 32, "interpolation": "bilinear", "crop_pct": 1,
            "mean": [0, 0, 0], "std": [1, 1, 1], "color_mode": "RGB",
        }, 0.5, {})

    monkeypatch.setattr(inference, "load_model", load)
    session = PredictionSession()
    settings = Inference(cache_models=1, cache_mb=16).model_dump()

    def predict(name):
        folder = tmp_path / "jobs" / str(len(list((tmp_path / "jobs").glob("*"))))
        folder.mkdir(parents=True)
        return session.execute({"root": str(tmp_path), "directory": str(folder), "run_id": name,
                                "uploads": ["image"], "inference": settings})

    assert not predict("a")["cache_hit"]
    assert predict("a")["cache_hit"]
    checkpoint = tmp_path / "runs/a/checkpoints/best.pt"
    stamp = checkpoint.stat()
    checkpoint.write_bytes(b"modified")  # Same length and explicitly preserved modification time.
    import os

    os.utime(checkpoint, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not predict("a")["cache_hit"]
    assert not predict("b")["cache_hit"]
    assert not predict("a")["cache_hit"]  # The least recently used model was evicted.
    assert len(loaded) == 4
    for _ in range(2):
        result = predict("large")
        assert not result["cache_hit"] and result["cache_models"] == 0


def test_report_failure_does_not_discard_predictions(tmp_path):
    queue = JobQueue(tmp_path)
    job = queue.submit("predict", "prediction", {"run_id": "model/run", "top_k": 2})
    job.update(status="running", started_at=now())
    queue.save(job)
    folder = queue.directory / "jobs" / job["id"]
    (folder / "result.json").write_text(json.dumps({"predictions": [{"label": "blue"}]}), encoding="utf-8")
    queue.finish(job["id"], 0)
    parent = queue.get(job["id"])
    child = queue.get(parent["result"]["report_job_id"])
    (queue.directory / "jobs" / child["id"] / "result.json").write_text(
        json.dumps({"error": "report failed"}), encoding="utf-8")
    queue.finish(child["id"], 1)
    parent = queue.get(job["id"])
    assert parent["status"] == "succeeded"
    assert parent["result"]["predictions"] == [{"label": "blue"}]
    assert parent["result"]["report_status"] == "failed"
    assert parent["result"]["report_error"] == "report failed"
    queue.delete([job["id"]])
    assert queue.page()["total"] == 0
