"""Temporary image lifetime and shared-upload ownership."""

import json

import pytest
from test_api import workspace as workspace_fixture

from clsweb.app import PredictRequest
from clsweb.jobs import JobQueue


@pytest.fixture
def workspace(tmp_path):
    return workspace_fixture.__wrapped__(tmp_path)


def uploaded(queue):
    upload_id = "a" * 32
    folder = queue.directory / "uploads"
    folder.mkdir(exist_ok=True)
    path = folder / f"{upload_id}.png"
    path.write_bytes(b"test image placeholder")
    path.with_suffix(".json").write_text(json.dumps({"name": "test.png"}))
    return upload_id, path


@pytest.mark.parametrize("save_other", [True, False])
def test_cancel_preserves_shared_upload_until_last_transient_job(workspace, save_other):
    queue = JobQueue(workspace)
    upload_id, original = uploaded(queue)
    spec = {"run_id": "unused", "uploads": [upload_id], "save_images": False}
    first = queue.submit("predict", "temporary", spec)
    second = queue.submit("predict", "shared", dict(spec, save_images=save_other))
    inputs = queue.directory / "jobs" / first["id"] / "inputs"
    inputs.mkdir()
    (inputs / original.name).write_bytes(original.read_bytes())
    queue.cancel(first["id"])
    assert original.exists() and not list(inputs.glob("*.png"))
    queue.cancel(second["id"])
    assert original.exists() is save_other
    assert original.with_suffix(".json").exists() is save_other
    assert queue.specification(first["id"])["upload_names"][upload_id] == "test.png"


def test_failure_and_restart_release_temporary_images(workspace):
    queue = JobQueue(workspace)
    upload_id, original = uploaded(queue)
    spec = {"run_id": "unused", "uploads": [upload_id], "save_images": False}
    failed = queue.submit("predict", "failure", spec)
    queue.finish(failed["id"], 1)
    assert queue.get(failed["id"])["status"] == "failed" and not original.exists()
    uploaded(queue)
    interrupted = queue.submit("predict", "interruption", spec)
    interrupted.update(status="running")
    queue.save(interrupted)
    queue.start()
    try:
        assert queue.get(interrupted["id"])["status"] == "interrupted" and not original.exists()
    finally:
        queue.stop()


def test_save_images_is_strict_opt_in():
    assert PredictRequest(run_id="model", uploads=["a" * 32]).save_images is False
    with pytest.raises(ValueError):
        PredictRequest(run_id="model", uploads=["a" * 32], save_images="false")
