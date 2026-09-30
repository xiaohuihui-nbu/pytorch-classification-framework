"""CPU inference session with a bounded, version-aware model cache."""

import json
import os
import shutil
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def version(source):
    # Atomic checkpoint replacement and legacy metadata changes invalidate the cache.
    files = sorted(source.iterdir()) if source.is_dir() else [source]
    status = source.parent.parent / "status.json"
    if source.parent.name == "checkpoints" and status.is_file():
        files.append(status)
    return tuple(
        (str(p), s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
        for p in files
        if p.is_file()
        for s in [p.stat()]
    )


class PredictionSession:
    def __init__(self):
        self.cache = OrderedDict()

    def trim(self, settings):
        budget = settings["cache_mb"] * 1024**2
        while self.cache and (
            len(self.cache) > settings["cache_models"] or sum(v[2] for v in self.cache.values()) > budget
        ):
            self.cache.popitem(last=False)

    def execute(self, spec):
        started = perf_counter()
        worker_started_at = datetime.now(UTC).isoformat()
        import psutil
        import torch

        from clsframework.inference import load_model, model_source, predict
        from clsframework.utils import file_hash

        import_seconds = perf_counter() - started
        root, folder = Path(spec["root"]), Path(spec["directory"])
        settings = spec["inference"]
        torch.set_num_threads(settings["cpu_threads"])
        source = model_source(root / "runs" / spec["run_id"])
        key = str(source.resolve())
        checking = perf_counter()
        files = sorted(source.iterdir()) if source.is_dir() else [source]
        status = source.parent.parent / "status.json"
        if source.parent.name == "checkpoints" and status.is_file():
            files.append(status)
        stamp = (version(source), tuple((str(p), file_hash(p)) for p in files if p.is_file()))
        verify_seconds = perf_counter() - checking
        self.trim(settings)
        previous = self.cache.pop(key, None)
        hit = previous is not None and previous[0] == stamp
        loading = perf_counter()
        if hit:
            data = previous[1]
        else:
            del previous
            data = load_model(source)
            if version(source) != stamp[0]:
                raise ValueError("模型文件在加载期间变化，请重新提交推理")
        load_seconds = perf_counter() - loading
        model = data[0]
        size = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
        self.cache[key] = (stamp, data, size)
        self.trim(settings)
        inputs = folder / "inputs"
        inputs.mkdir()
        snapshot_started = perf_counter()
        for upload in dict.fromkeys(spec["uploads"]):  # Preserve legacy queued jobs with repeated IDs.
            original = root / "runs/web/uploads" / f"{upload}.png"
            target = inputs / original.name
            try:
                os.link(original, target)
            except OSError:
                shutil.copy2(original, target)
        snapshot_seconds = perf_counter() - snapshot_started
        timings = {}
        predict(
            source,
            inputs,
            folder / "predictions.jsonl",
            batch_size=settings["batch_size"],
            model_data=data,
            timings=timings,
        )
        timings.update(
            model_load_seconds=load_seconds,
            model_verify_seconds=verify_seconds,
            snapshot_seconds=snapshot_seconds,
            runtime_import_seconds=import_seconds,
            worker_seconds=perf_counter() - started,
        )
        predictions = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return {
            "predictions": predictions,
            "timings": timings,
            "cache_hit": hit,
            "worker_started_at": worker_started_at,
            "worker_rss_mb": psutil.Process().memory_info().rss / 1024**2,
            "cache_models": len(self.cache),
            "report_status": "queued" if spec.get("save_images", True) else "skipped",
        }
