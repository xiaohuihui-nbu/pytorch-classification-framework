"""Execute one trusted server-generated job specification in a fresh process."""

import json
import os
import sys
import threading
import traceback
from pathlib import Path
from time import perf_counter


def execute(spec):
    if spec["kind"] == "predict":
        from .predictor import PredictionSession
        from .settings import Inference

        spec.setdefault("inference", Inference().model_dump())
        return PredictionSession().execute(spec)
    if spec.get("parent_job_id"):
        started = perf_counter()
        from clsframework.visualization import PredictionResult, prediction_report

        root = Path(spec["root"])
        parent_folder = root / "runs/web/jobs" / spec["parent_job_id"]
        predictions = [
            PredictionResult(json.loads(line))
            for line in (parent_folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        report = prediction_report(
            predictions, parent_folder / "predictions_visuals", top_k=spec["top_k"], max_images=20
        )
        return {
            "report": report.relative_to(root / "runs").as_posix(),
            "report_seconds": perf_counter() - started,
        }
    from clsframework.api import Classifier
    from clsframework.config import Config

    root, folder = Path(spec["root"]), Path(spec["directory"])
    kind = spec["kind"]
    if kind == "train":
        os.environ["CLS_ACTIVE_RUN"] = str(root / "runs" / spec["run_id"])
        cfg = Config.model_validate(spec["config"])
        options = (
            {"finetune_from": cfg.checkpoint.finetune_from}
            if cfg.checkpoint.finetune_from
            else {"fresh": True}
        )
        run = Classifier(cfg).train(**options)
        return {"run_id": run.relative_to(root / "runs").as_posix()}
    model = Classifier(root / "runs" / spec["run_id"])
    if kind == "evaluate":
        metrics = model.val(split=spec["split"], output=folder / "evaluation")
        return {
            "metrics": metrics,
            "report": str(model.last_report.relative_to(root / "runs")) if model.last_report else None,
        }
    if kind == "report":
        report = model.report(output=folder / "report")
        return {"report": str(Path(report).relative_to(root / "runs"))}
    raise ValueError(f"Unsupported job kind: {kind}")


def watch_parent():
    # Avoid orphaned training if the API process crashes or is forcibly closed.
    import psutil

    parent = psutil.Process(os.getppid())

    def watch():
        while parent.is_running():
            threading.Event().wait(1)
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        os._exit(2)

    threading.Thread(target=watch, daemon=True).start()


def write_result(target, result):
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(target)


def main():
    watch_parent()
    import psutil

    primary = int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    target = Path(spec["directory"]) / "result.json"
    stopped = threading.Event()
    peak_rss = [0]
    process = psutil.Process()

    def sample_memory():
        while not stopped.is_set():
            try:
                processes = [process, *process.children(recursive=True)]
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss[0] = max(peak_rss[0], rss)
            except psutil.NoSuchProcess:
                return
            stopped.wait(0.2)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        result = execute(spec)
    except Exception as exc:
        if primary:
            write_result(target, {"error": str(exc)})
        traceback.print_exc()
        return 1
    finally:
        stopped.set()
        sampler.join(timeout=1)
    result["peak_worker_rss_mb"] = peak_rss[0] / 1024**2
    if primary:
        write_result(target, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
