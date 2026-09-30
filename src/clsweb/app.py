"""HTTP API for the local or LAN classification workbench."""

import argparse
import asyncio
import codecs
import io
import ipaddress
import json
import os
import re
import sys
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .devices import resolve_device
from .jobs import JobQueue
from .resources import gpu_devices, snapshot
from .settings import Concurrency, WebSettings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def within(base: Path, relative: str):
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base.resolve()) or candidate == base.resolve():
        raise HTTPException(400, "路径必须位于指定目录内")
    return candidate


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobEdit(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    parameters: dict | None = None

    @field_validator("title")
    @classmethod
    def trim_title(cls, value):
        if value is not None and not value.strip():
            raise ValueError("任务名称不能为空")
        return value.strip() if value is not None else value


class JobDelete(StrictModel):
    job_ids: list[str] = Field(min_length=1, max_length=200)
    stop_running: bool = False


class TrainRequest(StrictModel):
    model_id: str
    dataset_id: str
    experiment_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=512)
    learning_rate: float = Field(default=0.0003, gt=0, le=1)
    num_workers: int = Field(default=0, ge=0, le=16)
    cpu_threads: int = Field(default=2, ge=1, le=32)
    device: Literal["global", "cpu", "gpu", "auto"] = "global"
    seed: int = Field(default=42, ge=0, le=2147483647)
    training_mode: Literal["full_finetune", "linear_probe"] = "full_finetune"
    initialization: Literal["last", "official"] = "last"
    offline: bool = False
    priority: int = Field(default=0, ge=0, le=10)
    timeout_minutes: int = Field(default=0, ge=0, le=10080)
    gpu_index: int | None = Field(default=None, ge=0, le=127)


class RunRequest(StrictModel):
    run_id: str
    priority: int = Field(default=0, ge=0, le=10)
    timeout_minutes: int = Field(default=0, ge=0, le=10080)


class PredictRequest(RunRequest):
    uploads: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    save_images: bool = Field(default=False, strict=True)

    @field_validator("uploads")
    @classmethod
    def unique_uploads(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("同一张上传图片不能在一个推理任务中重复提交")
        return value


class EvaluateRequest(RunRequest):
    split: Literal["val", "test"] = "val"


def create_app(root: Path | None = None, *, host: str = "127.0.0.1", port: int = 8000):
    address = ipaddress.IPv4Address(host)
    if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
        raise ValueError("Web 监听地址必须是具体的本机或局域网 IPv4 地址")
    if not 1 <= port <= 65535:
        raise ValueError("Web 端口必须在 1～65535 之间")
    extra_origins = set()
    extra_hosts = set()
    if raw_origins := os.environ.get("CLS_WEB_ORIGINS", ""):
        for item in raw_origins.split(","):
            origin = item.strip()
            try:
                parsed = urlsplit(origin)
                ip = ipaddress.IPv4Address(parsed.hostname)
                valid = (
                    parsed.scheme == "http"
                    and parsed.port is not None
                    and 1 <= parsed.port <= 65535
                    and origin == f"http://{ip}:{parsed.port}"
                    and (ip.is_private or ip.is_loopback)
                    and not ip.is_unspecified
                    and not ip.is_multicast
                )
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise ValueError("CLS_WEB_ORIGINS 必须是逗号分隔的 http://本机或局域网IPv4:端口")
            extra_origins.add(origin)
            extra_hosts.add(str(ip))
    root = (root or Path.cwd()).resolve()
    queue = JobQueue(root)
    catalog = {}

    @asynccontextmanager
    async def lifespan(app):
        from clsframework.config import load_config

        for path in sorted((root / "configs/flower").glob("flower_*.yaml")):
            if path.stem == "flower_base":
                continue
            cfg = load_config(path)
            if cfg.runtime.plugins:
                raise ValueError("Web 模型目录暂不支持 plugins 配置")
            catalog[path.stem] = cfg
        queue.start()
        try:
            yield
        finally:
            queue.stop()

    app = FastAPI(title="Classification Studio", version="1.0", lifespan=lifespan)
    app.state.queue = queue
    app.state.catalog = catalog
    web_port = int(os.environ.get("CLS_WEB_PORT", "5173"))
    if not 1 <= web_port <= 65535:
        raise ValueError("CLS_WEB_PORT 必须在 1～65535 之间")
    origins = {
        f"http://{name}:{number}"
        for name in ("localhost", "127.0.0.1")
        for number in (web_port, 8000, port)
    }
    origins.add(f"http://{host}:{port}")
    origins.update(extra_origins)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", host, *sorted(extra_hosts)],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def allowed_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"detail": "仅接受已配置的 Web 页面请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def datasets():
        result = []
        for path in sorted((root / "data").glob("*")):
            if not path.is_dir() or not (path / "train").is_dir() or not (path / "val").is_dir():
                continue
            if not path.resolve().is_relative_to((root / "data").resolve()):
                continue
            classes = sorted(p.name for p in (path / "train").iterdir() if p.is_dir())
            counts = {
                split: sum(
                    1 for p in (path / split).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                )
                for split in ("train", "val", "test")
            }
            result.append(dict(id=path.name, classes=classes, counts=counts))
        return result

    def run_folder(run_id):
        path = within(root / "runs", run_id)
        if not ((path / "checkpoints/best.pt").is_file() or (path / "bundle/bundle_manifest.json").is_file()):
            raise HTTPException(404, "未找到可推理的模型，请先完成训练")
        return path

    def artifact(path):
        return "/api/artifacts/" + quote(path.relative_to(root / "runs").as_posix(), safe="/")

    def run_info(path, report_jobs):
        run_id = path.relative_to(root / "runs").as_posix()
        generated = next(
            (job.get("result", {}).get("report") for job in report_jobs if job.get("run_id") == run_id), None
        )
        reports = [artifact(p) for p in path.glob("visualizations/*.html")]
        reports += [artifact(p) for p in path.glob("visuals/*.html")]
        images = [
            artifact(p)
            for directory in ("visuals", "visualizations")
            for p in (path / directory).glob("*.png")
        ]
        if generated:
            report_path = within(root / "runs", generated)
            if report_path.is_file():
                reports = [artifact(report_path)]
                images = [artifact(p) for p in report_path.parent.glob("*.png")]
        return dict(
            id=run_id,
            name=path.parent.name,
            classes=read_json(path / "classes.json", []),
            status=read_json(path / "status.json", {}),
            metrics=rows(path / "metrics.jsonl"),
            reports=reports,
            images=images,
            config=read_json(path / "config.requested.json", {}),
        )

    @app.get("/api/catalog")
    def get_catalog():
        return dict(
            models=[
                dict(id=key, name=cfg.model.name, provider=cfg.model.provider) for key, cfg in catalog.items()
            ],
            datasets=datasets(),
            defaults=TrainRequest.model_fields["epochs"].default,
            scheduler=queue.scheduler_state(),
            gpus=gpu_devices(),
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "workspace": str(root), "scheduler": queue.scheduler_state()}

    @app.get("/api/scheduler")
    def scheduler():
        return queue.scheduler_state()

    @app.get("/api/settings")
    def get_settings():
        return queue.settings_state()

    @app.post("/api/settings")
    def save_settings(body: WebSettings):
        validate_device(body.training.device, body.training.gpu_indices)
        return queue.configure_settings(body)

    @app.get("/api/status")
    def status():
        return {
            "scheduler": queue.scheduler_state(),
            "settings": queue.settings_state(),
            "system": snapshot(),
            "workspace": root.name,
            "server_time": datetime.now(UTC).isoformat(),
        }

    @app.post("/api/scheduler")
    def configure_scheduler(body: Concurrency):
        return queue.configure(body)

    @app.get("/api/runs")
    def runs():
        paths = {
            p.parent.parent
            for pattern in ("*/*/checkpoints/best.pt", "*/*/bundle/bundle_manifest.json")
            for p in (root / "runs").glob(pattern)
        }
        report_jobs = [
            job
            for job in queue.list()
            if job["kind"] == "report" and job["status"] == "succeeded" and not job.get("hidden")
        ]
        return [
            run_info(p, report_jobs)
            for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
            if p.resolve().is_relative_to(root / "runs")
        ]

    @app.get("/api/jobs")
    def jobs(summary: bool = False):
        visible = [job for job in queue.list() if not job.get("hidden")]
        return [queue.summary(job) for job in visible] if summary else visible

    @app.get("/api/jobs/page")
    def jobs_page(
        limit: int = Query(12, ge=1, le=200),
        offset: int = Query(0, ge=0),
        query: str = Query("", max_length=200),
        kind: Literal["", "train", "predict", "evaluate", "report"] = "",
        status: Literal["", "queued", "running", "succeeded", "failed", "cancelled", "interrupted"] = "",
        order: Literal["newest", "oldest", "name"] = "newest",
    ):
        return queue.page(limit, offset, query, kind, status, order)

    @app.get("/api/events")
    async def events(request: Request, job_id: str = Query("", max_length=64)):
        if job_id and not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(422, "无效任务 ID")

        async def stream():
            previous_detail = None
            counter = 0
            offset = None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            # Every connection starts with current state, so reconnects recover missed changes.
            while not await request.is_disconnected():
                packet = {"status": await asyncio.to_thread(status), "revision": queue.revision}
                if job_id:
                    try:
                        detail = await asyncio.to_thread(job_detail, job_id, False)
                    except HTTPException as exc:
                        if exc.status_code != 404:
                            raise
                        detail = {"id": job_id, "deleted": True}
                    log_path = root / "logs/web" / f"{job_id}.log"
                    if log_path.is_file():
                        size = log_path.stat().st_size
                        reset = offset is None or size < offset or size - offset > 48000
                        start = max(0, size - 48000) if reset else offset
                        if size > start or reset:
                            if reset:
                                decoder.reset()
                            with log_path.open("rb") as stream_file:
                                stream_file.seek(start)
                                data = stream_file.read(size - start)
                            packet["log"] = {"reset": reset, "text": decoder.decode(data)}
                        offset = size
                    encoded = json.dumps(detail, ensure_ascii=False)
                    if encoded != previous_detail:
                        packet["detail"] = detail
                        previous_detail = encoded
                counter += 1
                yield f"id: {counter}\nevent: snapshot\ndata: {json.dumps(packet, ensure_ascii=False)}\n\n"
                await asyncio.sleep(queue.settings.refresh_interval_seconds)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def parameters(job):
        spec = read_json(queue.directory / "jobs" / job["id"] / "spec.json", {})
        if "request" in spec:
            request = dict(spec["request"])
            if job["kind"] == "train":
                request.setdefault(
                    "cpu_threads", spec.get("config", {}).get("trainer", {}).get("cpu_threads", 2)
                )
            return request
        if job["kind"] != "train":
            allowed = {
                "predict": ("run_id", "uploads", "top_k", "save_images"),
                "evaluate": ("run_id", "split"),
                "report": ("run_id",),
            }[job["kind"]]
            return {key: spec[key] for key in (*allowed, "priority", "timeout_minutes") if key in spec}
        cfg = spec.get("config")
        if not cfg:
            return None
        model_id = next(
            (
                key
                for key, value in catalog.items()
                if value.model.name == cfg["model"]["name"]
                and value.model.provider == cfg["model"]["provider"]
            ),
            None,
        )
        return dict(
            model_id=model_id,
            dataset_id=Path(cfg["dataset"]["root"]).name,
            experiment_name=cfg["experiment"]["name"],
            epochs=cfg["trainer"]["max_epochs"],
            batch_size=cfg["loader"]["batch_size_per_device"],
            learning_rate=cfg["optimizer"]["lr"],
            num_workers=cfg["loader"]["num_workers"],
            cpu_threads=cfg["trainer"]["cpu_threads"],
            device=cfg["trainer"]["accelerator"],
            seed=cfg["experiment"]["seed"],
            training_mode=cfg["model"]["training_mode"],
            initialization="last" if cfg["checkpoint"].get("finetune_from") else "official",
            offline=cfg["runtime"]["offline"],
        )

    def job_action(action):
        try:
            return action()
        except KeyError as exc:
            raise HTTPException(404, "任务不存在，列表可能已更新") from exc
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/batch-delete")
    def batch_delete(body: JobDelete):
        return job_action(lambda: queue.delete(body.job_ids, body.stop_running))

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, stop_running: bool = False):
        return job_action(lambda: queue.delete([job_id], stop_running))

    @app.post("/api/jobs/{job_id}/clone", status_code=202)
    def clone_job(job_id: str):
        return job_action(lambda: queue.clone(job_id))

    @app.patch("/api/jobs/{job_id}")
    def edit_job(job_id: str, body: JobEdit):
        def edit():
            with queue.lock:
                job = queue.get(job_id)
                spec = None
                if body.parameters is not None:
                    if job["status"] != "queued":
                        raise ValueError("任务已开始或结束，只能改名；请复制后调整执行参数")
                    if job["kind"] == "train":
                        request = TrainRequest.model_validate(body.parameters)
                        spec = training_spec(request)
                    else:
                        schema = {
                            "predict": PredictRequest,
                            "evaluate": EvaluateRequest,
                            "report": RunRequest,
                        }
                        request = schema[job["kind"]].model_validate(body.parameters)
                        run_folder(request.run_id)
                        if job["kind"] == "predict":
                            validate_uploads(request.uploads)
                        spec = request.model_dump()
                return queue.update(job_id, body.title, spec)

        return job_action(edit)

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str, include_log: bool = True):
        try:
            job = queue.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc
        log_path = root / "logs/web" / f"{job['id']}.log"
        if include_log and log_path.exists():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 48000))
                job["log"] = stream.read().decode("utf-8", errors="replace")
        elif include_log:
            job["log"] = "等待队列调度…"
        job["metrics"] = (
            rows(within(root / "runs", job["run_id"]) / "metrics.jsonl")
            if job["kind"] == "train" and job["run_id"]
            else []
        )
        if job["started_at"]:
            end = datetime.fromisoformat(job["finished_at"]) if job["finished_at"] else datetime.now(UTC)
            job["elapsed_seconds"] = max(0, (end - datetime.fromisoformat(job["started_at"])).total_seconds())
        if job["kind"] == "predict":
            spec = read_json(queue.directory / "jobs" / job["id"] / "spec.json", {})
            job["save_images"] = spec.get("save_images", True)
            discarded = not job["save_images"] and job["status"] not in {"queued", "running"}
            job["input_count"] = len(spec.get("uploads", []))
            inputs = []
            for upload_id in spec.get("uploads", []):
                if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
                    continue
                uploaded = root / "runs/web/uploads" / f"{upload_id}.png"
                snapshot = queue.directory / "jobs" / job["id"] / "inputs" / f"{upload_id}.png"
                image_path = snapshot if snapshot.is_file() else uploaded
                metadata = read_json(uploaded.with_suffix(".json"), {})
                inputs.append(
                    {
                        "id": upload_id,
                        "name": spec.get("upload_names", {}).get(upload_id)
                        or metadata.get("name")
                        or f"{upload_id}.png",
                        "path": str(snapshot if discarded and job["status"] == "succeeded" else image_path),
                        "url": artifact(image_path) if not discarded and image_path.is_file() else None,
                    }
                )
            job["inputs"] = inputs
            # Match by the uploaded ID, not prediction order (directory inference sorts filenames).
            known = {item["id"] for item in inputs}
            for prediction in job.get("result", {}).get("predictions", []):
                input_id = Path(
                    (prediction.get("path") or prediction.get("source") or "").replace("\\", "/")
                ).stem
                prediction["input_id"] = input_id if input_id in known else None
        if job["kind"] == "train":
            spec = read_json(queue.directory / "jobs" / job["id"] / "spec.json", {})
            total = spec.get("config", {}).get("trainer", {}).get("max_epochs", 0)
            job["progress"] = {"completed": len(job["metrics"]), "total": total}
            events = rows(within(root / "runs", job["run_id"]) / "events.jsonl") if job["run_id"] else []
            completed = [event for event in events if event.get("event") == "epoch_completed"]
            if completed and "epoch_seconds" in completed[-1]:
                event = completed[-1]
                job["performance"] = {
                    key: event[key] for key in ("epoch", "epoch_seconds", "train_images_per_second")
                }
        result = job.get("result", {})
        job["blocked_reason"] = queue.blocked.get(job_id)
        job["parameters"] = parameters(job)
        if result.get("report"):
            job["report_url"] = artifact(within(root / "runs", result["report"]))
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return queue.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc

    def validate_device(device, indices):
        import torch

        try:
            return resolve_device(device, indices, gpu_devices(), torch.cuda.is_available())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def training_spec(body: TrainRequest):
        from clsframework.config import Config

        if body.model_id not in catalog:
            raise HTTPException(400, "请选择目录中的模型")
        if body.device == "global":
            if body.gpu_index is not None:
                raise HTTPException(422, "跟随全局设置时不能单独指定 GPU")
            selection = queue.settings.training
            device, indices = validate_device(selection.device, selection.gpu_indices)
        else:
            device, indices = validate_device(
                body.device, [] if body.gpu_index is None else [body.gpu_index]
            )
        if body.dataset_id not in {item["id"] for item in datasets()}:
            raise HTTPException(400, "请选择包含 train/val 类别目录的数据集")
        cfg = catalog[body.model_id].model_dump(mode="json")
        cfg["experiment"].update(output_root=str(root / "runs"), seed=body.seed)
        if body.experiment_name:
            cfg["experiment"]["name"] = body.experiment_name
        elif Path(cfg["dataset"]["root"]).resolve() != within(root / "data", body.dataset_id):
            cfg["experiment"]["name"] += "_" + re.sub(r"[^\w-]", "_", body.dataset_id)
        cfg["dataset"].update(
            provider="imagefolder",
            root=str(within(root / "data", body.dataset_id)),
            name=body.dataset_id,
            manifest=None,
            classes_file=None,
            num_classes="auto",
            limit_per_split=None,
            splits={"train": "train", "val": "val", "test": "test"},
        )
        cfg["model"].update(training_mode=body.training_mode, num_classes="auto")
        cfg["trainer"].update(
            max_epochs=body.epochs,
            accelerator=device,
            devices=max(1, len(indices)),
            enable_progress_bar=False,
            cpu_threads=body.cpu_threads,
        )
        cfg["loader"].update(batch_size_per_device=body.batch_size, num_workers=body.num_workers)
        cfg["optimizer"]["lr"] = body.learning_rate
        cfg["scheduler"].update(
            warmup_epochs=min(cfg["scheduler"]["warmup_epochs"], body.epochs - 1),
            min_lr=min(cfg["scheduler"]["min_lr"], body.learning_rate),
        )
        cfg["checkpoint"].update(
            resume_from=None, finetune_from="last" if body.initialization == "last" else None
        )
        cfg["runtime"].update(
            offline=body.offline,
            cache_dir=str(root / "cache"),
            weights_dir=str(root / "weights"),
            cache_image_validation=True,
        )
        cfg["logging"].update(directory=str(root / "logs/web"), tensorboard=False)
        cfg["visualization"]["enabled"] = True
        try:
            cfg = Config.model_validate(cfg).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "config": cfg,
            "request": body.model_dump(),
            "device_selection": {"device": device, "gpu_indices": indices},
        }

    @app.post("/api/train", status_code=202)
    def train(body: TrainRequest):
        spec = training_spec(body)
        return queue.submit(
            "train",
            f"{catalog[body.model_id].model.name} · {body.epochs} epochs",
            spec,
        )

    @app.post("/api/uploads", status_code=201)
    async def upload(file: UploadFile):
        content = await file.read(20 * 1024 * 1024 + 1)
        await file.close()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "单张图片不能超过 20 MB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as source:
                    if source.width * source.height > 20_000_000:
                        raise ValueError("图片不能超过 2000 万像素")
                    picture = ImageOps.exif_transpose(source).convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(400, f"无效图片：{exc}") from exc
        upload_id = uuid4().hex
        target = root / "runs/web/uploads" / f"{upload_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.save(target)
        name = (file.filename or "图片").replace("\\", "/").rsplit("/", 1)[-1]
        target.with_suffix(".json").write_text(
            json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8"
        )
        return {"id": upload_id, "name": name, "url": artifact(target)}

    def validate_uploads(uploads):
        for upload_id in uploads:
            if (
                not re.fullmatch(r"[0-9a-f]{32}", upload_id)
                or not (root / "runs/web/uploads" / f"{upload_id}.png").is_file()
            ):
                raise HTTPException(400, "图片不存在，请重新上传")

    @app.post("/api/predict", status_code=202)
    def predict(body: PredictRequest):
        with queue.lock:
            run_folder(body.run_id)
            validate_uploads(body.uploads)
            return queue.submit("predict", f"图片推理 · {len(body.uploads)} 张", body.model_dump())

    @app.post("/api/evaluate", status_code=202)
    def evaluate(body: EvaluateRequest):
        run_folder(body.run_id)
        return queue.submit("evaluate", f"独立评估 · {body.split}", body.model_dump())

    @app.post("/api/report", status_code=202)
    def report(body: RunRequest):
        run_folder(body.run_id)
        return queue.submit("report", "生成可视化报告", body.model_dump())

    @app.get("/api/artifacts/{relative:path}")
    def get_artifact(relative: str):
        path = within(root / "runs", relative)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".html", ".csv"} or not path.is_file():
            raise HTTPException(404, "文件不存在或不允许访问")
        headers = {
            "Content-Security-Policy": "sandbox; default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
        }
        return FileResponse(path, headers=headers)

    return app


def main():
    import uvicorn

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="本机分类实验室后端")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if not (args.workspace / "configs/flower").is_dir():
        parser.error("请在项目根目录运行，或使用 --workspace 指定项目路径")
    uvicorn.run(create_app(args.workspace), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
