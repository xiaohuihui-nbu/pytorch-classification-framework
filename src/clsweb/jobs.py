"""Persistent multi-process scheduler with independent per-kind capacity."""

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil
import yaml
from filelock import FileLock

from .devices import choose_gpus, gpu_wait_reason
from .resources import snapshot
from .settings import Concurrency, WebSettings


def now():
    return datetime.now(UTC).isoformat()


def terminate(process):
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        process.wait(timeout=10)
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass


class JobQueue:
    def __init__(self, root: Path):
        self.root = root
        self.settings_path = root / "configs/web.yaml"
        self.settings = WebSettings.load(self.settings_path)
        self.limits = self.settings.concurrency
        self.directory = root / "runs/web"
        self.directory.mkdir(parents=True, exist_ok=True)
        (root / "logs/web").mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "jobs.sqlite3"
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.wakeup = threading.Event()
        self.active = {}
        self.residents = {}
        self.resident_errors = {}
        self.revision = 0
        self.blocked = {}
        self.reservations = {}
        self.lease = FileLock(str(self.directory / "server.lock"), timeout=0)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            for field in ("status", "kind", "created_at"):
                db.execute(
                    f"CREATE INDEX IF NOT EXISTS jobs_{field} ON jobs(json_extract(payload, '$.{field}'))"
                )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list(self, status=None):
        with self.connect() as db:
            query = "SELECT payload FROM jobs"
            arguments = ()
            if status is not None:
                query += " WHERE json_extract(payload, '$.status') = ?"
                arguments = (status,)
            return [json.loads(row[0]) for row in db.execute(query + " ORDER BY rowid DESC", arguments)]

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def save(self, job):
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (job["id"], json.dumps(job)),
            )
        self.revision += 1

    def summary(self, job):
        result = {key: value for key, value in job.items() if key != "result"}
        result["prediction_count"] = len(job.get("result", {}).get("predictions", []))
        result["blocked_reason"] = self.blocked.get(job["id"])
        return result

    def page(self, limit=12, offset=0, query="", kind="", status="", order="newest"):
        clauses, params = ["coalesce(json_extract(payload, '$.hidden'), 0) = 0"], []
        for field, value in (("kind", kind), ("status", status)):
            if value:
                clauses.append(f"json_extract(payload, '$.{field}') = ?")
                params.append(value)
        if query:
            clauses.append(
                "instr(lower(coalesce(json_extract(payload, '$.title'), '') || ' ' || id || ' ' || coalesce(json_extract(payload, '$.run_id'), '')), lower(?)) > 0"
            )
            params.append(query)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sorting = {
            "newest": "rowid DESC",
            "oldest": "rowid ASC",
            "name": "json_extract(payload, '$.title') COLLATE NOCASE, rowid",
        }[order]
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM jobs" + where, params).fetchone()[0]
            rows = db.execute(
                "SELECT payload FROM jobs" + where + " ORDER BY " + sorting + " LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return dict(
            items=[self.summary(json.loads(row[0])) for row in rows], total=total, limit=limit, offset=offset
        )

    def specification(self, job_id):
        job = self.get(job_id)
        return json.loads((self.directory / "jobs" / job["id"] / "spec.json").read_text(encoding="utf-8"))

    def update(self, job_id, title=None, spec=None):
        with self.lock:
            job = self.get(job_id)
            if spec is not None:
                if job["status"] != "queued":
                    raise ValueError("只有排队中的任务可以修改执行参数；请复制任务后调整")
                spec = json.loads(json.dumps(spec))
                if job["kind"] == "predict":
                    self.prepare_prediction(spec)
                if job["kind"] == "train":
                    self.freeze_initialization(spec)
                    spec["run_id"] = f"{spec['config']['experiment']['name']}/web-{job_id}"
                folder = self.directory / "jobs" / job_id
                spec.update(kind=job["kind"], root=str(self.root), directory=str(folder))
                temporary = folder / "spec.json.tmp"
                temporary.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
                temporary.replace(folder / "spec.json")
                job["run_id"] = spec.get("run_id")
                job["priority"] = spec.get("request", spec).get("priority", 0)
                job["timeout_minutes"] = spec.get("request", spec).get("timeout_minutes", 0)
            if title is not None:
                job["title"] = title
            job["updated_at"] = now()
            self.save(job)
            self.wakeup.set()
            return job

    def delete(self, job_ids, stop_running=False):
        with self.lock:
            ids = list(dict.fromkeys(job_ids))
            jobs = [self.get(job_id) for job_id in ids]
            if not stop_running and any(job["status"] == "running" for job in jobs):
                raise ValueError("包含运行中的任务，请确认停止后删除")
            for job in jobs:
                for child in self.list():
                    if child.get("parent_job_id") == job["id"]:
                        self.cancel(child["id"])
                if job["status"] in {"queued", "running"}:
                    self.cancel(job["id"])
            with self.connect() as db:
                db.executemany("DELETE FROM jobs WHERE id = ?", [(job_id,) for job_id in ids])
            self.revision += 1
            for job_id in ids:
                self.blocked.pop(job_id, None)
            return {"deleted": ids, "artifacts_preserved": True}

    def clone(self, job_id):
        with self.lock:
            job = self.get(job_id)
            spec = self.specification(job_id)
            return self.submit(job["kind"], job["title"][:110] + "（副本）", spec)

    def submit(self, kind, title, spec):
        if kind not in {"train", "predict", "evaluate", "report"}:
            raise ValueError(f"Unsupported job kind: {kind}")
        with self.lock:
            spec = json.loads(json.dumps(spec))
            if kind == "predict":
                self.prepare_prediction(spec)
            if kind == "train":
                self.freeze_initialization(spec)
            job_id = uuid.uuid4().hex
            folder = self.directory / "jobs" / job_id
            folder.mkdir(parents=True)
            spec.update(kind=kind, root=str(self.root), directory=str(folder))
            if kind == "train":
                spec["run_id"] = f"{spec['config']['experiment']['name']}/web-{job_id}"
            (folder / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            job = dict(
                id=job_id,
                kind=kind,
                title=title,
                status="queued",
                created_at=now(),
                started_at=None,
                finished_at=None,
                run_id=spec.get("run_id"),
                error=None,
                priority=spec.get("request", spec).get("priority", 0),
                timeout_minutes=spec.get("request", spec).get("timeout_minutes", 0),
            )
            if spec.get("parent_job_id"):
                job.update(hidden=True, parent_job_id=spec["parent_job_id"])
            self.save(job)
            self.wakeup.set()
            return job

    def prepare_prediction(self, spec):
        names = {}
        for upload_id in spec.get("uploads", []):
            if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
                raise ValueError("无效图片 ID")
            path = self.directory / "uploads" / f"{upload_id}.png"
            if not path.is_file():
                raise ValueError("推理图片已清理或不存在，请重新上传图片")
            metadata = path.with_suffix(".json")
            names[upload_id] = (
                json.loads(metadata.read_text(encoding="utf-8")).get("name", path.name)
                if metadata.is_file()
                else path.name
            )
        spec["upload_names"] = names

    def cleanup_prediction(self, job):
        """Release only this opt-out job's images, respecting other jobs' references."""
        if job["kind"] != "predict" or job["status"] in {"queued", "running"}:
            return
        spec = self.specification(job["id"])
        if spec.get("save_images", True):  # Legacy jobs retain their original contract.
            return
        retained = set()
        for other in self.list():
            if other["id"] == job["id"] or other["kind"] != "predict":
                continue
            other_spec = self.specification(other["id"])
            if other["status"] in {"queued", "running"} or other_spec.get("save_images", True):
                retained.update(other_spec.get("uploads", []))
        for upload_id in spec.get("uploads", []):
            if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
                continue
            (self.directory / "jobs" / job["id"] / "inputs" / f"{upload_id}.png").unlink(missing_ok=True)
            if upload_id not in retained:
                path = self.directory / "uploads" / f"{upload_id}.png"
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)

    def freeze_initialization(self, spec):
        checkpoint = spec["config"]["checkpoint"]
        if checkpoint.get("finetune_from") != "last":
            return
        experiment = spec["config"]["experiment"]
        directory = Path(experiment["output_root"]) / experiment["name"]
        states = {job["run_id"]: job["status"] for job in self.list() if job["kind"] == "train"}
        candidates = []
        for suffix in ("pt", "ckpt"):
            for path in directory.glob(f"*/checkpoints/last.{suffix}"):
                run = path.parent.parent
                web_state = states.get(run.relative_to(self.root / "runs").as_posix())
                if web_state in {"queued", "running"}:
                    continue
                if web_state is None:
                    status = run / "status.json"
                    if status.exists():
                        try:
                            state = json.loads(status.read_text(encoding="utf-8")).get("state")
                        except (OSError, ValueError):
                            continue
                        if state in {"PREPARING", "RUNNING"}:
                            continue
                candidates.append(path)
        source = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))) if candidates else None
        checkpoint["finetune_from"] = str(source.resolve()) if source else None

    @staticmethod
    def category(kind):
        return kind if kind in {"train", "predict"} else "auxiliary"

    def scheduler_state(self):
        with self.lock:
            running = dict.fromkeys(self.limits.model_dump(), 0)
            queued = dict(running)
            with self.connect() as db:
                counts = db.execute(
                    "SELECT json_extract(payload, '$.kind'), json_extract(payload, '$.status'), count(*) "
                    "FROM jobs WHERE json_extract(payload, '$.status') IN ('queued', 'running') "
                    "GROUP BY json_extract(payload, '$.kind'), json_extract(payload, '$.status')"
                )
                for kind, state, count in counts:
                    target = queued if state == "queued" else running
                    target[self.category(kind)] += count
            return {"limits": self.limits.model_dump(), "running": running, "queued": queued}

    def configure(self, limits: Concurrency):
        with self.lock:
            self.configure_settings(self.settings.model_copy(update={"concurrency": limits}))
            return self.scheduler_state()

    def settings_state(self):
        with self.lock:
            return self.settings.model_dump()

    def configure_settings(self, settings: WebSettings):
        with self.lock:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix(".yaml.tmp")
            temporary.write_text(
                "# Web 全局设置；页面保存后立即生效。降低并发上限不终止活动任务。\n"
                + yaml.safe_dump(settings.model_dump(), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            temporary.replace(self.settings_path)
            self.settings = settings
            self.limits = settings.concurrency
            self.revision += 1
            self.wakeup.set()
            return self.settings_state()

    def start(self):
        self.lease.acquire()
        for job in self.list("running"):
            if job["status"] == "running":
                job.update(status="interrupted", finished_at=now(), error="服务中断，请重新提交任务")
                self.cleanup_prediction(job)
                self.save(job)
                self.sync_report(job)
        self.thread = threading.Thread(target=self.run, daemon=True, name="web-job-queue")
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wakeup.set()
        with self.lock:
            for job_id in list(self.active):
                self.cancel(job_id)
            for process in list(self.residents):
                self.close_resident(process)
        self.thread.join(timeout=15)
        self.lease.release()

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] in {"queued", "running"}:
                if job_id in self.active:
                    process, log, _ = self.active.pop(job_id)
                    if process in self.residents:
                        self.close_resident(process)
                    else:
                        terminate(process)
                    if log:
                        log.close()
                job.update(status="cancelled", finished_at=now())
                self.cleanup_prediction(job)
                self.save(job)
                self.reservations.pop(job_id, None)
                self.sync_report(job)
                self.wakeup.set()
            return job

    def close_resident(self, process):
        terminate(process)
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        self.residents.pop(process, None)
        self.resident_errors.pop(process, None)

    def watch_resident(self, process):
        try:
            for _ in process.stdout:
                self.wakeup.set()
        except (OSError, ValueError):
            pass
        self.wakeup.set()

    def trim_residents(self):
        busy = {process for process, _, _ in self.active.values()}
        for process, idle_since in list(self.residents.items()):
            if process not in busy and (
                process.poll() is not None
                or not self.settings.inference.resident
                or len(self.residents) > self.limits.predict
                or time.monotonic() - idle_since > self.settings.inference.idle_seconds
            ):
                self.close_resident(process)

    def run(self):
        while not self.stopping.is_set():
            self.wakeup.wait(0.3 if self.active else 1)
            self.wakeup.clear()
            if self.stopping.is_set():
                break
            with self.lock:
                for job_id, (process, log, _) in list(self.active.items()):
                    job = self.get(job_id)
                    timeout = job.get("timeout_minutes", 0)
                    if (
                        timeout
                        and (datetime.now(UTC) - datetime.fromisoformat(job["started_at"])).total_seconds()
                        > timeout * 60
                    ):
                        self.cancel(job_id)
                        job.update(status="failed", finished_at=now(), error="任务超过配置的运行时限，已停止")
                        self.save(job)
                        self.sync_report(job)
                        continue
                    completed = (
                        process in self.residents
                        and (self.directory / "jobs" / job_id / "result.json").is_file()
                    )
                    if completed or process.poll() is not None:
                        if log:
                            log.close()
                        if not completed and process in self.resident_errors:
                            job_log = self.root / "logs/web" / f"{job_id}.log"
                            error_log = self.resident_errors[process]
                            if error_log.is_file():
                                with job_log.open("ab") as stream:
                                    stream.write(error_log.read_bytes())
                        self.finish(job_id, 0 if completed else process.returncode)
                        del self.active[job_id]
                        self.reservations.pop(job_id, None)
                        if process in self.residents:
                            self.residents[process] = time.monotonic()
                self.trim_residents()
                pending = sorted(
                    reversed(self.list("queued")),
                    key=lambda j: -j.get("priority", 0),
                )
                previous_blocked = self.blocked.copy()
                self.blocked = {}
                resources = snapshot() if pending else None
                reserved = sum(
                    self.settings.resources.launch_reserve_gb
                    for active_id in self.active
                    if (
                        datetime.now(UTC) - datetime.fromisoformat(self.get(active_id)["started_at"])
                    ).total_seconds()
                    < 15
                )
                for job in pending:
                    if self.stopping.is_set():
                        break
                    category = self.category(job["kind"])
                    if category == "train" and any(
                        group == "train"
                        and self.get(active_id)["run_id"].rsplit("/", 1)[0] == job["run_id"].rsplit("/", 1)[0]
                        for active_id, (_, _, group) in self.active.items()
                    ):
                        self.blocked[job["id"]] = "同名实验正在运行"
                        continue
                    count = sum(group == category for _, _, group in self.active.values())
                    limit = getattr(self.limits, category)
                    if limit == 0 or count < limit:
                        cfg = self.settings.resources
                        if cfg.enabled and (
                            resources["memory_percent"] >= cfg.max_memory_percent
                            or resources["available_gb"] - reserved
                            < cfg.min_available_gb + cfg.launch_reserve_gb
                        ):
                            self.blocked[job["id"]] = f"等待可用内存：当前 {resources['available_gb']:.1f} GB"
                            continue
                        spec = self.specification(job["id"])
                        try:
                            gpus = choose_gpus(spec, resources["gpus"], self.reservations, cfg)
                        except ValueError as exc:
                            job.update(status="failed", error=str(exc), finished_at=now())
                            self.save(job)
                            continue
                        if gpus is None:
                            self.blocked[job["id"]] = gpu_wait_reason(
                                spec, resources["gpus"], self.reservations, cfg
                            )
                            continue
                        self.launch(job, category, gpus)
                        reserved += cfg.launch_reserve_gb
                    else:
                        self.blocked[job["id"]] = "等待该类型的并发额度"
                if previous_blocked != self.blocked:
                    self.revision += 1

    def launch(self, job, category, gpus=None):
        folder = self.directory / "jobs" / job["id"]
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        env.pop("CLS_ACTIVE_RUN", None)
        log = None
        process = None
        try:
            # Also migrate queued jobs created by the older single-task service.
            if job["kind"] == "train":
                spec_path = folder / "spec.json"
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                self.freeze_initialization(spec)
                spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            if gpus:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            if job["kind"] == "predict":
                spec_path = folder / "spec.json"
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                spec["inference"] = self.settings.inference.model_dump()
                spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
                if self.settings.inference.resident:
                    busy = {p for p, _, _ in self.active.values()}
                    process = next((p for p in self.residents if p not in busy and p.poll() is None), None)
                    if process is None:
                        error_path = self.root / "logs/web" / f"inference-{uuid.uuid4().hex}.log"
                        with error_path.open("wb") as error_log:
                            process = subprocess.Popen(
                                [sys.executable, "-m", "clsweb.inference_worker"],
                                cwd=self.root,
                                env=env,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=error_log,
                                text=True,
                                encoding="utf-8",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                            )
                        self.residents[process] = time.monotonic()
                        self.resident_errors[process] = error_path
                        threading.Thread(target=self.watch_resident, args=(process,), daemon=True).start()
                    process.stdin.write(
                        json.dumps(
                            {"spec": str(spec_path), "log": str(self.root / "logs/web" / f"{job['id']}.log")}
                        )
                        + "\n"
                    )
                    process.stdin.flush()
                    self.active[job["id"]] = (process, None, category)
                    job.update(status="running", started_at=now(), pid=process.pid)
                    self.save(job)
                    return
            log = (self.root / "logs/web" / f"{job['id']}.log").open("wb")
            process = subprocess.Popen(
                [sys.executable, "-m", "clsweb.worker", str(folder / "spec.json")],
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.active[job["id"]] = (process, log, category)
            if gpus:
                self.reservations[job["id"]] = list(gpus)
            job.update(status="running", started_at=now(), pid=process.pid)
            job["gpu_indices"] = list(gpus or [])
            job["gpu_index"] = gpus[0] if gpus and len(gpus) == 1 else None
        except Exception as exc:
            if process in self.residents:
                self.close_resident(process)
            if log:
                log.close()
            job.update(status="failed", error=str(exc), finished_at=now())
            self.cleanup_prediction(job)
        self.save(job)

        self.sync_report(job)

    def finish(self, job_id, code):
        job = self.get(job_id)
        result_path = self.directory / "jobs" / job_id / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Invalid worker result")
        except (OSError, ValueError):
            result = {"error": "工作进程未生成完整结果，请查看日志"}
        job.update(
            status="succeeded" if code == 0 and not result.get("error") else "failed",
            finished_at=now(),
            result=result,
            error=result.get("error") or (f"进程退出码 {code}" if code else None),
        )
        if job["status"] == "succeeded" and job["kind"] == "predict":
            timings = result.setdefault("timings", {})
            timings["queue_seconds"] = (
                datetime.fromisoformat(job["started_at"]) - datetime.fromisoformat(job["created_at"])
            ).total_seconds()
            if result.get("worker_started_at"):
                timings["startup_seconds"] = max(
                    0,
                    (
                        datetime.fromisoformat(result.pop("worker_started_at"))
                        - datetime.fromisoformat(job["started_at"])
                    ).total_seconds(),
                )
            spec = self.specification(job_id)
            if spec.get("save_images", True):
                report = self.submit(
                    "report",
                    job["title"][:100] + " · 推理报告",
                    {"parent_job_id": job_id, "run_id": job["run_id"], "top_k": spec["top_k"]},
                )
                result.update(report_job_id=report["id"], report_status="queued")
            else:
                result["report_status"] = "skipped"
        self.cleanup_prediction(job)
        self.save(job)
        self.sync_report(job)

    def sync_report(self, job):
        parent_id = job.get("parent_job_id")
        if not parent_id:
            return
        try:
            parent = self.get(parent_id)
        except KeyError:
            return
        result = parent.setdefault("result", {})
        result["report_status"] = job["status"]
        if job["status"] == "succeeded":
            result["report"] = job["result"]["report"]
            result.setdefault("timings", {})["report_seconds"] = job["result"]["report_seconds"]
        elif job.get("error"):
            result["report_error"] = job["error"]
        self.save(parent)
