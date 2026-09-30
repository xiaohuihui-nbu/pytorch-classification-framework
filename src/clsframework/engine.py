import importlib.metadata
import logging
import math
import os
import platform
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter

import lightning as L
import numpy as np
import torch
import yaml
from filelock import FileLock, Timeout
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from timm.data import Mixup
from torch.nn import functional as F
from torchmetrics.functional.classification import multiclass_f1_score

from .continuation import apply_auto_resume, apply_finetune, requested_training_config
from .data import loader, prepare_data, save_prepared
from .losses import build_loss
from .metrics import evaluate_arrays
from .models import build_model, classifier_module
from .registry import load_plugins
from .utils import append_json, digest, file_hash, utc_now, write_json


def environment():
    packages = ["torch", "torchvision", "timm", "lightning", "torchmetrics", "hydra-core", "pydantic"]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {x: importlib.metadata.version(x) for x in packages},
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
    }


def resolve_runtime(cfg):
    load_plugins(cfg.runtime.plugins)
    torch.set_num_threads(cfg.trainer.cpu_threads)
    L.seed_everything(cfg.experiment.seed, workers=True)
    if cfg.trainer.accelerator == "auto":
        cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if cfg.trainer.accelerator == "gpu" and not torch.cuda.is_available():
        raise ValueError("GPU requested but CUDA is unavailable")
    if cfg.trainer.devices > 1 and (os.name == "nt" or cfg.trainer.accelerator != "gpu"):
        raise ValueError("Multi-device mode is currently restricted to Linux CUDA DDP")
    if cfg.trainer.devices > torch.cuda.device_count() and cfg.trainer.accelerator == "gpu":
        raise ValueError("Requested more GPUs than are visible")
    if cfg.trainer.precision == "auto":
        cfg.trainer.precision = (
            ("bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed")
            if cfg.trainer.accelerator == "gpu"
            else "32-true"
        )
    if cfg.loader.pin_memory == "auto":
        cfg.loader.pin_memory = cfg.trainer.accelerator == "gpu"
    if cfg.loader.persistent_workers == "auto":
        cfg.loader.persistent_workers = cfg.loader.num_workers > 0
    if cfg.loader.num_workers == 0:
        cfg.loader.persistent_workers = False


def primary_process():
    return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0


def contract(cfg, data, architecture, preprocessing):
    raw = cfg.model_dump(mode="json")
    relevant = {
        k: raw[k]
        for k in (
            "task",
            "model",
            "augmentation",
            "loss",
            "optimizer",
            "scheduler",
            "trainer",
            "loader",
            "evaluation",
        )
    }
    relevant.update(
        seed=cfg.experiment.seed,
        fingerprint=data.fingerprint,
        architecture=architecture,
        preprocessing=preprocessing,
        packages=environment()["packages"],
        source_digest=digest({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}),
    )
    # Display-only settings must not prevent resuming the same experiment.
    relevant["trainer"].pop("enable_progress_bar", None)
    return relevant


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class ClassificationTask(L.LightningModule):
    def __init__(self, model, criterion, cfg, classes, signature, directory):
        super().__init__()
        self.model, self.criterion, self.cfg = model, criterion, cfg
        self.classes, self.signature, self.directory = classes, signature, Path(directory)
        self.validation_outputs = []
        self.last_report = None
        self.mixup = None
        if cfg.loss.name == "soft_target_ce":
            self.mixup = Mixup(
                mixup_alpha=cfg.augmentation.mixup_alpha,
                cutmix_alpha=cfg.augmentation.cutmix_alpha,
                num_classes=len(classes),
                label_smoothing=cfg.loss.label_smoothing,
            )

    def forward(self, inputs):
        output = self.model(inputs)
        if isinstance(output, dict):
            output = output.get("logits")
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("ModelAdapter must return [N,C] logits or {'logits': tensor}")
        return output

    def on_train_epoch_start(self):
        if self.cfg.model.training_mode == "linear_probe":
            self.model.eval()
            classifier_module(self.model, self.cfg.model.provider).train()

    def training_step(self, batch, batch_idx):
        x, target = batch["image"], batch["target"]
        if self.mixup is not None:
            # timm pair/batch mixing expects even batches. An odd final batch uses
            # valid smoothed one-hot labels instead of discarding a real sample.
            if len(x) % 2 == 0:
                x, target = self.mixup(x, target)
            else:
                smoothing = self.cfg.loss.label_smoothing
                target = F.one_hot(target, len(self.classes)).float() * (1 - smoothing) + smoothing / len(
                    self.classes
                )
        logits = self(x)
        loss = self.criterion(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=len(x), sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
            raise FloatingPointError("Nonfinite gradients")

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation logits")
        probabilities = logits.softmax(1) if self.cfg.task.type == "multiclass" else logits.sigmoid()
        for sid, target, prob in zip(
            batch["sample_id"], batch["target"].cpu(), probabilities.cpu(), strict=True
        ):
            self.validation_outputs.append((sid, target.tolist(), prob.tolist()))

    def on_validation_epoch_end(self):
        rows = self.validation_outputs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        rows = list({row[0]: row for row in rows}.values())  # remove DDP sampler padding
        self.validation_outputs.clear()
        y, p = [r[1] for r in rows], [r[2] for r in rows]
        report = evaluate_arrays(self.cfg.task.type, y, p, self.classes, self.cfg.evaluation.threshold)
        if self.cfg.task.type == "multiclass":
            # Match the report's fixed class list, including absent classes with F1=0.
            score = multiclass_f1_score(
                torch.tensor(p), torch.tensor(y), len(self.classes), average=None, zero_division=0
            ).mean()
            if abs(float(score) - report["macro_f1"]) > 1e-5:
                raise RuntimeError("TorchMetrics and report F1 disagree")
        self.last_report = report
        for key, value in report.items():
            if isinstance(value, (int, float)) and key not in {"samples", "threshold", "ap_defined_classes"}:
                self.log(f"val/{key}", float(value), sync_dist=False, prog_bar=key == "macro_f1")
        monitor = self.cfg.evaluation.monitor.removeprefix("val/")
        if not isinstance(report.get(monitor), (int, float)):
            raise ValueError(f"Undefined/unknown monitored metric: {self.cfg.evaluation.monitor}")
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            write_json(self.directory / "evaluation" / "val_metrics.json", report)
            append_json(
                self.directory / "metrics.jsonl",
                {"epoch": self.current_epoch, "global_step": self.global_step, "time": utc_now(), **report},
            )

    def configure_optimizers(self):
        cfg = self.cfg
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                (
                    no_decay
                    if cfg.optimizer.exclude_norm_and_bias_from_decay
                    and (param.ndim <= 1 or name.endswith(".bias"))
                    else decay
                ).append(param)
        groups = [
            {"params": decay, "weight_decay": cfg.optimizer.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
            cfg.optimizer.name
        ]
        kwargs = {"momentum": cfg.optimizer.momentum} if cfg.optimizer.name == "sgd" else {}
        optimizer = cls(groups, lr=cfg.optimizer.lr, **kwargs)
        if cfg.scheduler.name == "none":
            return optimizer
        total = int(self.trainer.estimated_stepping_batches)
        warmup = round(total * cfg.scheduler.warmup_epochs / cfg.trainer.max_epochs)
        floor = cfg.scheduler.min_lr / cfg.optimizer.lr

        def multiplier(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cls_contract"] = digest(self.signature)
        checkpoint["cls_rng"] = capture_rng()
        checkpoint["cls_run_dir"] = str(self.directory)
        checkpoint["cls_inference"] = {
            "schema_version": 1,
            "model_spec": {
                **self.signature["architecture"],
                "task": self.cfg.task.type,
                "plugins": self.cfg.runtime.plugins,
            },
            "classes": self.classes,
            "preprocessing": self.signature["preprocessing"],
            "threshold": self.cfg.evaluation.threshold,
            "dataset_fingerprint": self.signature["fingerprint"],
        }

    def on_load_checkpoint(self, checkpoint):
        if checkpoint["cls_contract"] != digest(self.signature):
            raise ValueError("E_RESUME_MISMATCH: data/config/code/dependency contract changed")
        restore_rng(checkpoint["cls_rng"])


class Events(Callback):
    def __init__(self, directory, stop_after_epoch=None):
        self.directory, self.stop_after_epoch = Path(directory), stop_after_epoch

    def on_train_epoch_start(self, trainer, module):
        self.started = perf_counter()
        self.samples = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.samples += len(batch["image"])

    def on_train_epoch_end(self, trainer, module):
        if trainer.is_global_zero:
            seconds = perf_counter() - self.started
            logging.getLogger(__name__).info(
                "Epoch %d/%d 完成：step=%d, lr=%.8g, %s=%.6g",
                trainer.current_epoch + 1,
                trainer.max_epochs,
                trainer.global_step,
                trainer.optimizers[0].param_groups[0]["lr"],
                module.cfg.evaluation.monitor,
                float(trainer.callback_metrics.get(module.cfg.evaluation.monitor, float("nan"))),
            )
            append_json(
                self.directory / "events.jsonl",
                {
                    "event": "epoch_completed",
                    "time": utc_now(),
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "epoch_seconds": seconds,
                    "train_images_per_second": self.samples * trainer.world_size / seconds,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                    "train_loss": float(trainer.callback_metrics["train/loss"])
                    if "train/loss" in trainer.callback_metrics
                    else None,
                },
            )
        if self.stop_after_epoch is not None and trainer.current_epoch + 1 >= self.stop_after_epoch:
            trainer.should_stop = True


class EpochCheckpoint(ModelCheckpoint):
    """Always save latest training state, even when the monitored score plateaus.

    Upstream save_last follows top-k save events in some Lightning versions.
    Our resume contract requires a full checkpoint at every epoch boundary.
    """

    FILE_EXTENSION = ".pt"

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        destination = Path(self.dirpath) / "last.pt"
        self.last_model_path = str(destination)
        temporary = destination.with_suffix(".pt.tmp")
        # Strict resume needs optimizer/scheduler/callback state as well as model weights.
        trainer.save_checkpoint(str(temporary), weights_only=False)
        if trainer.is_global_zero:
            os.replace(temporary, destination)
        trainer.strategy.barrier()


def train(cfg, original_config=None, smoke=False, stop_after_epoch=None):
    root = cfg.experiment.output_root / cfg.experiment.name
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / ".train.lock")) if primary_process() else None
    if lock is not None:
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise ValueError("E_TRAIN_ACTIVE: 同一实验已有训练进程，请先停止它或使用不同实验名") from exc
    try:
        return _train(cfg, original_config, smoke, stop_after_epoch)
    finally:
        if lock is not None:
            lock.release()


def _train(cfg, original_config=None, smoke=False, stop_after_epoch=None):
    log = logging.getLogger(__name__)
    requested = requested_training_config(cfg)
    if not smoke:
        apply_auto_resume(cfg)
    log.info(
        "准备训练：model=%s/%s, epochs=%d, batch=%d, resume=%s",
        cfg.model.provider,
        cfg.model.name,
        cfg.trainer.max_epochs,
        cfg.loader.batch_size_per_device,
        cfg.checkpoint.resume_from,
    )
    resolve_runtime(cfg)
    data = prepare_data(cfg)
    log.info("数据准备完成：classes=%s, fingerprint=%s", data.classes, data.fingerprint)
    if smoke:
        cfg.checkpoint.finetune_from = None
    finetune = apply_finetune(cfg, data)
    model, architecture, preprocessing, provenance = build_model(cfg, data.classes)
    if finetune is not None:
        provenance["finetune"] = finetune
    criterion = build_loss(cfg, data)
    cfg.preprocessing = cfg.preprocessing.model_validate(preprocessing)
    signature = contract(cfg, data, architecture, preprocessing)
    resume = cfg.checkpoint.resume_from
    if resume:
        # Only framework-created, trusted local checkpoints are supported for resume.
        previous = torch.load(resume, map_location="cpu", weights_only=False)
        if previous.get("cls_contract") != digest(signature):
            raise ValueError("E_RESUME_MISMATCH: data/config/code/dependency contract changed")
        directory = Path(previous["cls_run_dir"])
        if not directory.is_dir():
            raise ValueError("Original run directory required for strict resume")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        directory = Path(
            os.environ.get("CLS_ACTIVE_RUN", str(cfg.experiment.output_root / cfg.experiment.name / run_id))
        )
    os.environ["CLS_ACTIVE_RUN"] = str(directory)
    log.info("运行目录：%s", directory)
    if primary_process():
        directory.mkdir(parents=True, exist_ok=bool(resume) or "LOCAL_RANK" in os.environ)
        if not resume:
            save_prepared(data, directory)
            write_json(directory / "config.requested.json", requested)
            (directory / "config.resolved.yaml").write_text(
                yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            if original_config:
                (directory / "config.original.yaml").write_text(
                    Path(original_config).read_text(encoding="utf-8-sig"), encoding="utf-8"
                )
            write_json(directory / "environment.json", environment())
            write_json(directory / "provenance.json", provenance)
            write_json(directory / "contract.json", signature)
        write_json(directory / "status.json", {"state": "PREPARING", "time": utc_now()})
    task = ClassificationTask(model, criterion, cfg, data.classes, signature, directory)
    checkpoint = EpochCheckpoint(
        dirpath=directory / "checkpoints",
        filename="best",
        monitor=cfg.evaluation.monitor,
        mode=cfg.evaluation.mode,
        save_last=False,
        save_top_k=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    callbacks = [Events(directory, stop_after_epoch), checkpoint]
    logger = False
    if cfg.logging.tensorboard:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(str(directory), name="tensorboard")
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy="ddp" if cfg.trainer.devices > 1 else "auto",
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.trainer.deterministic,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=False,
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=2 if smoke else 1.0,
    )
    try:
        if primary_process():
            write_json(directory / "status.json", {"state": "RUNNING", "time": utc_now()})
        trainer.fit(
            task,
            train_dataloaders=loader(cfg, data, "train", preprocessing),
            val_dataloaders=loader(cfg, data, "val", preprocessing),
            ckpt_path=str(resume) if resume else None,
        )
        if trainer.is_global_zero:
            if trainer.interrupted:
                write_json(directory / "status.json", {"state": "CANCELLED", "time": utc_now()})
                return directory
            best = checkpoint.best_model_path
            if not best:
                raise RuntimeError("Training produced no best checkpoint")
            finished = trainer.current_epoch >= cfg.trainer.max_epochs
            log.info(
                "训练状态=%s；best=%s",
                "SUCCEEDED" if finished else "PAUSED",
                best,
            )
            write_json(
                directory / "status.json",
                {
                    "state": "SUCCEEDED" if finished else "PAUSED",
                    "time": utc_now(),
                    "epochs_completed": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "smoke": smoke,
                    "best_checkpoint": best,
                    "best_checkpoint_sha256": file_hash(best),
                },
            )
            if cfg.visualization.enabled:
                from .visualization import training_report

                log.info("可视化报告：%s", training_report(directory))
    except BaseException as exc:
        if primary_process():
            write_json(
                directory / "status.json",
                {
                    "state": "CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                    "time": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    finally:
        os.environ.pop("CLS_ACTIVE_RUN", None)
    return directory
