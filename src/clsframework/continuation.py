"""Start a new fine-tuning run from a previous trusted framework checkpoint."""

import logging
from pathlib import Path

import torch
from filelock import FileLock
from safetensors.torch import save_file

from .config import load_config
from .data import class_names
from .utils import file_hash, read_json, write_json


def latest_checkpoint(root):
    """Find the latest full state, including checkpoints written by older releases."""
    candidates = [p for suffix in ("pt", "ckpt") for p in root.glob(f"*/checkpoints/last.{suffix}")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.parent.parent.name, p.suffix)).resolve()


def requested_training_config(cfg):
    """Compare user intent before runtime/preprocessing/finetune normalization."""
    raw = cfg.model_dump(mode="json")
    for key in ("checkpoint", "logging", "visualization", "runtime"):
        raw.pop(key)
    raw["plugins"] = cfg.runtime.plugins
    raw["trainer"].pop("enable_progress_bar")
    return raw


def apply_auto_resume(cfg):
    """Resume the latest saved incomplete run, never silently ignore mismatches."""
    if not cfg.checkpoint.auto_resume or cfg.checkpoint.resume_from is not None:
        return
    root = cfg.experiment.output_root / cfg.experiment.name
    source = latest_checkpoint(root)
    log = logging.getLogger(__name__)
    if source is None:
        log.info("没有可恢复的完整轮次断点，按配置开始训练")
        return
    directory = source.parent.parent
    previous = torch.load(source, map_location="cpu", weights_only=False)
    snapshot = load_config(directory / "config.resolved.yaml")
    if previous["epoch"] + 1 >= snapshot.trainer.max_epochs:
        log.info("最近断点已完成原训练预算，按配置开始新的训练")
        return
    request_path = directory / "config.requested.json"
    if not request_path.is_file():
        raise ValueError(
            "E_AUTO_RESUME_LEGACY: 旧运行缺少 config.requested.json，无法验证自动恢复配置；"
            "请用原版本和 --resume 恢复，或用 --finetune-from last 仅继承权重"
        )
    if read_json(request_path) != requested_training_config(cfg):
        raise ValueError(
            "E_AUTO_RESUME_CONFIG: 本次配置与最近未完成训练不同；"
            "请恢复原配置，或显式使用 --finetune-from last / --fresh"
        )
    # Keep display/runtime preferences, but recover the effective initialization and preprocessing.
    snapshot.logging = cfg.logging
    snapshot.visualization = cfg.visualization
    snapshot.runtime = cfg.runtime
    snapshot.trainer.enable_progress_bar = cfg.trainer.enable_progress_bar
    snapshot.checkpoint.auto_resume = False
    snapshot.checkpoint.finetune_from = None
    snapshot.checkpoint.resume_from = source
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    log.info(
        "自动严格续训：%s；已保存 %d 轮，总预算 %d 轮", source, previous["epoch"] + 1, cfg.trainer.max_epochs
    )


def select_checkpoint(cfg):
    source = cfg.checkpoint.finetune_from
    if source is None:
        return None
    if source == "last":
        root = cfg.experiment.output_root / cfg.experiment.name
        checkpoint = latest_checkpoint(root)
        if checkpoint is None:
            logging.getLogger(__name__).info("没有上次训练权重，按当前 YAML 的初始权重设置开始训练")
            return None
        return checkpoint
    path = Path(source)
    if path.is_dir():
        current = path / "checkpoints/last.pt"
        path = current if current.is_file() else path / "checkpoints/last.ckpt"
    if not path.is_file() or path.suffix not in {".pt", ".ckpt"}:
        raise ValueError(f"finetune_from must be a framework run directory or .pt/.ckpt file: {path}")
    return path.resolve()


def apply_finetune(cfg, data):
    if cfg.checkpoint.resume_from is not None and cfg.checkpoint.finetune_from is not None:
        raise ValueError("Choose strict resume or weight-only finetuning, not both")
    source = select_checkpoint(cfg)
    cfg.checkpoint.finetune_from = None  # Resolved snapshots must not select another run on resume.
    if source is None:
        return None
    source_run = source.parent.parent
    log = logging.getLogger(__name__)
    log.info(
        "继续微调：加载上次权重 %s；新建实验，重新初始化优化器/调度器，训练 %d 轮",
        source,
        cfg.trainer.max_epochs,
    )
    if class_names(source_run / "classes.json", []) != data.classes:
        raise ValueError("E_FINETUNE_CLASSES: class names/order differ from the source run")
    checksum = file_hash(source)
    # These paths refer to trusted, locally produced framework runs, as with strict resume.
    previous = torch.load(source, map_location="cpu", weights_only=False)
    if file_hash(source) != checksum:
        raise ValueError("Source checkpoint changed while loading; wait for the previous training to finish")
    metadata = previous.get("cls_inference")
    if metadata is not None:
        if metadata.get("schema_version") != 1:
            raise ValueError("E_FINETUNE_MODEL: unsupported checkpoint inference metadata")
        spec = metadata["model_spec"]
        architecture = {key: spec[key] for key in ("provider", "name", "channels", "outputs")}
        task = spec["task"]
        preprocessing = metadata["preprocessing"]
    else:
        contract = previous.get("cls_contract", {})
        if not isinstance(contract, dict):
            raise ValueError("E_FINETUNE_MODEL: source checkpoint lacks model metadata")
        architecture = contract.get("architecture")
        task = contract.get("task", {}).get("type")
        preprocessing = contract.get("preprocessing")
    expected = {
        "provider": cfg.model.provider,
        "name": cfg.model.name,
        "channels": cfg.model.input_channels,
        "outputs": 1 if cfg.task.type == "binary" else len(data.classes),
    }
    if architecture != expected or task != cfg.task.type:
        raise ValueError(
            "E_FINETUNE_MODEL: source architecture/task differs; choose matching weights or fresh"
        )
    if cfg.preprocessing.source == "model_weights":
        preprocess = dict(preprocessing)
        size = cfg.preprocessing.image_size
        if size != "auto" and size != preprocess["image_size"]:
            rounding = int if cfg.model.provider == "timm" else round
            preprocess.update(image_size=size, resize_size=rounding(size / preprocess["crop_pct"]))
        preprocess["source"] = "explicit"
        cfg.preprocessing = cfg.preprocessing.model_validate(preprocess)
    weights = {
        key.removeprefix("model."): value
        for key, value in previous["state_dict"].items()
        if key.startswith("model.")
    }
    if not weights:
        raise ValueError("Source checkpoint has no model weights")
    directory = cfg.runtime.weights_dir or cfg.runtime.cache_dir
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"finetune-{checksum}.safetensors"
    metadata = cache.with_suffix(".json")
    with FileLock(str(cache) + ".lock"):
        if cache.exists() or metadata.exists():
            if (
                not cache.is_file()
                or not metadata.is_file()
                or file_hash(cache) != read_json(metadata)["sha256"]
            ):
                raise ValueError(f"Fine-tuning weight cache is incomplete or corrupt: {cache}")
        else:
            temporary = cache.with_suffix(".tmp")
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in weights.items()}, str(temporary)
            )
            temporary.replace(cache)
            write_json(
                metadata, {"source": str(source), "source_sha256": checksum, "sha256": file_hash(cache)}
            )
    cfg.model.weights.source = "local"
    cfg.model.weights.path = cache
    return {
        "source_checkpoint": str(source),
        "source_sha256": checksum,
        "source_run": str(source_run),
        "source_epoch": previous.get("epoch"),
        "optimizer_reset": True,
        "scheduler_reset": True,
    }
