"""Start a new fine-tuning run from a previous trusted framework checkpoint."""

import logging
from pathlib import Path

import torch
from filelock import FileLock
from safetensors.torch import save_file

from .data import class_names
from .utils import file_hash, read_json, write_json


def select_checkpoint(cfg):
    source = cfg.checkpoint.finetune_from
    if source is None:
        return None
    if source == "last":
        root = cfg.experiment.output_root / cfg.experiment.name
        candidates = list(root.glob("*/checkpoints/last.ckpt"))
        if not candidates:
            logging.getLogger(__name__).info("没有上次训练权重，按当前 YAML 的初始权重设置开始训练")
            return None
        return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.parent.parent.name)).resolve()
    path = Path(source)
    if path.is_dir():
        path = path / "checkpoints/last.ckpt"
    if not path.is_file() or path.suffix != ".ckpt":
        raise ValueError(f"finetune_from must be a framework run directory or .ckpt file: {path}")
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
    contract = previous.get("cls_contract", {})
    expected = {
        "provider": cfg.model.provider,
        "name": cfg.model.name,
        "channels": cfg.model.input_channels,
        "outputs": 1 if cfg.task.type == "binary" else len(data.classes),
    }
    if contract.get("architecture") != expected or contract.get("task", {}).get("type") != cfg.task.type:
        raise ValueError(
            "E_FINETUNE_MODEL: source architecture/task differs; choose matching weights or fresh"
        )
    if cfg.preprocessing.source == "model_weights":
        preprocess = dict(contract["preprocessing"])
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
