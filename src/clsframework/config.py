from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Experiment(Strict):
    name: str = "classification"
    seed: int = 42
    output_root: Path = Field(default_factory=lambda: Path("runs"))


class Task(Strict):
    type: Literal["multiclass", "binary", "multilabel"] = "multiclass"


class SplitPolicy(Strict):
    val_fraction: float = Field(default=0.1, gt=0, lt=1)
    seed: int = 42


class Data(Strict):
    provider: str = "imagefolder"
    name: str | None = None
    root: Path = Field(default_factory=lambda: Path("data"))
    manifest: Path | None = None
    classes_file: Path | None = None
    splits: dict[str, str] = Field(default_factory=lambda: {"train": "train", "val": "val", "test": "test"})
    download: bool = False
    num_classes: int | Literal["auto"] = "auto"
    split_policy: SplitPolicy = Field(default_factory=SplitPolicy)
    integrity: Literal["strict", "paths"] = "strict"
    limit_per_split: int | None = Field(default=None, gt=0)


class Weights(Strict):
    source: Literal["none", "provider_default", "local"] = "provider_default"
    path: Path | None = None
    required: Literal[True] = True


class Model(Strict):
    provider: str = "timm"
    name: str = "resnet18"
    weights: Weights = Field(default_factory=Weights)
    num_classes: int | Literal["auto"] = "auto"
    input_channels: int = Field(default=3, ge=1, le=3)
    training_mode: Literal["full_finetune", "linear_probe"] = "full_finetune"


class Preprocessing(Strict):
    source: Literal["explicit", "model_weights"] = "model_weights"
    image_size: int | Literal["auto"] = "auto"
    color_mode: Literal["RGB", "L"] = "RGB"
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    interpolation: Literal["bilinear", "bicubic", "nearest"] = "bilinear"
    crop_pct: float = Field(default=0.875, gt=0, le=1)
    resize_size: int | None = Field(default=None, ge=8)


class Augmentation(Strict):
    preset: Literal["none", "baseline", "finetune", "strong"] = "baseline"
    horizontal_flip: float = Field(default=0.5, ge=0, le=1)
    mixup_alpha: float = Field(default=0.0, ge=0)
    cutmix_alpha: float = Field(default=0.0, ge=0)


class Loader(Strict):
    batch_size_per_device: int = Field(default=16, gt=0)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool | Literal["auto"] = "auto"


class Loss(Strict):
    name: str = "cross_entropy"
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    class_weight: list[float] | Literal["balanced"] | None = None
    pos_weight: list[float] | None = None
    gamma: float = Field(default=2.0, ge=0)
    alpha: float | None = Field(default=None, ge=0, le=1)


class Optimizer(Strict):
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    exclude_norm_and_bias_from_decay: bool = True


class Scheduler(Strict):
    name: Literal["none", "warmup_cosine"] = "none"
    interval: Literal["step"] = "step"
    warmup_epochs: int = Field(default=0, ge=0)
    min_lr: float = Field(default=0.0, ge=0)


class Trainer(Strict):
    engine: Literal["lightning"] = "lightning"
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int = Field(default=1, ge=1)
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"] = "auto"
    max_epochs: int = Field(default=3, gt=0)
    accumulate_grad_batches: int = Field(default=1, gt=0)
    gradient_clip_val: float = Field(default=1.0, ge=0)
    deterministic: bool = False
    cpu_threads: int = Field(default=2, gt=0)
    enable_progress_bar: bool = True


class Evaluation(Strict):
    monitor: str = "val/macro_f1"
    mode: Literal["max", "min"] = "max"
    threshold: float = Field(default=0.5, gt=0, lt=1)
    test_after_fit: Literal[False] = False


class Checkpoint(Strict):
    resume_from: Path | None = None
    resume_policy: Literal["strict"] = "strict"
    finetune_from: Path | Literal["last"] | None = None


class Logging(Strict):
    tensorboard: bool = False
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = True
    file: bool = True
    directory: Path | None = None


class Runtime(Strict):
    offline: bool = False
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))
    weights_dir: Path | None = Field(default_factory=lambda: Path("weights"))
    plugins: list[str] = Field(default_factory=list)


class Visualization(Strict):
    enabled: bool = True
    top_k: int = Field(default=5, ge=1, le=100)
    max_images: int = Field(default=64, ge=1, le=1000)


class Config(Strict):
    schema_version: Literal[1] = 1
    experiment: Experiment = Field(default_factory=Experiment)
    task: Task = Field(default_factory=Task)
    dataset: Data = Field(default_factory=Data)
    model: Model = Field(default_factory=Model)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    augmentation: Augmentation = Field(default_factory=Augmentation)
    loader: Loader = Field(default_factory=Loader)
    loss: Loss = Field(default_factory=Loss)
    optimizer: Optimizer = Field(default_factory=Optimizer)
    scheduler: Scheduler = Field(default_factory=Scheduler)
    trainer: Trainer = Field(default_factory=Trainer)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)
    logging: Logging = Field(default_factory=Logging)
    runtime: Runtime = Field(default_factory=Runtime)
    visualization: Visualization = Field(default_factory=Visualization)

    @model_validator(mode="after")
    def compatible(self):
        if self.checkpoint.resume_from is not None and self.checkpoint.finetune_from is not None:
            raise ValueError("Choose strict resume or weight-only finetuning, not both")
        soft = self.augmentation.mixup_alpha > 0 or self.augmentation.cutmix_alpha > 0
        if self.task.type == "multiclass" and self.loss.name in {"bce_with_logits", "sigmoid_focal"}:
            raise ValueError("E_TASK_LOSS: multiclass requires a multiclass loss")
        if self.task.type != "multiclass" and self.loss.name in {
            "cross_entropy",
            "soft_target_ce",
            "softmax_focal",
        }:
            raise ValueError("E_TASK_LOSS: binary/multilabel requires BCE or sigmoid focal")
        if soft and (self.task.type != "multiclass" or self.loss.name != "soft_target_ce"):
            raise ValueError("E_TASK_LOSS: Mixup/CutMix requires multiclass + soft_target_ce")
        if self.loss.name == "soft_target_ce" and not soft:
            raise ValueError("soft_target_ce requires Mixup or CutMix")
        if self.loss.name not in {"cross_entropy", "soft_target_ce"} and self.loss.label_smoothing:
            raise ValueError("label_smoothing is supported only by cross_entropy/soft_target_ce")
        if self.loss.class_weight is not None and self.loss.name not in {"cross_entropy", "softmax_focal"}:
            raise ValueError("class_weight is supported only by cross_entropy/softmax_focal")
        if self.loss.pos_weight is not None and self.loss.name != "bce_with_logits":
            raise ValueError("pos_weight is supported only by bce_with_logits")
        if self.model.input_channels != (3 if self.preprocessing.color_mode == "RGB" else 1):
            raise ValueError("input_channels and color_mode disagree")
        if self.model.weights.source == "local" and self.model.weights.path is None:
            raise ValueError("local weights require weights.path")
        if self.preprocessing.source == "model_weights" and self.model.weights.source == "none":
            raise ValueError("model_weights preprocessing requires selected pretrained weights")
        if self.scheduler.warmup_epochs >= self.trainer.max_epochs:
            raise ValueError("warmup_epochs must be less than max_epochs")
        if self.scheduler.min_lr > self.optimizer.lr:
            raise ValueError("min_lr must not exceed optimizer.lr")
        if self.trainer.accelerator == "cpu" and self.trainer.precision not in {"auto", "32-true"}:
            raise ValueError("CPU currently supports fp32 only")
        if not self.experiment.name or any(x in self.experiment.name for x in "/\\:"):
            raise ValueError("experiment.name must be a simple directory name")
        return self


def load_config(path: Path, overrides: list[str] | None = None) -> Config:
    path = path.resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        composed = compose(config_name=path.stem)
    raw = OmegaConf.to_container(composed, resolve=True)
    return resolve_config(raw, path.parent, overrides)


def resolve_config(raw: dict, base_dir: Path, overrides: list[str] | None = None) -> Config:
    """Validate merged configuration and resolve paths for both CLI and Python callers."""
    if overrides:
        raw = OmegaConf.to_container(OmegaConf.merge(raw, OmegaConf.from_dotlist(overrides)), resolve=True)
    cfg = Config.model_validate(raw)
    for obj, names in [
        (cfg.experiment, ["output_root"]),
        (cfg.dataset, ["root", "manifest", "classes_file"]),
        (cfg.model.weights, ["path"]),
        (cfg.checkpoint, ["resume_from"]),
        (cfg.logging, ["directory"]),
        (cfg.runtime, ["cache_dir", "weights_dir"]),
    ]:
        for name in names:
            value = getattr(obj, name)
            if value is not None:
                setattr(obj, name, (base_dir / value).resolve())
    if isinstance(cfg.checkpoint.finetune_from, Path):
        cfg.checkpoint.finetune_from = (base_dir / cfg.checkpoint.finetune_from).resolve()
    return cfg
