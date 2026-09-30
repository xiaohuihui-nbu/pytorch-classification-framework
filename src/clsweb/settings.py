"""Web scheduling settings, independent of the training configuration schema."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    device: Literal["auto", "cpu", "gpu"] = "auto"
    gpu_indices: list[Annotated[int, Field(ge=0, le=127)]] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_selection(self):
        if len(set(self.gpu_indices)) != len(self.gpu_indices):
            raise ValueError("GPU 编号不能重复")
        if self.device == "cpu" and self.gpu_indices:
            raise ValueError("CPU 模式不能指定 GPU")
        return self


class Concurrency(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    train: int = Field(default=0, ge=0, le=8, description="0 表示不限制训练任务数量")
    predict: int = Field(default=1, ge=1, le=8)
    auxiliary: int = Field(default=1, ge=1, le=4)


class Inference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    resident: bool = True
    cpu_threads: int = Field(default=2, ge=1, le=32)
    batch_size: int = Field(default=8, ge=1, le=64)
    cache_models: int = Field(default=1, ge=1, le=4)
    cache_mb: int = Field(default=256, ge=16, le=2048)
    idle_seconds: int = Field(default=300, ge=30, le=3600)


class Appearance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    theme: Literal["cyber", "forest", "ocean", "violet", "amber"] = "cyber"
    mode: Literal["light", "dark", "system"] = "dark"


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    max_memory_percent: int = Field(default=90, ge=20, le=99)
    min_available_gb: float = Field(default=2.0, ge=0, le=1024)
    launch_reserve_gb: float = Field(default=1.0, ge=0, le=1024)
    min_gpu_free_gb: float = Field(default=2.0, ge=0, le=1024)
    max_jobs_per_gpu: int = Field(default=0, ge=0, le=8, description="0 表示仅按显存判断，不限制任务数量")


class WebSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    training: TrainingDevice = Field(default_factory=TrainingDevice)
    concurrency: Concurrency = Field(default_factory=Concurrency)
    inference: Inference = Field(default_factory=Inference)
    appearance: Appearance = Field(default_factory=Appearance)
    resources: Resources = Field(default_factory=Resources)
    refresh_interval_seconds: int = Field(default=3, ge=1, le=30)

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw and set(raw) <= set(Concurrency.model_fields):
            raw = {"concurrency": raw}  # Preserve existing values from the previous flat YAML.
        return cls.model_validate(raw)
