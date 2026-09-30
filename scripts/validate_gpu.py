"""单卡 CUDA 训练验收；使用合成数据，不衡量真实分类性能。"""

import argparse
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image

from clsframework.config import Config
from clsframework.engine import ClassificationTask, environment, train
from clsframework.inference import evaluate_model
from clsframework.utils import read_json


def make_data(root):
    rng = np.random.default_rng(42)
    for split, count in [("train", 32), ("val", 8), ("test", 8)]:
        for label, channel in [("blue", 2), ("red", 0)]:
            directory = root / split / label
            directory.mkdir(parents=True)
            for index in range(count):
                pixels = rng.integers(0, 40, (32, 32, 3), dtype=np.uint8)
                pixels[:, :, channel] += 180
                Image.fromarray(pixels).save(directory / f"{index:03d}.png")


def run_case(root, provider, name, precision, resume):
    case = f"{provider}_{name}_{precision}"
    cfg = Config.model_validate({
        "experiment": {"name": case, "output_root": str(root / "runs")},
        "dataset": {"root": str(root / "data")},
        "model": {"provider": provider, "name": name, "weights": {"source": "none"}},
        "preprocessing": {
            "source": "explicit", "image_size": 32,
            "mean": [0, 0, 0], "std": [1, 1, 1], "crop_pct": 1,
        },
        "augmentation": {"preset": "none"},
        "trainer": {
            "max_epochs": 2, "accelerator": "gpu", "devices": 1,
            "precision": precision, "deterministic": True, "enable_progress_bar": False,
        },
        "optimizer": {"lr": 0.002},
        "scheduler": {"name": "warmup_cosine", "min_lr": 0.0001},
        "runtime": {
            "offline": True, "cache_dir": str(root / "cache"),
            "weights_dir": str(root / "weights"),
        },
    })
    observations = []
    original = ClassificationTask.training_step

    def observe(task, batch, batch_idx):
        loss = original(task, batch, batch_idx)
        assert batch["image"].is_cuda and loss.is_cuda
        assert torch.isfinite(loss).all()
        observations.append(str(loss.device))
        return loss

    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with patch.object(ClassificationTask, "training_step", observe):
        reference = train(cfg.model_copy(deep=True))
        checkpoint = torch.load(reference / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["global_step"] == 8
        assert read_json(reference / "status.json")["state"] == "SUCCEEDED"
        assert checkpoint["optimizer_states"][0]["state"]
        for tensor in checkpoint["state_dict"].values():
            assert torch.isfinite(tensor).all()
        if precision == "16-mixed":
            assert checkpoint["MixedPrecision"]["scale"] > 0
        if resume:
            other = cfg.model_copy(deep=True)
            other.experiment.name += "_resumed"
            interrupted = train(other.model_copy(deep=True), stop_after_epoch=1)
            assert read_json(interrupted / "status.json")["state"] == "PAUSED"
            other.checkpoint.resume_from = interrupted / "checkpoints/last.pt"
            resumed = train(other)
            restored = torch.load(resumed / "checkpoints/last.pt", map_location="cpu", weights_only=False)
            assert restored["global_step"] == checkpoint["global_step"]
            assert restored["lr_schedulers"] == checkpoint["lr_schedulers"]
            for key, value in checkpoint["state_dict"].items():
                torch.testing.assert_close(value, restored["state_dict"][key], rtol=0, atol=0)
            if precision == "16-mixed":
                assert restored["MixedPrecision"] == checkpoint["MixedPrecision"]
    assert len(observations) == (16 if resume else 8)
    metrics = evaluate_model(reference, cfg.model_copy(deep=True), split="test")
    return {
        "case": case, "status": "passed", "run_dir": str(reference),
        "epochs": 2, "optimizer_steps": checkpoint["global_step"],
        "observed_training_devices": sorted(set(observations)),
        "strict_resume_zero_tolerance": resume,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": perf_counter() - started,
        "synthetic_test_metrics_cpu": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的验收输出目录，拒绝覆盖")
    parser.add_argument("--precision", choices=["all", "32-true", "16-mixed"], default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝回退到 CPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("请通过 CUDA_VISIBLE_DEVICES 显式选择一张 GPU")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    make_data(root / "data")
    report = {
        "scope": "合成数据单卡训练、FP16 AMP、严格续训；独立测试集评估在 CPU 执行",
        "environment": environment(), "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cases": [],
    }
    for provider, name, precision, resume in [
        ("builtin", "tiny_cnn", "32-true", True),
        ("builtin", "tiny_cnn", "16-mixed", True),
        ("torchvision", "resnet18", "32-true", False),
        ("torchvision", "resnet18", "16-mixed", False),
    ]:
        if args.precision not in {"all", precision}:
            continue
        try:
            result = run_case(root, provider, name, precision, resume)
        except Exception as exc:
            result = {
                "case": f"{provider}_{name}_{precision}", "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            }
        report["cases"].append(result)
        report["status"] = "failed" if any(c["status"] == "failed" for c in report["cases"]) else "passed"
        (root / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
