"""Resolve and reserve physical GPUs for Web training jobs."""

import sys


def resolve_device(device, indices, available, cuda_available):
    indices = list(indices)
    if device == "cpu":
        if indices:
            raise ValueError("CPU 模式不能指定 GPU")
        return "cpu", []
    usable = cuda_available and bool(available)
    if not usable:
        if device == "gpu" or indices:
            raise ValueError("当前环境没有可用的 CUDA GPU 或驱动监控")
        return "cpu", []
    if not set(indices) <= {gpu["index"] for gpu in available}:
        raise ValueError("所选 GPU 不可用，请刷新设备列表")
    if len(indices) > 1 and sys.platform != "linux":
        raise ValueError("多卡 DDP 训练仅支持 Linux CUDA 环境")
    return "gpu", indices


def choose_gpus(spec, available, reservations, limits):
    """Return all cards atomically, or None to keep the job queued."""
    trainer = spec.get("config", {}).get("trainer", {})
    if trainer.get("accelerator") != "gpu":
        return []
    selection = spec.get("device_selection")
    if selection is None:
        legacy = spec.get("request", {}).get("gpu_index")
        requested = [] if legacy is None else [legacy]
    else:
        requested = selection["gpu_indices"]
    count = trainer.get("devices", 1)
    if count > 1 and len(requested) != count:
        raise ValueError("多卡任务必须明确指定全部 GPU")
    eligible = [
        gpu
        for gpu in available
        if (
            limits.max_jobs_per_gpu == 0
            or sum(gpu["index"] in cards for cards in reservations.values()) < limits.max_jobs_per_gpu
        )
        and (not limits.enabled or gpu["free_gb"] >= limits.min_gpu_free_gb)
    ]
    if requested:
        return list(requested) if set(requested) <= {gpu["index"] for gpu in eligible} else None
    if not eligible:
        return None
    return [max(eligible, key=lambda gpu: gpu["free_gb"])["index"]]


def gpu_wait_reason(spec, available, reservations, limits):
    selection = spec.get("device_selection", {})
    requested = selection.get("gpu_indices", [])
    if not selection and spec.get("request", {}).get("gpu_index") is not None:
        requested = [spec["request"]["gpu_index"]]
    cards = {gpu["index"]: gpu for gpu in available}
    reasons = []
    for index in requested or list(cards):
        card = cards.get(index)
        if card is None:
            reasons.append(f"GPU {index} 未检测到")
            continue
        count = sum(index in used for used in reservations.values())
        if limits.max_jobs_per_gpu and count >= limits.max_jobs_per_gpu:
            reasons.append(f"GPU {index} 任务数 {count}/{limits.max_jobs_per_gpu}")
        if limits.enabled and card["free_gb"] < limits.min_gpu_free_gb:
            reasons.append(f"GPU {index} 空闲 {card['free_gb']:.1f} GB，低于 {limits.min_gpu_free_gb:.1f} GB")
    return "等待 GPU：" + ("；".join(reasons) or "未检测到可用显卡")
