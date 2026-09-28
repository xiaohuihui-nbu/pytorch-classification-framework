import hashlib
import logging

import timm
import torch
from filelock import FileLock
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision import models as tv_models

from .registry import lookup
from .utils import digest, file_hash, read_json, write_json


class TinyCNN(nn.Module):
    """Small offline reference model for contract tests and learning examples."""

    def __init__(self, channels, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def create_architecture(provider, name, channels, outputs):
    if provider == "builtin" and name == "tiny_cnn":
        return TinyCNN(channels, outputs)
    if provider == "timm":
        return timm.create_model(name, pretrained=False, in_chans=channels, num_classes=outputs)
    if provider == "torchvision":
        if channels != 3:
            raise ValueError("torchvision adapter currently requires 3 channels")
        return tv_models.get_model(name, weights=None, num_classes=outputs)
    return lookup("model", provider)(name=name, in_channels=channels, num_classes=outputs)


def _replace_tv_head(model, outputs):
    if isinstance(getattr(model, "fc", None), nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, outputs)
    elif isinstance(getattr(model, "classifier", None), nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, outputs)
    elif isinstance(getattr(model, "classifier", None), nn.Sequential) and isinstance(
        model.classifier[-1], nn.Linear
    ):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, outputs)
    elif isinstance(getattr(model, "head", None), nn.Linear):
        model.head = nn.Linear(model.head.in_features, outputs)
    elif isinstance(getattr(model, "heads", None), nn.Sequential) and isinstance(
        getattr(model.heads, "head", None), nn.Linear
    ):
        model.heads.head = nn.Linear(model.heads.head.in_features, outputs)
    else:
        raise ValueError("torchvision head not supported by this adapter; use timm or a plugin")


def classifier_module(model, provider):
    if provider == "timm":
        return model.get_classifier()
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    if isinstance(getattr(model, "head", None), nn.Linear):
        return model.head
    if isinstance(getattr(model, "heads", None), nn.Sequential):
        return model.heads
    raise ValueError("linear_probe needs an adapter exposing classifier/fc/get_classifier")


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def build_model(cfg, classes):
    spec = cfg.model
    if spec.num_classes != "auto" and spec.num_classes != len(classes):
        raise ValueError("E_CLASS_MAP: model.num_classes differs from dataset classes")
    spec.num_classes = len(classes)
    outputs = 1 if cfg.task.type == "binary" else len(classes)
    architecture = dict(provider=spec.provider, name=spec.name, channels=spec.input_channels, outputs=outputs)
    preprocess = cfg.preprocessing.model_dump()
    model = None
    provenance = {"source": spec.weights.source}
    if spec.weights.source == "provider_default":
        weights_dir = cfg.runtime.weights_dir or cfg.runtime.cache_dir
        weights_dir.mkdir(parents=True, exist_ok=True)
        key = digest({**architecture, "seed": cfg.experiment.seed})
        cache = weights_dir / f"init-{key}.safetensors"
        metadata = cache.with_suffix(".json")
        with FileLock(str(cache) + ".lock"):
            if cache.exists() and metadata.exists():
                logging.getLogger(__name__).info("加载本地预训练初始化缓存：%s", cache)
                meta = read_json(metadata)
                if file_hash(cache) != meta["sha256"]:
                    raise ValueError("Pretrained cache checksum mismatch")
                model = create_architecture(**architecture)
                model.load_state_dict(load_file(str(cache)), strict=True)
                weight_preprocess = meta["preprocess"]
                provenance.update(meta)
            else:
                if cfg.runtime.offline:
                    raise ValueError(
                        f"E_PRETRAINED_MISSING: offline cache missing; run prepare online: {cache}"
                    )
                logging.getLogger(__name__).info(
                    "初始化缓存未命中，加载发布方权重（允许下载）：%s/%s", spec.provider, spec.name
                )
                if spec.provider == "timm":
                    model = timm.create_model(
                        spec.name,
                        pretrained=True,
                        in_chans=spec.input_channels,
                        num_classes=outputs,
                        cache_dir=str(weights_dir),
                    )
                    data_cfg = timm.data.resolve_model_data_config(model)
                    if data_cfg.get("crop_mode", "center") != "center" or (
                        data_cfg["input_size"][-2] != data_cfg["input_size"][-1]
                    ):
                        raise ValueError(
                            "Only square inputs with center-crop weight preprocessing are supported"
                        )
                    weight_preprocess = {
                        "image_size": data_cfg["input_size"][-1],
                        "mean": list(data_cfg["mean"]),
                        "std": list(data_cfg["std"]),
                        "interpolation": data_cfg["interpolation"],
                        "crop_pct": data_cfg.get("crop_pct", 0.875),
                        "resize_size": int(data_cfg["input_size"][-1] / data_cfg.get("crop_pct", 0.875)),
                    }
                    provenance["provider_config"] = {
                        k: v
                        for k, v in model.pretrained_cfg.items()
                        if isinstance(v, (str, int, float, bool, list, tuple, type(None)))
                    }
                elif spec.provider == "torchvision":
                    if spec.input_channels != 3:
                        raise ValueError("torchvision pretrained adapter requires RGB")
                    torch.hub.set_dir(str(weights_dir / "torch"))
                    weights = tv_models.get_model_weights(spec.name).DEFAULT
                    model = tv_models.get_model(spec.name, weights=weights)
                    _replace_tv_head(model, outputs)
                    transform = weights.transforms()
                    weight_preprocess = {
                        "image_size": transform.crop_size[0],
                        "resize_size": transform.resize_size[0],
                        "crop_pct": transform.crop_size[0] / transform.resize_size[0],
                        "mean": transform.mean,
                        "std": transform.std,
                        "interpolation": transform.interpolation.value,
                    }
                    provenance["weight_id"] = str(weights)
                    provenance["url"] = weights.url
                else:
                    raise ValueError("provider_default weights supported only by timm/torchvision")
                save_file(
                    {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(cache)
                )
                provenance.update({"sha256": file_hash(cache), "preprocess": weight_preprocess})
                write_json(metadata, provenance)
        if preprocess["source"] == "model_weights":
            explicit_size = preprocess["image_size"]
            preprocess.update(weight_preprocess)
            if explicit_size != "auto":
                preprocess["image_size"] = explicit_size
                if explicit_size != weight_preprocess["image_size"]:
                    rounding = int if spec.provider == "timm" else round
                    preprocess["resize_size"] = rounding(explicit_size / preprocess["crop_pct"])
    if model is None:
        model = create_architecture(**architecture)
    if spec.weights.source == "local":
        path = spec.weights.path
        state = (
            load_file(str(path))
            if path.suffix == ".safetensors"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        model.load_state_dict(state, strict=True)
        provenance.update({"path": str(path), "sha256": file_hash(path)})
        if preprocess["source"] == "model_weights":
            raise ValueError("Local weights require explicit preprocessing; use a bundle for prediction")
    if preprocess["image_size"] == "auto":
        preprocess["image_size"] = 224
    if not isinstance(preprocess["image_size"], int) or preprocess["image_size"] < 8:
        raise ValueError("image_size must be an integer >= 8")
    if len(preprocess["mean"]) != spec.input_channels or len(preprocess["std"]) != spec.input_channels:
        raise ValueError("Normalization channels do not match model; configure mean/std explicitly")
    if any(s <= 0 for s in preprocess["std"]):
        raise ValueError("Normalization std must be positive")
    if spec.training_mode == "linear_probe":
        model.requires_grad_(False)
        classifier_module(model, spec.provider).requires_grad_(True)
    provenance["initial_state_sha256"] = state_hash(model)
    return model, architecture, preprocess, provenance
