import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from .data import loader, make_transform, prepare_data
from .metrics import evaluate_arrays
from .models import create_architecture
from .registry import load_plugins
from .utils import append_json, file_hash, read_json, write_json


def model_source(path):
    """Select a self-contained checkpoint or a legacy bundle from a saved run."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() in {".pt", ".ckpt"}:
        return path
    if path.is_dir():
        # Old .pt checkpoints rely on their bundle metadata. New runs have no bundle.
        for candidate in (path / "bundle", path):
            if (candidate / "bundle_manifest.json").is_file():
                return candidate
        if (path / "checkpoints/best.pt").is_file():
            return path / "checkpoints/best.pt"
    raise ValueError(f"Model must be a framework .pt checkpoint or saved run directory: {path}")


def model_directory(source):
    source = Path(source)
    if source.is_file() and source.parent.name == "checkpoints":
        return source.parent.parent
    return source.parent


def load_model(path, allow_plugins=False):
    source = model_source(path)
    if source.is_dir():
        return load_bundle(source, allow_plugins)
    status_path = source.parent.parent / "status.json" if source.parent.name == "checkpoints" else None
    if source.name == "best.pt" and status_path is not None and status_path.is_file():
        checksum = read_json(status_path).get("best_checkpoint_sha256")
        if checksum is not None and file_hash(source) != checksum:
            raise ValueError(f"Model integrity check failed: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    metadata = checkpoint.get("cls_inference")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(
            "E_MODEL_METADATA: checkpoint lacks supported inference metadata; use the original run"
        )
    spec = metadata["model_spec"]
    model = inference_architecture(spec, allow_plugins)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return (model, spec, metadata["classes"], metadata["preprocessing"], metadata["threshold"], metadata)


def inference_architecture(spec, allow_plugins):
    if spec["plugins"]:
        if not allow_plugins:
            raise ValueError("This model requires installed Python plugins; pass --allow-plugins explicitly")
        load_plugins(spec["plugins"])
    return create_architecture(**{k: spec[k] for k in ("provider", "name", "channels", "outputs")})


def load_bundle(path, allow_plugins=False):
    """Read legacy inference bundles; new training writes self-contained .pt files."""
    path = Path(path)
    manifest = read_json(path / "bundle_manifest.json")
    version = manifest.get("schema_version")
    required = {"model_spec.json", "classes.json", "preprocess.json", "thresholds.json"}
    if version != 1:
        raise ValueError(f"Unsupported bundle schema_version: {version}")
    weights_path = "model.safetensors"
    required.add(weights_path)
    if not required.issubset(manifest["checksums"]):
        raise ValueError("Incomplete bundle checksums")
    for name, checksum in manifest["checksums"].items():
        if Path(name).name != name or not (path / name).is_file() or file_hash(path / name) != checksum:
            raise ValueError(f"Bundle integrity check failed: {name}")
    spec = read_json(path / "model_spec.json")
    model = inference_architecture(spec, allow_plugins)
    state = load_file(str(path / weights_path))
    model.load_state_dict(state, strict=True)
    model.eval()
    return (
        model,
        spec,
        read_json(path / "classes.json"),
        read_json(path / "preprocess.json"),
        read_json(path / "thresholds.json")["threshold"],
        manifest,
    )


def decoded(probabilities, task, classes, threshold):
    if task == "multiclass":
        i = int(np.argmax(probabilities))
        return {
            "label": classes[i],
            "class_index": i,
            "probabilities": dict(zip(classes, map(float, probabilities), strict=True)),
        }
    if task == "binary":
        p = float(probabilities[0])
        return {
            "label": classes[int(p >= threshold)],
            "positive_class": classes[1],
            "probabilities": {classes[0]: 1 - p, classes[1]: p},
            "threshold": threshold,
        }
    return {
        "labels": [label for label, value in zip(classes, probabilities, strict=True) if value >= threshold],
        "probabilities": dict(zip(classes, map(float, probabilities), strict=True)),
        "threshold": threshold,
    }


def predict(source, inputs, output, allow_plugins=False, batch_size=16, *, model_data=None, timings=None):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    timings = timings if timings is not None else {}
    started = perf_counter()
    model, spec, classes, preprocess, threshold, _ = (
        load_model(source, allow_plugins) if model_data is None else model_data
    )
    timings["model_load_seconds"] = perf_counter() - started
    timings.update(preprocess_seconds=0.0, forward_seconds=0.0, write_seconds=0.0)
    inputs, output = Path(inputs), Path(output)
    if output.exists():
        raise FileExistsError(f"Prediction output already exists: {output}")
    files = (
        [inputs]
        if inputs.is_file()
        else sorted(
            p
            for p in inputs.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        )
    )
    if not files:
        raise ValueError("No input images found")
    transform = make_transform(preprocess)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream, torch.inference_mode():
            for start in range(0, len(files), batch_size):
                paths = files[start : start + batch_size]
                tensors = []
                started = perf_counter()
                for path in paths:
                    with Image.open(path) as image:
                        tensors.append(
                            transform(ImageOps.exif_transpose(image).convert(preprocess["color_mode"]))
                        )
                images = torch.stack(tensors)
                timings["preprocess_seconds"] += perf_counter() - started
                started = perf_counter()
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite prediction logits")
                probs = logits.softmax(1) if spec["task"] == "multiclass" else logits.sigmoid()
                timings["forward_seconds"] += perf_counter() - started
                started = perf_counter()
                for path, probability in zip(paths, probs.tolist(), strict=True):
                    stream.write(
                        json.dumps(
                            {"path": str(path), **decoded(probability, spec["task"], classes, threshold)},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                timings["write_seconds"] += perf_counter() - started
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"samples": len(files), "output": str(output)}


def save_report(directory, report, classes, plots=True):
    directory = Path(directory)
    write_json(directory / "metrics.json", report)
    with (directory / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for label, row in report["per_class"].items():
            writer.writerow({"class": label, **row})
    write_json(directory / "classes.json", classes)
    if plots:
        from .visualization import evaluation_report

        evaluation_report(directory, report, classes)


def evaluate_model(source, cfg, split="test", output=None, allow_plugins=False):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    model, spec, classes, preprocess, threshold, manifest = load_model(source, allow_plugins)
    if cfg.task.type != spec["task"]:
        raise ValueError("Dataset task differs from model task")
    data = prepare_data(cfg)
    if data.classes != classes or data.fingerprint != manifest["dataset_fingerprint"]:
        raise ValueError("E_DATA_MISMATCH: class mapping/data fingerprint differs from training model")
    cfg.loader.pin_memory = False
    cfg.loader.persistent_workers = cfg.loader.num_workers > 0
    batches = loader(cfg, data, split, preprocess)
    if len(batches.dataset) == 0:
        raise ValueError(f"No {split} split available; test results cannot be invented")
    directory = Path(output) if output else model_directory(model_source(source)) / "evaluation" / split
    directory.mkdir(parents=True, exist_ok=False)
    targets, probabilities = [], []
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch["image"])
            if isinstance(logits, dict):
                logits = logits["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            probs = logits.softmax(1) if spec["task"] == "multiclass" else logits.sigmoid()
            targets.extend(batch["target"].tolist())
            probabilities.extend(probs.tolist())
            for sid, target, p in zip(
                batch["sample_id"], batch["target"].tolist(), probs.tolist(), strict=True
            ):
                append_json(
                    directory / "predictions.jsonl",
                    {"sample_id": sid, "target": target, **decoded(p, spec["task"], classes, threshold)},
                )
    report = evaluate_arrays(spec["task"], targets, probabilities, classes, threshold)
    report.update(split=split, dataset_fingerprint=data.fingerprint, output_dir=str(directory.resolve()))
    save_report(directory, report, classes, cfg.visualization.enabled)
    return report


def export_onnx(source, output, allow_plugins=False):
    import onnx
    import onnxruntime as ort

    model, spec, classes, preprocess, threshold, _ = load_model(source, allow_plugins)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = preprocess["image_size"]
    sample = torch.randn(1, spec["channels"], size, size)
    torch.onnx.export(
        model,
        sample,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    with torch.inference_mode():
        for n in (1, 3):
            x = torch.randn(n, spec["channels"], size, size)
            reference = model(x).numpy()
            exported = session.run(None, {"images": x.numpy()})[0]
            np.testing.assert_allclose(reference, exported, rtol=1e-3, atol=1e-4)
            errors.append({"batch": n, "max_abs_error": float(np.max(np.abs(reference - exported)))})
    metadata = {
        "preprocess": preprocess,
        "classes": classes,
        "task": spec["task"],
        "threshold": threshold,
        "sha256": file_hash(output),
        "validation": errors,
        "rtol": 1e-3,
        "atol": 1e-4,
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata
