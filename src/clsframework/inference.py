import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from .data import loader, make_transform, prepare_data
from .metrics import evaluate_arrays
from .models import create_architecture
from .registry import load_plugins
from .utils import append_json, file_hash, read_json, write_json


def load_bundle(path, allow_plugins=False):
    path = Path(path)
    manifest = read_json(path / "bundle_manifest.json")
    required = {"model.safetensors", "model_spec.json", "classes.json", "preprocess.json", "thresholds.json"}
    if not required.issubset(manifest["checksums"]):
        raise ValueError("Incomplete bundle checksums")
    for name, checksum in manifest["checksums"].items():
        if Path(name).name != name or file_hash(path / name) != checksum:
            raise ValueError(f"Bundle integrity check failed: {name}")
    spec = read_json(path / "model_spec.json")
    if spec["plugins"]:
        if not allow_plugins:
            raise ValueError("This bundle requires installed Python plugins; pass --allow-plugins explicitly")
        load_plugins(spec["plugins"])
    model = create_architecture(**{k: spec[k] for k in ("provider", "name", "channels", "outputs")})
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
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


def predict(bundle, inputs, output, allow_plugins=False, batch_size=16):
    model, spec, classes, preprocess, threshold, _ = load_bundle(bundle, allow_plugins)
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
                for path in paths:
                    with Image.open(path) as image:
                        tensors.append(
                            transform(ImageOps.exif_transpose(image).convert(preprocess["color_mode"]))
                        )
                logits = model(torch.stack(tensors))
                if isinstance(logits, dict):
                    logits = logits["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite prediction logits")
                probs = logits.softmax(1) if spec["task"] == "multiclass" else logits.sigmoid()
                for path, probability in zip(paths, probs.tolist(), strict=True):
                    stream.write(
                        json.dumps(
                            {"path": str(path), **decoded(probability, spec["task"], classes, threshold)},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
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


def test_bundle(bundle, cfg, split="test", output=None, allow_plugins=False):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    model, spec, classes, preprocess, threshold, manifest = load_bundle(bundle, allow_plugins)
    if cfg.task.type != spec["task"]:
        raise ValueError("Dataset task differs from bundle task")
    data = prepare_data(cfg)
    if data.classes != classes or data.fingerprint != manifest["dataset_fingerprint"]:
        raise ValueError("E_DATA_MISMATCH: class mapping/data fingerprint differs from training bundle")
    cfg.loader.pin_memory = False
    cfg.loader.persistent_workers = cfg.loader.num_workers > 0
    batches = loader(cfg, data, split, preprocess)
    if len(batches.dataset) == 0:
        raise ValueError(f"No {split} split available; test results cannot be invented")
    directory = Path(output) if output else Path(bundle).parent / "evaluation" / split
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


def export_onnx(bundle, output, allow_plugins=False):
    import onnx
    import onnxruntime as ort

    model, spec, classes, preprocess, threshold, _ = load_bundle(bundle, allow_plugins)
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
