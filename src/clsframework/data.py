import csv
import hashlib
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .registry import DATASET_CATALOG, lookup
from .utils import digest, file_hash, read_json, write_json


@dataclass
class PreparedData:
    rows: list[dict]
    classes: list[str]
    root: Path
    fingerprint: str
    report: dict
    sources: dict = field(default_factory=dict, repr=False)


def class_names(path, inferred):
    if path:
        raw = read_json(path)
        if isinstance(raw, dict):
            if sorted(raw.values()) != list(range(len(raw))):
                raise ValueError("classes.json indices must be unique and contiguous from zero")
            names = sorted(raw, key=raw.get)
        else:
            names = raw
    else:
        names = sorted(inferred)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("classes.json must be a string list or name-to-index mapping")
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("At least two unique classes required")
    return names


def _custom_rows(cfg):
    spec = cfg.dataset
    rows = []
    if spec.provider == "imagefolder":
        if cfg.task.type == "multilabel":
            raise ValueError("ImageFolder cannot represent multilabel targets; use JSONL")
        for split, dirname in spec.splits.items():
            folder = spec.root / dirname
            if not folder.exists():
                continue
            for label in sorted(folder.iterdir()):
                if label.is_dir():
                    for path in sorted(label.rglob("*")):
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                            relative = path.relative_to(spec.root).as_posix()
                            rows.append(
                                {"sample_id": relative, "path": relative, "label": label.name, "split": split}
                            )
    else:
        if not spec.manifest:
            raise ValueError("manifest provider requires dataset.manifest")
        if spec.manifest.suffix.lower() == ".csv":
            with spec.manifest.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if cfg.task.type == "multilabel":
                raise ValueError("Use JSONL with labels arrays for multilabel")
        elif spec.manifest.suffix.lower() == ".jsonl":
            import json

            rows = [
                json.loads(line)
                for line in spec.manifest.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        else:
            raise ValueError("manifest must be .csv or .jsonl")
    inferred = set()
    for row in rows:
        if row.get("split") == "train":
            if cfg.task.type == "multilabel":
                if not isinstance(row.get("labels"), list):
                    raise ValueError("Multilabel rows require a labels array; null is not a negative label")
                inferred.update(row["labels"])
            else:
                inferred.add(str(row["label"]))
    names = class_names(spec.classes_file, inferred)
    mapping = {name: i for i, name in enumerate(names)}
    seen_ids, seen_paths, seen_hashes, groups = set(), set(), {}, {}
    normalized = []
    for row in rows:
        split = row.get("split")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        path = (spec.root / row["path"]).resolve()
        if not path.is_relative_to(spec.root.resolve()):
            raise ValueError(f"Image path escapes dataset.root: {row['path']}")
        sid = str(row.get("sample_id") or row["path"])
        if sid in seen_ids or path in seen_paths:
            raise ValueError(f"Duplicate sample_id/path: {sid}")
        seen_ids.add(sid)
        seen_paths.add(path)
        with Image.open(path) as image:
            image.load()
            if min(image.size) < 1:
                raise ValueError(f"Empty image: {path}")
        checksum = file_hash(path) if spec.integrity == "strict" else str(path.stat().st_size)
        if spec.integrity == "strict" and checksum in seen_hashes and seen_hashes[checksum] != split:
            raise ValueError(f"E_DATA_LEAKAGE: identical image across splits: {sid}")
        seen_hashes[checksum] = split
        group = row.get("group_id")
        if group and group in groups and groups[group] != split:
            raise ValueError(f"E_DATA_LEAKAGE: group {group} crosses splits")
        if group:
            groups[group] = split
        labels = row.get("labels") if cfg.task.type == "multilabel" else [str(row["label"])]
        if not isinstance(labels, list) or len(labels) != len(set(labels)):
            raise ValueError(f"Invalid/duplicate labels for {sid}")
        unknown = set(labels) - set(mapping)
        if unknown:
            raise ValueError(f"E_CLASS_MAP: {sid} has unknown labels {unknown}")
        target = [mapping[x] for x in labels] if cfg.task.type == "multilabel" else mapping[labels[0]]
        normalized.append(
            {
                "sample_id": sid,
                "path": path.relative_to(spec.root).as_posix(),
                "split": split,
                "target": target,
                "group_id": group or None,
                "sha256": checksum,
            }
        )
    return normalized, names, {}


def _torchvision_rows(cfg):
    spec = cfg.dataset
    if cfg.task.type != "multiclass" or spec.name not in DATASET_CATALOG:
        raise ValueError(f"torchvision datasets require multiclass and name in {DATASET_CATALOG}")
    download = spec.download and not cfg.runtime.offline
    spec.root.mkdir(parents=True, exist_ok=True)
    constructors = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    with FileLock(str(spec.root / ".download.lock")):
        if spec.name in constructors:
            ctor = constructors[spec.name]
            sources = {
                "train": ctor(str(spec.root), train=True, download=download),
                "test": ctor(str(spec.root), train=False, download=download),
            }
        elif spec.name == "pets":
            sources = {
                "train": datasets.OxfordIIITPet(str(spec.root), split="trainval", download=download),
                "test": datasets.OxfordIIITPet(str(spec.root), split="test", download=download),
            }
        else:
            sources = {
                s: datasets.Flowers102(str(spec.root), split=s, download=download)
                for s in ("train", "val", "test")
            }
    labels = {}
    for split, dataset in sources.items():
        if hasattr(dataset, "targets"):
            labels[split] = np.asarray(dataset.targets).astype(int).tolist()
        else:
            labels[split] = [int(dataset[i][1]) for i in range(len(dataset))]
    count = max(labels["train"]) + 1
    names = list(getattr(sources["train"], "classes", [str(i) for i in range(count)]))
    if spec.classes_file:
        names = class_names(spec.classes_file, [])
        if len(names) != count:
            raise ValueError("Explicit classes length disagrees with torchvision labels")
    train_indices = list(range(len(labels["train"])))
    selection = {s: (s, list(range(len(v)))) for s, v in labels.items()}
    if "val" not in sources:
        ti, vi = train_test_split(
            train_indices,
            test_size=spec.split_policy.val_fraction,
            random_state=spec.split_policy.seed,
            stratify=labels["train"],
        )
        selection["train"], selection["val"] = ("train", sorted(ti)), ("train", sorted(vi))
    rows = []
    for split, (source, indices) in selection.items():
        if spec.limit_per_split:
            buckets = defaultdict(list)
            for i in indices:
                buckets[labels[source][i]].append(i)
            indices = [
                i for batch in itertools.zip_longest(*buckets.values()) for i in batch if i is not None
            ][: spec.limit_per_split]
        for i in indices:
            image, target = sources[source][i]
            checksum = (
                hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                if spec.integrity == "strict"
                else None
            )
            rows.append(
                {
                    "sample_id": f"{spec.name}/{source}/{i}",
                    "source": source,
                    "index": i,
                    "split": split,
                    "target": int(target),
                    "sha256": checksum,
                }
            )
    return rows, names, sources


def prepare_data(cfg):
    spec = cfg.dataset
    if spec.provider == "torchvision":
        rows, names, sources = _torchvision_rows(cfg)
    elif spec.provider in {"imagefolder", "manifest"}:
        rows, names, sources = _custom_rows(cfg)
    else:
        return lookup("dataset", spec.provider)(cfg)
    if spec.limit_per_split and spec.provider != "torchvision":
        rows = [
            r
            for s in ("train", "val", "test")
            for r in [x for x in rows if x["split"] == s][: spec.limit_per_split]
        ]
    counts = dict(Counter(r["split"] for r in rows))
    if not counts.get("train") or not counts.get("val"):
        raise ValueError("Nonempty train and val splits required")
    if cfg.task.type == "binary" and len(names) != 2:
        raise ValueError("binary single-logit mode requires exactly two classes")
    if spec.num_classes != "auto" and spec.num_classes != len(names):
        raise ValueError("dataset.num_classes disagrees with class mapping")
    spec.num_classes = len(names)
    frequency = {}
    for split in counts:
        c = Counter()
        for row in rows:
            if row["split"] == split:
                c.update(row["target"] if isinstance(row["target"], list) else [row["target"]])
        frequency[split] = {name: c[i] for i, name in enumerate(names)}
    hash_splits = defaultdict(set)
    for row in rows:
        if row.get("sha256"):
            hash_splits[row["sha256"]].add(row["split"])
    report = {
        "split_counts": counts,
        "class_counts": frequency,
        "integrity": spec.integrity,
        "official_cross_split_duplicate_hashes": sum(len(x) > 1 for x in hash_splits.values()),
    }
    fingerprint = digest({"classes": names, "rows": rows, "task": cfg.task.type, "provider": spec.provider})
    return PreparedData(rows, names, spec.root, fingerprint, report, sources)


def save_prepared(data, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "classes.json", {name: i for i, name in enumerate(data.classes)})
    write_json(directory / "dataset_fingerprint.json", {"sha256": data.fingerprint, **data.report})
    import json

    (directory / "dataset_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data.rows), encoding="utf-8"
    )


def make_transform(preprocess, augmentation=None):
    interpolation = InterpolationMode(preprocess["interpolation"])
    size = preprocess["image_size"]
    ops = []
    if augmentation and augmentation.preset != "none":
        ops += [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(augmentation.horizontal_flip),
        ]
        if augmentation.preset == "strong":
            ops += [transforms.RandAugment()]
    else:
        ops += [
            transforms.Resize(
                preprocess.get("resize_size") or round(size / preprocess["crop_pct"]),
                interpolation=interpolation,
            ),
            transforms.CenterCrop(size),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(preprocess["mean"], preprocess["std"])]
    if augmentation and augmentation.preset == "strong":
        ops += [transforms.RandomErasing(p=0.1)]
    return transforms.Compose(ops)


class ClassificationDataset(Dataset):
    def __init__(self, data, split, task, preprocess, augmentation=None):
        self.data, self.task, self.preprocess = data, task, preprocess
        self.rows = [row for row in data.rows if row["split"] == split]
        self.transform = make_transform(preprocess, augmentation)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if "source" in row:
            image = self.data.sources[row["source"]][row["index"]][0]
        else:
            with Image.open(self.data.root / row["path"]) as opened:
                image = ImageOps.exif_transpose(opened).copy()
        image = image.convert(self.preprocess["color_mode"])
        if self.task == "multilabel":
            target = torch.zeros(len(self.data.classes))
            target[row["target"]] = 1
        elif self.task == "binary":
            target = torch.tensor([float(row["target"])])
        else:
            target = torch.tensor(row["target"], dtype=torch.long)
        return {"image": self.transform(image), "target": target, "sample_id": row["sample_id"]}


def loader(cfg, data, split, preprocess):
    ds = ClassificationDataset(
        data, split, cfg.task.type, preprocess, cfg.augmentation if split == "train" else None
    )
    return DataLoader(
        ds,
        batch_size=cfg.loader.batch_size_per_device,
        shuffle=split == "train",
        num_workers=cfg.loader.num_workers,
        pin_memory=bool(cfg.loader.pin_memory),
        persistent_workers=bool(cfg.loader.persistent_workers),
    )
