"""下载或读取 flower_photos，去重后生成固定分层划分与训练清单。"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
# 已确认且由用户指定排除的跨类别重复；路径与内容摘要必须同时匹配。
KNOWN_LABEL_CONFLICTS = {
    "roses/15922772266_1167a06620.jpg": "6784c09ff4eeec0e0ac148cdc74682f99c424ac89d85321d95161e20da9204e9",
    "tulips/15922772266_1167a06620.jpg": "6784c09ff4eeec0e0ac148cdc74682f99c424ac89d85321d95161e20da9204e9",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(archive) + ".lock"):
        if archive.is_file():
            return
        with TemporaryDirectory(prefix="flower-download-", dir=archive.parent) as temporary:
            target = Path(temporary) / "download.tgz"
            with urllib.request.urlopen(URL, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0)) or None
                with (
                    target.open("wb") as stream,
                    tqdm(total=total, unit="B", unit_scale=True, desc="下载") as bar,
                ):
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        bar.update(len(block))
                if total is not None and target.stat().st_size != total:
                    raise OSError("下载不完整，请重新运行")
            target.rename(archive)


def extract(archive, destination):
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f"压缩包包含不安全路径或链接：{member.name}")
        package.extractall(destination, members=members, filter="data")
    source = destination / "flower_photos"
    if not source.is_dir():
        raise ValueError("压缩包内缺少 flower_photos 目录")
    return source


def split_dataset(source, output, *, seed=42, train_ratio=0.7, val_ratio=0.15, provenance=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train-ratio、val-ratio 和剩余 test 比例必须大于 0")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有数据目录：{output}；复用现有数据或指定新的 --output")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source 与 output 不能相互包含")
    if not source.is_dir() or sorted(p.name for p in source.iterdir() if p.is_dir()) != CLASSES:
        raise ValueError(f"source 必须包含五类目录：{CLASSES}")
    files = sorted(
        p
        for label in CLASSES
        for p in (source / label).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    records, excluded, seen = [], [], {}
    for path in tqdm(files, desc="校验与去重", unit="张"):
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"图片路径越过 source：{path}")
        relative = path.relative_to(source).as_posix()
        if relative in KNOWN_LABEL_CONFLICTS and sha256(path) == KNOWN_LABEL_CONFLICTS[relative]:
            excluded.append({
                "source_path": relative, "reason": "known_label_conflict",
                "sha256": KNOWN_LABEL_CONFLICTS[relative],
            })
            continue
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
                pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            excluded.append({"source_path": relative, "reason": "invalid_image", "error": str(exc)})
            continue
        label = path.parent.name
        if pixel_hash in seen:
            kept = seen[pixel_hash]
            if kept["label"] != label:
                raise ValueError(f"相同像素存在类别冲突，请人工检查：{relative} 与 {kept['source_path']}")
            excluded.append(
                {"source_path": relative, "reason": "duplicate_pixels", "kept": kept["source_path"]}
            )
            continue
        record = {
            "sample_id": relative,
            "source_path": relative,
            "label": label,
            "sha256": sha256(path),
            "pixel_sha256": pixel_hash,
            "group_id": pixel_hash,
        }
        seen[pixel_hash] = record
        records.append(record)
    rng = random.Random(seed)
    rows = []
    for label in CLASSES:
        items = [r for r in records if r["label"] == label]
        rng.shuffle(items)
        train_count, val_count = int(len(items) * train_ratio), round(len(items) * val_ratio)
        if min(train_count, val_count, len(items) - train_count - val_count) < 1:
            raise ValueError(f"类别 {label} 的有效图片不足以按指定比例分为三个非空集合")
        for i, row in enumerate(items):
            split = "train" if i < train_count else "val" if i < train_count + val_count else "test"
            rows.append({**row, "split": split, "path": f"{split}/{row['source_path']}"})
    rows.sort(key=lambda row: row["sample_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # 在同一文件系统暂存，成功后才发布整个数据目录；异常不留下半份划分。
    with TemporaryDirectory(prefix="flower-split-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for row in tqdm(rows, desc="写入划分", unit="张"):
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row["source_path"], target)
        license_file = source / "LICENSE.txt"
        if license_file.is_file():
            shutil.copy2(license_file, staging / "LICENSE.txt")
        with (staging / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "source": str(source),
            "provenance": provenance or {},
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
            "images_found": len(files),
            "images_kept": len(rows),
            "excluded": excluded,
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "class_counts": {
                label: dict(Counter(r["split"] for r in rows if r["label"] == label)) for label in CLASSES
            },
            "manifest_sha256": sha256(staging / "samples.csv"),
        }
        write_json(staging / "classes.json", {label: i for i, label in enumerate(CLASSES)})
        write_json(staging / "split_report.json", report)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="已解压的 flower_photos 目录；指定后不下载或解压")
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/flower_photos.tgz", help="压缩包缓存位置"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/flower_photos_split", help="输出目录，不允许覆盖"
    )
    parser.add_argument("--offline", action="store_true", help="禁止下载，要求 source 或本地压缩包存在")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="测试比例为剩余部分")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"输出目录已存在：{args.output}；复用现有数据或指定新的 --output")
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        parser.error("train-ratio、val-ratio 和剩余 test 比例必须大于 0")
    options = dict(seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    if args.source:
        report = split_dataset(args.source, args.output, **options)
    else:
        if args.offline and not args.archive.is_file():
            parser.error(f"离线模式缺少压缩包：{args.archive}")
        download(args.archive)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="flower-extract-", dir=args.output.parent) as temporary:
            source = extract(args.archive, Path(temporary))
            report = split_dataset(
                source,
                args.output,
                **options,
                provenance={
                    "url": URL,
                    "archive": str(args.archive.resolve()),
                    "archive_sha256": sha256(args.archive),
                },
            )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split_counts": report["split_counts"],
                "excluded": len(report["excluded"]),
                "report": str(args.output / "split_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
