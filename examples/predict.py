"""加载训练结果，对一张图片或图片目录进行分类推理。"""

import argparse
import json
import sys
from pathlib import Path

from clsframework import Classifier


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="通过 Classifier Python API 分类推理")
    parser.add_argument("--model", type=Path, required=True, help="训练 run 或 bundle 目录")
    parser.add_argument("--source", type=Path, required=True, help="本地图片或目录")
    parser.add_argument("--output", type=Path, help="可选 JSONL 保存路径，不覆盖已有文件")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True, help="保存预测图片和 HTML"
    )
    parser.add_argument("--project", type=Path, help="预测输出根目录，默认 runs/predict")
    parser.add_argument("--name", help="本次预测目录名，默认自动生成")
    parser.add_argument("--topk", type=int, help="图片中显示的最高概率类别数")
    parser.add_argument("--max-images", type=int, help="报告最多保存的图片数")
    args = parser.parse_args()
    model = Classifier(args.model)
    results = model.predict(
        args.source,
        output=args.output,
        batch=args.batch,
        save=args.save,
        project=args.project,
        name=args.name,
        top_k=args.topk,
        max_images=args.max_images,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"JSONL: {model.last_output}\nHTML: {model.last_report}")


if __name__ == "__main__":
    main()
