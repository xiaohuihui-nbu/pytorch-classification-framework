"""加载训练结果，对一张图片或图片目录进行分类推理。"""

import argparse
import json
from pathlib import Path

from clsframework import Classifier


def main():
    parser = argparse.ArgumentParser(description="通过 Classifier Python API 分类推理")
    parser.add_argument("--model", type=Path, required=True, help="训练 run 或 bundle 目录")
    parser.add_argument("--source", type=Path, required=True, help="本地图片或目录")
    parser.add_argument("--output", type=Path, help="可选 JSONL 保存路径，不覆盖已有文件")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    model = Classifier(args.model)
    results = model.predict(args.source, output=args.output, batch=args.batch)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
