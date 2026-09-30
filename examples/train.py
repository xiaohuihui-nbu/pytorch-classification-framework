"""通过模型 YAML 训练：uv run examples/train.py --config configs/flower/flower_resnet18.yaml。"""

import argparse
import sys
from pathlib import Path

from clsframework import Classifier
from clsframework.api import normalize_overrides
from clsframework.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "flower"


def model_configs():
    """以项目中的具体模型 YAML 为准，避免与实际配置维护两份模型清单。"""
    return {
        path.stem.removeprefix("flower_"): path
        for path in sorted(CONFIG_DIR.glob("flower_*.yaml"))
        if path.stem not in {"flower_photos", "flower_base"}
    }


def list_models(configs):
    print("可选花卉模型（对应项目现有配置；模型支持范围并非仅限此列表）：")
    print("模型简称 | provider | 网络名称 | 配置文件")
    for name, path in configs.items():
        cfg = load_config(path)
        print(f"{name} | {cfg.model.provider} | {cfg.model.name} | configs/flower/{path.name}")
    print("使用 --model <简称> 选择模型；自定义模型使用 --config <YAML>。")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    configs = model_configs()
    parser = argparse.ArgumentParser(
        description="正式训练；花卉配置默认自动恢复最近未完成断点；未指定的参数沿用 YAML 配置",
        epilog="示例：--list-models；--model resnet50 --epochs 20 --batch 4；--model vit_tiny --show-config",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--config",
        "-c",
        type=Path,
        help="自定义模型 YAML；不指定 config/model 时使用 flower_resnet18.yaml",
    )
    selection.add_argument("--model", "-m", choices=list(configs), help="选择现有模型配置，与 --config 互斥")
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument(
        "--list-models", action="store_true", help="列出模型、provider 和配置路径，不训练"
    )
    inspection.add_argument(
        "--show-config", action="store_true", help="显示继承和参数覆盖后的最终配置，不训练或下载"
    )
    parser.add_argument("--epochs", "--epoch", type=int, help="训练轮数")
    parser.add_argument("--batch", "--batch-size", type=int, help="每个设备的批量大小")
    parser.add_argument("--lr", "--lr0", dest="lr0", type=float, help="初始学习率")
    parser.add_argument("--workers", type=int, help="数据加载进程数，当前机器建议 0")
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], help="计算设备")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--name", help="实验名称，结果保存到 runs/<name>/")
    parser.add_argument("--warmup-epochs", type=int, help="预热轮数，必须小于 epochs")
    parser.add_argument("--weight-decay", type=float, help="优化器权重衰减")
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], help="优化器")
    parser.add_argument("--scheduler", choices=["warmup_cosine", "none"], help="学习率调度器")
    parser.add_argument(
        "--augmentation", choices=["none", "baseline", "finetune", "strong"], help="训练增强方案"
    )
    parser.add_argument(
        "--training-mode",
        choices=["full_finetune", "linear_probe"],
        help="full_finetune 微调全部参数；linear_probe 只训练分类头",
    )
    parser.add_argument("--imgsz", type=int, help="输入图像大小；不指定时按配置自动匹配权重")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="保存训练曲线、混淆矩阵和 HTML 报告；默认沿用 YAML",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="只使用本地权重；--no-offline 允许下载；未指定则沿用 YAML",
    )
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, help="从受信任的本框架 checkpoint 严格续训")
    continuation.add_argument("--finetune-from", help="仅继承权重：last 自动选上次，或指定 run 目录/.pt")
    continuation.add_argument("--fresh", action="store_true", help="关闭自动接续，按 YAML 初始权重开始新实验")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖其他 YAML 字段，可重复使用；重复或冲突字段会报错",
    )
    args = parser.parse_args(argv)
    if args.list_models:
        list_models(configs)
        return
    config = args.config or configs[args.model or "resnet18"]

    # 只传入用户明确指定的值，避免覆盖 YAML 中的配置。
    options = {
        key: getattr(args, key)
        for key in (
            "epochs",
            "batch",
            "lr0",
            "workers",
            "device",
            "seed",
            "name",
            "offline",
            "imgsz",
            "plots",
        )
        if getattr(args, key) is not None
    }
    overrides = list(args.overrides)
    if args.warmup_epochs is not None:
        overrides.append(f"scheduler.warmup_epochs={args.warmup_epochs}")
    if args.weight_decay is not None:
        overrides.append(f"optimizer.weight_decay={args.weight_decay}")
    for argument, key in {
        "optimizer": "optimizer.name",
        "scheduler": "scheduler.name",
        "augmentation": "augmentation.preset",
        "training_mode": "model.training_mode",
        "log_level": "logging.level",
    }.items():
        if (value := getattr(args, argument)) is not None:
            overrides.append(f"{key}={value}")

    if args.show_config:
        import yaml

        cfg = load_config(config, normalize_overrides(overrides, options))
        if args.resume is not None or args.finetune_from is not None or args.fresh:
            cfg.checkpoint.auto_resume = False
        if args.resume is not None:
            cfg.checkpoint.resume_from = args.resume.resolve()
            cfg.checkpoint.finetune_from = None
        if args.finetune_from is not None:
            cfg.checkpoint.finetune_from = (
                "last" if args.finetune_from == "last" else Path(args.finetune_from).resolve()
            )
        if args.fresh:
            cfg.checkpoint.resume_from = None
            cfg.checkpoint.finetune_from = None
        print(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False))
        return

    model = Classifier(config)
    directory = model.train(
        resume=args.resume, finetune_from=args.finetune_from, fresh=args.fresh, overrides=overrides, **options
    )
    print(f"训练结果：{directory}")
    print(f"最佳模型：{model.checkpoint}")
    print(f"日志文件：{model.last_log}")
    print(f"可视化报告：{model.last_report}")


if __name__ == "__main__":
    main()  # Windows 多进程入口必须加此保护。
