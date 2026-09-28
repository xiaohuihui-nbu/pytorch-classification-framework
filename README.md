# PyTorch 通用图像分类框架

使用 YAML 选择网络和训练参数，通过统一的 CLI 与 Python API 完成 **训练、验证、测试、推理、ONNX 导出和可视化报告**。

接口形式参考 [Ultralytics CLI](https://docs.ultralytics.com/usage/cli/) 与 [Python API](https://docs.ultralytics.com/usage/python/)。本项目使用 `Classifier` 对象和 `cls` 命令，专注图像分类，无需安装 ultralytics。

当前版本为 **0.1.0**。基于 PyTorch、torchvision、timm、Lightning、Hydra、Pydantic 和 TorchMetrics；当前锁定依赖为 Windows CPU 验收环境。

**目录**

- [1. 功能概览](#1-功能概览)
- [2. 快速开始](#2-快速开始)
  - [2.1 安装项目环境](#21-安装项目环境)
  - [2.2 准备数据并检查配置](#22-准备数据并检查配置)
  - [2.3 正式训练](#23-正式训练)
  - [2.4 评估与推理](#24-评估与推理)
- [3. 数据准备](#3-数据准备)
  - [3.1 下载地址](#31-下载地址)
  - [3.2 一条命令完成处理](#32-一条命令完成处理)
  - [3.3 本地数据与自定义划分](#33-本地数据与自定义划分)
  - [3.4 输出目录与清单](#34-输出目录与清单)
  - [3.5 检查数据并训练](#35-检查数据并训练)
- [4. 模型与配置](#4-模型与配置)
- [5. 命令行](#5-命令行)
- [6. Python API](#6-python-api)
- [7. 预训练与继续训练](#7-预训练与继续训练)
- [8. 可视化与输出目录](#8-可视化与输出目录)
- [9. 日志](#9-日志)
- [10. 后台批量训练](#10-后台批量训练)
- [11. 常见问题](#11-常见问题)
- [12. 项目结构与验收](#12-项目结构与验收)

## 1. 功能概览

| 模块 | 已实现能力 |
|---|---|
| 任务 | 单标签多分类、单 logit 二分类、多标签分类 |
| 模型 | torchvision、timm、内置 TinyCNN；8 个花卉模型配方 |
| 权重 | 默认加载发布方预训练权重，缺失时下载；本地缓存、离线运行、历史权重微调 |
| 数据 | ImageFolder、CSV/JSONL manifest、torchvision 数据集适配器；类别映射和数据泄漏检查 |
| 训练 | 全参数微调、冻结主干、数据增强、Mixup/CutMix、进度条和 checkpoint |
| 评估 | Top-1/Top-5、Macro-F1、逐类指标、混淆矩阵；多标签 mAP/Micro-F1 |
| 推理与导出 | 本地图片或目录批量推理、JSONL、safetensors bundle、ONNX 数值验证 |
| 可视化与记录 | 训练曲线、预测概率图片、离线 HTML、CSV、根目录 logs 日志 |

模型可以加载，不代表所有模型和数据组合均完成收敛或精度评估。当前验证范围见[项目结构与验收](#12-项目结构与验收)。

## 2. 快速开始

### 2.1 安装项目环境

需要已安装 Git 和 uv。在 PowerShell 中执行：

```powershell
git clone https://github.com/xiaohuihui-nbu/pytorch-classification-framework.git
cd pytorch-classification-framework

# 创建项目独立的 Python 3.12 环境，并安装项目依赖
uv sync
uv run cls --help
```

已有本地项目时，直接进入项目目录并执行 `uv sync`。无需激活或修改 Conda 环境。

- 日常使用 `uv run cls ...` 或 `uv run 脚本.py`；uv 自动选择项目环境及 Python。仓库的 `.python-version` 指定 Python 3.12。
- 仅 ONNX 导出命令使用 `uv run --extra export cls export ...`，按需安装导出依赖。
- 当前 `uv.lock` 指向 PyTorch 官方 CPU wheel 源。GPU 训练需另外配置匹配驱动的 CUDA 环境并验收，不能仅修改 `device=gpu`。

### 2.2 准备数据并检查配置

按下一节放置数据，再运行：

```powershell
# 列出可选模型，不训练
uv run examples/train.py --list-models

# 检查配置和环境，不下载或训练
uv run cls doctor model=configs/flower/flower_resnet18.yaml

# 检查数据并准备所选模型的预训练缓存，不训练
uv run cls prepare model=configs/flower/flower_resnet18.yaml
```

`prepare` 可以提前发现数据或权重问题，也可省略，由 `train` 自动准备。它不会一次下载所有网络的权重。

### 2.3 正式训练

```powershell
uv run cls train model=configs/flower/flower_resnet18.yaml epochs=10 batch=4
```

训练完成后终端返回 `run_dir`、`report` 和 `log`。记录 `run_dir`，后续验证与推理使用该目录。

### 2.4 评估与推理

以下命令中的 `RUN_DIR` 是占位符，请替换为实际的 `runs/flower_resnet18/<run_id>` 路径：

```powershell
uv run cls val model=RUN_DIR
uv run cls test model=RUN_DIR
uv run cls predict model=RUN_DIR source=data/flower_photos_split/test
```

打开终端返回的 `report.html` 即可查看图片和指标，无需启动 Web 服务。

## 3. 数据准备

### 3.1 下载地址

本项目花卉配方使用 **flower_photos 五分类数据集**，原始压缩包包含 3,670 张图片，类别为 daisy、dandelion、roses、sunflowers、tulips。[TensorFlow 官方教程](https://www.tensorflow.org/tutorials/images/classification)提供了该数据集及下载地址；它与 102 类的 Flowers102 是不同的数据集。

- [下载 flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- 下载包保存在 `data/flower_photos.tgz`；使用现有包时无需再次下载。
- 无需安装 TensorFlow。处理脚本使用项目环境中的 Pillow 和 tqdm。

数据、权重、日志和训练产物不包含在 Git 仓库中。训练入口使用已划分的数据，不会在每次训练时重新下载或划分。

### 3.2 一条命令完成处理

首次使用、尚未创建 `data/flower_photos_split/` 时，在项目根目录执行：

```powershell
uv run scripts/prepare_flowers.py
```

脚本 [prepare_flowers.py](scripts/prepare_flowers.py) 会依次执行：

1. 复用本地压缩包，缺失时从上述 HTTPS 地址下载，显示字节进度。
2. 解压至临时目录，拒绝越界路径、符号链接等不安全条目。
3. 逐张解码，按 EXIF 方向转换为 RGB 计算像素摘要；损坏图片记录后排除，同类别的相同像素图片只保留一张，跨类别重复则报错供人工检查。
4. 使用固定随机种子 `42`，在每个类别内部按约 **70% / 15% / 15%** 分为 train / val / test；整数取整使比例存在小幅差异。
5. 复制原始图片到划分目录，生成 `samples.csv`、`classes.json` 和 `split_report.json`，保留压缩包中的 `LICENSE.txt`。
6. 输出实际数量与报告路径。校验和复制阶段均显示进度条，全部成功后才发布最终数据目录。

处理只改变图片归属，不修改原始图片尺寸或编码。训练时由框架执行预训练权重配套的缩放、裁剪和归一化；随机增强只用于训练集。像素去重不等同于相似图片识别，其他数据集若包含同一主体或拍摄组，仍需提供合理的 group_id 并按组划分。

**已有划分目录时脚本会拒绝覆盖。** 当前本机已有数据可直接跳到“检查数据并训练”。需要重建时指定一个新的 `--output`，并在模型 YAML 中更新 `dataset.root`；ImageFolder 无需配置 manifest 或 classes_file。不要覆盖正在使用的实验数据。

### 3.3 本地数据与自定义划分

```powershell
# 已手动下载压缩包：仅使用本地资源，不联网
uv run scripts/prepare_flowers.py --archive data/flower_photos.tgz --offline

# 已解压：source 指向直接包含五个类别文件夹的目录
uv run scripts/prepare_flowers.py --source data/flower_photos --offline

# 另建一份 80% / 10% / 10% 划分，不影响已有目录
uv run scripts/prepare_flowers.py --source data/flower_photos --output data/flower_photos_split_80_10_10 --seed 42 --train-ratio 0.8 --val-ratio 0.1
```

这些是不同输入方式，选择一种即可。`--source` 跳过下载和解压；`--offline` 禁止脚本下载，uv 自身的依赖安装仍受 uv 参数控制。自动解压目录在处理完成后清理，不会另外留下 `data/flower_photos/`；若使用 `--source`，该原始目录完整保留。

同一输入、参数与依赖环境会产生相同清单。`split_report.json` 记录种子、比例、每类数量、被排除图片及原因、清单 SHA256；压缩包模式还记录下载地址与压缩包 SHA256，用于追溯本次输入，不代表额外获得了官方校验值。

### 3.4 输出目录与清单

```text
data/flower_photos_split/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── roses/
│   ├── sunflowers/
│   └── tulips/
├── val/                 # 同样的五个类别目录
├── test/                # 同样的五个类别目录
├── samples.csv          # 可选：处理脚本生成的追溯清单，ImageFolder 不读取
├── classes.json         # 可选：处理脚本记录的类别映射，默认从目录推断
├── split_report.json    # 实际数量、划分参数、去重/损坏记录及摘要
└── LICENSE.txt          # 原始包包含时保留
```

默认花卉配方使用 `dataset.provider: imagefolder`，只需图片目录即可训练；不依赖 CSV 或类别 JSON 文件。实际配置如下：

```yaml
dataset:
  provider: imagefolder
  root: ../../data/flower_photos_split
  manifest: null
  classes_file: null
  splits: {train: train, val: val, test: test}
  num_classes: 5
  integrity: strict
```

框架从 `train/` 下的类别目录名称按字母顺序推断标签，验证集和测试集使用同一映射，未知类别会报错。当前映射为：

```json
{"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}
```

本地此前准备的花卉划分为 train=2564、val=551、test=551；这是历史实验记录，不是本脚本承诺复现的逐文件划分。新处理的有效图片数和各集合数量以 `split_report.json` 为准。多个模型应共用同一份 train/val/test 目录，以便比较；重建划分会改变数据指纹，不能拿新数据严格恢复旧实验。

数据处理脚本仍生成清单、类别表和划分报告，用于去重审计与追溯，不作为 ImageFolder 的输入依赖。ImageFolder 校验图片内容的跨集合重复，不读取清单中的 group_id；需要按拍摄组检查或进行多标签训练时，改用 manifest，并在清单中提供 group_id 或 labels 数组。

从旧 manifest 配方切换后，样本 ID 和 group 信息可能使数据指纹改变。已有实验的独立评估与严格续训应继续使用该 run 的配置快照；新的 ImageFolder 配方可以按兼容的类别顺序加载旧权重继续微调。

### 3.5 检查数据并训练

```powershell
# 检查图片、目录类别映射及跨集合重复，同时准备预训练权重
uv run cls prepare model=configs/flower/flower_resnet18.yaml

# 开始正式训练；其他网络共用上述固定目录划分
uv run cls train model=configs/flower/flower_resnet18.yaml epochs=10 batch=4
```

验证集用于每轮评估和选择最佳权重，测试集通过独立的 `cls test model=RUN_DIR` 做最终评估。处理脚本已通过 6 项小型数据回归，覆盖去重、可复现划分、框架读取、离线压缩包、不安全路径拒绝及下载缓存/中断清理；下载分支使用模拟响应验证，本轮没有重新下载完整数据集，不把数据处理成功当作模型性能验收。

## 4. 模型与配置

公共配置 [flower_base.yaml](configs/flower/flower_base.yaml) 保存数据、优化器、预处理、权重目录、日志和可视化设置。具体模型 YAML 继承公共配置并覆盖差异，日常切换模型无需修改 base。**base 是模板，不能单独训练。**

| 示例脚本的模型简称 | provider | 网络名称 | 配置文件 |
|---|---|---|---|
| `resnet18` | torchvision | `resnet18` | [flower_resnet18.yaml](configs/flower/flower_resnet18.yaml) |
| `resnet50` | torchvision | `resnet50` | [flower_resnet50.yaml](configs/flower/flower_resnet50.yaml) |
| `mobilenetv3_small` | torchvision | `mobilenet_v3_small` | [flower_mobilenetv3_small.yaml](configs/flower/flower_mobilenetv3_small.yaml) |
| `efficientnet_b0` | torchvision | `efficientnet_b0` | [flower_efficientnet_b0.yaml](configs/flower/flower_efficientnet_b0.yaml) |
| `convnext_tiny` | torchvision | `convnext_tiny` | [flower_convnext_tiny.yaml](configs/flower/flower_convnext_tiny.yaml) |
| `vit_tiny` | timm | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | [flower_vit_tiny.yaml](configs/flower/flower_vit_tiny.yaml) |
| `deit_tiny` | timm | `deit_tiny_patch16_224.fb_in1k` | [flower_deit_tiny.yaml](configs/flower/flower_deit_tiny.yaml) |
| `swin_tiny` | torchvision | `swin_t` | [flower_swin_tiny.yaml](configs/flower/flower_swin_tiny.yaml) |

6 个 torchvision 配方采用官方 torchvision 权重，ViT-Tiny 与 DeiT-Tiny 采用 timm 对应发布方权重。模型类别数从数据类别推断；预训练主干保留，新的分类头需要微调。每次运行的权重来源记录在该 run 的 provenance.json 中。

例如 ResNet50 的配置可以写为：

```yaml
defaults: [flower_base, _self_]
experiment:
  name: flower_resnet50
model:
  provider: torchvision
  name: resnet50
logging:
  directory: ../../logs/flower_resnet50
# 仅在需要时覆盖公共参数
trainer:
  max_epochs: 20
optimizer:
  lr: 0.0001
```

配置中的路径相对于最终传入的 YAML 所在目录解析，`configs/flower/` 中的 `../../weights` 指向项目根目录。复制模型 YAML 时修改实验名、provider、模型名和日志目录；保持 `_self_` 在 defaults 最后。需要完整字段定义时，执行 `uv run cls schema --output config.schema.json` 生成当前版本的 JSON Schema。

## 5. 命令行

支持 `cls <mode> key=value` 和传统 `--option` 参数。可选任务前缀 `classify` 与省略时行为相同：

```powershell
uv run cls classify train model=configs/flower/flower_resnet50.yaml epochs=20 lr0=0.0001
uv run cls train --config configs/flower/flower_resnet50.yaml --set trainer.max_epochs=20
```

PowerShell 自带的 `cls` 是清屏别名，请使用 `uv run ... cls`、`.venv\Scripts\cls.exe`，或 `python -m clsframework.cli`。含空格的参数整体加引号，例如 `"model=E:\my project\configs\flower\flower_resnet18.yaml"`。

| 命令 | 用途 |
|---|---|
| `doctor` | 检查配置与环境，不执行数据完整性检查或下载 |
| `prepare` | 检查数据并准备所选模型的初始化权重 |
| `train` | 正式训练，每轮在验证集上评估 |
| `val` / `test` | 使用保存的 bundle，独立评估验证集 / 测试集 |
| `predict` | 对本地图片或目录推理 |
| `report` | 从已有训练记录生成图片和 HTML，无需重新训练 |
| `export` | 导出 ONNX，并检查与 PyTorch 输出的数值一致性 |
| `list models` | 查看模型目录；花卉配方列表使用示例脚本的 `--list-models` |
| `smoke` | 独立的两批训练/验证短流程检查，不代表正式训练完成 |

训练常用参数如下；未传入的值沿用 YAML，覆盖只影响当前调用，不改写配置文件。

| CLI 参数 | Python 参数 / YAML 字段 | 花卉默认值或说明 |
|---|---|---|
| `epochs=10` | `epochs` / `trainer.max_epochs` | 10；必须大于 warmup_epochs |
| `batch=4` | `batch` / `loader.batch_size_per_device` | 4，每设备 batch |
| `lr0=0.0003` | `lr0` / `optimizer.lr` | 0.0003，不低于 scheduler.min_lr |
| `workers=0` | `workers` / `loader.num_workers` | 0 |
| `device=cpu` | `device` / `trainer.accelerator` | cpu；还可选 auto、gpu |
| `imgsz=224` | `imgsz` / `preprocessing.image_size` | 默认 auto，采用权重输入尺寸 |
| `name=my_run` | `name` / `experiment.name` | 决定实验目录及自动接续的搜索范围 |
| `project=...` | `project` / `experiment.output_root` | 默认项目根目录 runs |
| `offline=true` | `offline` / `runtime.offline` | 默认 false，允许下载缺失权重 |
| `plots=false` | `plots` / `visualization.enabled` | 默认 true，生成训练/评估报告 |

更多字段用 `optimizer.weight_decay=0.02` 或 `--set optimizer.weight_decay=0.02`。推理使用独立参数集：`batch`、`save`、`project`、`name`、`topk`、`max_images`、`output`。未知参数、重复或冲突覆盖会报错。

`examples/` 仅保留两个脚本：

```powershell
uv run examples/train.py --model resnet50 --epochs 20 --batch 4 --lr 0.0001
uv run examples/train.py --model resnet18 --show-config
uv run examples/train.py --help
uv run examples/predict.py --model RUN_DIR --source path/to/image.jpg
```

`examples/train.py` 的 `--model` 接受表中的简称，`cls train` 的 `model=` 接受 YAML 路径；两者参数形式不要混用。

## 6. Python API

CLI 与 Python API 共用实现。将以下代码保存为 Python 脚本，在项目环境中运行：

```python
from clsframework import Classifier


def main():
    model = Classifier("configs/flower/flower_resnet18.yaml")
    run_dir = model.train(epochs=10, batch=4, lr0=0.0003)
    print("训练目录：", run_dir)
    print("训练报告：", model.last_report)

    validation = model.val()
    print("验证 Macro-F1：", validation["macro_f1"])
    test_metrics = model.test()  # 独立测试集，训练期间不用于选模型

    results = model.predict("data/flower_photos_split/test", save=True, max_images=32)
    print(results[0]["label"], results[0]["probabilities"])
    print("预测 JSONL：", model.last_output)
    print("预测报告：", model.last_report)
    results[0].save("prediction.jpg", top_k=3)  # 拒绝覆盖已有文件
    image = results[0].plot(top_k=3)  # 返回 PIL.Image，不自动弹出窗口

    model.export(format="onnx", output="exports/resnet18.onnx")


if __name__ == "__main__":
    main()  # Windows 多进程脚本保留入口保护
```

仅推理时无需重新训练，也无需原训练数据：

```python
from clsframework import Classifier

model = Classifier("RUN_DIR/bundle")
results = model("path/to/image.jpg")  # 等价于 model.predict(...)
```

训练返回 run 目录 `Path`；评估返回指标字典；推理返回兼容 dict/JSON 的结果对象列表。多标签结果使用 `labels` 数组，二分类/多标签按 bundle 中的阈值做决策。

Python 的 `predict()` 默认不保存，`save=True` 才保存图片和 HTML；CLI 与 `examples/predict.py` 默认保存。复制出的独立 bundle 若需评估，应显式提供 `model.val(config="configs/flower/flower_resnet18.yaml")`，且数据指纹须与训练时一致。

## 7. 预训练与继续训练

预训练原始权重和适配后初始化缓存统一放在根目录 `weights/`。首次缺失时自动下载，失败直接报错；不会静默改用随机初始化。`offline=true` 只使用已准备的本地资源，缺缓存时报错。模型离线开关不控制 uv 的依赖安装联网行为。

花卉公共配置默认 `checkpoint.finetune_from: last`，因此日常再次训练会自动继承同一实验最近一次 `last.ckpt` 的参数。

| 方式 | 继承内容 | epoch 与优化器 | 适用场景 |
|---|---|---|---|
| 默认训练 / `finetune_from=last` | 上次模型参数，包含已训练分类头；无历史时按配置初始化 | 新建 run，计数重置，可改轮数和学习率 | 接着上次权重继续微调 |
| `fresh=true` | YAML 指定的初始权重，花卉配方为发布方预训练权重 | 新建 run，重新开始 | 独立实验，不接续历史结果 |
| `resume=.../last.ckpt` | 模型、优化器、调度器、计数及随机状态 | 延续原 run 与总训练预算 | 同环境下严格断点恢复 |

```powershell
# 同一实验已有历史时，本次再微调 5 轮
uv run cls train model=configs/flower/flower_resnet18.yaml epochs=5 lr0=0.0001

# 按配方初始权重开始一个独立实验
uv run cls train model=configs/flower/flower_resnet18.yaml fresh=true

# 明确选择某次运行的权重继续微调
uv run cls train model=configs/flower/flower_resnet18.yaml finetune_from=RUN_DIR epochs=5

# 严格恢复：使用该运行的配置快照及 checkpoint
uv run cls train model=RUN_DIR/config.resolved.yaml resume=RUN_DIR/checkpoints/last.ckpt
```

严格恢复要求训练配置、数据、依赖和核心源码契约一致；改变训练预算或学习率应使用微调方式。跨核心源码版本的旧 checkpoint 可能无法严格恢复，参数兼容时仍可用于微调，标准 bundle 仍可推理。这三种显式模式互斥。

比较多个模型时使用相同数据划分和训练预算，并通过 `fresh=true` 或独立实验名控制初始化条件，避免无意接续历史结果。

## 8. 可视化与输出目录

所有报告均可离线打开。训练和评估默认生成报告，Python 对象的 `last_report` 返回最近操作的报告路径。

```text
runs/flower_resnet18/<run_id>/
├── config.resolved.yaml       # 实际运行配置
├── checkpoints/
│   ├── last.ckpt              # 最新完整训练状态
│   └── epoch-*.ckpt           # 选出的最佳 checkpoint
├── bundle/                   # 最佳权重的推理交付包
│   ├── model.safetensors
│   ├── model_spec.json
│   ├── classes.json
│   ├── preprocess.json
│   ├── thresholds.json
│   └── bundle_manifest.json
├── visuals/
│   ├── report.html
│   ├── results.csv
│   ├── results.png
│   ├── per_class.png
│   ├── confusion_matrix.png
│   └── confusion_matrix_normalized.png
├── evaluation/val/           # 独立 val 命令执行后生成；test 类似
├── metrics.jsonl             # 各轮验证指标
├── events.jsonl              # epoch、学习率、训练 loss 等
└── status.json               # 运行状态与最佳 checkpoint
```

训练曲线展示已记录的 train loss、验证 Top-1 或 mAP、Macro-F1 和学习率。训练报告的逐类指标与混淆矩阵来自**最后记录轮**的验证集；`val`/`test` 才是保存的最佳 bundle 的独立评估。历史缺失的 loss 留空，当前不计算验证 loss。二分类/多标签不生成不适用的多分类混淆矩阵。

预测默认输出 `runs/predict/<时间-ID>/predictions.jsonl`，报告位于同目录下的 `predictions_visuals/report.html`，图片放在 `predictions_visuals/images/`。

```powershell
# 自定义预测输出目录、Top-K 与图片保存上限
uv run cls predict model=RUN_DIR source=path/to/images project=runs/predict name=flower_demo topk=3 max_images=32

# 仅保存完整 JSONL，不生成图片或 HTML
uv run cls predict model=RUN_DIR source=path/to/images save=false output=predictions.jsonl

# 重建历史训练报告，不加载模型权重
uv run cls report model=RUN_DIR

# 导出 ONNX，自动校验 batch=1 和 3 的数值一致性
uv run --extra export cls export model=RUN_DIR format=onnx output=exports/resnet18.onnx
```

预测默认最多保存 64 张图，JSONL 始终保留全部结果。显式指定 `output=path/result.jsonl` 并开启保存时，图片和 HTML 写入 `path/result_visuals/`，不能同时传 project/name。已有预测输出、评估目录和导出文件拒绝覆盖；再次执行请指定新路径。`report` 默认允许重建该 run 的 visuals，显式 `output` 则要求新目录。

训练/评估通过 `plots=false` 关闭图片和 HTML；预测通过 `save=false` 控制。两者不影响模型决策。

## 9. 日志

花卉模型的日志保存在根目录 `logs/flower_<模型>/`，每次操作生成包含时间、操作名和随机 ID 的 UTF-8 文件，记录加载权重、轮次指标、耗时、输出路径和异常堆栈。`model.last_log` 返回最近操作的日志路径。

```powershell
uv run cls train model=configs/flower/flower_resnet18.yaml logging.level=DEBUG logging.console=false
```

训练产物和数值指标保存在 `runs/`，操作日志保存在 `logs/`。普通文件日志不复制终端动态进度条；后台批量脚本另行完整捕获各模型 stdout/stderr。

## 10. 后台批量训练

`scripts/train.sh` 默认后台**同时运行 8 个模型**，每个模型正式训练 10 轮并逐轮验证。先同步一次环境，再并发执行，不会并发修改 `.venv`。

在 Git Bash 中执行：

```bash
bash scripts/train.sh          # 启动，终端随即返回
bash scripts/train.sh status   # 查看批次及逐模型状态
bash scripts/train.sh stop     # 停止本脚本启动的活动批次及训练子进程
EPOCHS=20 bash scripts/train.sh  # 下一次启动使用 20 轮
```

Windows PowerShell 中应指定实际安装的 Git Bash；本机路径示例：

```powershell
& "D:\software\Git\bin\bash.exe" scripts/train.sh
& "D:\software\Git\bin\bash.exe" scripts/train.sh status
& "D:\software\Git\bin\bash.exe" scripts/train.sh stop
```

每批日志保存到 `logs/train_<时间>_<唯一编号>/`，包含 `batch.log`、`summary.csv`、`status.txt` 和 `train_resnet18.log` 等逐模型日志。启动返回只表示任务已提交，完成情况以状态及退出码为准。

可以在脚本末尾为每个模型单独修改参数，或注释不需要的模型。8 模型并发会叠加 CPU 和内存占用；后台脚本默认限制线程、使用 0 个 DataLoader worker 并关闭动态进度条。单个模型失败不终止其他任务，全部结束后根据退出码汇总批次状态；stop 立即请求终止本批次训练进程树。

## 11. 常见问题

| 问题 | 处理方式 |
|---|---|
| PowerShell 输入 `cls` 后清屏 | 使用 `uv run cls ...` 或项目 `.venv` 内的 `cls.exe` |
| `bash` 报 WSL `/bin/bash` 不存在 | 当前命令指向 WSL；改用 Git Bash 的实际路径 |
| Lightning 提示建议 `num_workers=27` | 这是性能建议，当前配置采用 0；结合内存和吞吐实测调整，不直接照搬提示 |
| OpenBLAS 内存分配失败 | 保持 workers=0，降低 batch 或并发模型数；后台脚本已限制 BLAS/OMP/MKL 线程 |
| 离线运行提示权重不存在 | 先在线对相同模型、类别数和 seed 执行 prepare；离线模式不会自动下载或随机回退 |
| 只训练 1 轮时报预热参数错误 | 同时设置 `epochs=1 scheduler.warmup_epochs=0`；示例脚本用 `--epochs 1 --warmup-epochs 0` |
| 直接替换官方模型分类数能否免训练 | 新分类头需要在目标数据上微调；已有训练 bundle 可直接推理 |
| 再次运行评估或预测报目录已存在 | 指定新的 output；自动生成名称的预测不复用已有目录 |
| 五分类的 Top-5 一直为 100% | 类别数就是 5，比较模型时关注 Top-1、Macro-F1 和逐类指标 |
| 旧 checkpoint 严格恢复被拒绝 | 使用匹配的原源码、依赖、配置和数据；若要改变参数则选择仅加载权重微调 |

## 12. 项目结构与验收

```text
configs/flower/      公共配置和各模型配置
src/clsframework/   CLI、Python API、训练、数据、模型、推理和可视化
examples/           train.py、predict.py
scripts/            prepare_flowers.py 数据处理、train.sh 后台批量入口
docs/               Git 常用命令文档与项目规划
data/               本地数据（不入 Git）
weights/            权重与初始化缓存（不入 Git）
cache/              数据准备等缓存（不入 Git）
runs/               训练、评估、预测产物（不入 Git）
logs/               操作和批量日志（不入 Git）
```

开发阶段曾完成 **57 项完整回归测试**，包含严格恢复的参数零容差对齐；ImageFolder 切换后另有 26 项相关回归通过。Ruff 检查和 wheel/sdist 构建通过。这些是已有本地验收结果，历史验收附件已移除，不表示每次修改文档都会重新训练。

在包含 `tests/` 的完整开发工作区执行：

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests examples
.venv\Scripts\ruff.exe check scripts/prepare_flowers.py
uv build
```

当前 `.gitignore` 排除了 `tests/`，仅克隆仓库的副本可能不包含本地验收测试。GPU/AMP/Linux DDP、全模型长训练与精度排名尚未完成全面验收；合成数据测试和短流程检查不代表真实花卉性能。推理目前支持 CPU 上的本地图片/目录，导出仅支持 ONNX；不支持视频、摄像头、URL 输入或任意 Ultralytics 参数。

进一步阅读：[GitHub 上传与拉取常用命令](docs/GIT_GUIDE.md)。[项目规划](docs/PLAN.md)记录设计目标，具体已实现功能以本 README 和当前代码为准。
