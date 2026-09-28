# PyTorch 通用图像分类框架

独立的 Python 包和 CLI：用 YAML 选择模型、数据集、损失、增强、优化器与训练参数，完成训练、恢复、独立评估、预测和 ONNX 导出。

当前版本：**0.1.0，可运行并已完成 CPU 验收**。基于 PyTorch、timm、torchvision、Lightning、Hydra、Pydantic 与 TorchMetrics。项目不依赖原有 Web MVP，也不需要数据库或在线追踪服务。

批量后台训练 8 个模型及停止命令见[运行指南](docs/RUNNING.md#bash-批量训练)，入口为 `scripts/train.sh`。

## 1. 立即运行

本机已经建立 `.venv`、安装锁定依赖，并生成了离线演示数据。PowerShell：

```powershell
cd pytorch-classification-framework
.\scripts\run-demo.ps1 -SkipSync
```

脚本完成训练、测试和批量预测，最后输出本次 `runs/demo_colors/<run_id>` 路径。每次新建 run，不覆盖旧结果。演示数据是有独立 train/val/test 的合成颜色图片，只用于验证功能，不代表真实业务准确率。

在一台新机器上首次运行：

```powershell
uv sync --frozen --managed-python --extra export
.\scripts\run-demo.ps1 -SkipSync
```

本机最初使用 Conda Python 3.11 创建的环境无法加载 PyTorch DLL，已改为 uv 管理的独立 Python 3.12.14 并验证。无需激活或修改现有 Conda 环境。

**当前 `uv.lock` 使用官方 CPU wheel 源。** 它适用于本次 Windows CPU 交付；GPU 机器需要单独建立匹配驱动的 CUDA torch/torchvision 环境并验收，不能只改 `accelerator: gpu` 就让 CPU wheel 获得 CUDA 功能。GPU/AMP/Linux DDP 分支已接入，但本机只有 Intel UHD 770，无 CUDA，未宣称通过 GPU 验收。

## 2. 已有能力

| 模块 | 当前实现 |
|---|---|
| 任务 | 单标签多分类、两类多分类、单 logit 二分类、多标签分类 |
| 模型 | timm 通用工厂；torchvision 工厂和已明确的分类头适配；内置 TinyCNN；显式 Python 插件 |
| 模型目录 | ResNet18/50、MobileNetV3-Small、EfficientNet-B0、ConvNeXt-Tiny、ViT-Tiny、DeiT-Tiny、Swin-Tiny |
| 公开数据 | MNIST、FashionMNIST、CIFAR-10/100、Oxford-IIIT Pet、Flowers102 的 torchvision 适配器 |
| 自定义数据 | ImageFolder、单标签 CSV、单标签/多标签 JSONL；固定类别映射和 group 检查 |
| 损失 | CE、类别加权/标签平滑、Soft-target CE、BCEWithLogits、softmax focal、sigmoid focal |
| 增强 | 基础/强增强、Mixup、CutMix；验证和预测使用相同的确定性变换 |
| 优化 | AdamW、Adam、SGD；warmup + cosine；梯度累积、裁剪、冻结主干 |
| 配置 | Hydra defaults 组合、`--set key=value` 覆盖、严格 Pydantic 校验、可导出 JSON Schema |
| 训练 | CPU、已接入的 GPU/AMP/DDP 分支；best 模型；每个 epoch 完整 last checkpoint |
| 评估 | Top-1/Top-5、Macro-F1、逐类指标、混淆矩阵；多标签 mAP/Micro-F1/Macro-F1 |
| 交付 | safetensors bundle、校验和、批量 JSONL 预测、ONNX + 数值对齐报告 |
| 记录 | 配置快照、依赖版本、权重身份、数据指纹、状态、事件和指标 JSONL；可选 TensorBoard |

模型目录可加载不等于所有模型/数据/损失组合已训练到收敛。具体实测范围见 [验收记录](docs/VALIDATION.md)。MLflow、HPO、Web 平台接入、蒸馏与新模态仍是后续模块。

## 3. CLI 用法

```powershell
# 全部命令
.venv\Scripts\cls.exe --help

# 查看目录
.venv\Scripts\cls.exe list models
.venv\Scripts\cls.exe list datasets
.venv\Scripts\cls.exe list losses

# 创建一份完整配置（不会覆盖已有文件）
.venv\Scripts\cls.exe init --recipe custom_resnet18 --output experiment.yaml

# 只读检查，不下载数据或权重
.venv\Scripts\cls.exe doctor -c experiment.yaml

# 准备数据、预训练缓存和指纹
.venv\Scripts\cls.exe prepare -c experiment.yaml

# 训练前可用真实数据跑两个 batch 检查
.venv\Scripts\cls.exe smoke -c experiment.yaml

# 完整训练、参数覆盖
.venv\Scripts\cls.exe train -c experiment.yaml --set trainer.max_epochs=10 --set optimizer.lr=0.0003

# 恢复原实验；这里只接受本框架生成且受信任的 checkpoint
.venv\Scripts\cls.exe train -c experiment.yaml --resume runs/my_run/checkpoints/last.ckpt

# 独立评估：将下面路径替换为训练命令返回的实际路径
.venv\Scripts\cls.exe test -c experiment.yaml --bundle runs/my_run/bundle --split test
.venv\Scripts\cls.exe predict --bundle runs/my_run/bundle --input images --output predictions.jsonl

# ONNX；需要安装 export extra，默认固定空间分辨率、动态 batch
.venv\Scripts\cls.exe export --bundle runs/my_run/bundle --output exports/model.onnx
```

`cls` 已注册为虚拟环境命令；也可使用 `.venv\Scripts\python.exe -m clsframework.cli`。如果使用 `uv run` 并需要 ONNX，请带 `--extra export`，避免默认同步移除可选依赖。

`init` 支持 `custom_resnet18`、`cifar10_resnet18`、`mnist_resnet18`、`multilabel`。生成配置默认使用预训练模型，首次训练需要联网准备。示例 `configs/demo.yaml` 使用无外部权重的 TinyCNN，适合完全离线。

`train --stop-after-epoch 1` 可在第一个 epoch 边界主动停止并保留训练状态，便于验证恢复。保持原始 `max_epochs`、数据、batch 与调度器不变，然后用 `--resume` 继续。

## 4. 配置示例

仓库内有四份示例：

| 文件 | 目的 |
|---|---|
| [demo.yaml](configs/demo.yaml) | 完全离线的单标签功能演示 |
| [mnist_resnet18.yaml](configs/mnist_resnet18.yaml) | 真实公开数据下载；各 split 限制 100 张、2 epoch 的 CPU 验证 |
| [cifar10_finetune.yaml](configs/cifar10_finetune.yaml) | 完整 CIFAR-10 的 ResNet18 预训练微调起点 |
| [multilabel_demo.yaml](configs/multilabel_demo.yaml) | Hydra 继承基础配置，多标签 BCE 训练 |

多标签演示首次准备：

```powershell
.venv\Scripts\python.exe examples/make_demo_data.py --multilabel --output data/multilabel_demo
.venv\Scripts\cls.exe train -c configs/multilabel_demo.yaml
```

生成器拒绝覆盖已有目录；本机已经生成，无需重复执行。公开数据配方中的 `limit_per_split` 仅用于冒烟/工程验证，正式训练应设为 `null`，并重新产生数据指纹与 run。

切换模型示例：

```powershell
.venv\Scripts\cls.exe smoke -c configs/mnist_resnet18.yaml --set model.name=mobilenetv3_small_100
.venv\Scripts\cls.exe smoke -c configs/mnist_resnet18.yaml --set model.name=vit_tiny_patch16_224 --set preprocessing.image_size=224
```

精确支持字段由 [JSON Schema](docs/config.schema.json) 定义。设计稿中的高级字段不一定已实现；未知字段会报错，不会静默忽略。配置值的含义和限制见 [配置与数据说明](docs/USAGE.md)。

## 5. 自定义数据

单标签目录结构：

```text
data/my_dataset/
  train/cat/001.jpg
  train/dog/002.jpg
  val/cat/101.jpg
  val/dog/102.jpg
  test/cat/201.jpg        # test 可选
  test/dog/202.jpg
```

默认按训练集类别名称排序生成固定映射；也可指定 `dataset.classes_file` 为 JSON 类别列表或 `{"cat":0,"dog":1}`。验证/测试出现未知类会报错。

CSV 列：`sample_id,path,label,split,group_id`。其中 `group_id` 可省略。多标签 JSONL 使用 `labels` 数组：

```json
{"sample_id":"001","path":"images/001.jpg","labels":["cat","indoor"],"split":"train","group_id":"scene1"}
```

数据必须有明确的 train/val；test 可省略，但不能生成虚构的测试成绩。自定义数据当前不自动随机切分，以便用户明确 group/时间边界。相同内容跨 split、相同 group 跨 split、重复样本 ID、坏图像及未知标签默认报错。

## 6. 产物与恢复

每个 run 包含 `config.resolved.yaml`、`classes.json`、数据 manifest/指纹、`environment.json`、`provenance.json`、`metrics.jsonl`、`events.jsonl`、`status.json`、checkpoint 和推理 bundle。

`bundle/` 独立包含模型结构、safetensors 权重、类别顺序、预处理、阈值和文件校验和。预测无需训练进程，也不再下载预训练权重。测试输出写入 `evaluation/test/`，含指标 JSON、逐类 CSV、逐样本预测与混淆矩阵 PNG。

恢复采用严格契约：比较数据指纹、标签、模型、预处理、优化器、学习率日程、关键依赖和核心源码摘要。修改任一核心 `.py` 文件或关键训练参数后会拒绝旧实验的严格续训；推理 bundle 仍可按其结构与环境要求加载。这是为了避免把一次新的实验误认为原实验延续。

本版保证的实测范围是 **Windows CPU、相同环境、num_workers=0、epoch 边界恢复**。连续训练与中断恢复的模型参数已通过零容差对比。多 worker、GPU、DDP 的精确恢复仍需目标环境验收，任意 batch 恢复尚未实现。

## 7. 扩展与开发

扩展示例在 [custom_plugin.py](examples/custom_plugin.py)。配置 `runtime.plugins: [examples.custom_plugin]` 显式导入已安装/本地 Python 模块，再使用它注册的 model/dataset/loss provider。新模型输出必须是 `[N,C]` logits 或 `{"logits": tensor}`，不应在模型末尾对 CE/BCE 额外做 softmax/sigmoid。

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests examples
.venv\Scripts\ruff.exe format --check src tests examples
.venv\Scripts\python.exe examples/validate_models.py
uv build
```

测试覆盖配置冲突、数据泄漏、类别映射、指标、loss 梯度、公开数据 adapter 切分语义、离线权重缓存、单标签/多标签/二分类训练、模型包校验及严格恢复。部分公开数据 adapter 测试使用小型可控替身；真实下载与训练范围另列于验收记录。

## 8. 当前边界

- 当前环境和 lockfile 是 CPU 版本；CUDA/AMP/DDP 尚未实机验收。
- 真实公开数据端到端验收为 MNIST；另外五个 provider 已接入并有切分契约测试，未全部下载长训。
- 8 个 timm 模型通过 CPU 前向/反向测试；仅部分模型完成实际数据训练/导出，不能据此宣称全部组合认证。
- ONNX 数值验收覆盖 TinyCNN 和 ResNet18；其他模型导出按实际结果判断。
- 验证指标当前在 CPU 累积整轮预测以正确聚合、去除 DDP 补齐样本；超大验证集需要后续实现分片/流式方案。
- checkpoint 为受信任的本地训练状态；分发推理使用带校验的 safetensors bundle。
- 暂未实现 Web/MLflow 接入、HPO、模型蒸馏、外部测试集重映射、阈值自动寻优和多模态训练。

原始调研方案见 [PLAN.md](docs/PLAN.md)；本次实际交付以本 README、配置 Schema 和 [VALIDATION.md](docs/VALIDATION.md) 为准。
