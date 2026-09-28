# 预训练模型下载与花卉五分类测试

首次训练使用预训练权重，先查找本地初始化缓存，不存在时自动下载。后续训练默认接着同一实验最新 last.ckpt 的模型权重微调，--fresh 可重新使用预训练初始化。所有下列配方将权重存到项目根目录 `weights/`，并使用配套的预处理及 5 类分类头。
训练主干从预训练参数开始，新分类头需要在花卉数据上学习。默认全参数微调，尚未完成这些模型的长时间训练或精度对比。

## 模型与来源

| 脚本 Model 参数 | provider / 模型名 | 权重版本 | 输入 / 验证短边 |
|---|---|---|---|
| resnet18 | torchvision / resnet18 | IMAGENET1K_V1 | 224 / 256 |
| resnet50 | torchvision / resnet50 | IMAGENET1K_V2 | 224 / 232 |
| mobilenetv3_small | torchvision / mobilenet_v3_small | IMAGENET1K_V1 | 224 / 256 |
| efficientnet_b0 | torchvision / efficientnet_b0 | IMAGENET1K_V1 | 224 / 256 |
| convnext_tiny | torchvision / convnext_tiny | IMAGENET1K_V1 | 224 / 236 |
| vit_tiny | timm / vit_tiny_patch16_224.augreg_in21k_ft_in1k | AugReg，21k 预训练后适配 1k | 224 / 248 |
| deit_tiny | timm / deit_tiny_patch16_224.fb_in1k | Facebook/Meta ImageNet-1k | 224 / 248 |
| swin_tiny | torchvision / swin_t | IMAGENET1K_V1 | 224 / 232 |

6 个 torchvision 模型使用 [PyTorch 官方 torchvision 权重](https://docs.pytorch.org/vision/stable/models.html)。ViT-Tiny 和 DeiT-Tiny 使用 [timm 发布方模型仓库](https://huggingface.co/docs/timm/en/quickstart)，不称为 torchvision 官方权重。
准确下载来源、模型标签、适配后权重 SHA256、运行目录见[逐模型验收记录](evidence/pretrained-models.json)。

## 常用命令

也支持 `cls.exe train model=...` 与 `Classifier(...).train()`；命令行/Python 对照和日志设置见[运行指南](RUNNING.md)。

在项目根目录执行 PowerShell。脚本使用项目 `.venv`，限制 BLAS 线程、启用 UTF-8，并固定 `num_workers=0`。

```powershell
# 直接训练：默认预训练；本地缺失会自动下载。各模型独立 run 目录。
.\scripts\run-flower.ps1 -Config configs/flower/flower_resnet50.yaml
.\scripts\run-flower.ps1 -Config configs/flower/flower_mobilenetv3_small.yaml
.\scripts\run-flower.ps1 -Config configs/flower/flower_vit_tiny.yaml

# 只下载并检查数据/模型初始化，不训练
.\scripts\run-flower.ps1 -Config configs/flower/flower_swin_tiny.yaml -Prepare

# 已下载后，离线跑两个训练 batch 和两个验证 batch
.\scripts\run-flower.ps1 -Config configs/flower/flower_convnext_tiny.yaml -Smoke -Offline

# 批量准备八个模型，每个模型使用独立进程
$models = 'resnet18', 'resnet50', 'mobilenetv3_small', 'efficientnet_b0', 'convnext_tiny', 'vit_tiny', 'deit_tiny', 'swin_tiny'
foreach ($model in $models) {
    .venv\Scripts\cls.exe prepare "model=configs/flower/flower_$model.yaml"
    if ($LASTEXITCODE -ne 0) { throw "Prepare failed: $model" }
}

# 缓存准备后，可把上面的 prepare 换成 smoke，并添加 offline=true 做短流程检查
```

配置在 `configs/flower/flower_<Model>.yaml`，统一继承 `flower_base.yaml` 的固定数据划分、类别表和预训练设置。
这些多模型配方默认 batch=4；批量验收时临时使用 batch=2。训练轮数仍为 10，只是可编辑起点。
`-Config` 支持绝对路径或相对于调用位置的路径，含空格时加引号。配置内部的文件路径相对于所选 YAML 所在目录。
仍支持 `-Model resnet50` 快捷方式，但不能与 `-Config` 同时使用；两者都不传时使用 `configs/flower/flower_photos.yaml`。

## 修改配置选择模型和训练参数

公共配置在 [configs/flower/flower_base.yaml](../configs/flower/flower_base.yaml)，日常切换模型无需修改。它集中保存当前花卉数据集、训练、预处理、权重目录等设置。
模型配置只需声明继承关系、实验名及网络，例如 `configs/flower/flower_resnet50.yaml`：

```yaml
# 继承公共数据、5 类分类头、预训练权重、预处理及 weights/ 目录设置
defaults: [flower_base, _self_]
experiment:
  name: flower_resnet50  # 输出到 runs/flower_resnet50/<run_id>
model:
  provider: torchvision  # 与 name 配套；ViT-Tiny / DeiT-Tiny 配方使用 timm
  name: resnet50
```

八个模型文件均采用上述结构；`flower_photos.yaml` 也继承 base，保留原 ResNet18 入口及 batch=16。
`flower_base.yaml` 的实验名、provider 和模型名留空，直接启动会报错，避免误用模板训练。
模型输出类别数设为 `auto`，准备数据后从类别表推断，当前得到 5 类。

若某个模型需要不同参数，只在其 YAML 末尾增加对应字段，例如：

```yaml
loader:
  batch_size_per_device: 8
optimizer:
  lr: 0.0001
trainer:
  max_epochs: 20
```

新增模型时，在 `configs/flower/` 中复制一个模型 YAML，修改 `experiment.name`、`model.provider` 和 `model.name`，然后用 `-Config` 指向新文件。
保持 `_self_` 在 defaults 最后，确保本文件参数覆盖公共设置。切换数据集时也可在子配置中覆盖 dataset 的路径、名称和类别数；不同任务还需配套调整损失及评估设置。
只有确实要改变所有模型的公共基线时才编辑 `flower_base.yaml`；修改 `flower_photos.yaml` 现在只影响该入口。
同名字段由子配置覆盖，未写字段继续继承。为了防止此前的 Windows 内存不足，脚本固定覆盖 `loader.num_workers=0`；其余训练参数从配置读取。
若需要完全按 YAML 中的 worker 数运行，可使用同一 CLI 实现（先设置进程级线程限制）：

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.venv\Scripts\python.exe -X utf8 -m clsframework.cli train -c configs/flower/flower_resnet50.yaml
```

## 权重存储和默认行为

```text
weights/
  torch/checkpoints/*.pth       # torchvision 官方原始权重
  models--timm--*/              # timm Hugging Face 快照与引用
  blobs/                       # Hugging Face 共享权重内容
  init-<key>.safetensors        # 保留预训练主干、适配 5 类的新初始化状态
  init-<key>.json               # 来源、预处理和校验和
  finetune-<sha256>.safetensors # 从上次 checkpoint 提取的完整模型参数，包含已训练分类头
  manifest.json                # 八个模型的文件、大小、SHA256 清单
```

本次八个模型的原始及五分类初始化权重合计约 841.3 MiB（按实际物理文件去重），均在 `weights/`；目录已被 Git 忽略。
Hugging Face 快照使用符号链接，部分 Windows 文件列表可能显示链接本身长度为 0；已校验其目标文件非空及 SHA256。
此前 `cache/` 的历史缓存保留；当前配方只从 `weights/` 读写模型权重，`cache/` 保留数据准备记录和测试日志。

默认值为 `model.weights.source: provider_default`、`preprocessing.source: model_weights`、`image_size: auto`、`runtime.offline: false`。
权重下载失败、损坏或 provider 不支持时直接报错，不回退到随机主干。设置 `offline: true` 后缺缓存即报错，避免违背离线要求。
只有显式指定 `weights.source: none` 和 `preprocessing.source: explicit` 才从随机参数开始。测试中的 TinyCNN 使用该显式设置。
改变架构、类别数或 seed 会产生新的初始化缓存；原始官方权重若已缓存，通常无需重复下载。

## 验收范围

2026-09-28：8/8 模型在线准备与短流程检查通过，随后 8/8 完整离线复测通过。
自动化测试 33 项通过（含严格续训零容差检查），13 份 YAML 配置验证通过，Schema 同步、Ruff 和构建通过。

每个模型均检查：

1. 加载真实发布权重和新建 5 类分类头，实际输出为 `[2, 5]` 且数值有限。
2. 每类取一张真实验证图，框架预处理与 provider 自带验证预处理逐像素零容差对比。
3. 禁止网络连接后重载初始化权重，参数摘要及 logits 零容差一致；检查 linear_probe 只开放分类头参数。
4. 使用真实 flower_photos 数据进行两步全参数微调、两批验证，确认模型状态更新。
5. 离线保存/加载 bundle 并预测真实图片。

这不是八个模型的完整训练认证，也不提供基于两个 batch 的精度排名。当前验收是 Windows CPU / FP32 / num_workers=0，GPU、AMP 和 ONNX 未包含在本轮测试。

本轮增加了独立权重目录、默认预训练配置、Swin/ViT 分类头适配，以及精确记录验证 resize_size，避免 timm 与 torchvision 的缩放取整差异。
核心源码摘要已改变，旧版本 checkpoint 不可在新版下直接严格续训；新实验直接复用预训练权重缓存。
