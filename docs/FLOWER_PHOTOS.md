# flower_photos 固定划分与模型测试

原始数据位于 `data/flower_photos`，是五类花卉图片，不是 Flowers102。
2026-09-28 已生成 `data/flower_photos_split`，采用按类别分层的 70%/15%/15% 划分，种子为 42。
每类验证、测试数量各为 `round(类别有效样本数 × 0.15)`，其余用于训练，因此比例有整数舍入差异。

| 类别 | 原始 | 排除 | train | val | test |
|---|---:|---:|---:|---:|---:|
| daisy（雏菊） | 633 | 0 | 443 | 95 | 95 |
| dandelion（蒲公英） | 898 | 0 | 628 | 135 | 135 |
| roses（玫瑰） | 641 | 1 | 448 | 96 | 96 |
| sunflowers（向日葵） | 699 | 2 | 487 | 105 | 105 |
| tulips（郁金香） | 799 | 1 | 558 | 120 | 120 |
| 合计 | 3670 | 4 | 2564 | 551 | 551 |

原始图片全部保留。扫描未发现坏图，但发现三组文件内容完全重复：

- `roses/15922772266_1167a06620.jpg` 与 `tulips/15922772266_1167a06620.jpg` 标签冲突，两份均排除，未擅自重标。
- `sunflowers/14889392928_9742aed45b_m.jpg` 与 `sunflowers/15066430311_fb57fa92b0_m.jpg` 相同，保留前者。
- `sunflowers/15069459615_7e0fd61914_n.jpg` 与 `sunflowers/15072973261_73e2912ef2_n.jpg` 相同，保留前者。

去重检查同时计算文件 SHA256 和经过 EXIF 方向修正的 RGB 像素摘要。
未进行感知近重复、同植株/场景或拍摄者分组审核；不能据此保证所有语义相似图片都隔离。这是本项目固定实验划分，不是官方 benchmark。

## 文件布局

```text
data/flower_photos_split/
  train/<类别>/*.jpg
  val/<类别>/*.jpg
  test/<类别>/*.jpg
  classes.json          # daisy=0, dandelion=1, roses=2, sunflowers=3, tulips=4
  samples.csv           # 固定划分、原始路径、标签、文件/像素摘要
  excluded.json         # 排除理由及重复图片保留对象
  split_report.json     # 数量、算法、种子、源数据和清单摘要
  LICENSE.txt           # 原数据许可和归属说明
  validation/           # 本次框架数据检查记录
```

图片为独立副本，保持原始分辨率与编码；增强、缩放在训练读取时完成。
目录兼容 torchvision ImageFolder。项目配置使用同一目录的 CSV manifest，以保留稳定 sample_id 和 group_id。
数据及验证产物均位于 Git 忽略的 `data/` 中。

## 运行模型

八种预训练模型、权重目录和切换命令见[多模型预训练使用说明](PRETRAINED_MODELS.md)。
例如 `.\scripts\run-flower.ps1 -Model resnet50` 会使用对应预训练权重，缺失时自动下载。

本机运行优先使用下列脚本：在 Python 导入前将 OpenBLAS/OMP/MKL 线程限制为 1，
并固定 `loader.num_workers=0`。脚本退出后恢复当前 PowerShell 的环境变量，不修改系统或 Conda。
PyTorch 的计算线程数仍由 `trainer.cpu_threads` 控制（默认 2）。
脚本同时使用 Python UTF-8 模式，避免 Windows GBK 输出环境在打印进度条字符时失败。

```powershell
.\scripts\run-flower.ps1 -Prepare  # 下载/缓存官方权重并检查数据，不训练
.\scripts\run-flower.ps1 -Smoke  # 两个训练和验证 batch，检查流程
.\scripts\run-flower.ps1         # 从官方预训练权重开始花卉五分类微调
```

本机曾将 `num_workers` 改为 8，训练进入验证时出现 OpenBLAS 内存分配失败。
现场仍有 15 个加载子进程，每个私有提交内存约 1.8 GiB，系统剩余提交空间约 0.6 GiB。
两套加载器最多共创建 16 个 worker。
训练/验证加载器各自创建 worker，默认持久化，可能同时驻留；因此已恢复为 0。
Lightning 的 worker 数提示是按 CPU 数量估算，没有测量本机内存，不能直接采用。
修改配置不会影响已启动的进程；报错的旧进程须先退出才能释放资源。
更改 worker 数后必须开始新实验，不能用新配置严格恢复旧的多 worker checkpoint。
本次调整仅涉及配置、启动脚本和文档，未修改核心包源码。
线程变量含义见 [OpenBLAS 官方说明](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)。

本次修复验证（2026-09-28）：`runs/flower_photos_resnet18/20260928-125845-f3874d96`
完成 161 个训练 batch、551 张验证图片及 epoch-000/last checkpoint 保存，未再出现 OpenBLAS 错误。
该次运行最终因 GBK 无法输出进度条字符而标为 FAILED；随后启动脚本增加 `-X utf8`。
修改后的 `runs/flower_photos_resnet18-smoke/20260928-130227-11986383` 完成两批训练/验证、
bundle 保存并以 SUCCEEDED 退出，脚本环境变量恢复检查通过。尚未完成修复后的 10 epoch 长训练。

当前默认配置为 CPU、224×224、5 类、完整数据，使用 torchvision 官方 ResNet18 预训练权重。
`ResNet18_Weights.DEFAULT` 当前对应 `IMAGENET1K_V1`：先加载完整官方权重，再将
`fc: Linear(512, 1000)` 替换为 `Linear(512, 5)`，主干参数保留，新分类头随机初始化。
当前 `training_mode: full_finetune` 微调全部参数；改成 `linear_probe` 则只训练新分类头。
`preprocessing.source: model_weights`、`image_size: auto` 使用权重配套预处理。
首次允许下载权重，原始与初始化权重放在 Git 忽略的 `weights/`；数据准备记录仍放 `cache/`。
prepare 完成后可设 `runtime.offline: true`。
官方权重说明见 [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。
10 epoch 和当前优化器参数仅为可编辑的实验起点，未证明训练收敛。
训练和验证默认显示进度条；需要关闭时添加 `--set trainer.enable_progress_bar=false`。

```powershell
# 两个训练/验证 batch 的流程检查
.\scripts\run-flower.ps1 -Smoke

# ResNet18 完整数据训练
.\scripts\run-flower.ps1

# 换模型，沿用同一份数据划分
.venv\Scripts\python.exe -X utf8 -m clsframework.cli train -c configs/flower/flower_photos.yaml --set model.name=mobilenet_v3_small --set experiment.name=flower_photos_mobilenetv3
.venv\Scripts\python.exe -X utf8 -m clsframework.cli train -c configs/flower/flower_photos.yaml --set model.provider=timm --set model.name=vit_tiny_patch16_224 --set experiment.name=flower_photos_vit_tiny

# 训练结束后，把路径替换为该次训练返回的 run 目录
.venv\Scripts\python.exe -X utf8 -m clsframework.cli test -c configs/flower/flower_photos.yaml --bundle runs/flower_photos_resnet18_finetune/<run_id>/bundle --split test
```

torchvision 模型名称示例包括 `resnet50`、`efficientnet_b0`、`convnext_tiny`。
timm 模型名称示例包括 `deit_tiny_patch16_224`、`swin_tiny_patch4_window7_224`，切换时必须指定 provider。
上述模型未在本次任务中逐一训练；较大模型在当前 CPU 环境可能耗时较长。

当前已默认启用预训练微调；若只训练新分类头，可使用：

```powershell
.venv\Scripts\python.exe -X utf8 -m clsframework.cli train -c configs/flower/flower_photos.yaml --set experiment.name=flower_photos_resnet18_probe --set model.training_mode=linear_probe
```

官方权重已实际下载并校验：其 100 个主干参数/缓冲张量与加载后的模型零容差一致，
旧权重未提供的 BatchNorm 计数器按 PyTorch 兼容规则初始化为 0；分类头为 `Linear(512, 5)`。
离线读取准备好的缓存也已通过，详见[权重核验记录](evidence/flower-resnet18-pretrained.json)。
真实花卉数据短微调运行 `runs/flower_photos_resnet18_finetune-smoke/20260928-131133-9af2e319`
已成功完成两批训练、两批验证和 bundle 保存。这只证明流程可运行，未证明收敛或测试集性能。

该检查暴露并修复了缺类验证 batch 的 Macro-F1 一致性问题：TorchMetrics 校验现按完整类别表
计算逐类 F1 再取均值，与原有报告的规则一致。新增回归测试后，29 项测试通过，
包含相同环境的续训参数零容差检查；Ruff 与构建通过。
此次指标修复修改了核心源码，旧版本 checkpoint 不能在新版中直接严格续训；请从预训练缓存开始新实验。

模型对比时固定本次清单，统一训练预算、增强和预训练策略，记录输入尺寸与预处理。
使用 val 选模型、调参数；方案确定后再报告 test 的 Top-1、Macro-F1 和逐类指标。
5 类任务的 Top-5 恒为 100%，不适合用于区分模型效果。
可改变 `experiment.seed` 重复训练并汇总均值/标准差；这不会改变已保存的划分。
不要为每个模型重新划分数据，也不要依据 test 成绩反复调参。

## 划分记录与复核

当前固定划分已完成，训练直接复用 `data/flower_photos_split`。目录整理后不再提供独立划分脚本。
`samples.csv` 保存每个样本的 split、原始路径和摘要，`excluded.json` 保存排除记录，`split_report.json` 保存算法、种子与数量，可用于复核本次划分。

本次验证范围为：全量图像解码、复制校验、完整且互斥的样本分配、重新计算划分一致性、框架严格数据检查及三个集合的真实 batch 读取。
详细结果见 `data/flower_photos_split/validation/verification.json`。本次没有执行模型训练或性能评估。
初次划分未修改核心包；后续增加训练进度显示修改了核心源码与配置 schema。
因此修改前的 checkpoint 无法在新版下直接严格续训，需使用原源码恢复，或以新版开始新的实验。
