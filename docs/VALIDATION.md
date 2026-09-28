# 实施与验收记录

日期：2026-09-28。交付版本：0.1.0。项目路径：`E:\xhh\projects\pytorch-classification-framework`。

## 1. 本次实际完成

已建立独立 Git 仓库、Python 包、项目隔离环境、依赖 lockfile、配置/数据协议、CLI、训练与预测闭环、checkpoint 恢复、ONNX 导出、示例、测试及使用文档。未修改相邻 Web MVP，未上传远程仓库。

本次已完成核心框架首版，并提前实现部分稳定版功能（多标签、Focal/Mixup、导出等）。原方案中的完整 v1.0 全平台验收没有被混称为全部完成。

## 2. 实际环境

| 项目 | 结果 |
|---|---|
| OS | Windows 11，10.0.26200 |
| Python | uv 管理的 CPython 3.12.14；独立 .venv |
| PyTorch / torchvision | 2.14.0+cpu / 0.29.0+cpu |
| timm / Lightning | 1.0.30 / 2.6.6 |
| TorchMetrics / Hydra | 1.9.0 / 1.3.7 |
| 设备 | Intel UHD Graphics 770；CUDA 不可用 |
| 计算设置 | 验收 CPU 2 threads，DataLoader num_workers=0 |

完整记录见 [acceptance-summary.json](evidence/acceptance-summary.json)，准确依赖见项目 `uv.lock`。

初次用 Conda 3.11.8 派生环境导入 torch 出现 c10.dll 初始化失败；切换至 uv 管理的 Python 3.12 后导入与训练成功。没有修改现有 Conda 安装，也未通过忽略异常继续。

## 3. 自动化检查

| 检查 | 结果 | 证据 |
|---|---|---|
| pytest | **28 passed**，9.08 秒 | [JUnit XML](evidence/pytest.xml) |
| Ruff lint | 通过 | 对 src/tests/examples 执行 |
| Ruff format | 通过 | 所有源文件格式化检查 |
| 构建 | wheel + sdist 成功 | dist/xhh_classification-0.1.0-* |
| CLI/Schema | help、list、init、doctor、schema 已运行 | [config.schema.json](config.schema.json)、[doctor.txt](evidence/doctor.txt) |
| 一键脚本 | 实际训练→test→predict 完成 | scripts/run-demo.ps1 |

测试包含：未知配置/非法任务组合拒绝、Hydra 组合与路径解析、数据内容与 group 泄漏、未知类、CSV 与 ImageFolder 标签一致性、loss 有限梯度、缺正负样本的指标语义、六个公开数据 adapter 的 split 契约、预训练缓存隔离、缺失离线权重报错、`.pt` 参数零容差往返、完整恢复、单标签/二分类/多标签与 Mixup/linear probe。

pytest 的 22 条 warning 来自 Lightning/PyTorch 的 TreeSpec 弃用提示及 DataLoader worker 数性能建议；测试没有跳过失败项，也没有把异常吞掉。CPU 下使用 0 个 worker 是本次明确选择。

## 4. 模型检查

8 个 timm 模型均完成 `batch=2, channels=3, 224×224` 的 CPU 前向、交叉熵反向及有限梯度检查：

- resnet18、resnet50、mobilenetv3_small_100、efficientnet_b0。
- convnext_tiny、vit_tiny_patch16_224、deit_tiny_patch16_224、swin_tiny_patch4_window7_224。

另验证 torchvision resnet18。它们使用随机初始化；**此项是模型工厂/计算兼容检查，不是所有模型的完整训练认证**。[模型检查明细](evidence/model-smoke.json)

timm ResNet18 已另行完成真实 MNIST 短训练、默认预训练权重在线下载后的 smoke，以及使用相同缓存的 `offline: true` smoke。后者未重新获取权重；单元测试还禁止了预训练下载入口，以验证缓存读取不会调用该入口。

## 5. 数据与训练实际结果

| 实验 | 数据与训练预算 | 独立 test 结果 | 解释 |
|---|---|---|---|
| TinyCNN 单标签 | 合成颜色图 train=64、val=16、test=16；5 epoch | Top-1=1.0，Macro-F1=1.0 | 验证可学习、标签映射和完整流程；不是业务性能 |
| timm ResNet18 | 真实 MNIST 下载；每个 split 取100张、64×64 RGB、随机初始化、2 epoch | Top-1=0.44，Top-5=0.91，Macro-F1≈0.3844 | 短训练工程验收，未调优、未训练到收敛 |
| TinyCNN 多标签 | 合成 RGB 属性 train=64、val=16、test=16；5 epoch | mAP=1.0，Micro-F1=0.96，Macro-F1≈0.9630 | 验证 BCE、阈值、多标签指标与 bundle |

每次训练只使用 val；test 由独立命令执行。当前没有据测试成绩继续选超参。

真实下载并训练的公开数据仅为 MNIST。FashionMNIST/CIFAR-10/CIFAR-100/Pets/Flowers102 的适配器有受控数据替身测试，验证官方 split 参数和标签协议；本次没有下载这五个完整数据集，不能写成它们已经过真实长训。

## 6. 断点恢复发现与修复

本次自动化测试实际发现：当前 Lightning 的 `save_last=True` 在某些未更新 top-k checkpoint 的 epoch 中，没有产生本框架期望的最新完整状态。测试因此读到了旧 global_step。

已实现 EpochCheckpoint：每个 epoch 边界先处理 best 模型，再独立保存完整 last 状态，通过临时文件和原子替换更新。记录随机状态、优化器、调度器、epoch/step 与数据/配置/源码契约。

修复后，在固定 CPU 环境中比较“连续训练2 epoch”和“训练1 epoch后停止，再恢复到2 epoch”：

- global_step 均为 8，调度器步数一致。
- state_dict 每个参数及缓冲区通过 `rtol=0, atol=0` 比较。
- 更改学习率后严格恢复被正确拒绝。

GPU、多 worker、DDP 精确恢复尚未验收。恢复边界为 epoch，不承诺任意 batch 恢复。

## 7. 权重格式与导出

推理产物为 `model.safetensors`；完整训练状态为 `.ckpt`；本地初始化也接受纯 state_dict `.pt`。`.pt` 往返测试已通过零容差参数比较，详见 [格式说明](USAGE.md)。

ONNX 使用固定空间分辨率、动态 batch。已执行结构检查，并在随机 Tensor 输入 batch=1、3 上与 PyTorch 对比：

| 模型 | 最大绝对误差（两次取最大） | 判定 |
|---|---|---|
| TinyCNN | 约 7.08×10⁻⁷ | 通过 rtol=1e-3、atol=1e-4 |
| ResNet18 | 约 6.85×10⁻⁷ | 通过相同容差 |

证据：[TinyCNN ONNX](evidence/tinycnn-onnx.json)、[ResNet18 ONNX](evidence/resnet18-onnx.json)。未验证所有 timm 模型、TensorRT、量化或其他硬件。随机 Tensor 对齐不替代目标部署环境的真实图片验收。

## 8. 可直接查看的运行产物

最终源码版本的代表运行：

| 实验 | 本地目录 |
|---|---|
| 单标签训练/测试/预测 | [demo_colors](../runs/demo_colors/20260928-113545-3c398c54/status.json) |
| 真实 MNIST | [mnist_resnet18](../runs/mnist_resnet18/20260928-113601-296239b0/status.json) |
| 多标签 | [demo_multilabel](../runs/demo_multilabel/20260928-113610-f4d73013/status.json) |
| 预训练缓存离线 smoke | [offline smoke](../runs/mnist_pretrained_offline-smoke/20260928-113555-4c9ea11c/status.json) |

旧的调试运行保留在 runs 中，未删除；需要恢复时使用与当前源码契约匹配的新运行。runs/data/cache/.venv 被 Git 忽略，复制仓库不会自动携带它们。

## 9. 后续验收项目

1. 在 NVIDIA GPU 主机锁定 CUDA 环境，验证 AMP、真实显存/吞吐、DDP 与多 worker 恢复。
2. 按真实业务选择公开数据和自定义样本完成完整训练，确定基线与推荐超参。
3. 增加更多模型的 bundle/ONNX、真实数据迁移学习和长时间恢复检查。
4. 需要平台化时再接入现有 Worker/MLflow/Web；这些未包含在本次首版中。
5. 大规模验证集的分片聚合、HPO、蒸馏、自动阈值优化与新模态单独迭代。

## 10. 花卉官方预训练微调补充验收（2026-09-28）

- 配置 `configs/flower_photos.yaml` 使用 torchvision 官方 ResNet18 `IMAGENET1K_V1`，保留主干并新建 `Linear(512, 5)` 分类头，默认全参数微调。
- 已下载官方权重、准备 3666 张图片的固定划分；官方权重提供的 100 个主干参数/缓冲张量零容差一致，离线缓存加载通过。见[证据](evidence/flower-resnet18-pretrained.json)。
- 两批真实训练/验证及 bundle 保存成功，运行目录为 `runs/flower_photos_resnet18_finetune-smoke/20260928-131133-9af2e319`；未完成长训练或测试集性能验收。
- 修复短验证缺类时 TorchMetrics 宏平均与固定类别报告不一致的问题：统一逐类 F1 后对全部类别取平均，保留原有报告语义。
- 本轮 pytest **29 passed**，含新增缺类回归与参数零容差恢复测试；Ruff lint/format、wheel/sdist 构建通过。
- 核心源码摘要已改变，旧源码 checkpoint 不能在本版直接严格续训。

## 11. 多模型预训练与 weights 目录补充验收（2026-09-28）

已下载并测试 ResNet18/50、MobileNetV3-Small、EfficientNet-B0、ConvNeXt-Tiny、ViT-Tiny、DeiT-Tiny、Swin-Tiny。
其中 6 个使用 torchvision 官方权重，ViT-Tiny/DeiT-Tiny 使用 timm 发布方仓库。原始权重及五分类初始化权重合计约 841.3 MiB，均位于 Git 忽略的 `weights/`。

- 8/8 在线准备和短流程验收通过，随后 8/8 离线复测通过；每个模型两步真实训练、两批验证，确认参数状态更新及 bundle 保存/加载/预测。
- 每类选一张真实花卉图片，框架预处理与 provider 配套预处理逐像素零容差一致；初始参数及 logits 离线重载零容差一致。
- 每个模型验证 linear_probe 仅开放分类头参数；未单独进行所有模型的长时间线性探测训练。
- 默认启用预训练，在线缺缓存自动下载；显式 none 保留随机初始化，offline 缺缓存仍拒绝。新增回归验证在线缓存命中不重复获取预训练权重。
- 增加独立 weights_dir、Swin/ViT head 适配和精确 resize_size，未静默替换权重、架构或预处理。
- pytest **33 passed**（含严格恢复零容差对齐）、13 个配置加载、Schema 一致性、Ruff 和构建通过。
- 本轮仅 Windows CPU / FP32 / num_workers=0；不宣称长训练收敛、精度排名、GPU 或 ONNX 验收。

来源与结果见[模型报告](evidence/pretrained-models.json)、[权重清单](evidence/weights-inventory.json)、[使用说明](PRETRAINED_MODELS.md)。
核心 Python 文件已修改，旧 checkpoint 不能在新版下直接严格续训。

## 12. 按配置文件启动训练（2026-09-28）

`scripts/run-flower.ps1` 新增 `-Config`，保留 `-Model` 快捷方式；二者互斥。
本次仅修改脚本、ResNet18 配方和文档，没有修改核心 Python 或配置 Schema。

- `-Config configs/flower_resnet18.yaml -Smoke -Offline` 通过，两批真实训练及验证，运行目录 `runs/flower_resnet18-smoke/20260928-133234-2d103811`。
- 从 `configs` 目录传相对路径 `flower_vit_tiny.yaml`，离线短流程通过，运行目录 `runs/flower_vit_tiny-smoke/20260928-133303-82ebb6e9`；执行后调用目录和线程环境变量恢复。
- 同时指定 Config/Model、文件不存在、目录路径、非 YAML 文件四种情况均拒绝。
- 本轮验证配置入口和短训练流程，不代表完整 10 轮训练或精度验收。

## 13. 公共配置、统一入口与日志（2026-09-28）

- 新增 `configs/base.yaml`，8 个模型 YAML 直接继承，只保留实验名和模型差异。旧 flower_photos 入口继续使用 ResNet18/batch=16，其余模型保持 batch=4。
- 模型类别数改为 auto，从当前类别表得到 5 类；原数据划分、优化器、训练轮数和权重位置不变。公共模板拒绝单独运行，13 份具体配置均能加载。
- 新增 `Classifier` Python 接口，CLI 与其共用 prepare/train/smoke/val/test/predict/export 流程；已安装的 `cls.exe` 支持 `mode key=value`，保留原 `-c/--set` 形式。
- 新增可配置的日志级别、终端/文件开关及目录，每次操作独立 UTF-8 文件。验证异常堆栈、中文、级别过滤、文件关闭、handler 清理及失败后继续执行；更改日志设置后续训参数零容差对齐。
- pytest **43 passed**；Ruff lint/format、Schema 与代码一致性、wheel/sdist 构建通过。已用 uv frozen 同步项目本身的新命令入口，未更换系统/Conda 环境。
- Python ResNet18：两批真实花卉训练、完整 551 张验证图片、单图五分类预测、ONNX batch=1/3 数值检查及四种操作日志通过。ONNX 最大绝对误差分别约 3.35e-7、9.09e-7；见 [Python 验收证据](evidence/python-interface.json)。
- 已安装 CLI 的 ViT-Tiny：key=value 形式完成两批真实训练、完整 551 张验证图片及单图预测，训练日志包含每轮指标；见 [CLI 验收证据](evidence/cli-interface.json)。
- 本轮为 Windows CPU/FP32；没有完成长训练收敛或所有模型 ONNX 验收。TinyCNN 的测试数据只用于软件回归，不代表花卉识别性能。

完整用法见[运行与日志指南](RUNNING.md)。核心源码摘要已经改变，旧版本 checkpoint 不能在本版直接严格续训；标准 bundle 推理仍可使用。

日志目录随后统一调整到项目根目录 `logs/`：base 显式设置 `logging.directory: ../logs`，未指定目录的 API 使用 output_root 同级的 logs。
已迁移原 runs/logs 中的 11 个日志文件，并同步运行目录中的日志索引和上述验收证据路径；历史日志正文保持原记录。
再次确认 43 项测试、Ruff 和构建通过；真实 ResNet18 离线 prepare 验证新日志直接写入根目录，未重新创建 runs/logs。

## 14. 正式训练示例脚本参数（2026-09-28）

`examples/train.py` 移除 smoke 分支，仅调用 Classifier.train；支持 epochs、batch、学习率、worker、设备、seed、实验名、warmup、weight decay、离线开关、resume 和通用 --set。未提供的参数继续使用 YAML，--no-offline 可显式允许下载。
使用离线演示数据完成正式一轮训练，运行目录 `runs/train_script_check/20260928-140643-bf8f99cb`；确认 status 为 SUCCEEDED、smoke=false、global_step=8，覆盖后的参数正确保存。
契约与训练测试 **19 passed**（含续训零容差），脚本 Ruff 检查通过。本轮只修改示例脚本和文档，未修改核心源码或 Schema；演示数据仅用于流程验证，不代表花卉识别性能。

## 15. 参数查看与 checkpoint 提示（2026-09-28）

训练示例新增 --list-models、--model/-m、--show-config，以及优化器、调度器、增强方案、微调模式、输入尺寸和日志级别选项；保留 --config/-c。已验证 8 个模型配置预览、只读列表、非法选择/互斥参数拒绝及参数转发。
Lightning 的 `` `weights_only` was not set, defaulting to `False` `` 是保存 last.ckpt 时的 INFO 提示。本框架已在 trainer.save_checkpoint 显式设置 weights_only=False，继续保存完整训练状态，不屏蔽全局日志。
pytest **43 passed**，增加断言确认默认值提示消失、优化器/调度器/callback 状态保留，严格续训参数仍零容差一致；Ruff 和构建通过。
本轮修改了核心 engine.py，源码摘要变化；此前版本 checkpoint 的严格续训需使用对应旧源码。

## 16. 权重接续与目录整理（2026-09-28）

- 花卉配置统一迁入 `configs/flower/`，公共模板改名为 `flower_base.yaml`。9 份可运行配置迁移前后解析结果一致；模型选择、PowerShell 入口及使用文档已同步。数据、权重、运行、缓存和日志仍定位项目根目录。
- `examples/` 仅保留 train.py、predict.py；测试所需的数据生成器放入 tests/helpers。旧演示配置和辅助脚本不再作为当前使用入口，前文记录保留当时的验收路径。
- 花卉配置默认从同一实验最新 last.ckpt 的完整模型参数开始新一轮微调，允许重新设置轮数和学习率，优化器/调度器重置。没有历史记录时采用配置的初始权重；--fresh 关闭自动接续，--finetune-from 支持显式选择。
- 新增 5 项回归，覆盖完整参数（含分类头）零容差加载、新运行与旧 checkpoint 保留、计数重置、配置路径、架构/类别不符及损坏权重拒绝，以及微调运行的严格恢复零容差对齐。修复 YAML 快照排序改变 split 遍历顺序的问题。
- 全量 pytest **48 passed**；Ruff lint/format、Schema 一致性、wheel/sdist 构建通过。模型列表与参数预览验证通过，公共模板拒绝单独运行。
- 验证范围为 Windows CPU 软件回归，未新增花卉长训练或性能评估。核心源码摘要已改变，旧 checkpoint 的严格恢复仍需对应旧源码；参数兼容时可使用本次新增的权重微调入口。

检查摘要见 [目录与接续验收](evidence/config-layout-continuation.json)。

## 17. Bash 批量训练入口（2026-09-28）

新增独立 `scripts/train-models.sh`，通过 uv 串行调用正式训练示例，支持模型选择、轮数、批量、学习率、离线、重新初始化、命令预览及失败策略。沿用训练流程的每轮验证，不使用 smoke、不自动评估 test。

Windows Git Bash 下语法检查通过；隔离的模拟命令验证了 8 模型发现、参数拒绝、失败后继续、首错停止、CSV 汇总、含空格路径和线程/离线参数传递。实际 uv 命令的 ResNet18 单轮配置预览通过（warmup=0）。没有在本轮启动全部模型长训练；模拟状态不作为真实训练通过证明。未修改核心训练源码或配置 Schema。

检查摘要见 [Bash 脚本验证](evidence/bash-model-tests.json)。

### 批量脚本简化

按用户要求，将上述批量入口简化为逐行 `uv run python examples/train.py --config ... --epochs 10`，直接编辑命令，不再提供脚本参数解析、失败继续或 CSV 汇总。保留 8 模型串行训练、失败停止、项目目录定位与 UTF-8 输出。Bash 语法和 8 条命令的配置路径检查通过，未执行全部模型长训练；上一节模拟测试记录对应简化前版本。

## 18. 后台批量训练与明确日志命名（2026-09-28）

在用户已改名的 `scripts/train.sh` 上增加 nohup 后台提交，保留末尾逐模型可编辑命令；8 个模型串行训练。日志目录为根目录 `logs/train_<时间>_<随机后缀>`，每个模型输出独立 `train_flower_<模型>_<时间>.log`，同时保存按模型划分的框架日志、PID、批次状态和 CSV 汇总。后台关闭动态进度条，启用 UTF-8 与无缓冲输出；失败停止，STOP 文件请求当前模型结束后停止。

Git Bash 隔离模拟验收通过：父命令在任务完成前返回、8 模型成功串行执行、第二模型失败后不再执行后续项、STOP 请求、含空格路径、stdout/stderr 捕获及明确日志命名。所有花卉配置日志路径均解析到项目根目录。未启动真实长训练；未修改核心训练源码或 Schema。证据见 [后台脚本验证](evidence/background-train-script.json)。

## 19. 8 模型并发与立即停止（2026-09-28）

按用户明确选择，train.sh 改为后台同时运行 8 个模型。环境只 uv sync 一次，各模型 uv run --no-sync；保留逐模型可编辑命令、清晰日志及退出码，增加 status/stop、原子状态更新和批次启动锁。单模型失败不停止其余并发任务，汇总批次标记 FAILED；stop 立即终止活动批次的训练进程树。

新增 tests/test_train_shell.py，Windows Git Bash 下 **4 passed**：8 任务运行区间重叠、失败汇总及其他任务继续、重复启动拒绝、整批立即停止且确认所有 Python 子孙进程退出、重复停止、环境同步失败后释放锁。使用隔离模拟进程，未进行真实模型长训练。发现并修复 MSYS exec 改变 Windows 父子关系导致只杀 Bash 而遗留 Python 的问题；以批次唯一日志参数定位原生 uv/Python 进程，再终止其子树。

Ruff 和 Bash 语法检查通过。本轮没有修改 src 核心 Python 或配置 Schema，现有 checkpoint 的源码契约不因本轮脚本改动而变化。结果见 [并发脚本验收](evidence/parallel-train-script.json)。
