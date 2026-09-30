# gpu-server 部署与验收

2026-09-29 在 `gpu-server` 的 `/home/jyh/projects/pytorch-classification-framework` 部署当前工作区快照，包含未提交源码与本地 `tests/`。不是仅克隆 Git 提交。

## 环境与启动

- Ubuntu 24.04，Python 3.12.3；NVIDIA 驱动 570.211.01；Tesla V100-SXM2-16GB。
- 服务器 `.venv` 由 uv 正常管理，锁定 PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128 及 CUDA 12.8 依赖。GPU 包使用已有 uv 缓存安装，以普通目录和文件硬链接保存，不再使用跨项目的目录符号链接。
- 服务器 `pyproject.toml` 和 `uv.lock` 已改为 Linux x86_64 CUDA 配置，默认包含 dev、web 依赖组。Windows 本地工作区继续保留原 CPU 配置。
- 使用物理 GPU 6，通过 `CUDA_VISIBLE_DEVICES=6` 映射为进程内的 `cuda:0`。启动前查看 `nvidia-smi`，确认所选 GPU 当前可用。

```bash
ssh gpu-server
cd /home/jyh/projects/pytorch-classification-framework
nvidia-smi
uv sync
uv run cls --help
```

**服务器现在可以直接执行 `uv sync` 和 `uv run`。** PyTorch 源为 `https://download.pytorch.org/whl/cu128`，版本显式固定；执行普通同步也会保留 Web 依赖。不要把本地 CPU 版 `pyproject.toml`、`uv.lock` 覆盖到服务器；服务器专用副本已下载到本地 `dist/gpu-server-validation/pyproject.gpu-server.toml` 和 `uv.gpu-server.lock`。

早期目录符号链接方案不兼容 uv 卸载，已撤销。用户执行自动同步后，原 DIT 环境的部分 torch/torchvision 文件被沿链接删除；这些文件已从现有缓存恢复，并通过两包 RECORD 的完整 SHA256 检查与 CUDA 前向/反向计算检查。损坏的项目环境已移至 `.venv-before-uv-repair-20260929-170342`，其中跨项目目录链接已解除。当前项目环境通过离线 `uv sync --locked --offline` 从已有缓存重建，随后普通 `uv sync`、连续两次 `uv run cls --help` 和依赖检查均通过；不再依赖 DIT 目录存在。

## 运行单卡训练验收

无需下载数据或预训练权重，脚本生成小型合成 ImageFolder 数据集，并执行两轮真实训练及逐轮验证。

```bash
CUDA_VISIBLE_DEVICES=6 uv run python scripts/validate_gpu.py \
  --precision 32-true --output "runs/gpu-check-$(date +%Y%m%d-%H%M%S)"
```

`--output` 必须是新目录，拒绝覆盖。结果包含模型 checkpoint、配置、训练报告和 `validation.json`。脚本断言训练输入和 loss 在 CUDA，检查优化器状态、参数有限性，并对 TinyCNN 的连续训练与断点恢复执行 `rtol=0, atol=0` 参数比较。

使用 `--precision all` 同时检查 FP32 和 FP16；任一用例失败会保留错误详情并返回非零退出码。

## 实测结果

| 用例 | 两轮训练 | 严格续训参数零容差 |
|---|---|---|
| TinyCNN，FP32 | 通过 | 通过 |
| TinyCNN，FP16 AMP | 通过 | 通过，AMP scaler 状态也一致 |
| torchvision ResNet18，FP32 | 通过 | 未测 |
| torchvision ResNet18，FP16 AMP | 失败：`FloatingPointError: Nonfinite gradients` | 未测 |

ResNet18 FP16 的异常发生在 `ClassificationTask.on_before_optimizer_step`：非有限梯度被直接拒绝，执行没有到达 Lightning 的 GradScaler 跳步及缩放调整阶段。因此目前使用显式 `trainer.precision=32-true`；不能把这次测试称为完整 AMP 认证。本次未更改核心训练语义。

既有完整测试套件首轮为 **73 通过、1 失败、5 跳过**。失败来自 CLI 测试将 Linux 保留的 `.train.lock` 文件误计为训练目录；训练本身已成功。测试已改为仅统计目录，随后该失败用例单独重测通过。5 项跳过为未安装 ONNX 导出依赖的测试及 4 项 Windows Git Bash 专用测试。Ruff 检查通过；以上既有测试主要使用 CPU fixture，与表中的实际 GPU 测试分开记录。

验收样本为 train=64、val=16、test=16、32×32、两类颜色图片。各成功用例运行两轮、8 次优化器更新；独立 test 评估在 CPU 执行。合成数据结果不代表真实花卉精度；未验收多卡 DDP、真实数据长训练、GPU 推理或所有模型。

服务器产物：

- `runs/gpu-validation-report/validation.json`：全部 GPU 用例和错误详情。
- `logs/gpu-validation/training-report.log`：GPU 训练日志。
- `logs/gpu-validation/pytest.log`：既有测试套件结果。
- `logs/gpu-validation/pytest-cli-recheck.log`：CLI 测试修正后的单项复测。
- `runs/gpu-validation-fp32/validation.json`：仅 FP32 的完整通过结果。
- `runs/gpu-validation-uv-sync/validation.json`：改为 uv 正常管理后，使用 `uv run` 重新通过 FP32 训练和 TinyCNN 严格恢复。
- `logs/gpu-validation/shared-environment-repair.json`：原 DIT 包文件恢复与完整性检查结果。
- `logs/gpu-validation/doctor.json`：配置与环境检查。
- `requirements-gpu-server.txt`：实际环境版本清单。
- `source-manifest.json`：初次同步的 105 个文件的 SHA256；新增验收脚本、本文及 CLI 测试修正另行同步。

## 使用真实数据训练

先准备合法的 ImageFolder 数据（`train/val/test/<类别>/图片`），再显式指定其路径。花卉脚本已支持自动排除已确认的两张同像素异类图片：`roses/15922772266_1167a06620.jpg`、`tulips/15922772266_1167a06620.jpg`；路径和 SHA256 必须同时匹配，其他跨类别冲突仍报错。原始压缩包保留；排除项写入 `split_report.json`。以下示例中的路径是占位符：

本次服务器已执行 `uv run python scripts/prepare_flowers.py --offline`，生成 `data/flower_photos_split`：train=2564、val=551、test=551，共 3666 张。已排除上述两张冲突图片及两张同类重复图片；本地两张指定源图已删除，本地已有划分保持不变。7 项数据脚本回归在本地和服务器均通过。脚本两端 SHA256 相同。

```bash
CUDA_VISIBLE_DEVICES=6 uv run cls train \
  model=configs/flower/flower_resnet18.yaml \
  dataset.root=/absolute/path/to/your/imagefolder \
  device=gpu trainer.precision=32-true batch=16 workers=0
```

花卉配方默认加载官方预训练权重；缺缓存时会下载。严格续训应保持同一源码、依赖、配置和数据；已有本地 CPU 实验不能作为本次 GPU 零容差恢复的依据。

### Web 默认 GPU 与真实数据短训练验证

Web 表单默认设备已从 CPU 改为“自动选择 · 优先 GPU”，并移除固定的“CPU · 当前环境推荐”提示。服务器 CUDA 与 GPU 监控可用时，后端将 `device=auto` 解析为 GPU，再由调度器分配显卡。显式选择 CPU 仍会使用 CPU；此前已提交的任务不会自动切换。前端源码已同步本地和服务器，两端生产构建通过；刷新浏览器加载新版即可生效，无需重启后端或中断已有任务。

2026-09-29 通过运行中的 `POST /api/train` 提交 `device=auto`，使用缓存的 MobileNetV3 Small 官方权重、真实花卉 train=2564 / val=551、batch=16、FP32，完整训练一轮及验证成功。任务 `f9af33b1615e4a0cbb4c447772801298` 自动分配物理 GPU 6，日志显示 `GPU available: True (cuda), used: True`、`CUDA_VISIBLE_DEVICES: [6]`，完成 161 次优化器更新并生成 checkpoint 与报告。此验证覆盖 Web API 到实际 CUDA 训练的路径，不代表所有模型、多卡训练或充分收敛认证。

记录：`logs/gpu-validation/web-gpu-default.json`；实验：`runs/web_gpu_check_20260929_092624/web-f9af33b1615e4a0cbb4c447772801298`。

### 全局训练设备、指定显卡和多卡 DDP

后续更新将训练表单默认改为“跟随全局设置”。设置页新增“全局训练设备”：选择 CPU / 自动 / GPU，勾选一张卡用于固定单卡，勾选多张卡用于同一任务的 DDP。空列表表示自动分配单卡。设置持久化到 `configs/web.yaml`，服务器当前为 `training: {device: gpu, gpu_indices: [6]}`；本地默认保持自动选择。设备配置在提交时固定，显存不足或所选任一卡额度不足时整个任务排队；运行期间不自动换卡。

本地 36 项设备、训练、契约和 API 测试通过（含严格恢复参数零容差用例），服务器 7 项新增设备与调度测试通过。两端前端构建、Ruff、`uv build` 通过；浏览器验证保存 GPU 6、7、刷新保留、表单提交 `device=global`。未修改核心 `src/clsframework` 源码。

通过 Web API 省略 `device`（使用默认全局配置），分别在 GPU 6 单卡与 GPU 6、7 双卡完成 MobileNetV3 Small 的真实花卉一轮 FP32 训练：每卡 batch=16，单卡 161 次更新、双卡 81 次更新；两者验证均为 551 个独立样本，并生成 checkpoint 与报告。双卡日志确认 NCCL、两个 rank 与 `CUDA_VISIBLE_DEVICES: [6,7]`，任务成功退出。仅主 rank 发布 Web 结果，调度器预留及释放全部所选卡。该结果限定于本次模型及单轮配置，不代表所有模型、多卡严格恢复或长训练收敛认证。记录为 `logs/gpu-validation/global-gpu-training.json`。

部署重启中停止了当时正在运行的 MobileNetV3 Small 和 EfficientNet-B0 两项任务，两者 `best.pt`、`last.pt` 均保留。备份与中断记录位于 `logs/gpu-validation/before-global-gpu-20260929-173601/`。可以从 Web“接着上次权重微调”创建新任务；该操作不是严格续训。新服务 PID 为 714424，首次重启后健康检查及真实单卡、双卡训练均通过；末次可连接时全局配置已保存为 GPU 6、活动任务与排队任务均为零。随后服务器在 SSH 握手前主动断开连接，额外的第二次重启复查未执行；上述新增文档尚未再次同步到服务器。

## 从本机访问服务器 Web

### 2026-09-30 ViT / DeiT 权重离线缓存修复

服务器访问 Hugging Face 时连接被重置，ViT-Tiny 和 DeiT-Tiny 因缺少本地权重而失败。本地工作区已有两款官方快照和默认花卉配置（5 类、seed=42）的初始化缓存，已核对 SHA256 后同步到服务器 `weights/`，不重新下载、不改变模型或回退到随机权重。HF 快照文件使用实际文件保存，不依赖 Windows 符号链接；初始化缓存通过项目原有文件锁与临时文件原子发布。

ViT 官方快照 revision `7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106`，DeiT 为 `80e968688553f219e4a86f940ed945a23709c16f`。服务器在 `HF_HUB_OFFLINE=1` 与 `runtime.offline=true` 下，两款模型均通过官方快照查找、真实数据 `prepare`、严格权重加载及 CPU 前向检查（输出 1×5 且有限）；这些检查不等于完整训练验收。当前花卉配置重新提交后直接加载初始化缓存，无需重启 Web。旧失败任务的日志不会自动清除；ViT 原失败任务已从列表中删除，未重建其十轮训练。

记录：`logs/gpu-validation/vit-tiny-cache-manifest.json`、`deit-tiny-cache-manifest.json`、`timm-offline-cache-validation.json`。本次未修改核心训练代码，也未中断正在运行的其他训练。

### 2026-09-30 按显存启动，共享 GPU

本地和服务器新增 `concurrency.train: 0`、`resources.max_jobs_per_gpu: 0`，0 表示取消对应任务数量限制。服务器已保存这两个值，保留用户选择的 GPU 6、7，以及每卡最低空闲显存 2 GB、系统内存保护和同名实验锁。空闲显存达标时允许多个不同实验共享所选显卡。显存不足的排队信息会具体显示 GPU 编号、实际空闲量和阈值；2 GB 是启动门槛，不是各模型峰值显存的预测。

部署在 Swin 正常完成、ViT 因 Hugging Face 权重下载失败退出后执行，没有中断运行中的训练。当前 Web 服务 PID 1572861，健康检查通过。部署前备份：`logs/gpu-validation/before-memory-mode-20260930-095037/`。

9 项调度回归在本地和服务器通过，包含同时启动超过 8 个模拟任务、取消每卡数量限制后仍拒绝显存不足、明确显示显存等待原因。两端生产构建、浏览器表单保存及刷新、Ruff 和本地 `uv build` 通过。真实验证提交两个不同实验的 MobileNetV3 Small 双卡任务，运行时间重叠、共同使用 GPU 6 和 7；两者均完成一轮真实花卉 FP32 训练和 551 个验证样本，并成功退出。记录：`logs/gpu-validation/memory-scheduling-validation.json`。这是指定模型与配置的短测，不代表任意并发数量或长训练稳定性认证；NCCL 共享 GPU 的限制参见 [PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed)。

服务器 `jyh` 用户已通过 [nvm 官方安装方式](https://github.com/nvm-sh/nvm#installing-and-updating) 安装 nvm 0.40.8，安装并默认使用 Node.js v24.21.0 / npm 11.19.0。项目 `.nvmrc` 指定 24。新开交互式 Bash 会加载 nvm；非交互 SSH 命令需要显式加载：

```bash
source ~/.nvm/nvm.sh
cd /home/jyh/projects/pytorch-classification-framework
nvm use
cd web
npm ci
npm run build
```

服务通过 `scripts/serve_web.py` 同时提供已构建的前端和原有 API，不依赖 Vite 开发服务器。2026-09-30 改为监听服务器局域网地址 `192.168.4.3:8000`，浏览器直接访问 **http://192.168.4.3:8000**，无需 SSH 隧道。Web Python 依赖已补齐；现有 CUDA 版 PyTorch 保持不变。

当前服务已在后台启动，日志为 `logs/web/server.log`，当前进程记录为 `runs/web-server.json`。环境修复后服务已重启。关闭浏览器不会停止服务；尚未配置服务器重启后的自动启动。需要重新启动时，先确认端口未占用，然后在项目根目录运行：

```bash
nohup .venv/bin/python scripts/serve_web.py --host 192.168.4.3 --port 8000 \
  > logs/web/server.log 2>&1 < /dev/null &
```

API 文档为 **http://192.168.4.3:8000/docs**。本机需要能路由到服务器局域网地址。应用只允许本机及指定服务器地址的 Host/Origin；不使用通配白名单。当前没有登录认证，供可信局域网使用。

如果所在网络无法直接访问 8000 端口，但 SSH 可达，也可以手动使用隧道，转发目标需使用新的监听地址：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:5173:192.168.4.3:8000 gpu-server
```

不指定 `--host` 时仍默认监听 `127.0.0.1`，适合本地 CPU 环境。页面操作的模型、任务和数据属于服务器工作区；数据放在服务器项目的 `data/` 下，准备成功后点击页面的“刷新数据集”。

### 2026-09-30 Docker 部署：复用服务器环境

当前服务已改为轻量容器 `classification-studio-studio-1`，只读挂载本项目现有 `.venv` 与源码，未使用独立 CUDA 镜像。旧手工 Python 服务已停止，后续运维使用 [Docker 部署说明](DOCKER.md) 中的发布目录与 Compose 命令，不要同时启动旧服务。

现有 15 条任务、18 个实验、GPU 6/7 及页面设置全部保留，健康检查和重启持久化验证通过。复用环境下 GPU 6 的 TinyCNN、ResNet18 两轮 FP32 合成训练及 TinyCNN 零容差严格恢复通过。本机局域网直连仍发生网络中断，已恢复后台 SSH 隧道，当前可通过 **http://127.0.0.1:5173** 访问。服务器同时发布回环和局域网 8000 端口。

随后已恢复 SSH 连接，完成 116 个共同项目文件的同步与摘要核对，并补齐本机局域网隧道来源白名单。当前 **http://192.168.0.42:5173** 与 **http://127.0.0.1:5173** 均实测可访问页面和 API。`scripts/web_tunnel.ps1` 提供断线重连；原容器网络连接丢失问题已通过重新创建容器修复，详细验收及备份记录见 [Docker 部署说明](DOCKER.md)。
