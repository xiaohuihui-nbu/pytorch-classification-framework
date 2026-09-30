# Docker 部署

服务器使用轻量 Ubuntu 24.04 / Python 3.12 容器，**直接只读挂载服务器现有 `.venv` 和 `src`**。`Dockerfile.server` 不安装 PyTorch、不安装 CUDA、不同步 Python 依赖。前端在构建时生成静态文件；API 与训练子进程均调用挂载环境的 Python。服务器原 `.venv` 和 CUDA 配置保持不变。

另提供 `Dockerfile` 独立 CPU 镜像，使用本地根目录 CPU `uv.lock` 安装依赖，供没有现成环境的机器使用。

## 服务器部署：复用现有环境

当前方案适配 Ubuntu 24.04、Python 3.12，现有 `.venv/bin/python` 应解析到 `/usr/bin/python3.12`，且已安装项目 Web 依赖。其他发行版、Python ABI 或引用外部目录的虚拟环境须先适配并验证，不做自动回退。

服务器需要 Docker Engine、Compose 和已配置的 NVIDIA Container Toolkit。Toolkit 将宿主驱动与设备提供给容器；PyTorch/CUDA 用户态库来自现有 `.venv`，不是独立 CUDA 镜像。

在部署源码目录中创建 `.env`（按实际环境修改）：

```dotenv
CLS_UID=1008
CLS_GID=1008
CLS_WORKSPACE=/home/jyh/projects/pytorch-classification-framework
CLS_DATA_ROOT=/home/jyh/projects/pytorch-classification-framework
CLS_IMAGE=classification-studio:server-env
CLS_PORT=8000
CLS_WEB_PORT=5173
CLS_BIND_IP=127.0.0.1
CLS_ALLOWED_HOST=127.0.0.1
```

`CLS_UID` / `CLS_GID` 使用部署用户的 `id -u` / `id -g`。`CLS_WORKSPACE` 保持原项目绝对路径，避免历史任务、配置和模型路径失效。默认从 `CLS_DATA_ROOT` 读取 `.venv` 与 `src`，独立测试目录可用 `CLS_ENV_ROOT` 指向原项目。

```bash
docker compose -f compose.yaml -f compose.server.yaml build
# 先确认原 Web 服务没有运行或排队任务，再停止原服务。
docker compose -f compose.yaml -f compose.server.yaml up -d --wait
docker compose -f compose.yaml -f compose.server.yaml ps
docker compose -f compose.yaml -f compose.server.yaml exec studio nvidia-smi
docker compose -f compose.yaml -f compose.server.yaml exec studio \
  /home/jyh/projects/pytorch-classification-framework/.venv/bin/python \
  -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

一个工作目录只能运行一个 Web 服务，不要增加 Uvicorn workers 或 Compose replicas。服务会自动启动训练子进程。Compose 暴露全部 GPU 以保留物理编号，实际训练卡仍由设置页选择；现有 GPU 6、7 设置可继续使用。这不是多租户显卡隔离方案。

容器内源码直接来自宿主 `src`，更新宿主源码会影响容器；更新前等待任务结束并备份，更新后重启。不要在服务运行时对宿主 `.venv` 执行同步或升级。挂载是只读的，但宿主操作仍会改变环境。

## 访问与持久化

默认只发布服务器 `127.0.0.1:8000`，远程访问使用 SSH 隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:5173:127.0.0.1:8000 gpu-server
```

打开 <http://127.0.0.1:5173>。已有局域网部署可将 `CLS_BIND_IP` 与 `CLS_ALLOWED_HOST` 均设为服务器局域网 IP，使用 8000 端口访问。`CLS_WEB_PORT` 是允许的浏览器来源端口。应用保留 Host/Origin 校验；`--container` 仅改变容器内监听接口。系统没有登录和多用户权限，不直接发布到公网。

如果要从本机局域网地址 `http://192.168.0.42:5173` 访问服务器，需同时放通隧道监听和后端白名单。在服务器 `.env` 增加 `CLS_WEB_ORIGINS=http://192.168.0.42:5173` 并用 Compose 重新创建服务；此变量只接受精确的 `http://本机或局域网IPv4:端口`，多个来源用逗号分隔。然后在 Windows 项目目录运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/web_tunnel.ps1
```

该脚本同时监听 `127.0.0.1:5173` 与 `192.168.0.42:5173`，转发至服务器回环 8000 端口，连接断开后每 15 秒重试；前台可用 Ctrl+C 退出。其他本地 IP 可传 `-BindAddress`，并同步修改服务器白名单。监听地址必须属于本机；Windows 防火墙只需允许 SSH 程序在该地址的 TCP 5173 入站，来源限制为本地子网。日志保存在 `logs/remote-web/lan-tunnel.log`。SSH 握手失败时无法访问页面，自动重连不会绕过网络或服务器拒绝。

`configs`、`data`、`weights`、`cache`、`runs`、`logs` 挂载到宿主原目录；数据集只读，其余目录可写。页面设置采用原子替换，所以挂载整个 `configs/`，不能只挂载 `web.yaml`。构建上下文排除数据、权重、虚拟环境和运行记录，构建不会下载数据或预训练权重。

## 独立 CPU 部署

在使用根目录 CPU `uv.lock` 的本地源码副本运行：

```bash
cp deploy/docker.env.example .env
# 将 .env 中 UID/GID 改为当前用户。
mkdir -p data weights cache runs logs
docker compose build
docker compose up -d --wait
```

此模式不需要挂载 `.venv` 或源码。CLI 示例：`docker compose exec studio cls doctor model=configs/flower/flower_resnet18.yaml`。服务器现有根目录依赖已是 CUDA 专用配置，不能把该配置误当 CPU 锁构建。

## 迁移、更新与回滚

1. 在独立发布目录构建轻量镜像，保留原服务、源码和环境。
2. 使用独立测试目录与端口验证镜像，不与旧服务同时打开同一 `runs/web`。
3. 确认运行和排队任务均为空，备份 SQLite、配置与启动参数，再停止原服务并启动容器。
4. 检查页面、历史任务、模型、GPU 与设置；重启容器再次检查持久化。

服务器命令均在对应发布目录执行：

```bash
docker compose -f compose.yaml -f compose.server.yaml logs --tail=100 -f
docker compose -f compose.yaml -f compose.server.yaml restart
docker compose -f compose.yaml -f compose.server.yaml down
```

`restart: unless-stopped` 在 Docker 服务重启后自动拉起容器；宿主 Docker 服务需启用开机启动。停止容器会停止活动训练，任务会标记中断，不自动续训。首次切换失败时，先 `down` 再用原 `.venv/bin/python scripts/serve_web.py` 与原 host/port 参数启动旧服务。镜像、宿主源码和 `.venv` 都是回滚对象，仅切换镜像不能撤回宿主源码或环境升级。

训练核心源码摘要属于严格续训契约；本次 Docker 支持不修改核心训练代码。后续改变源码、依赖或数据可能使旧 checkpoint 无法严格恢复，不能宣称跨环境零容差一致。

容器日志保留 3 个 10 MB 文件，训练日志和产物不自动删除。备份包括配置、完整 `runs`、所需数据与权重；在线 SQLite 备份使用 backup API。不要用全局 Docker prune 清理其他项目。

## 本地验收（2026-09-30）

- 独立 CPU 镜像构建与 Compose 健康检查通过。
- 真实 HTTP 请求完成 TinyCNN 合成数据一轮训练、上传图片推理、设置保存，Host/Origin 拒绝检查通过。重启后任务、模型和设置保留，测试容器已移除。
- 核心测试 95 项、Web 测试 26 项通过；修正了 Web 测试中过时的并发默认值和 GPU 列表参数断言。Ruff 与 `uv build` 通过。
- 记录：`logs/docker-build-cpu.log`、`logs/docker-core-tests.log`、`logs/docker-web-tests.log`、`dist/docker-cpu-check/validation.json`，均不入 Git。合成数据检查不代表真实分类性能。

实现参考：[uv Docker 集成](https://docs.astral.sh/uv/guides/integration/docker/)、[Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## gpu-server 实际部署（2026-09-30）

已切换为 `classification-studio:server-env-20260930`，容器名 `classification-studio-studio-1`，镜像约 188 MB（压缩内容约 46 MB）。现有 `.venv` 与源码只读挂载，核心源码摘要与切换前一致；没有使用独立 CUDA 镜像。已清理本次废弃 CUDA 构建的约 15.2 GB 专属缓存，未清理其他项目缓存。

发布目录为 `/home/jyh/projects/pytorch-classification-framework/deploy/releases/docker-20260930`。额外的 `compose.access.yaml` 同时保留回环和局域网端口：

```bash
cd /home/jyh/projects/pytorch-classification-framework/deploy/releases/docker-20260930
docker compose -f compose.yaml -f compose.server.yaml -f compose.access.yaml ps
docker compose -f compose.yaml -f compose.server.yaml -f compose.access.yaml logs --tail=100
```

服务在 `192.168.4.3:8000` 与 `127.0.0.1:8000` 发布。Windows 后台运行 `scripts/web_tunnel.ps1`，可通过 <http://192.168.0.42:5173> 或 <http://127.0.0.1:5173> 访问。服务器已配置 `CLS_WEB_ORIGINS=http://192.168.0.42:5173`，两处页面、API 和设置请求来源校验均实测通过。隧道进程记录为 `logs/remote-web/lan-tunnel-supervisor.pid`，连接日志为 `lan-tunnel.log`；本机重启后重新运行隧道脚本。

独立测试容器复用现有服务器环境，在 GPU 6 上通过 TinyCNN 与 ResNet18 各两轮 FP32 合成数据训练，TinyCNN 严格恢复参数 `rtol=0, atol=0` 对齐通过。该结果不是 AMP、多卡或真实数据精度认证。验证报告位于发布目录 `check/runs/gpu-validation/validation.json`，日志 `validation-server-2.log`。曾修正任意 UID 缺少用户名导致的 PyTorch 缓存初始化错误，Compose 显式设置容器用户名环境变量，不修改宿主用户。

切换时没有活动或排队任务。15 条历史任务、18 个实验、GPU 6/7 与全部页面设置保持，容器重启后再次核对通过；Docker 开机启动已启用。备份位于原项目 `logs/docker-deployment/before-20260930-105051/`，包括 SQLite、配置与原启动参数。`deployment.json` 和 `container-inspect.json` 保存部署结果与挂载记录，原项目 `runs/web-server.json` 已更新为容器部署信息。

### 本地与服务器同步

2026-09-30 后续同步核对 116 个代码、文档、配方和测试文件，更新 15 个差异文件，包含局域网来源白名单、自动重连脚本及 Docker 配置。服务器容器当时处于健康状态，但网络连接为空、端口未实际发布；确认无运行和排队任务后，使用 `up -d --no-build --force-recreate --wait` 恢复网络和端口，未构建 CUDA 镜像或更换 Python 环境。

两端 `pyproject.toml`、`uv.lock`、`configs/web.yaml` 保持各自 CPU/GPU 环境和设备设置，不互相覆盖；数据、权重、虚拟环境、运行记录不参与代码同步。备份为 `logs/docker-deployment/before-sync-20260930-114523/`，同步清单和结果为服务器 `deploy/sync-manifest.json`、`deploy/sync-result.json`。本次没有改变核心训练源码。

服务器 Web 回归首轮为 36 项通过、1 项平台断言失败：反斜线在 Windows 表示目录分隔符，在 Linux 则是普通文件名，两端分别正确拒绝为 400/404。测试已兼容这两种拒绝响应，并增加跨平台正斜线越界路径的严格 400 检查；该用例与来源白名单的 12 项定向测试在两端均通过。两端 Ruff 通过，最后再次核对 116 个共同文件摘要一致。
