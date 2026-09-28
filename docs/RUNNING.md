# 命令行与 Python 统一入口

接口形式参考 [Ultralytics Python](https://docs.ultralytics.com/usage/python/) 和 [CLI](https://docs.ultralytics.com/usage/cli/)：创建一个模型对象，再调用 train、val、predict、export。
本项目的对象是 `Classifier`，处理图像分类；训练模型由本项目 YAML 指定，推理模型使用本项目的 bundle。无需安装 ultralytics。

## 配置只写差异

`configs/flower/flower_base.yaml` 保存共用的数据、优化器、训练、预处理和权重目录设置；正常切换模型不用修改。
8 个 `configs/flower/flower_<模型>.yaml` 均直接继承 base。例如 ResNet50 文件只需：

```yaml
defaults: [flower_base, _self_]
experiment:
  name: flower_resnet50
model:
  provider: torchvision
  name: resnet50
```

其他参数从 base 继承；在该文件增加 loader、optimizer、trainer 等字段即可覆盖。
base 是模板，必须选择具体模型配置后才能训练。当前共用花卉五分类数据，模型类别数从类别表自动推断。
首次训练的初始权重优先从项目根目录 `weights/` 加载，缺失时自动下载；`offline=true` 禁止下载，缺缓存会报错。同一实验已有训练记录时，花卉配置默认从上次权重开始新一轮微调。

## 命令行

在项目根目录执行，无需激活环境。PowerShell 自带 `cls` 是清屏别名，务必使用 `cls.exe` 的路径：

```powershell
# 完整训练：参数默认来自 YAML，也可覆盖常用选项
.venv\Scripts\cls.exe train model=configs/flower/flower_resnet18.yaml
.venv\Scripts\cls.exe train model=configs/flower/flower_resnet50.yaml epochs=10 batch=4 lr0=0.0003

# 只准备数据/权重；或用两批真实数据检查训练和验证流程
.venv\Scripts\cls.exe prepare model=configs/flower/flower_vit_tiny.yaml
.venv\Scripts\cls.exe smoke model=configs/flower/flower_vit_tiny.yaml offline=true

# 将 <run_id> 替换为 train/smoke 返回的实际目录；也可传该目录下的 bundle
.venv\Scripts\cls.exe val model=runs/flower_resnet18/<run_id>
.venv\Scripts\cls.exe test model=runs/flower_resnet18/<run_id>
.venv\Scripts\cls.exe predict model=runs/flower_resnet18/<run_id>/bundle source=data/flower_photos_split/test output=predictions.jsonl batch=16
.venv\Scripts\cls.exe export model=runs/flower_resnet18/<run_id>/bundle format=onnx output=exports/resnet18.onnx

# 原有参数形式继续支持，行为与上面相同
.venv\Scripts\cls.exe train -c configs/flower/flower_resnet50.yaml --set trainer.max_epochs=10
```

命令入口将输出设为 UTF-8，并在导入数值库前为未设置的 OpenBLAS/OMP/MKL 线程变量设置 1；保留调用方已设置的值。
base 的 workers 为 0。`scripts/run-flower.ps1 -Config ...` 仍可使用，它额外强制 workers=0 及线程数=1。
含空格参数需整体加引号，例如 `"model=E:\my project\configs\flower\flower_resnet18.yaml"`。
也可把上述 `.venv\Scripts\cls.exe` 替换成 `.venv\Scripts\python.exe -X utf8 -m clsframework.cli`。

## Bash 批量训练

`scripts/train.sh` 默认后台同时运行 8 个模型，每个模型正式训练 10 轮并在每轮验证。终端关闭不影响 nohup 任务，电脑关机/重启仍会终止任务。

本机 PowerShell 的 `bash` 指向 WSL，请明确使用 Git Bash：

```powershell
# 启动，终端随即返回
& "D:\software\Git\bin\bash.exe" scripts/train.sh
# 查看所有批次和逐模型状态
& "D:\software\Git\bin\bash.exe" scripts/train.sh status
# 立即停止活动批次及其训练子进程
& "D:\software\Git\bin\bash.exe" scripts/train.sh stop
```

若已打开 Git Bash：

```bash
bash scripts/train.sh
bash scripts/train.sh status
bash scripts/train.sh stop
# 本次统一训练 20 轮
EPOCHS=20 bash scripts/train.sh
```

也可直接编辑脚本顶部 EPOCHS，或文件末尾每个模型的命令；用 `#` 注释整行即可跳过该模型：

```bash
run_model resnet50 uv run --no-sync python examples/train.py --config configs/flower/flower_resnet50.yaml --epochs "$EPOCHS"
```

各行可增加 `--batch 4`、`--lr 0.0001`、`--offline` 或 `--fresh`。默认继承 YAML 的上次权重微调设置，`--fresh` 从配置的预训练权重开始。统一 EPOCHS=1 时自动设 warmup=0；单独将某行改为一轮时，应同时加 `--warmup-epochs 0`。

后台先执行一次 `uv sync --frozen --extra export`，之后所有模型使用 `uv run --no-sync`，避免并发修改 `.venv`，也保留 ONNX 导出依赖。`--offline` 仅针对模型命令；需要依赖安装也离线时，在脚本的 uv sync 行加 `--offline`。数据加载进程固定为 0，BLAS/OMP/MKL 线程固定为 1。后台关闭动态进度条，逐轮指标、异常和 stdout/stderr 完整写入日志。

每次启动使用独立目录，模型日志命名明确：

```text
logs/train_20260928_180000_Ab12Cd/
  batch.log                  # 环境准备、任务启动与批次结果
  batch.pid                  # Git Bash 监督进程 PID
  status.txt                 # STARTING/RUNNING/STOPPING/SUCCEEDED/FAILED/STOPPED
  exit_code.txt              # 完成后的批次退出码
  summary.csv                # 所有模型的状态、退出码和日志路径
  train_resnet18.log          # ResNet18 标准输出与错误
  train_resnet50.log          # ResNet50 标准输出与错误
  resnet18.status            # 单模型状态
  resnet18.exit              # 单模型退出码
  resnet18.pid               # 单模型 Git Bash 任务 PID
  resnet18/                  # 框架结构化日志
```

启动输出会显示实际目录。使用 `tail -f <目录>/train_resnet50.log` 追踪某个模型日志；Ctrl+C 仅退出查看，不停止训练。

单个模型失败不会中止其他并发模型；全部结束后，任一失败使批次标记 FAILED。`stop` 会请求监督进程立即终止本批次训练进程树，并等待停止确认；无活动任务时重复调用无副作用。Git Bash 下停止操作内部使用 Windows 原生进程查询及 taskkill，已验证不会只结束外层 Bash 而遗留 Python 子进程。

脚本通过 `logs/.train.lock` 拒绝重复启动。正常完成或 stop 后自动释放；若断电或外部强制终止留下锁，先核实对应批次没有存活进程，再移除锁中的 batch 文件和空锁目录。不要通过删锁绕过正在运行的批次。

启动返回成功仅表示任务提交成功，实际结果以 status/summary 为准。本轮仅在 Windows Git Bash 下进行模拟任务的并发和进程管理验证，未启动真实 8 模型长训练；未修改核心训练实现或配置 Schema。

## Python 脚本

```python
from clsframework import Classifier


def main():
    model = Classifier("configs/flower/flower_resnet18.yaml")
    run_dir = model.train()  # 首次加载预训练主干；有历史权重时接着上次微调
    # 也可以 model.train(epochs=10, batch=4, lr0=0.0003)
    print(run_dir, model.bundle)

    validation = model.val()  # 默认验证集，返回指标字典
    test_metrics = model.test()  # 独立测试集
    print(validation["macro_f1"], test_metrics["macro_f1"])

    results = model.predict("data/flower_photos_split/test", output="predictions.jsonl")
    print(results[0]["label"], results[0]["probabilities"])
    # model("path/to/image.jpg") 等价于 model.predict(...)
    model.export(format="onnx", output="exports/resnet18.onnx")


if __name__ == "__main__":
    main()  # Windows 多进程脚本需保留入口保护
```

用项目 Python 的 UTF-8 模式执行脚本：`.venv\Scripts\python.exe -X utf8 your_script.py`。
Python API 在创建对象时设置未指定的 BLAS 线程限制；应在导入其他数值计算库前创建对象，已初始化的数值库不会因此重新配置线程。
训练返回 run 目录 `Path`，对象保存 `run_dir` 和 `bundle`；评估返回指标字典；推理返回列表，每项包含图片路径、标签和类别概率。
二分类/多标签沿用各自任务的结果字段。每次推理会加载并校验 bundle；当前接口不缓存常驻模型，也不提供流式返回。
只推理时不需要训练配置或训练数据：

```python
model = Classifier("runs/flower_resnet18/<run_id>/bundle")
results = model("path/to/image.jpg")  # 不传 output 时返回结果，不保留 JSONL 临时文件
```

重新打开 run/bundle 后，val/test 会读取相邻的 `config.resolved.yaml`。独立复制的 bundle 需显式传 `model.val(config="configs/flower/flower_resnet18.yaml")`。
评估仍严格检查训练时的数据指纹与类别顺序，不能把不同数据静默当作原测试集。已有预测文件、评估目录和导出文件拒绝覆盖；再次运行时指定新的 output。
模型来源为 YAML 时，必须先训练生成 bundle 才能调用推理；单独的官方 ImageNet 权重不能直接完成花卉五分类。

已提供可直接运行的 Python 示例：

```powershell
.venv\Scripts\python.exe -X utf8 examples/train.py --config configs/flower/flower_resnet18.yaml
.venv\Scripts\python.exe -X utf8 examples/train.py --config configs/flower/flower_vit_tiny.yaml --epochs 20 --batch 4 --lr 0.0001 --offline
.venv\Scripts\python.exe -X utf8 examples/predict.py --model runs/flower_resnet18/<run_id>/bundle --source path/to/image.jpg
```

`examples/train.py` 只调用正式 train，不提供 smoke。未传入的参数沿用 YAML；可通过 `--help` 查看全部选项。
例如训练一轮时同时设置 `--epochs 1 --warmup-epochs 0`，确保预热轮数小于总轮数；其余字段可用 `--set optimizer.weight_decay=0.02` 覆盖。

列出模型、参数及预览配置：

```powershell
# 读取现有模型 YAML，列出简称、provider、网络名称和配置路径
.venv\Scripts\python.exe -X utf8 examples/train.py --list-models
# 列出全部参数及可选值
.venv\Scripts\python.exe -X utf8 examples/train.py --help
# 预览 YAML 继承和参数覆盖结果，不下载、不训练（auto 尚未解析数据/权重）
.venv\Scripts\python.exe -X utf8 examples/train.py --model resnet50 --epochs 20 --show-config
# 使用模型简称启动正式训练，与选择对应 YAML 等价
.venv\Scripts\python.exe -X utf8 examples/train.py --model resnet50 --epochs 20 --batch 4 --optimizer adamw
```

当前模型简称为 `resnet18`、`resnet50`、`mobilenetv3_small`、`efficientnet_b0`、`convnext_tiny`、`vit_tiny`、`deit_tiny`、`swin_tiny`。
`--model/-m` 与 `--config/-c` 互斥；都不指定时使用 ResNet18。简称清单来自 configs/flower 中的模型文件；自定义模型仍可用 --config 指向其他 YAML。

| 参数 | 可选值 / 用途 |
|---|---|
| --epochs / --epoch | 正整数，训练轮数 |
| --batch / --batch-size | 正整数，每设备批量 |
| --lr / --lr0 | 大于 0 的学习率，不能低于 scheduler.min_lr |
| --optimizer | adamw / adam / sgd |
| --scheduler | warmup_cosine / none |
| --augmentation | none / baseline / finetune / strong |
| --training-mode | full_finetune（全部参数）/ linear_probe（仅分类头） |
| --device | cpu / gpu / auto；当前环境为 CPU 版 |
| --workers | 大于等于 0，当前机器建议 0 |
| --imgsz | 图像边长，至少 8；须满足所选模型输入要求 |
| --log-level | DEBUG / INFO / WARNING / ERROR |
| --offline / --no-offline | 禁止 / 允许下载缺失资源 |
| --warmup-epochs / --weight-decay | 预热轮数 / 权重衰减 |
| --seed / --name / --resume | 随机种子 / 实验名 / 严格续训 checkpoint |
| --finetune-from / --fresh | 指定上次权重 / 关闭自动接续 |
| --set KEY=VALUE | 覆盖其余字段，可重复传入不同字段 |

## 参数、续训与支持范围

| 简写（Python / key=value CLI） | 对应 YAML 字段 |
|---|---|
| epochs | trainer.max_epochs |
| batch | loader.batch_size_per_device；predict 时为推理批量 |
| workers | loader.num_workers |
| device | trainer.accelerator：cpu / gpu / auto |
| imgsz | preprocessing.image_size |
| lr0 | optimizer.lr |
| seed / name / project | experiment.seed / name / output_root |
| offline | runtime.offline |

完整字段：CLI 使用 `optimizer.weight_decay=0.02` 或 `--set optimizer.weight_decay=0.02`；Python 使用 `model.train(overrides={"optimizer.weight_decay": 0.02})`。
覆盖仅影响本次调用，不改写 YAML，也不会累积到下一次 train/smoke。val/test 默认沿用最近一次训练实际使用的配置。
未知字段、重复/冲突覆盖、非法参数会报错。epochs 必须大于 warmup_epochs；单轮快速检查请用 smoke。

```python
model = Classifier("configs/flower/flower_resnet18.yaml")
run_dir = model.train(stop_after_epoch=1)
Classifier(run_dir / "config.resolved.yaml").train(resume=run_dir / "checkpoints/last.ckpt")
```

CLI 对应 `stop_after_epoch=1` 和 `resume=.../checkpoints/last.ckpt`。续训要求相同源码、依赖、数据及训练参数，resume 接受受信任的本框架 checkpoint 路径，不接受 `True`。
花卉配置重新调用 train 且不传 resume，会默认从上次权重开始新一轮微调，详细行为见下一节。

## 接着上次权重微调

flower_base.yaml 默认设置：

```yaml
checkpoint:
  finetune_from: last
```

普通 train 会在 `runs/<experiment.name>/*/checkpoints/last.ckpt` 中按修改时间选最新可用文件。
首次没有历史权重时按 YAML 的初始权重策略开始；有历史权重则保留完整主干及已训练分类头，并新建 run。此前的 run 保留。
`--epochs 5` 表示本次再训练 5 轮，优化器、调度器和 epoch 计数重新初始化，可以修改学习率、batch 等。
更换 --name 会进入另一组实验；如需跨实验接续，显式传入 --finetune-from。

```powershell
# 首次使用官方权重，以后从同一实验上次 last.ckpt 继续微调
.venv\Scripts\python.exe -X utf8 examples/train.py --model resnet18 --epochs 5 --lr 0.0001
# 主动按 YAML 初始权重重新开始，不使用历史训练结果
.venv\Scripts\python.exe -X utf8 examples/train.py --model resnet18 --fresh
# 选定某次受信任的运行目录，也支持直接传 .ckpt
.venv\Scripts\python.exe -X utf8 examples/train.py --model resnet18 --finetune-from runs/flower_resnet18/<run_id> --epochs 5

# 通用 CLI 的等价方式
.venv\Scripts\cls.exe train model=configs/flower/flower_resnet18.yaml epochs=5
.venv\Scripts\cls.exe train model=configs/flower/flower_resnet18.yaml fresh=true
```

```python
model = Classifier("configs/flower/flower_resnet18.yaml")
model.train(epochs=5, lr0=0.0001)  # 根据 base 自动选上次
model.train(finetune_from="last", epochs=5)  # 显式要求自动选择
model.train(fresh=True)  # 使用 YAML 的初始权重设置
```

仅接受本框架生成的受信任 run/checkpoint；模型架构、任务和类别顺序必须匹配，否则报错，不跳到旧权重或随机主干。
完整模型参数被提取为 `weights/finetune-<来源SHA256>.safetensors`，复用前校验；日志及新 run 的 provenance.json 记录来源。
预处理默认继承上次权重配套设置；若显式选择 preprocessing.source=explicit，则由当前配置负责提供预处理。
只加载模型参数时允许来自旧源码版本；它不提供“与不中断训练完全等价”的保证。

严格断点恢复仍使用 --resume，且与 --fresh/--finetune-from 互斥。对于新一轮微调的断点，请选择该 run 保存的 config.resolved.yaml，再传 last.ckpt，以保留实际初始化权重配置。
prepare 仍准备 YAML 的初始权重；smoke 用于独立的初始模型检查，不自动接续历史训练。

目前推理只支持本地图片/目录与 CPU，export 只支持 ONNX（需 export extra）。视频、摄像头、URL 输入、TensorRT 及任意 Ultralytics 参数不在此接口范围，不会静默忽略。
本轮修改了核心 Python 源码，之前版本的 checkpoint 无法在新版严格续训；已保存的标准 bundle 仍可用于推理。

## 日志

`flower_base.yaml` 已启用终端和 UTF-8 文件日志：

```yaml
logging:
  level: INFO  # DEBUG / INFO / WARNING / ERROR
  console: true
  file: true
  directory: ../../logs  # 相对于 configs/flower/ 中的模型配置，指向项目根目录 logs/
  tensorboard: false
```

训练、短流程检查、数据/权重准备、验证、测试、推理和导出每次使用独立日志文件，例如：

```text
logs/20260928-140000-train-a1b2c3d4.log
```

日志记录操作开始/结束、耗时、数据指纹、权重缓存命中或加载、训练运行目录、每轮学习率及监控指标、最佳 checkpoint、bundle 路径和异常堆栈。
也会收集操作期间 Lightning 发出的日志；这是结构化事件日志，不是终端逐字录屏，动态进度条和直接写 stderr 的下载进度不复制进文件。
终端中的本框架日志写入 stderr，CLI JSON 结果仍写 stdout。日志级别同时筛选本框架终端和文件输出；Lightning 自己的终端输出由其控制。

Python 中 `model.last_log` 返回最近一次操作的日志路径。训练成功或正常暂停后，run 内的 `operation_logs.jsonl` 记录对应日志位置；原有 `metrics.jsonl`、`events.jsonl`、`status.json` 继续保存。
日志在数据/权重准备前打开，因此该阶段失败也会保存堆栈；模型配置解析、参数校验等尚未进入操作的错误直接报给调用方。
续训会新建一个操作日志，保留之前的记录；修改日志级别/开关不影响同一源码下的严格续训契约。

```powershell
.venv\Scripts\cls.exe train model=configs/flower/flower_resnet18.yaml logging.level=DEBUG logging.console=false
```

```python
model = Classifier("configs/flower/flower_resnet18.yaml")
model.train(overrides={"logging.level": "DEBUG", "logging.directory": "../../logs"})
print(model.last_log)

# 独立 bundle 没有 YAML 时，也可以在构造对象时显式设置日志
predictor = Classifier("path/to/bundle", log_config={"level": "INFO", "console": False})
```

每次操作结束都会关闭文件并移除临时 handler，不替换宿主应用原有 handler。日志文件使用时间与随机 ID 区分，长期实验可按需归档 `logs/`。
花卉模型继承 base 中的 `../../logs`，统一写入项目根目录；其他配置未指定 directory 时，使用 output_root 同级的 `logs/<experiment.name>/`。独立 bundle 无配置时使用当前目录下的 `logs/inference/`。
底层 `engine.train` 等内部函数不主动配置日志 handler；应用入口建议使用统一的 `Classifier` 或 CLI。
