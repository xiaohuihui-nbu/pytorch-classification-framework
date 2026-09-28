# 配置与数据协议

训练、验证、推理与导出的统一入口及日志示例见[运行指南](RUNNING.md)。

## 配置解析

Hydra `defaults` 组合后应用 CLI `--set` 覆盖，再进行 Pydantic 校验。所有根配置中的文件路径相对于该配置文件所在目录解析；manifest 内图片路径相对于 `dataset.root`。不会改变当前工作目录。

主配置生成：`cls init --recipe custom_resnet18 --output experiment.yaml`。完整字段可查看 `config.schema.json`，或执行 `cls schema --output schema.json`。

原始配置和展开配置保存在 run 中。`auto` 的 accelerator/precision/类别数/输入大小会解析为具体值。CPU 默认 fp32；显式要求 GPU 但不可用时失败。`num_workers=0` 会关闭 persistent workers。

花卉多模型训练可直接选择配置文件：`.\scripts\run-flower.ps1 -Config configs/flower/flower_resnet50.yaml`。
替换 YAML 路径即可切换网络；脚本使用项目虚拟环境、限制 BLAS 线程，并固定覆盖 `loader.num_workers=0`。
模型、学习率、批量大小和训练轮数等从 YAML 读取，详细示例见[多模型配置说明](PRETRAINED_MODELS.md)。
`configs/flower/flower_base.yaml` 集中保存公共设置，各模型通过 `defaults: [flower_base, _self_]` 继承，只需指定实验名、provider 和模型名。
公共模板不能单独训练；模型配置放在同一 `configs/flower/` 目录，日常调参在模型文件内覆盖，无需修改 base。

## 任务与损失

| task.type | 标签形式 | loss.name | 说明 |
|---|---|---|---|
| multiclass | 单个类别索引 | cross_entropy | 二分类默认也可按两类处理；支持 label_smoothing/class_weight |
| multiclass | 单个类别索引，经 batch 增强生成软标签 | soft_target_ce | 需同时启用 mixup_alpha 或 cutmix_alpha |
| multiclass | 单个类别索引 | softmax_focal | gamma 调难样本权重；类别加权用 class_weight，禁止 scalar alpha |
| binary | 两类原始标签，转成 `[N,1]` 浮点标签 | bce_with_logits / sigmoid_focal | 第二个类别是正类，只输出一个 logit |
| multilabel | JSONL labels 数组，转为 multi-hot | bce_with_logits / sigmoid_focal | 类别独立，阈值默认 0.5 |

`class_weight: balanced` 只根据训练集计算，训练集中缺类时失败。BCE `pos_weight` 是每个输出正例的权重，不能当作 softmax 类别权重。softmax focal 与 sigmoid focal 明确区分。

多标签空数组表示所有标签均已确认阴性；未知/部分标注不应填空数组，本版不提供 label mask。多标签数据建议显式提供 classes 文件，确保列顺序稳定。

Mixup/CutMix 只支持单标签 soft-target CE；标签平滑在 batch 标签生成处应用一次。奇数个样本的最后 batch 使用平滑 one-hot，而非丢弃最后一张样本。

## 模型与权重

`model.provider` 支持 timm、torchvision、builtin 或注册的插件 provider。timm 模型名来自安装版本的注册表，目录中列出的模型是当前检查过的代表项。

`weights.source`：

默认是 `provider_default`，`preprocessing.source` 默认 `model_weights`，`image_size` 默认 `auto`。
执行 train 会优先加载本地预训练初始化缓存，缺失时自动下载对应发布方权重；无需先执行 prepare。
默认允许联网；显式 `runtime.offline: true` 时缺失权重直接报错，不下载、不随机回退。

- `none`：从零初始化，必须显式选择预处理。
- `provider_default`：使用上游默认预训练权重，下载后将该模型/类别数/seed 对应的完整初始状态缓存在 `runtime.weights_dir` 中。记录 provider 元信息、权重文件 SHA256 和初始参数摘要；不会因下载失败而改成随机初始化。
- `local`：加载完整兼容 state_dict 或 safetensors，必须显式填写预处理；不会自动过滤不匹配的分类头。

`runtime.weights_dir` 默认 `weights`，路径相对于配置文件所在目录；本项目 `configs/flower/` 内的配方均指定 `../../weights`，统一写入项目根目录。
原始权重与五分类初始化权重都在这里，数据准备记录仍在 `runtime.cache_dir`。
显式设 `weights_dir: null` 可使用旧的 cache_dir 位置；旧目录中的权重不会被自动删除。

`preprocessing.source: model_weights` 会读取权重对应 mean/std/interpolation/crop_pct/resize_size。image_size 设置为 auto 时沿用权重输入尺寸；显式数值可覆盖，最终值落盘。固定尺寸 Transformer 对不合法大小会在真实前向检查中报错。
`resize_size` 记录验证/预测短边缩放尺寸，避免各 provider 的取整规则差异；显式预处理未设置时沿用 `round(image_size/crop_pct)`。
timm 权重当前仅接受方形输入及 center crop，不支持的权重预处理配置直接报错。

使用一通道输入时同时设置 input_channels=1、color_mode=L，以及单元素 mean/std。torchvision 适配器目前只接受三通道；timm 单通道预训练的具体模型仍需单独验证。

`training_mode: linear_probe` 冻结主干参数，并在训练每轮把主干置为 eval，只训练分类头，避免 BatchNorm 统计漂移。

## 数据适配器

ImageFolder 从训练目录推断类别顺序，manifest 从训练行推断类别集合；也可以显式提供固定的 classes JSON。manifest 的 split 必须为 train/val/test，sample_id 缺省时使用相对路径。

自定义数据严格检查文件内容 SHA256、跨 split 重复、group_id 泄漏、样本 ID 重复和图像解码。`integrity: paths` 只记录路径及文件大小，不能作为内容未变化的强证明。

torchvision 的 MNIST/FashionMNIST/CIFAR 从官方训练集分层划 val，官方 test 保留。Pets 从 trainval 划 val；Flowers102 直接使用官方三个 split。官方基准数据中的内容重复会在 `official_cross_split_duplicate_hashes` 中报告，保留官方协议，不自行改写官方 split。

`limit_per_split` 仅用于工程烟测。torchvision 子集轮流选取各类样本；自定义数据取各 split 前 N 条，可能造成类别不均衡，报告会显示逐类 support。移除限制即形成新的数据指纹，不能直接严格续训原子集实验。

prepare 会把归一化后的 manifest、classes 和指纹保存到 cache/prepared。运行 train/test 时仍会重新检查本地文件，避免用户修改数据后误用过期指纹。这个版本优先保证可核对性，超大数据增量扫描后续优化。

## 权重文件扩展名

默认推理文件为 `model.safetensors`，完整训练恢复文件为 `.ckpt`。PyTorch 不要求必须使用 `.pt` 后缀；`.pt` 通常只是 `torch.save` 的惯例。

本框架本地加载支持 `torch.save(model.state_dict(), "model.pt")` 形式的参数文件，使用 `weights.source: local` 和 `weights.path`。它不支持把任意 pickle 的完整 Python 模型对象当成 state_dict 自动加载。`.pt` 权重往返加载有专门的零容差测试。

如果其他 PyTorch 工程需要 `.pt`，可自行将 bundle 权重无损转为 state_dict；类别和预处理仍需保留：

```python
import torch
from safetensors.torch import load_file

state_dict = load_file("runs/<run_id>/bundle/model.safetensors")
torch.save(state_dict, "model.pt")
```

这里转换的是参数文件容器，不改变参数精度或模型网络结构。完整续训请仍使用本框架的 checkpoint，而不是只加载这个 `.pt`。

## 离线运行

先在线执行 `prepare`，确保数据与所选预训练权重缓存齐全；随后设置 runtime.offline=true。此时 torchvision 不执行下载，provider_default 只从框架已准备的初始权重缓存读取；缺失即报错。缓存 key 包含模型、输出维度、输入通道和随机种子，修改 seed 后需要重新准备对应初始化。

使用 `weights.source: none` 或兼容的本地权重时可从首次运行就保持离线。

## 训练与事件

训练和验证默认显示进度条，包含 epoch、batch 进度、速度和预计剩余时间。
可通过 `--set trainer.enable_progress_bar=false` 关闭；该显示选项不参与训练恢复参数契约。
数据准备完成后才进入训练进度条。当前版本增加此选项修改了核心源码，旧源码版本的 checkpoint 仍会被源码摘要检查拒绝严格续训。

每个实验单独 run_id。metrics.jsonl 每轮记录完整验证指标；events.jsonl 记录 epoch/global_step/学习率；status.json 是当前状态。val_metrics.json 表示最后一次验证，不一定是 best checkpoint 的指标；最终部署模型由 status 中 best_checkpoint 指向，独立 test 对它评估。

优化器更新步数按真实 batch 和 accumulate_grad_batches 计算。warmup_cosine 使用 update 步进。class_weight、有效精度、类别数、预处理等解析结果均保存在 config.resolved.yaml。

控制台中的 Lightning DataLoader worker 数建议只是性能提示；Windows 功能验收使用 num_workers=0。不要把 CPU 上提示的几十个 worker 直接套到真实项目。

## 评估与导出

训练过程中只读取 val；test 命令才读取测试 split，并要求类别表及完整数据指纹与 bundle 一致。新外部测试集的映射策略尚未实现，需要独立设计而非关闭检查。

多分类 Macro-F1 使用完整类别表和 zero_division=0；多标签 AP 在某类无正样本时置 null，mAP 排除该类并记录有效类数。AUC 缺正/负样本时置 null。二分类正类固定为 classes[1]。

ONNX 支持固定高宽和动态 batch，导出后检查模型结构，并在 batch=1/3 上对比 PyTorch 数值（rtol=1e-3、atol=1e-4）。模型旁边的 JSON 保存类别、预处理、阈值、SHA256 与误差；不将文件生成成功单独当作导出通过。

## Python 插件

从 `clsframework.registry` 导入 register，通过 `register('model'|'dataset'|'loss', name)` 注册 factory。

- 模型 factory 接收 name/in_channels/num_classes，返回 nn.Module。
- 数据 factory 接收 Config，返回 PreparedData；应复用或实现同等级的数据完整性检查。
- 损失 factory 接收 cfg/data，返回 nn.Module，自行拒绝不支持的标签协议。

配置 runtime.plugins 显式加载插件。携带插件名的 bundle 在推理时需要 `--allow-plugins`，且目标环境已安装相应模块。普通 YAML 不支持任意 `_target_` 代码实例化。
