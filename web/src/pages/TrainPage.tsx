import {
  ArrowRight,
  BarChart3,
  Box,
  CheckCircle2,
  Layers3,
  LoaderCircle,
  Play,
  RefreshCw,
  Settings2,
} from "lucide-react";
import { descriptions, modelNames } from "../shared";
import { useWorkbench } from "../WorkbenchContext";

export default function TrainPage() {
  const {
    catalog,
    settings,
    experimentName,
    setExperimentName,
    runs,
    form,
    selected,
    online,
    busy,
    advanced,
    setAdvanced,
    dataset,
    run,
    update,
    refreshCatalog,
    perform,
    launch,
  } = useWorkbench();
  return (
    <>
      <div className="stats">
        <div>
          <span>可选模型</span>
          <strong>
            {catalog.models.length}
            <small>种网络架构</small>
          </strong>
          <Layers3 />
        </div>
        <div>
          <span>可用数据集</span>
          <strong>
            {catalog.datasets.length}
            <small>ImageFolder</small>
          </strong>
          <Box />
        </div>
        <div>
          <span>历史实验</span>
          <strong>
            {runs.length}
            <small>已生成模型</small>
          </strong>
          <BarChart3 />
        </div>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          perform(() =>
            launch("/train", {
              ...form,
              experiment_name: experimentName || null,
            }),
          );
        }}
      >
        <div className="train-grid">
          <div className="main-stack">
            <section className="panel">
              <div className="section-title">
                <span className="step">01</span>
                <h2>选择网络模型</h2>
                <span className="hint">官方预训练权重</span>
              </div>
              <div className="models">
                {catalog.models.map((m) => (
                  <button
                    type="button"
                    key={m.id}
                    className={`model ${form.model_id === m.id ? "selected" : ""}`}
                    onClick={() => update("model_id", m.id)}
                  >
                    <div>
                      <Layers3 size={19} />
                      <span className="radio" />
                    </div>
                    <strong>{modelNames[m.id] || m.name}</strong>
                    <small>{descriptions[m.id] || m.name}</small>
                    <span className="provider">{m.provider}</span>
                  </button>
                ))}
              </div>
            </section>
            <section className="panel">
              <div className="section-title">
                <span className="step">02</span>
                <h2>选择数据集</h2>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="刷新数据集"
                  onClick={() => perform(refreshCatalog)}
                >
                  <RefreshCw size={16} />
                </button>
              </div>
              <label>
                数据目录
                <select
                  value={form.dataset_id}
                  onChange={(e) => update("dataset_id", e.target.value)}
                  required
                >
                  <option value="" disabled>
                    选择已划分的数据集
                  </option>
                  {catalog.datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      data/{d.id}
                    </option>
                  ))}
                </select>
              </label>
              {dataset ? (
                <>
                  <div className="split-stats">
                    {[
                      ["train", "训练集"],
                      ["val", "验证集"],
                      ["test", "测试集"],
                    ].map(([key, label]) => (
                      <div key={key}>
                        <span>{label}</span>
                        <strong>
                          {dataset.counts[key].toLocaleString()}{" "}
                          <small>张</small>
                        </strong>
                      </div>
                    ))}
                  </div>
                  <div className="classes">
                    {dataset.classes.map((c) => (
                      <span key={c}>{c}</span>
                    ))}
                    <small>
                      {dataset.classes.length} 个类别 · 自动适配分类头
                    </small>
                  </div>
                </>
              ) : (
                <div className="notice">
                  将数据放入 data/数据集名/train/类别名 与
                  val/类别名。花卉数据可运行{" "}
                  <code>uv run python scripts/prepare_flowers.py</code>{" "}
                  自动下载划分，然后刷新。
                </div>
              )}
            </section>
          </div>
          <div className="config-stack">
            <section className="panel">
              <div className="section-title">
                <span className="step">03</span>
                <h2>训练配置</h2>
                <Settings2 size={18} />
              </div>
              <div className="fields">
                <label>
                  训练轮数 <span>Epochs</span>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    required
                    value={form.epochs}
                    onChange={(e) => update("epochs", Number(e.target.value))}
                  />
                </label>
                <label>
                  批次大小 <span>Batch size</span>
                  <input
                    type="number"
                    min="1"
                    max="512"
                    required
                    value={form.batch_size}
                    onChange={(e) =>
                      update("batch_size", Number(e.target.value))
                    }
                  />
                </label>
              </div>
              <label>
                初始权重
                <select
                  value={form.initialization}
                  onChange={(e) => update("initialization", e.target.value)}
                >
                  <option value="last">接着上次权重微调（推荐）</option>
                  <option value="official">从官方预训练权重开始</option>
                </select>
              </label>
              <p className="field-note">
                同一模型没有历史权重时，自动使用官方预训练权重。新任务可重新设置轮数与学习率。
              </p>
              <label>
                计算设备
                <select
                  value={form.device}
                  onChange={(e) => update("device", e.target.value)}
                >
                  <option value="global">跟随全局设置（推荐）</option>
                  <option value="auto">本次自动选择单卡</option>
                  <option value="gpu">GPU · 使用 CUDA</option>
                  <option value="cpu">CPU</option>
                </select>
              </label>
              <p className="field-note">
                {form.device === "global"
                  ? `全局设备：${settings.training.device.toUpperCase()} · ${settings.training.gpu_indices.length ? `GPU ${settings.training.gpu_indices.join("、")}${settings.training.gpu_indices.length > 1 ? "（多卡 DDP）" : ""}` : "自动分配单卡 / CPU"}。可在设置页面修改。`
                  : "本次选择覆盖全局设置；GPU 资源不足时排队等待。"}
              </p>
              {form.device === "gpu" && (
                <label>
                  GPU 设备
                  <select
                    value={form.gpu_index ?? ""}
                    onChange={(e) =>
                      update(
                        "gpu_index",
                        e.target.value === "" ? null : Number(e.target.value),
                      )
                    }
                  >
                    <option value="">调度器自动分配</option>
                    {catalog.gpus?.map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        GPU {gpu.index} · {gpu.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="field-row">
                <label>
                  任务优先级
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={form.priority}
                    onChange={(e) => update("priority", Number(e.target.value))}
                  />
                </label>
                <label>
                  运行时限（分钟）
                  <input
                    type="number"
                    min="0"
                    max="10080"
                    value={form.timeout_minutes}
                    onChange={(e) =>
                      update("timeout_minutes", Number(e.target.value))
                    }
                  />
                </label>
              </div>
              <p className="field-note">
                优先级越大越早启动；时限 0 表示不限。复制运行会创建新的任务。
              </p>
              <button
                className="advanced"
                type="button"
                onClick={() => setAdvanced(!advanced)}
              >
                <Settings2 size={15} />
                高级参数 <span>{advanced ? "−" : "+"}</span>
              </button>
              {advanced && (
                <div className="advanced-fields">
                  <label>
                    实验名称（可选）
                    <input
                      value={experimentName}
                      maxLength={64}
                      pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                      placeholder="例如 resnet18_lr001"
                      onChange={(e) => setExperimentName(e.target.value)}
                    />
                  </label>
                  <p className="field-note">
                    同名实验依次训练。填写不同名称，可以并行比较同一模型的参数。
                  </p>
                  <label>
                    学习率
                    <input
                      type="number"
                      min="0.00000001"
                      max="1"
                      step="any"
                      required
                      value={form.learning_rate}
                      onChange={(e) =>
                        update("learning_rate", Number(e.target.value))
                      }
                    />
                  </label>
                  <div className="fields">
                    <label>
                      数据进程数
                      <input
                        type="number"
                        min="0"
                        max="16"
                        required
                        value={form.num_workers}
                        onChange={(e) =>
                          update("num_workers", Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      CPU 线程数
                      <input
                        type="number"
                        min="1"
                        max="32"
                        required
                        value={form.cpu_threads}
                        onChange={(e) =>
                          update("cpu_threads", Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      随机种子
                      <input
                        type="number"
                        min="0"
                        max="2147483647"
                        value={form.seed}
                        onChange={(e) => update("seed", Number(e.target.value))}
                      />
                    </label>
                  </div>
                  <label>
                    微调方式
                    <select
                      value={form.training_mode}
                      onChange={(e) => update("training_mode", e.target.value)}
                    >
                      <option value="full_finetune">训练全部参数</option>
                      <option value="linear_probe">只训练分类头</option>
                    </select>
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={form.offline}
                      onChange={(e) => update("offline", e.target.checked)}
                    />
                    离线模式（仅使用已有权重）
                  </label>
                </div>
              )}
              <button
                className="button primary full"
                type="submit"
                disabled={busy || !online || !dataset}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <Play size={17} />
                )}
                开始训练
                <ArrowRight size={17} />
              </button>
              <p className="submit-note">
                任务后台运行，关闭页面不会停止训练。
              </p>
            </section>
            <div className="tip">
              <CheckCircle2 size={20} />
              <div>
                <strong>配置一次，专注实验</strong>
                <p>
                  自动匹配输入尺寸与归一化参数，保存最佳权重、训练曲线和 HTML
                  报告。权重缓存在 weights/。
                </p>
              </div>
            </div>
          </div>
        </div>
      </form>
    </>
  );
}
