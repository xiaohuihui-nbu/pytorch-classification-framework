import {
  Check,
  Layers3,
  LoaderCircle,
  Palette,
  RefreshCw,
  Save,
  Settings2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { defaultSettings, post, type Catalog, type WebSettings } from "./api";

const themes = [
  {
    id: "cyber",
    name: "科技渐变",
    color: "linear-gradient(135deg,#29d6e9,#7863ff)",
    background: "#111b35",
    caption: "蓝紫 · 未来",
  },
  {
    id: "forest",
    name: "森林绿",
    color: "linear-gradient(135deg,#29dbac,#49c8f4)",
    background: "#102c2d",
    caption: "极光青绿 · 数字森林",
  },
  {
    id: "ocean",
    name: "海洋蓝",
    color: "linear-gradient(135deg,#36d3eb,#668bff)",
    background: "#11243e",
    caption: "冰蓝电光 · 深海科技",
  },
  {
    id: "violet",
    name: "鸢尾紫",
    color: "linear-gradient(135deg,#a38aff,#f18ccf)",
    background: "#241735",
    caption: "星云紫粉 · 未来空间",
  },
  {
    id: "amber",
    name: "琥珀橙",
    color: "linear-gradient(135deg,#f6c765,#f58a83)",
    background: "#30221e",
    caption: "琥珀珊瑚 · 能量光谱",
  },
] as const;

export default function SettingsPage({
  settings,
  online,
  gpus = [],
  onSave,
}: {
  settings: WebSettings;
  online: boolean;
  gpus?: Catalog["gpus"];
  onSave: (value: WebSettings) => void;
}) {
  const [draft, setDraft] = useState({
    ...settings,
    training: settings.training || defaultSettings.training,
    resources: settings.resources || defaultSettings.resources,
    inference: settings.inference || defaultSettings.inference,
  });
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    if (!dirty)
      setDraft({
        ...settings,
        training: settings.training || defaultSettings.training,
        resources: settings.resources || defaultSettings.resources,
        inference: settings.inference || defaultSettings.inference,
      });
  }, [settings, dirty]);
  const change = (value: WebSettings) => {
    setDraft(value);
    setDirty(true);
    setNotice("");
  };
  async function save(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await post<WebSettings>("/settings", draft);
      onSave(result);
      setDirty(false);
      setNotice("全局设置已保存，立即生效");
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="settings-page" onSubmit={save}>
      <div className="settings-intro">
        <Settings2 size={21} />
        <div>
          <strong>你的工作空间，你的偏好</strong>
          <p>设置对所有页面和新打开的浏览器标签生效，重启后保留。</p>
        </div>
        <span>GLOBAL SETTINGS</span>
      </div>
      <section className="panel settings-section">
        <div className="section-title">
          <h2>全局训练设备</h2>
        </div>
        <label>
          默认计算设备
          <select
            value={draft.training.device}
            onChange={(e) =>
              change({
                ...draft,
                training: {
                  device: e.target.value as WebSettings["training"]["device"],
                  gpu_indices:
                    e.target.value === "cpu" ? [] : draft.training.gpu_indices,
                },
              })
            }
          >
            <option value="auto">自动选择 · 优先 GPU</option>
            <option value="gpu">GPU · 使用 CUDA</option>
            <option value="cpu">CPU</option>
          </select>
        </label>
        {draft.training.device !== "cpu" && (
          <fieldset>
            <legend>训练使用的显卡</legend>
            <label className="check">
              <input
                type="checkbox"
                checked={draft.training.gpu_indices.length === 0}
                onChange={() =>
                  change({
                    ...draft,
                    training: { ...draft.training, gpu_indices: [] },
                  })
                }
              />
              自动分配一张空闲显卡
            </label>
            {gpus.map((gpu) => (
              <label className="check" key={gpu.index}>
                <input
                  type="checkbox"
                  checked={draft.training.gpu_indices.includes(gpu.index)}
                  onChange={(e) =>
                    change({
                      ...draft,
                      training: {
                        ...draft.training,
                        gpu_indices: e.target.checked
                          ? [...draft.training.gpu_indices, gpu.index].sort(
                              (a, b) => a - b,
                            )
                          : draft.training.gpu_indices.filter(
                              (index) => index !== gpu.index,
                            ),
                      },
                    })
                  }
                />
                GPU {gpu.index} · {gpu.name} · 空闲 {gpu.free_gb.toFixed(1)} /{" "}
                {gpu.total_gb.toFixed(1)} GB
              </label>
            ))}
            {draft.training.gpu_indices
              .filter((index) => !gpus.some((gpu) => gpu.index === index))
              .map((index) => (
                <label className="check" key={index}>
                  <input
                    type="checkbox"
                    checked
                    onChange={() =>
                      change({
                        ...draft,
                        training: {
                          ...draft.training,
                          gpu_indices: draft.training.gpu_indices.filter(
                            (id) => id !== index,
                          ),
                        },
                      })
                    }
                  />
                  GPU {index} · 当前不可用，请取消选择或检查服务器
                </label>
              ))}
            {!gpus.length && (
              <p className="field-note">服务器尚未检测到可用的 GPU 设备。</p>
            )}
          </fieldset>
        )}
        <p className="settings-note">
          {draft.training.gpu_indices.length > 1
            ? `多卡 DDP：每个训练任务同时使用 GPU ${draft.training.gpu_indices.join("、")}。总 batch = 每卡 batch × ${draft.training.gpu_indices.length}；仅支持 Linux CUDA。`
            : "勾选一张卡固定使用该卡；勾选多张卡用于同一任务的 DDP 训练。"}
        </p>
        <p className="settings-description">
          新训练默认跟随全局设置。保存后对新提交任务生效，已提交任务保留原设备。
          所选显卡资源不足时排队等待；推理和独立评估仍使用 CPU。
        </p>
      </section>
      <section className="panel settings-section">
        <div className="section-title">
          <h2>资源保护与设备额度</h2>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={draft.resources.enabled}
            onChange={(e) =>
              change({
                ...draft,
                resources: { ...draft.resources, enabled: e.target.checked },
              })
            }
          />
          启用内存与显存启动检查
        </label>
        <p className="settings-description">
          不足时保留排队并显示原因，不终止已有任务。内存预留为估算值，不保证模型峰值占用。
        </p>
        <div className="settings-concurrency">
          {(
            [
              ["max_memory_percent", "内存使用率上限（%）", 20, 99, 1],
              ["min_available_gb", "系统保留内存（GB）", 0, 1024, 0.5],
              ["launch_reserve_gb", "每次启动预留（GB）", 0, 1024, 0.5],
              ["min_gpu_free_gb", "GPU 最低空闲显存（GB）", 0, 1024, 0.5],
              ["max_jobs_per_gpu", "每张 GPU 任务上限（0 不限）", 0, 8, 1],
            ] as const
          ).map(([key, label, min, max, step]) => (
            <label key={key}>
              {label}
              <input
                type="number"
                required
                min={min}
                max={max}
                step={step}
                value={draft.resources[key]}
                onChange={(e) =>
                  change({
                    ...draft,
                    resources: {
                      ...draft.resources,
                      [key]: Number(e.target.value),
                    },
                  })
                }
              />
            </label>
          ))}
        </div>
      </section>
      <section className="panel settings-section">
        <div className="section-title">
          <Layers3 size={19} />
          <h2>任务与并发</h2>
          <span className="hint">训练可按显存调度；推理和评估独立限额</span>
        </div>
        <p className="settings-description">
          训练、推理、评估分别使用独立额度。同名实验依次运行，其他任务可以并行。
        </p>
        <div className="settings-concurrency">
          {(
            [
              ["train", "模型训练", "训练不同实验", 8],
              ["predict", "图片推理", "图片批次独立预测", 8],
              ["auxiliary", "评估与报告", "验证、测试与报告生成", 4],
            ] as const
          ).map(([key, label, hint, max]) => (
            <div className="setting-limit" key={key}>
              <label>
                {label}并发数
                <input
                  type="number"
                  min={key === "train" ? 0 : 1}
                  max={max}
                  required
                  value={draft.concurrency[key]}
                  onChange={(e) =>
                    change({
                      ...draft,
                      concurrency: {
                        ...draft.concurrency,
                        [key]: Number(e.target.value),
                      },
                    })
                  }
                />
              </label>
              <small>
                {hint} · {key === "train" ? "0 表示不限；1" : "1"}–{max}
              </small>
            </div>
          ))}
        </div>
        <div className="settings-note">
          {draft.concurrency.train === 0
            ? "训练任务数量不限；设置每张 GPU 任务上限为 0 后，显存达到启动阈值即可共享显卡运行。"
            : `最多 ${Object.values(draft.concurrency).reduce((a, b) => a + b, 0)} 个任务同时运行。`}
          内存不足或同名实验正在运行时仍会等待。降低额度不会终止现有任务。
        </div>
      </section>
      <section className="panel settings-section">
        <div className="section-title">
          <h2>推理性能</h2>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={draft.inference.resident}
            onChange={(e) =>
              change({
                ...draft,
                inference: { ...draft.inference, resident: e.target.checked },
              })
            }
          />
          复用推理进程和已加载模型
        </label>
        <p className="settings-description">
          预测完成即展示结果，图片和 HTML 报告在后台生成。设置对下一任务生效。
        </p>
        <div className="settings-concurrency">
          {(
            [
              ["cpu_threads", "推理 CPU 线程数", 1, 32],
              ["batch_size", "推理批次大小", 1, 64],
              ["cache_models", "每个进程缓存模型数", 1, 4],
              ["cache_mb", "每个进程模型缓存（MB）", 16, 2048],
              ["idle_seconds", "空闲进程保留（秒）", 30, 3600],
            ] as const
          ).map(([key, label, min, max]) => (
            <label key={key}>
              {label}
              <input
                type="number"
                required
                min={min}
                max={max}
                value={draft.inference[key]}
                onChange={(e) =>
                  change({
                    ...draft,
                    inference: {
                      ...draft.inference,
                      [key]: Number(e.target.value),
                    },
                  })
                }
              />
            </label>
          ))}
        </div>
        <p className="settings-note">
          缓存上限只统计模型参数与缓冲区。运行时和图片另占内存；超过上限的模型仍可预测，每次重新加载。
        </p>
      </section>
      <section className="panel settings-section">
        <div className="section-title">
          <Palette size={19} />
          <h2>外观与主题</h2>
          <span className="hint">全局配色</span>
        </div>
        <div className="theme-options">
          {themes.map((theme) => (
            <button
              key={theme.id}
              type="button"
              aria-label={`${theme.name}主题`}
              aria-pressed={draft.appearance.theme === theme.id}
              className={`theme-option ${draft.appearance.theme === theme.id ? "selected" : ""}`}
              onClick={() =>
                change({
                  ...draft,
                  appearance: { ...draft.appearance, theme: theme.id },
                })
              }
            >
              <div
                className="theme-preview"
                style={{ background: theme.background }}
              >
                <i style={{ background: theme.color }} />
                <div>
                  <b style={{ background: theme.color }} />
                  <span />
                  <span />
                </div>
              </div>
              <div className="theme-caption">
                <strong>{theme.name}</strong>
                {draft.appearance.theme === theme.id && <Check size={15} />}
              </div>
              <small>{theme.caption}</small>
            </button>
          ))}
        </div>
        <div className="appearance-row">
          <div>
            <strong>显示模式</strong>
            <p>选择浅色、深色，或自动跟随系统外观。</p>
          </div>
          <div className="mode-options" role="group" aria-label="显示模式">
            {(
              [
                ["light", "浅色"],
                ["dark", "深色"],
                ["system", "跟随系统"],
              ] as const
            ).map(([mode, name]) => (
              <button
                key={mode}
                type="button"
                aria-pressed={draft.appearance.mode === mode}
                className={draft.appearance.mode === mode ? "selected" : ""}
                onClick={() =>
                  change({
                    ...draft,
                    appearance: { ...draft.appearance, mode },
                  })
                }
              >
                {name}
              </button>
            ))}
          </div>
        </div>
      </section>
      <section className="panel settings-section">
        <div className="section-title">
          <RefreshCw size={19} />
          <h2>状态与刷新</h2>
        </div>
        <div className="refresh-setting">
          <div>
            <strong>自动刷新间隔</strong>
            <p>任务列表、日志、进度和底部状态栏共用此间隔。</p>
          </div>
          <label>
            刷新间隔（秒）
            <input
              type="number"
              min={1}
              max={30}
              required
              value={draft.refresh_interval_seconds}
              onChange={(e) =>
                change({
                  ...draft,
                  refresh_interval_seconds: Number(e.target.value),
                })
              }
            />
          </label>
        </div>
      </section>
      <div className="settings-save">
        <span>{dirty ? "有尚未保存的修改" : "设置已同步"}</span>
        <button
          className="button secondary"
          type="button"
          disabled={busy}
          onClick={() => change(structuredClone(defaultSettings))}
        >
          恢复默认值
        </button>
        <button
          className="button primary"
          type="submit"
          disabled={busy || !online}
        >
          {busy ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Save size={17} />
          )}
          保存全局设置
        </button>
      </div>
      {notice && (
        <div className="saved-notice" role="status">
          {notice}
        </div>
      )}
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
    </form>
  );
}
