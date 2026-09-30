import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  Columns3,
  Copy,
  Eye,
  LayoutGrid,
  List,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  post,
  type Catalog,
  type Job,
  type JobPage,
  type Run,
} from "./api";

const states: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};
const kinds: Record<string, string> = {
  train: "训练",
  predict: "推理",
  evaluate: "评估",
  report: "报告",
};
type View = "table" | "cards" | "board";
const views = [
  { id: "table", label: "表格视图", icon: List },
  { id: "cards", label: "卡片视图", icon: LayoutGrid },
  { id: "board", label: "看板视图", icon: Columns3 },
] as const;

export default function TaskManager({
  jobs,
  catalog,
  runs,
  onSelect,
  onOpen,
  onRefresh,
  onNew,
}: {
  jobs: Job[];
  catalog: Catalog;
  runs: Run[];
  onSelect: (id: string) => void;
  onOpen: (id: string) => void;
  onRefresh: () => Promise<void>;
  onNew: (page: string) => void;
}) {
  const [view, setView] = useState<View>(() => {
    const saved = localStorage.getItem("cls.jobs.view");
    return saved === "cards" || saved === "board" ? saved : "table";
  });
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [status, setStatus] = useState("");
  const [order, setOrder] = useState("newest");
  const [page, setPage] = useState(1);
  const [checked, setChecked] = useState<string[]>([]);
  const [edit, setEdit] = useState<Job | null>(null);
  const [title, setTitle] = useState("");
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [changeParameters, setChangeParameters] = useState(false);
  const [deleting, setDeleting] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const filtered = useMemo(
    () =>
      jobs
        .filter(
          (j) =>
            (!query ||
              `${j.title} ${j.id} ${j.run_id || ""}`
                .toLowerCase()
                .includes(query.toLowerCase())) &&
            (!kind || j.kind === kind) &&
            (!status || j.status === status),
        )
        .sort((a, b) =>
          order === "name"
            ? a.title.localeCompare(b.title)
            : (order === "oldest" ? 1 : -1) *
              a.created_at.localeCompare(b.created_at),
        ),
    [jobs, query, kind, status, order],
  );
  const pageQuery = useQuery({
    queryKey: ["job-page", query, kind, status, order, page, view],
    queryFn: ({ signal }) =>
      api<JobPage>(
        `/jobs/page?${new URLSearchParams({ query, kind, status, order, limit: String(view === "board" ? 200 : 12), offset: String(view === "board" ? 0 : (page - 1) * 12) })}`,
        { signal },
      ),
    placeholderData: keepPreviousData,
    refetchInterval: 15000,
  });
  const total = pageQuery.data?.total ?? filtered.length;
  const pages = Math.max(1, Math.ceil(total / 12));
  const currentPage = Math.min(page, pages);
  const shown =
    pageQuery.data?.items ??
    (view === "board"
      ? filtered.slice(0, 200)
      : filtered.slice((currentPage - 1) * 12, currentPage * 12));
  useEffect(() => {
    if (page > pages) setPage(pages);
  }, [page, pages]);
  useEffect(() => {
    setPage(1);
  }, [query, kind, status, order, view]);
  useEffect(() => {
    const ids = new Set(jobs.map((j) => j.id));
    setChecked((previous) => previous.filter((id) => ids.has(id)));
  }, [jobs]);
  useEffect(() => {
    if (edit || deleting.length || creating) dialog.current?.showModal();
  }, [edit, deleting, creating]);
  function close() {
    setEdit(null);
    setDeleting([]);
    setCreating(false);
  }
  function chooseView(next: View) {
    setView(next);
    localStorage.setItem("cls.jobs.view", next);
  }
  function toggle(id: string) {
    setChecked((old) =>
      old.includes(id)
        ? old.filter((value) => value !== id)
        : [...old, id].slice(0, 200),
    );
  }
  async function action(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await fn();
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }
  async function openEditor(job: Job) {
    await action(async () => {
      const detail = await api<Job>(`/jobs/${job.id}`);
      setTitle(detail.title);
      setParameters(detail.parameters || {});
      setChangeParameters(false);
      setEdit(detail);
    });
  }
  async function clone(job: Job) {
    await action(async () => {
      const copy = await post<Job>(`/jobs/${job.id}/clone`, {});
      await onRefresh();
      onSelect(copy.id);
      setNotice("已创建副本并加入执行队列");
    });
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!edit) return;
    await action(async () => {
      await api(`/jobs/${edit.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          ...(changeParameters ? { parameters } : {}),
        }),
      });
      close();
      await onRefresh();
      onSelect(edit.id);
      setNotice("任务已更新");
    });
  }
  async function remove() {
    await action(async () => {
      const result = await post<{ deleted: string[] }>("/jobs/batch-delete", {
        job_ids: deleting,
        stop_running: true,
      });
      close();
      setChecked((old) => old.filter((id) => !result.deleted.includes(id)));
      await onRefresh();
      setNotice(
        `已删除 ${result.deleted.length} 条任务记录，训练产物和日志已保留`,
      );
    });
  }
  const setParameter = (key: string, value: unknown) =>
    setParameters((old) => ({ ...old, [key]: value }));
  const tools = (job: Job) => (
    <div className="task-row-actions">
      <button
        aria-label={`查看 ${job.title}`}
        title="查看详情"
        onClick={() => onOpen(job.id)}
      >
        <Eye size={15} />
      </button>
      <button
        aria-label={`编辑 ${job.title}`}
        title="编辑任务"
        disabled={busy}
        onClick={() => openEditor(job)}
      >
        <Pencil size={15} />
      </button>
      <button
        aria-label={`复制并运行 ${job.title}`}
        title="复制并运行"
        disabled={busy}
        onClick={() => clone(job)}
      >
        <Copy size={15} />
      </button>
      <button
        aria-label={`删除 ${job.title}`}
        title="删除记录"
        disabled={busy}
        onClick={() => {
          setError("");
          setDeleting([job.id]);
        }}
      >
        <Trash2 size={15} />
      </button>
    </div>
  );
  const selection = (job: Job) => (
    <input
      type="checkbox"
      aria-label={`选择任务 ${job.title}`}
      checked={checked.includes(job.id)}
      onChange={() => toggle(job.id)}
    />
  );
  const badge = (job: Job) => (
    <span className={`badge ${job.status}`}>
      <i />
      {states[job.status]}
    </span>
  );
  const card = (job: Job) => (
    <article className="task-card" key={job.id}>
      <div className="task-card-top">
        {selection(job)}
        <span>{kinds[job.kind]}</span>
        {badge(job)}
      </div>
      <button className="task-name" onClick={() => onOpen(job.id)}>
        {job.title}
      </button>
      <p>
        {job.id.slice(0, 8)} ·{" "}
        {new Date(job.created_at).toLocaleString("zh-CN")}
      </p>
      {job.error && (
        <p className="task-error-summary" title={job.error}>
          {job.error}
        </p>
      )}
      {job.blocked_reason && <p className="field-note">{job.blocked_reason}</p>}
      {tools(job)}
    </article>
  );
  const numericFields = [
    ["epochs", "训练轮数", 1, 1000, 1],
    ["batch_size", "批次大小", 1, 512, 1],
    ["learning_rate", "学习率", 0.00000001, 1, "any"],
    ["num_workers", "数据进程数", 0, 16, 1],
    ["cpu_threads", "CPU 线程数", 1, 32, 1],
    ["seed", "随机种子", 0, 2147483647, 1],
  ] as const;
  return (
    <section className="panel task-manager">
      <div className="task-manager-heading">
        <div>
          <h2>任务管理</h2>
          <p>管理实验的完整生命周期 · {jobs.length} 项记录</p>
        </div>
        <button
          className="button primary"
          onClick={() => {
            setError("");
            setCreating(true);
          }}
        >
          <Plus size={17} />
          新建任务
        </button>
      </div>
      <div className="task-toolbar">
        <label className="task-search">
          <Search size={16} />
          <input
            aria-label="搜索任务"
            placeholder="搜索任务名称、ID 或实验…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <select
          aria-label="任务类型"
          value={kind}
          onChange={(e) => setKind(e.target.value)}
        >
          <option value="">全部类型</option>
          {Object.entries(kinds).map(([key, name]) => (
            <option key={key} value={key}>
              {name}
            </option>
          ))}
        </select>
        <select
          aria-label="任务状态"
          value={status}
          onChange={(e) => setStatus(e.target.value)}
        >
          <option value="">全部状态</option>
          {Object.entries(states).map(([key, name]) => (
            <option key={key} value={key}>
              {name}
            </option>
          ))}
        </select>
        <select
          aria-label="任务排序"
          value={order}
          onChange={(e) => setOrder(e.target.value)}
        >
          <option value="newest">最新创建</option>
          <option value="oldest">最早创建</option>
          <option value="name">名称排序</option>
        </select>
        <div className="task-view-switch">
          {views.map((v) => (
            <button
              key={v.id}
              aria-label={v.label}
              title={v.label}
              aria-pressed={view === v.id}
              className={view === v.id ? "selected" : ""}
              onClick={() => chooseView(v.id)}
            >
              <v.icon size={17} />
            </button>
          ))}
        </div>
      </div>
      <div className="task-bulk">
        <label className="check">
          <input
            type="checkbox"
            checked={
              shown.length > 0 && shown.every((j) => checked.includes(j.id))
            }
            onChange={(e) =>
              setChecked((old) =>
                e.target.checked
                  ? [...new Set([...old, ...shown.map((j) => j.id)])].slice(
                      0,
                      200,
                    )
                  : old.filter((id) => !shown.some((j) => j.id === id)),
              )
            }
          />
          {view === "board" ? "选择已显示任务" : "全选当前页"}
        </label>
        <span>已选 {checked.length} 项</span>
        <button
          className="button danger"
          disabled={!checked.length || busy}
          onClick={() => {
            setError("");
            setDeleting(checked);
          }}
        >
          <Trash2 size={14} />
          批量删除
        </button>
        {checked.length > 0 && (
          <button className="clear-selection" onClick={() => setChecked([])}>
            取消选择
          </button>
        )}
      </div>
      {notice && (
        <p className="saved-notice" role="status">
          {notice}
        </p>
      )}
      {error && !(edit || deleting.length || creating) && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {!shown.length ? (
        <div className="empty">
          <Search size={28} />
          <p>没有符合条件的任务</p>
          <span>调整筛选条件，或新建一次实验。</span>
        </div>
      ) : view === "table" ? (
        <div className="task-table-scroll">
          <table className="task-table">
            <thead>
              <tr>
                <th>选择</th>
                <th>任务名称</th>
                <th>类型</th>
                <th>状态</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((job) => (
                <tr key={job.id}>
                  <td>{selection(job)}</td>
                  <td>
                    <button
                      className="task-name"
                      onClick={() => onOpen(job.id)}
                    >
                      {job.title}
                    </button>
                    <small>
                      {job.id.slice(0, 8)}
                      {job.pid && job.status === "running"
                        ? ` · PID ${job.pid}`
                        : ""}
                    </small>
                  </td>
                  <td>{kinds[job.kind]}</td>
                  <td>{badge(job)}</td>
                  <td>{new Date(job.created_at).toLocaleString("zh-CN")}</td>
                  <td>{tools(job)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : view === "cards" ? (
        <div className="task-cards">{shown.map(card)}</div>
      ) : (
        <div className="task-board">
          {[
            ["queued", "待执行"],
            ["running", "运行中"],
            ["succeeded", "已完成"],
            ["other", "其他状态"],
          ].map(([key, label]) => {
            const lane = shown.filter((j) =>
              key === "other"
                ? ["failed", "cancelled", "interrupted"].includes(j.status)
                : j.status === key,
            );
            return (
              <section key={key} className={`task-lane lane-${key}`}>
                <h3>
                  <i />
                  {label}
                  <span>{lane.length}</span>
                </h3>
                <div>
                  {lane.map(card)}
                  {!lane.length && <p className="lane-empty">暂无任务</p>}
                </div>
              </section>
            );
          })}
        </div>
      )}
      <div className="task-pagination">
        <span>
          匹配 {total} 项
          {view === "board" && total > 200
            ? " · 显示前 200 项，请使用筛选缩小范围"
            : ""}
        </span>
        {view !== "board" && (
          <>
            <button
              disabled={currentPage <= 1}
              onClick={() => setPage(currentPage - 1)}
            >
              上一页
            </button>
            <span>
              {currentPage} / {pages}
            </span>
            <button
              disabled={currentPage >= pages}
              onClick={() => setPage(currentPage + 1)}
            >
              下一页
            </button>
          </>
        )}
      </div>
      {(edit || deleting.length > 0 || creating) && (
        <dialog
          className="task-dialog"
          ref={dialog}
          onCancel={(event) => {
            if (busy) event.preventDefault();
            else close();
          }}
        >
          <div className="task-dialog-heading">
            <h2>
              {edit ? "编辑任务" : creating ? "新建任务" : "删除任务记录"}
            </h2>
            <button aria-label="关闭对话框" disabled={busy} onClick={close}>
              <X size={20} />
            </button>
          </div>
          {creating ? (
            <div className="task-create-options">
              {[
                ["train", "新建训练", "选择模型、数据集与训练参数"],
                ["predict", "新建推理", "上传图片并使用已有模型预测"],
                [
                  "runs",
                  "新建评估或报告",
                  "选择实验，进行验证、测试或生成报告",
                ],
              ].map(([page, label, hint]) => (
                <button
                  key={label}
                  onClick={() => {
                    close();
                    onNew(page);
                  }}
                >
                  <Plus size={18} />
                  <div>
                    <strong>{label}</strong>
                    <small>{hint}</small>
                  </div>
                </button>
              ))}
            </div>
          ) : edit ? (
            <form onSubmit={save}>
              <label>
                任务名称
                <input
                  autoFocus
                  required
                  maxLength={120}
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </label>
              <p className="settings-description">
                {edit.status === "queued"
                  ? "可修改尚未执行的参数。若保存前已开始运行，系统会拒绝参数修改。"
                  : "运行中或已结束的任务可以改名，执行参数保持不变。"}
              </p>
              {edit.status === "queued" && edit.parameters && (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={changeParameters}
                    onChange={(e) => setChangeParameters(e.target.checked)}
                  />
                  修改执行参数
                </label>
              )}
              {changeParameters && (
                <div className="task-edit-fields">
                  <label>
                    任务优先级
                    <input
                      type="number"
                      min="0"
                      max="10"
                      value={Number(parameters.priority || 0)}
                      onChange={(e) =>
                        setParameter("priority", Number(e.target.value))
                      }
                    />
                  </label>
                  <label>
                    运行时限（分钟，0 不限）
                    <input
                      type="number"
                      min="0"
                      max="10080"
                      value={Number(parameters.timeout_minutes || 0)}
                      onChange={(e) =>
                        setParameter("timeout_minutes", Number(e.target.value))
                      }
                    />
                  </label>
                  {edit.kind === "train" ? (
                    <>
                      <label>
                        模型
                        <select
                          value={String(parameters.model_id ?? "")}
                          onChange={(e) =>
                            setParameter("model_id", e.target.value)
                          }
                        >
                          {catalog.models.map((m) => (
                            <option key={m.id} value={m.id}>
                              {m.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        数据集
                        <select
                          value={String(parameters.dataset_id ?? "")}
                          onChange={(e) =>
                            setParameter("dataset_id", e.target.value)
                          }
                        >
                          {catalog.datasets.map((d) => (
                            <option key={d.id} value={d.id}>
                              {d.id}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        实验名称
                        <input
                          value={String(parameters.experiment_name ?? "")}
                          pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                          maxLength={64}
                          onChange={(e) =>
                            setParameter(
                              "experiment_name",
                              e.target.value || null,
                            )
                          }
                        />
                      </label>
                      {numericFields.map(([key, label, min, max, step]) => (
                        <label key={key}>
                          {label}
                          <input
                            required
                            type="number"
                            min={min}
                            max={max}
                            step={step}
                            value={Number(parameters[key])}
                            onChange={(e) =>
                              setParameter(key, Number(e.target.value))
                            }
                          />
                        </label>
                      ))}
                      <label>
                        设备
                        <select
                          value={String(parameters.device)}
                          onChange={(e) =>
                            setParameter("device", e.target.value)
                          }
                        >
                          {["global", "cpu", "gpu", "auto"].map((v) => (
                            <option key={v}>{v}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        初始化
                        <select
                          value={String(parameters.initialization)}
                          onChange={(e) =>
                            setParameter("initialization", e.target.value)
                          }
                        >
                          <option value="last">历史权重微调</option>
                          <option value="official">官方预训练</option>
                        </select>
                      </label>
                      <label>
                        训练方式
                        <select
                          value={String(parameters.training_mode)}
                          onChange={(e) =>
                            setParameter("training_mode", e.target.value)
                          }
                        >
                          <option value="full_finetune">全参数微调</option>
                          <option value="linear_probe">只训练分类头</option>
                        </select>
                      </label>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={Boolean(parameters.offline)}
                          onChange={(e) =>
                            setParameter("offline", e.target.checked)
                          }
                        />
                        离线模式
                      </label>
                    </>
                  ) : (
                    <>
                      <label>
                        模型实验
                        <select
                          required
                          value={String(parameters.run_id ?? "")}
                          onChange={(e) =>
                            setParameter("run_id", e.target.value)
                          }
                        >
                          {runs.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.id}
                            </option>
                          ))}
                        </select>
                      </label>
                      {edit.kind === "predict" && (
                        <label>
                          Top K
                          <input
                            type="number"
                            min={1}
                            max={20}
                            required
                            value={Number(parameters.top_k ?? 5)}
                            onChange={(e) =>
                              setParameter("top_k", Number(e.target.value))
                            }
                          />
                        </label>
                      )}
                      {edit.kind === "evaluate" && (
                        <label>
                          评估集合
                          <select
                            value={String(parameters.split)}
                            onChange={(e) =>
                              setParameter("split", e.target.value)
                            }
                          >
                            <option value="val">验证集</option>
                            <option value="test">测试集</option>
                          </select>
                        </label>
                      )}
                    </>
                  )}
                </div>
              )}
              <details className="task-parameters">
                <summary>查看完整任务参数</summary>
                <pre>{JSON.stringify(parameters, null, 2)}</pre>
              </details>
              <div className="task-dialog-actions">
                <button
                  type="button"
                  className="button secondary"
                  disabled={busy}
                  onClick={close}
                >
                  取消
                </button>
                <button className="button primary" disabled={busy}>
                  保存任务
                </button>
              </div>
            </form>
          ) : (
            <>
              <p>
                将删除 <strong>{deleting.length}</strong> 条任务记录，其中{" "}
                {
                  jobs.filter(
                    (j) => deleting.includes(j.id) && j.status === "running",
                  ).length
                }{" "}
                条正在运行。
              </p>
              <div className="settings-note">
                运行中的任务会先停止。训练权重、图片、报告和日志保留在磁盘，实验结果仍可访问。
              </div>
              <div className="task-dialog-actions">
                <button
                  className="button secondary"
                  disabled={busy}
                  onClick={close}
                >
                  取消
                </button>
                <button
                  className="button danger"
                  disabled={busy}
                  onClick={remove}
                >
                  确认删除
                </button>
              </div>
            </>
          )}
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
        </dialog>
      )}
    </section>
  );
}
