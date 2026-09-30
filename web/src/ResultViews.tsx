import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowRight, Columns2, LayoutGrid, List, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Job, type JobPage } from "./api";

export type ResultView = "detail" | "cards" | "table";
export function useResultView(key: string) {
  const [view, setView] = useState<ResultView>(() => {
    const saved = localStorage.getItem(key);
    return saved === "cards" || saved === "table" ? saved : "detail";
  });
  return [
    view,
    (next: ResultView) => {
      setView(next);
      localStorage.setItem(key, next);
    },
  ] as const;
}
export function ViewSwitch({
  view,
  onChange,
  label,
}: {
  view: ResultView;
  onChange: (view: ResultView) => void;
  label: string;
}) {
  return (
    <div className="result-view-switch" role="group" aria-label={label}>
      {(
        [
          { id: "detail", title: "详细", icon: Columns2 },
          { id: "cards", title: "卡片", icon: LayoutGrid },
          { id: "table", title: "表格", icon: List },
        ] as const
      ).map((item) => (
        <button
          key={item.id}
          aria-label={`${label}：${item.title}视图`}
          aria-pressed={view === item.id}
          className={view === item.id ? "selected" : ""}
          onClick={() => onChange(item.id)}
        >
          <item.icon size={16} />
          {item.title}
        </button>
      ))}
    </div>
  );
}
const kinds: Record<string, string> = {
  train: "训练",
  predict: "推理",
  evaluate: "评估",
  report: "报告",
};
const states: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};
export default function ResultCatalog({
  jobs,
  view,
  onOpen,
}: {
  jobs: Job[];
  view: ResultView;
  onOpen: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [page, setPage] = useState(1);
  const filtered = jobs.filter(
    (j) =>
      (!kind || j.kind === kind) &&
      `${j.title} ${j.id} ${j.run_id || ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const queryResult = useQuery({
    queryKey: ["result-page", query, kind, page],
    queryFn: ({ signal }) =>
      api<JobPage>(
        `/jobs/page?${new URLSearchParams({ query, kind, limit: "12", offset: String((page - 1) * 12) })}`,
        { signal },
      ),
    placeholderData: keepPreviousData,
    refetchInterval: 15000,
  });
  const total = queryResult.data?.total ?? filtered.length;
  const pages = Math.max(1, Math.ceil(total / 12));
  const current = Math.min(page, pages);
  useEffect(() => {
    if (!queryResult.isPlaceholderData && page > pages) setPage(pages);
  }, [page, pages, queryResult.isPlaceholderData]);
  const shown =
    queryResult.data?.items ?? filtered.slice((current - 1) * 12, current * 12);
  const badge = (job: Job) => (
    <span className={`badge ${job.status}`}>
      <i />
      {states[job.status]}
    </span>
  );
  const summary = (job: Job) =>
    job.kind === "predict" && (job.prediction_count || job.result?.predictions)
      ? `${job.prediction_count ?? job.result?.predictions?.length} 张图片 · 已生成预测`
      : job.status === "succeeded"
        ? "结果已生成，点击查看"
        : job.error ||
          (["queued", "running"].includes(job.status)
            ? "结果随任务进度更新"
            : "任务已结束，可查看已有输出和日志");
  return (
    <section className="panel result-catalog">
      <div className="result-toolbar">
        <label className="task-search">
          <Search size={16} />
          <input
            aria-label="搜索任务结果"
            placeholder="搜索结果名称、任务 ID 或实验…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
          />
        </label>
        <select
          aria-label="结果类型"
          value={kind}
          onChange={(e) => {
            setKind(e.target.value);
            setPage(1);
          }}
        >
          <option value="">全部类型</option>
          {Object.entries(kinds).map(([key, value]) => (
            <option key={key} value={key}>
              {value}
            </option>
          ))}
        </select>
        <span className="field-note">{total} 项结果</span>
      </div>
      {!shown.length ? (
        <div className="empty">没有符合条件的任务结果</div>
      ) : view === "cards" ? (
        <div className="result-card-grid">
          {shown.map((job) => (
            <article className="result-overview-card" key={job.id}>
              <div className="section-title">
                <span>{kinds[job.kind]}</span>
                {badge(job)}
              </div>
              <h3>{job.title}</h3>
              <p>{summary(job)}</p>
              <code>{job.run_id || job.id}</code>
              <small>{new Date(job.created_at).toLocaleString("zh-CN")}</small>
              <button
                className="button secondary"
                onClick={() => onOpen(job.id)}
                aria-label={`打开结果 ${job.title}`}
              >
                查看结果
                <ArrowRight size={15} />
              </button>
            </article>
          ))}
        </div>
      ) : (
        <div className="task-table-scroll">
          <table className="task-table result-table">
            <thead>
              <tr>
                <th>任务结果</th>
                <th>类型</th>
                <th>状态</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((job) => (
                <tr key={job.id}>
                  <td>
                    <strong>{job.title}</strong>
                    <small>{summary(job)}</small>
                  </td>
                  <td>{kinds[job.kind]}</td>
                  <td>{badge(job)}</td>
                  <td>{new Date(job.created_at).toLocaleString("zh-CN")}</td>
                  <td>
                    <button
                      className="button secondary"
                      onClick={() => onOpen(job.id)}
                      aria-label={`打开结果 ${job.title}`}
                    >
                      查看结果
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="task-pagination">
        <span>每页 12 项</span>
        <button disabled={current <= 1} onClick={() => setPage(current - 1)}>
          上一页
        </button>
        <span>
          {current} / {pages}
        </span>
        <button
          disabled={current >= pages}
          onClick={() => setPage(current + 1)}
        >
          下一页
        </button>
      </div>
    </section>
  );
}
