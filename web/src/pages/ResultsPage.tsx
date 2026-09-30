import { ArrowRight, ImagePlus, Square, Terminal } from "lucide-react";
import { api, post } from "../api";
import PredictionResults from "../PredictionResults";
import ResultCatalog, { ViewSwitch } from "../ResultViews";
import { Badge, Chart, Empty, statusText } from "../shared";
import { useWorkbench } from "../WorkbenchContext";

export default function ResultsPage() {
  const {
    jobs,
    settings,
    runs,
    selected,
    detail,
    previewBatch,
    setDetail,
    setRunId,
    error,
    busy,
    perform,
    reportLink,
    setPage,
    openResult,
    resultView,
    setResultView,
  } = useWorkbench();
  return (
    <div className="results-workspace">
      <section className="panel result-navigation">
        <button className="button secondary" onClick={() => setPage("jobs")}>
          返回任务管理
        </button>
        <label>
          选择任务结果
          <select value={selected} onChange={(e) => openResult(e.target.value)}>
            <option value="" disabled>
              请选择一个任务
            </option>
            {selected && !jobs.some((j) => j.id === selected) && (
              <option value={selected}>{selected}</option>
            )}
            {jobs.map((job) => (
              <option key={job.id} value={job.id}>
                {job.title} · {statusText[job.status]}
              </option>
            ))}
          </select>
        </label>
      </section>
      <div className="result-toolbar result-view-toolbar">
        <h2>任务结果</h2>
        <ViewSwitch
          label="任务结果"
          view={resultView}
          onChange={setResultView}
        />
      </div>
      {resultView !== "detail" ? (
        <ResultCatalog jobs={jobs} view={resultView} onOpen={openResult} />
      ) : (
        <div>
          {detail ? (
            <>
              <section className="panel">
                <div className="section-title">
                  <h2>{detail.title}</h2>
                  <Badge status={detail.status} />
                </div>
                <p className="field-note">任务 {detail.id}</p>
                {detail.gpu_indices && detail.gpu_indices.length > 0 && (
                  <p className="field-note">
                    使用 GPU {detail.gpu_indices.join("、")}
                    {detail.gpu_indices.length > 1 ? " · 多卡 DDP" : " · 单卡"}
                  </p>
                )}
                {detail.blocked_reason && (
                  <div className="notice">
                    排队原因：{detail.blocked_reason}
                  </div>
                )}
                {detail.error && <div className="error">{detail.error}</div>}
                <>
                  {detail.progress && (
                    <div className="progress-block">
                      <div>
                        <span>
                          训练进度 · Epoch {detail.progress.completed} /{" "}
                          {detail.progress.total}
                        </span>
                        <strong>
                          {Math.round(
                            (detail.progress.completed /
                              Math.max(1, detail.progress.total)) *
                              100,
                          )}
                          %
                        </strong>
                      </div>
                      <progress
                        max={detail.progress.total || 1}
                        value={detail.progress.completed}
                      />
                      <small>
                        {detail.status === "running" &&
                        !detail.progress.completed
                          ? "正在准备数据、加载权重或进行首轮训练，详情见下方日志。"
                          : "每轮验证完成后更新进度与曲线。"}
                      </small>
                    </div>
                  )}
                  {detail.kind === "train" && (
                    <Chart metrics={detail.metrics || []} />
                  )}
                  {detail.kind === "predict" && (
                    <div className="inference-summary">
                      <ImagePlus size={24} />
                      <div>
                        <strong>
                          图片推理 ·{" "}
                          {detail.result?.predictions?.length ??
                            detail.input_count ??
                            0}{" "}
                          张图片
                        </strong>
                        <p>只加载已训练模型进行预测，不进行训练。</p>
                      </div>
                      <span>
                        {detail.elapsed_seconds != null
                          ? `${detail.elapsed_seconds.toFixed(1)} 秒`
                          : "等待执行"}
                      </span>
                    </div>
                  )}
                </>
                {detail.kind === "predict" && detail.result?.timings && (
                  <div className="settings-note">
                    {detail.result.cache_hit
                      ? "复用缓存模型"
                      : "首次加载或模型已变化"}{" "}
                    ·{" "}
                    {(
                      [
                        ["queue_seconds", "排队"],
                        ["startup_seconds", "进程启动"],
                        ["runtime_import_seconds", "运行库导入"],
                        ["model_load_seconds", "模型加载"],
                        ["model_verify_seconds", "权重校验"],
                        ["preprocess_seconds", "预处理"],
                        ["forward_seconds", "预测"],
                        ["report_seconds", "报告"],
                      ] as const
                    )
                      .filter(([key]) => detail.result?.timings?.[key] != null)
                      .map(
                        ([key, label]) =>
                          `${label} ${detail.result!.timings![key].toFixed(3)} 秒`,
                      )
                      .join(" · ")}
                  </div>
                )}
                {detail.kind === "train" && detail.performance && (
                  <div className="settings-note">
                    第 {detail.performance.epoch + 1} 轮 ·{" "}
                    {detail.performance.epoch_seconds.toFixed(2)} 秒 · 训练吞吐{" "}
                    {detail.performance.train_images_per_second.toFixed(1)}{" "}
                    张/秒（含本轮验证时间）
                  </div>
                )}
                {(detail.result?.peak_worker_rss_mb != null ||
                  detail.result?.worker_rss_mb != null) && (
                  <div className="settings-note">
                    {detail.result.peak_worker_rss_mb != null
                      ? `工作进程树内存采样峰值 ${detail.result.peak_worker_rss_mb.toFixed(0)} MB`
                      : `推理进程当前内存 ${detail.result.worker_rss_mb!.toFixed(0)} MB`}
                  </div>
                )}
                <div className="actions">
                  {["queued", "running"].includes(detail.status) && (
                    <button
                      className="button danger"
                      disabled={busy}
                      onClick={() =>
                        perform(async () => {
                          await post(`/jobs/${detail.id}/cancel`, {});
                          setDetail(await api(`/jobs/${detail.id}`));
                        })
                      }
                    >
                      <Square size={15} />
                      停止任务
                    </button>
                  )}
                  {detail.report_url && reportLink(detail.report_url)}
                  {detail.kind === "predict" &&
                    detail.status === "succeeded" &&
                    !detail.report_url && (
                      <span className="hint">
                        {detail.result?.report_status === "skipped"
                          ? "仅推理：未保存图片及报告"
                          : ["queued", "running"].includes(
                                detail.result?.report_status || "",
                              )
                            ? "预测完成，报告正在后台生成…"
                            : `预测完成，报告${detail.result?.report_status === "failed" ? "生成失败" : "未生成"}`}
                        {detail.result?.report_error &&
                          `：${detail.result.report_error}`}
                      </span>
                    )}
                  {detail.status === "succeeded" && detail.kind === "train" && (
                    <button
                      className="button primary"
                      onClick={() => {
                        setRunId(detail.run_id || "");
                        setPage("runs");
                      }}
                    >
                      查看实验结果
                      <ArrowRight size={16} />
                    </button>
                  )}
                </div>
                {detail.kind === "predict" && (
                  <PredictionResults
                    job={detail}
                    previews={
                      previewBatch?.jobId === detail.id
                        ? Object.fromEntries(
                            previewBatch.files.map((file) => [
                              file.id,
                              file.previewUrl || "",
                            ]),
                          )
                        : {}
                    }
                  />
                )}
              </section>
              <section className="panel log-panel">
                <div className="section-title">
                  <Terminal size={17} />
                  <h2>实时日志</h2>
                  <span className="hint">最近 48 KB · 自动刷新</span>
                </div>
                <pre>{detail.log || "等待输出…"}</pre>
              </section>
            </>
          ) : (
            <section className="panel">
              <Empty>
                {selected
                  ? "正在加载任务结果；若记录已删除，请返回任务管理。"
                  : "请选择任务，或从任务管理页面打开结果。"}
              </Empty>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
