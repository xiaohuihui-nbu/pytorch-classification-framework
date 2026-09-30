import { ArrowRight } from "lucide-react";
import React from "react";
import { Chart, Empty } from "../shared";
import { useWorkbench } from "../WorkbenchContext";
const ExperimentComparison = React.lazy(
  () => import("../ExperimentComparison"),
);

export default function RunsPage() {
  const {
    runs,
    runId,
    setRunId,
    busy,
    run,
    perform,
    launch,
    reportLink,
    setPage,
  } = useWorkbench();
  return (
    <>
      <React.Suspense fallback={<p>正在加载实验对比…</p>}>
        <ExperimentComparison runs={runs} />
      </React.Suspense>
      <section className="panel">
        <div className="section-title">
          <h2>历史模型与可视化</h2>
          <span className="hint">{runs.length} 个模型</span>
        </div>
        {runs.length ? (
          <>
            <label>
              实验记录
              <select value={runId} onChange={(e) => setRunId(e.target.value)}>
                {runs.map((r) => (
                  <option value={r.id} key={r.id}>
                    {r.id}
                  </option>
                ))}
              </select>
            </label>
            {run && (
              <>
                <Chart metrics={run.metrics} />
                <div className="actions">
                  {run.reports.map((href) => (
                    <React.Fragment key={href}>
                      {reportLink(href)}
                    </React.Fragment>
                  ))}
                  <button
                    className="button secondary"
                    disabled={busy}
                    onClick={() =>
                      perform(() => launch("/report", { run_id: runId }))
                    }
                  >
                    重新生成报告
                  </button>
                  <button
                    className="button secondary"
                    disabled={busy}
                    onClick={() =>
                      perform(() =>
                        launch("/evaluate", {
                          run_id: runId,
                          split: "val",
                        }),
                      )
                    }
                  >
                    验证集评估
                  </button>
                  <button
                    className="button secondary"
                    disabled={busy}
                    onClick={() =>
                      perform(() =>
                        launch("/evaluate", {
                          run_id: runId,
                          split: "test",
                        }),
                      )
                    }
                  >
                    测试集评估
                  </button>
                  <button
                    className="button primary"
                    onClick={() => setPage("predict")}
                  >
                    使用模型推理
                    <ArrowRight size={16} />
                  </button>
                </div>
                <div className="report-images">
                  {run.images.map((src) => (
                    <a key={src} href={src} target="_blank" rel="noreferrer">
                      <img
                        src={src}
                        alt={decodeURIComponent(
                          src.split("/").pop() || "实验图表",
                        )}
                      />
                    </a>
                  ))}
                </div>
              </>
            )}
          </>
        ) : (
          <Empty>训练完成后，模型权重和图表会出现在这里。</Empty>
        )}
      </section>
    </>
  );
}
