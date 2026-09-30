import { LineChart } from "echarts/charts";
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useMemo, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import type { Run } from "./api";
echarts.use([
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  AriaComponent,
  CanvasRenderer,
]);

function flatten(value: unknown, prefix = ""): Record<string, string> {
  if (value && typeof value === "object" && !Array.isArray(value))
    return Object.fromEntries(
      Object.entries(value).flatMap(([key, item]) =>
        Object.entries(flatten(item, prefix ? `${prefix}.${key}` : key)),
      ),
    );
  return { [prefix]: JSON.stringify(value) ?? "—" };
}
export default function ExperimentComparison({ runs }: { runs: Run[] }) {
  const [params, setParams] = useSearchParams();
  const ids = [...new Set(params.getAll("compare"))]
    .filter((id) => runs.some((run) => run.id === id))
    .slice(0, 4);
  const chosen = runs.filter((r) => ids.includes(r.id));
  const container = useRef<HTMLDivElement>(null);
  const series = useMemo(
    () =>
      chosen.map((run) => ({
        name: run.id,
        type: "line",
        showSymbol: true,
        symbolSize: 5,
        connectNulls: false,
        data: run.metrics.map((m, i) => [
          Number(m.epoch ?? i) + 1,
          typeof (m.accuracy_top1 ?? m["val/accuracy_top1"]) === "number"
            ? Number(m.accuracy_top1 ?? m["val/accuracy_top1"]) * 100
            : null,
        ]),
      })),
    [runs, params.toString()],
  );
  useEffect(() => {
    if (!container.current || !chosen.length) return;
    const chart = echarts.init(container.current);
    const apply = () => {
      const style = getComputedStyle(document.documentElement);
      const text = style.getPropertyValue("--text").trim();
      chart.setOption({
        animation: false,
        aria: { enabled: true, label: { description: "多实验验证准确率对比" } },
        color: [
          style.getPropertyValue("--accent").trim(),
          "#d9864e",
          "#9b7be0",
          "#44aba3",
        ],
        tooltip: { trigger: "axis", renderMode: "richText" },
        legend: { type: "scroll", top: 5, textStyle: { color: text } },
        grid: { left: 55, right: 25, top: 65, bottom: 65 },
        xAxis: {
          type: "value",
          name: "Epoch",
          nameLocation: "middle",
          nameGap: 25,
          min: 1,
          minInterval: 1,
          axisLabel: { color: text },
        },
        yAxis: {
          type: "value",
          name: "准确率 %",
          min: 0,
          max: 100,
          axisLabel: { color: text },
          splitLine: {
            lineStyle: { color: style.getPropertyValue("--border").trim() },
          },
        },
        dataZoom: [
          { type: "inside" },
          { type: "slider", height: 20, bottom: 10 },
        ],
        series,
      });
    };
    apply();
    const resize = new ResizeObserver(() => chart.resize());
    resize.observe(container.current);
    const observer = new MutationObserver(apply);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-mode", "data-theme"],
    });
    return () => {
      resize.disconnect();
      observer.disconnect();
      chart.dispose();
    };
  }, [series]);
  const configs = chosen.map((run) => flatten(run.config || {}));
  const diffKeys = [...new Set(configs.flatMap((c) => Object.keys(c)))].filter(
    (key) => new Set(configs.map((c) => c[key] ?? "—")).size > 1,
  );
  return (
    <section className="panel experiment-comparison">
      <div className="section-title">
        <h2>实验对比</h2>
        <span className="hint">最多选择 4 个实验</span>
      </div>
      <div className="comparison-picker">
        {runs.map((run) => (
          <label className="check" key={run.id}>
            <input
              type="checkbox"
              checked={ids.includes(run.id)}
              disabled={!ids.includes(run.id) && chosen.length >= 4}
              onChange={(e) => {
                const next = new URLSearchParams(params);
                next.delete("compare");
                (e.target.checked
                  ? [...ids, run.id]
                  : ids.filter((id) => id !== run.id)
                )
                  .slice(0, 4)
                  .forEach((id) => next.append("compare", id));
                setParams(next, { replace: true });
              }}
            />
            {run.id}
          </label>
        ))}
      </div>
      {!chosen.length ? (
        <p className="field-note">
          选择实验，对比验证准确率、最后一轮指标和参数差异。
        </p>
      ) : (
        <>
          <div
            className="comparison-chart"
            ref={container}
            role="img"
            aria-label="多实验验证准确率对比"
          />
          <p className="field-note">
            曲线来自各实验的验证记录，缺失值不填零。数据集、划分或训练设置不同的实验不宜直接按准确率排名。
          </p>
          <div className="task-table-scroll">
            <table className="task-table">
              <thead>
                <tr>
                  <th>实验</th>
                  <th>已记录轮数</th>
                  <th>最高验证准确率</th>
                  <th>最后验证准确率</th>
                </tr>
              </thead>
              <tbody>
                {chosen.map((run) => {
                  const values = run.metrics
                    .map((m) => m.accuracy_top1 ?? m["val/accuracy_top1"])
                    .filter(
                      (v): v is number =>
                        typeof v === "number" && Number.isFinite(v),
                    );
                  const last = run.metrics.at(-1);
                  const lastValue =
                    last?.accuracy_top1 ?? last?.["val/accuracy_top1"];
                  return (
                    <tr key={run.id}>
                      <td>{run.id}</td>
                      <td>{run.metrics.length}</td>
                      <td>
                        {values.length
                          ? (Math.max(...values) * 100).toFixed(2) + "%"
                          : "—"}
                      </td>
                      <td>
                        {typeof lastValue === "number" &&
                        Number.isFinite(lastValue)
                          ? (lastValue * 100).toFixed(2) + "%"
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <h3>参数差异</h3>
          {diffKeys.length ? (
            <div className="task-table-scroll">
              <table className="task-table">
                <thead>
                  <tr>
                    <th>参数</th>
                    {chosen.map((run) => (
                      <th key={run.id}>{run.id}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {diffKeys.map((key) => (
                    <tr key={key}>
                      <td>{key}</td>
                      {configs.map((cfg, i) => (
                        <td key={chosen[i].id}>{cfg[key] ?? "—"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="field-note">
              所选实验没有可比较的参数差异；旧实验可能缺少配置快照。
            </p>
          )}
        </>
      )}
    </section>
  );
}
