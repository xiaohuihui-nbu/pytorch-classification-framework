import { ImagePlus } from "lucide-react";
import { useState } from "react";
import type { Job } from "./api";
import { ViewSwitch, useResultView } from "./ResultViews";

function SourceImage({
  url,
  name,
  saved,
}: {
  url: string | null;
  name: string;
  saved?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  return url && !failed ? (
    <a href={url} target="_blank" rel="noreferrer" title="打开输入图片">
      <img src={url} alt={`输入图片：${name}`} onError={() => setFailed(true)} />
    </a>
  ) : (
    <div className="source-unavailable">
      <ImagePlus />
      <span>{saved === false ? "未保存原图" : "原图文件不可用"}</span>
    </div>
  );
}

export default function PredictionResults({
  job,
  previews = {},
}: {
  job: Job;
  previews?: Record<string, string>;
}) {
  const [view, setView] = useResultView("cls.predictions.view");
  const predictions = job.result?.predictions || [];
  const inputs = (job.inputs || []).map((input) => ({
    ...input,
    url: previews[input.id] || input.url,
  }));
  const temporary = job.save_images === false;
  const cleaned = temporary && !["queued", "running"].includes(job.status);
  return (
    <div className={`predictions predictions-${view}`}>
      <div className="result-toolbar">
        <h3>图片预测 · {inputs.length} 张</h3>
        <ViewSwitch label="图片结果" view={view} onChange={setView} />
      </div>
      {temporary && (
        <p className="field-note">
          {Object.values(previews).some(Boolean)
            ? "图片为当前会话临时预览，未保存到服务器。仅保留最近一次推理的预览；刷新或关闭页面后不可恢复。"
            : "本次未保存图片，当前会话没有可用预览；可重新选图推理。"}
          {cleaned && " 下方为本次使用的临时路径，文件已清理。"}
        </p>
      )}
      {view === "table" && inputs.length > 0 ? (
        <div className="task-table-scroll">
          <table className="task-table prediction-table">
            <thead>
              <tr>
                <th>输入图片</th>
                <th>文件名与路径</th>
                <th>预测类别</th>
                <th>概率</th>
                <th>所有类别</th>
              </tr>
            </thead>
            <tbody>
              {inputs.map((input) => {
                const prediction = predictions.find(
                  (p) => p.input_id === input.id,
                );
                return (
                  <tr key={input.id}>
                    <td>
                      <div className="source-frame">
                        <SourceImage
                          key={input.url}
                          url={input.url}
                          name={input.name}
                          saved={job.save_images}
                        />
                      </div>
                    </td>
                    <td>
                      <strong className="source-name">{input.name}</strong>
                      <code className="source-path">{input.path}</code>
                      {cleaned && <small>临时文件已清理</small>}
                    </td>
                    <td>
                      {prediction?.label ||
                        (["queued", "running"].includes(job.status)
                          ? "等待预测"
                          : "暂无结果")}
                    </td>
                    <td>
                      {prediction
                        ? `${((prediction.probabilities[prediction.label] || 0) * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td>
                      {prediction && (
                        <details>
                          <summary>查看概率</summary>
                          {Object.entries(prediction.probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .map(([label, score]) => (
                              <p key={label}>
                                {label} · {(score * 100).toFixed(1)}%
                              </p>
                            ))}
                        </details>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="prediction-items">
          {inputs.map((input, index) => {
            const prediction = predictions.find((p) => p.input_id === input.id);
            const scores = Object.entries(prediction?.probabilities || {}).sort(
              (a, b) => b[1] - a[1],
            );
            return (
              <article
                className="prediction prediction-comparison"
                key={input.id}
              >
                <div className="prediction-source">
                  <h3>
                    <span>{String(index + 1).padStart(2, "0")}</span>输入图片
                  </h3>
                  <div className="source-frame">
                    <SourceImage
                      key={input.url}
                      url={input.url}
                      name={input.name}
                      saved={job.save_images}
                    />
                  </div>
                  <strong className="source-name">{input.name}</strong>
                  <span className="path-caption">
                    {cleaned ? "临时推理路径（文件已清理）" : "本机文件路径"}
                  </span>
                  <code className="source-path">{input.path}</code>
                </div>
                <div className="prediction-output">
                  <h3>预测结果</h3>
                  {prediction ? (
                    <>
                      <div className="prediction-winner">
                        <span>预测类别</span>
                        <strong>{prediction.label}</strong>
                        <b>
                          {(
                            (prediction.probabilities[prediction.label] || 0) *
                            100
                          ).toFixed(1)}
                          %
                        </b>
                      </div>
                      <p className="field-note">类别概率</p>
                      {scores.map(([label, score]) => (
                        <div className="probability" key={label}>
                          <span>{label}</span>
                          <div>
                            <i
                              style={{
                                width: `${Math.max(0, Math.min(100, score * 100))}%`,
                              }}
                            />
                          </div>
                          <b>{(score * 100).toFixed(1)}%</b>
                        </div>
                      ))}
                    </>
                  ) : (
                    <p className="prediction-pending">
                      {["queued", "running"].includes(job.status)
                        ? "正在等待预测结果…"
                        : "本图片未生成预测结果，请查看任务日志。"}
                    </p>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
      {!inputs.length && (
        <p className="field-note">此任务没有可用的原图记录。</p>
      )}
      <p className="field-note">
        路径为本机服务保存的图片位置。浏览器上传仅提供文件名，不提供上传前的完整磁盘路径。
      </p>
    </div>
  );
}
