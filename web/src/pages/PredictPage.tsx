import { ArrowRight, Play, Trash2, UploadCloud } from "lucide-react";
import { useWorkbench } from "../WorkbenchContext";

export default function PredictPage() {
  const {
    runs,
    runId,
    setRunId,
    files,
    setFiles,
    online,
    busy,
    fileInput,
    perform,
    launch,
    upload,
  } = useWorkbench();
  return (
    <div className="train-grid">
      <section className="panel">
        <div className="section-title">
          <span className="step">01</span>
          <h2>上传待识别图片</h2>
          <span className="hint">最多 20 张 · 单张 20 MB</span>
        </div>
        <button
          className="dropzone"
          disabled={busy}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (!busy) upload(e.dataTransfer.files);
          }}
        >
          <UploadCloud size={38} />
          <strong>{busy ? "正在上传…" : "拖拽图片到这里，或点击选择"}</strong>
          <span>JPG、PNG、WebP 等常见图片格式</span>
        </button>
        <input
          type="file"
          aria-label="选择推理图片"
          accept="image/*"
          multiple
          hidden
          ref={fileInput}
          onChange={(e) => {
            if (e.target.files) upload(e.target.files);
            e.target.value = "";
          }}
        />
        <div className="actions">
          <span className="field-note">已选 {files.length} 张图片</span>
          <button
            type="button"
            className="button secondary"
            disabled={busy || !files.length}
            onClick={() => {
              setFiles([]);
              if (fileInput.current) fileInput.current.value = "";
            }}
          >
            <Trash2 size={16} />
            清空图片
          </button>
        </div>
        <div className="upload-grid">
          {files.map((f) => (
            <div key={f.id}>
              <img src={f.previewUrl || f.url} alt={f.name} />
              <span>{f.name}</span>
              <button
                aria-label={`移除 ${f.name}`}
                onClick={() =>
                  setFiles((old) => old.filter((x) => x.id !== f.id))
                }
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </section>
      <section className="panel inference-config">
        <div className="section-title">
          <span className="step">02</span>
          <h2>选择分类模型</h2>
        </div>
        <label>
          已训练模型
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="" disabled>
              请选择模型
            </option>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
        <p className="field-note">
          使用训练完成的最佳模型权重，输出各类别概率。
        </p>
        <p className="field-note">
          结果页显示输入原图预览、路径、类别和概率。任务结束后清理服务器临时图片，刷新后预览消失。
        </p>
        {!runs.length && (
          <div className="notice">请先完成一次训练，再使用自己的分类模型。</div>
        )}
        <button
          className="button primary full"
          disabled={busy || !online || !runId || !files.length}
          onClick={() =>
            perform(async () => {
              await launch("/predict", {
                run_id: runId,
                uploads: files.map((f) => f.id),
                top_k: 5,
                save_images: false,
              });
              setFiles([]);
            })
          }
        >
          <Play size={17} />
          开始推理
          <ArrowRight size={17} />
        </button>
      </section>
    </div>
  );
}
