import {
  Activity,
  BarChart3,
  Box,
  FileText,
  ImagePlus,
  Layers3,
  Settings2,
} from "lucide-react";
import React from "react";
import type { Metric, TrainRequest } from "./api";
export const statusText: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};
export const modelNames: Record<string, string> = {
  flower_resnet18: "ResNet 18",
  flower_resnet50: "ResNet 50",
  flower_convnext_tiny: "ConvNeXt Tiny",
  flower_deit_tiny: "DeiT Tiny",
  flower_efficientnet_b0: "EfficientNet B0",
  flower_mobilenetv3_small: "MobileNet V3",
  flower_swin_tiny: "Swin Tiny",
  flower_vit_tiny: "ViT Tiny",
};
export const descriptions: Record<string, string> = {
  flower_resnet18: "轻量经典 · 推荐起点",
  flower_resnet50: "更深残差网络",
  flower_convnext_tiny: "现代卷积架构",
  flower_deit_tiny: "数据高效 Transformer",
  flower_efficientnet_b0: "精度与效率平衡",
  flower_mobilenetv3_small: "轻量模型 · 移动端",
  flower_swin_tiny: "层级窗口注意力",
  flower_vit_tiny: "视觉 Transformer",
};
export const nav = [
  { id: "train", label: "训练工作台", icon: Layers3 },
  { id: "jobs", label: "任务管理", icon: Activity },
  { id: "results", label: "任务结果", icon: FileText },
  { id: "runs", label: "实验结果", icon: BarChart3 },
  { id: "predict", label: "图片推理", icon: ImagePlus },
  { id: "settings", label: "设置", icon: Settings2 },
];
export const initialForm: TrainRequest = {
  priority: 0,
  timeout_minutes: 0,
  gpu_index: null as number | null,
  model_id: "flower_resnet18",
  dataset_id: "",
  epochs: 10,
  batch_size: 4,
  learning_rate: 0.0003,
  num_workers: 0,
  cpu_threads: 2,
  device: "global",
  seed: 42,
  training_mode: "full_finetune",
  initialization: "last",
  offline: false,
};
export function Badge({ status }: { status: string }) {
  return (
    <span className={`badge ${status}`}>
      <i />
      {statusText[status] || status}
    </span>
  );
}
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <Box size={32} />
      <p>{children}</p>
    </div>
  );
}
export function Chart({ metrics }: { metrics: Metric[] }) {
  const points = metrics
    .map((m, i) => ({
      x: Number(m.epoch ?? i) + 1,
      y: Number(m.accuracy_top1 ?? m["val/accuracy_top1"]),
    }))
    .filter((p) => Number.isFinite(p.y));
  if (!points.length)
    return <Empty>完成一个验证轮次后，这里会显示真实准确率曲线。</Empty>;
  const x = (v: number) =>
    45 + ((v - 1) / Math.max(1, points[points.length - 1].x - 1)) * 620;
  const y = (v: number) => 170 - v * 145;
  return (
    <div className="chart">
      <svg viewBox="0 0 700 205" role="img" aria-label="验证集准确率曲线">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1="45" y1={y(v)} x2="670" y2={y(v)} stroke="#e5ebe7" />
            <text x="2" y={y(v) + 4}>
              {v * 100}%
            </text>
          </g>
        ))}
        <polyline
          points={points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
          fill="none"
          stroke="#18765d"
          strokeWidth="3"
        />
        {points.map((p, i) => (
          <circle key={i} cx={x(p.x)} cy={y(p.y)} r="4" fill="#18765d">
            <title>
              Epoch {p.x}: {(p.y * 100).toFixed(2)}%
            </title>
          </circle>
        ))}
        <text x="45" y="197">
          Epoch 1
        </text>
        <text x="605" y="197">
          Epoch {points[points.length - 1].x}
        </text>
      </svg>
    </div>
  );
}
