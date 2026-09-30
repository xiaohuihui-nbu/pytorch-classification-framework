import type { components } from "./generated/api-schema";
export type Metric = Record<string, unknown>;
export type Job = {
  gpu_indices?: number[];
  save_images?: boolean;
  priority?: number;
  timeout_minutes?: number;
  blocked_reason?: string | null;
  prediction_count?: number;
  inputs?: { id: string; name: string; path: string; url: string | null }[];
  parameters?: Record<string, unknown> | null;
  pid?: number;
  elapsed_seconds?: number;
  input_count?: number;
  id: string;
  kind: string;
  title: string;
  status: string;
  created_at: string;
  run_id?: string;
  error?: string;
  log?: string;
  metrics?: Metric[];
  report_url?: string;
  progress?: { completed: number; total: number };
  performance?: {
    epoch: number;
    epoch_seconds: number;
    train_images_per_second: number;
  };
  result?: {
    predictions?: Prediction[];
    timings?: Record<string, number>;
    cache_hit?: boolean;
    peak_worker_rss_mb?: number;
    worker_rss_mb?: number;
    report_status?: string;
    report_error?: string;
  };
};
export type Prediction = {
  input_id?: string | null;
  source?: string;
  path?: string;
  label: string;
  probabilities: Record<string, number>;
};
export type Run = {
  config?: Record<string, unknown>;
  id: string;
  name: string;
  classes: string[];
  status: Record<string, unknown>;
  metrics: Metric[];
  reports: string[];
  images: string[];
};
export type Catalog = {
  gpus?: { index: number; name: string; free_gb: number; total_gb: number }[];
  models: { id: string; name: string; provider: string }[];
  datasets: { id: string; classes: string[]; counts: Record<string, number> }[];
};
export type Limits = { train: number; predict: number; auxiliary: number };
export type Scheduler = { limits: Limits; running: Limits; queued: Limits };
export type WebSettings = Omit<
  Required<components["schemas"]["WebSettings"]>,
  "training"
> & {
  training: Required<components["schemas"]["TrainingDevice"]>;
};
export type TrainRequest = components["schemas"]["TrainRequest"];
export type ServiceStatus = {
  scheduler: Scheduler;
  settings: WebSettings;
  system: { cpu_percent: number; memory_percent: number };
  workspace: string;
  server_time: string;
};
export const defaultSettings: WebSettings = {
  training: { device: "auto", gpu_indices: [] },
  resources: {
    enabled: true,
    max_memory_percent: 90,
    min_available_gb: 2,
    launch_reserve_gb: 1,
    min_gpu_free_gb: 2,
    max_jobs_per_gpu: 0,
  },
  concurrency: { train: 0, predict: 1, auxiliary: 1 },
  inference: {
    resident: true,
    cpu_threads: 2,
    batch_size: 8,
    cache_models: 1,
    cache_mb: 256,
    idle_seconds: 300,
  },
  appearance: { theme: "cyber", mode: "dark" },
  refresh_interval_seconds: 3,
};
export type JobPage = {
  items: Job[];
  total: number;
  limit: number;
  offset: number;
};
export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${url}`, options);
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
export function post<T>(url: string, body: unknown): Promise<T> {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
