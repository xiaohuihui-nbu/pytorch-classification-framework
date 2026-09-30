import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, FileText } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  api,
  defaultSettings,
  post,
  type Catalog,
  type Job,
  type Run,
  type Scheduler,
  type ServiceStatus,
  type WebSettings,
} from "./api";
import { useResultView } from "./ResultViews";
import { initialForm, nav } from "./shared";
import { useRealtime } from "./useRealtime";
type UploadedImage = {
  id: string;
  name: string;
  url: string;
  previewUrl?: string;
};
export function useWorkbenchState() {
  const [resultView, setResultView] = useResultView("cls.results.view");
  const navigate = useNavigate();
  const location = useLocation();
  const [routePage = "train", routeId = ""] = location.pathname
    .split("/")
    .filter(Boolean);
  const page = nav.some((n) => n.id === routePage) ? routePage : "not-found";
  function setPage(next: string) {
    navigate(
      next === "results" && selected ? `/results/${selected}` : `/${next}`,
    );
  }
  function openResult(id: string) {
    setResultView("detail");
    setSelected(id);
    navigate(`/results/${id}`);
  }
  const [catalog, setCatalog] = useState<Catalog>({ models: [], datasets: [] });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [scheduler, setScheduler] = useState<Scheduler | null>(null);
  const [settings, setSettings] = useState<WebSettings>(defaultSettings);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(
    null,
  );
  const [experimentName, setExperimentName] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [form, setForm] = useState(initialForm);
  const [selected, setSelected] = useState(routeId);
  const [detail, setDetail] = useState<Job | null>(null);
  const [runId, setRunId] = useState("");
  const [files, setFiles] = useState<UploadedImage[]>([]);
  const [previewBatch, setPreviewBatch] = useState<{
    jobId: string;
    files: UploadedImage[];
  } | null>(null);
  const previewUrls = useRef(new Set<string>());
  useEffect(() => {
    const used = new Set(
      [...files, ...(previewBatch?.files || [])].map((file) => file.previewUrl),
    );
    for (const url of previewUrls.current) {
      if (!used.has(url)) {
        URL.revokeObjectURL(url);
        previewUrls.current.delete(url);
      }
    }
  }, [files, previewBatch]);
  useEffect(() => {
    const urls = previewUrls.current;
    return () => {
      for (const url of urls) URL.revokeObjectURL(url);
      urls.clear();
    };
  }, []);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (page === "results") setSelected(routeId);
    setError("");
  }, [page, routeId]);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [page]);
  const queryClient = useQueryClient();
  const realtime = useRealtime(selected, page === "results");
  const interval = settings.refresh_interval_seconds * 1000;
  const catalogQuery = useQuery({
    queryKey: ["catalog"],
    queryFn: ({ signal }) => api<Catalog>("/catalog", { signal }),
    staleTime: 60000,
  });
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: ({ signal }) => api<Job[]>("/jobs?summary=true", { signal }),
    refetchInterval: realtime ? 30000 : interval,
  });
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: ({ signal }) => api<Run[]>("/runs", { signal }),
    staleTime: 10000,
    refetchInterval: 30000,
  });
  const statusQuery = useQuery({
    queryKey: ["status"],
    queryFn: ({ signal }) => api<ServiceStatus>("/status", { signal }),
    refetchInterval: realtime ? false : interval,
  });
  const detailQuery = useQuery({
    queryKey: ["job", selected],
    queryFn: ({ signal }) =>
      api<Job>(`/jobs/${encodeURIComponent(selected)}`, { signal }),
    enabled: !!selected && page === "results",
    refetchInterval: realtime ? false : interval,
    retry: false,
  });
  async function refreshCatalog() {
    await queryClient.invalidateQueries({ queryKey: ["catalog"] });
  }
  useEffect(() => {
    const c = catalogQuery.data;
    if (!c) return;
    setCatalog(c);
    setForm((f) => ({
      ...f,
      dataset_id:
        f.dataset_id ||
        c.datasets.find((d) => d.id === "flower_photos_split")?.id ||
        c.datasets[0]?.id ||
        "",
      model_id: c.models.some((m) => m.id === f.model_id)
        ? f.model_id
        : c.models[0]?.id || "",
    }));
  }, [catalogQuery.data]);
  useEffect(() => {
    if (jobsQuery.data) setJobs(jobsQuery.data);
  }, [jobsQuery.data]);
  useEffect(() => {
    if (runsQuery.data) {
      setRuns(runsQuery.data);
      setRunId((old) => old || runsQuery.data[0]?.id || "");
    }
  }, [runsQuery.data]);
  useEffect(() => {
    const s = statusQuery.data;
    if (s) {
      setSettings({
        ...s.settings,
        training: s.settings.training || defaultSettings.training,
      });
      setScheduler(s.scheduler);
      setServiceStatus(s);
    }
    setOnline(realtime || (!!s && !statusQuery.isError));
  }, [statusQuery.data, statusQuery.isError, realtime]);
  useEffect(() => {
    setDetail(detailQuery.data || null);
    if (detailQuery.error) setError(detailQuery.error.message);
  }, [detailQuery.data, detailQuery.error, selected]);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = settings.appearance.theme;
      document.documentElement.dataset.mode =
        settings.appearance.mode === "system"
          ? media.matches
            ? "dark"
            : "light"
          : settings.appearance.mode;
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [settings.appearance.theme, settings.appearance.mode]);
  const dataset = catalog.datasets.find((d) => d.id === form.dataset_id);
  const run = runs.find((r) => r.id === runId);
  const active = jobs.filter((j) =>
    ["running", "queued"].includes(j.status),
  ).length;
  const update = (
    key: keyof typeof form,
    value: string | number | boolean | null,
  ) =>
    setForm((f) => ({
      ...f,
      [key]: value,
      ...(key === "device" && value !== "gpu" ? { gpu_index: null } : {}),
    }));
  async function refreshJobs() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["jobs"] }),
      queryClient.invalidateQueries({ queryKey: ["job-page"] }),
      queryClient.invalidateQueries({ queryKey: ["result-page"] }),
      queryClient.invalidateQueries({ queryKey: ["job", selected] }),
    ]);
  }
  async function perform(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
      await refreshJobs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function launch(endpoint: string, body: unknown) {
    const j = await post<Job>(endpoint, body);
    if (endpoint === "/predict") {
      setPreviewBatch({ jobId: j.id, files: [...files] });
    }
    setSelected(j.id);
    setJobs((old) => [j, ...old]);
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    openResult(j.id);
  }
  async function upload(incoming: FileList | File[]) {
    await perform(async () => {
      if (files.length + incoming.length > 20)
        throw new Error("每次最多选择 20 张图片");
      const added: UploadedImage[] = [];
      try {
        for (const f of Array.from(incoming)) {
          const data = new FormData();
          data.append("file", f);
          const uploaded = await api<UploadedImage>("/uploads", {
            method: "POST",
            body: data,
          });
          const previewUrl = URL.createObjectURL(f);
          previewUrls.current.add(previewUrl);
          added.push({ ...uploaded, previewUrl });
        }
        setFiles((old) => [...old, ...added]);
      } catch (error) {
        for (const file of added) {
          URL.revokeObjectURL(file.previewUrl!);
          previewUrls.current.delete(file.previewUrl!);
        }
        throw error;
      }
    });
  }
  const reportLink = (href: string) => (
    <a
      className="button secondary"
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      <FileText size={16} />
      打开 HTML 报告 <ArrowRight size={16} />
    </a>
  );

  return {
    page,
    setPage,
    openResult,
    active,
    catalog,
    jobs,
    scheduler,
    setScheduler,
    settings,
    setSettings,
    serviceStatus,
    experimentName,
    setExperimentName,
    runs,
    form,
    selected,
    setSelected,
    detail,
    setDetail,
    runId,
    setRunId,
    files,
    previewBatch,
    setFiles,
    error,
    setError,
    online,
    busy,
    advanced,
    setAdvanced,
    fileInput,
    dataset,
    run,
    update,
    refreshJobs,
    refreshCatalog,
    perform,
    launch,
    upload,
    reportLink,
    resultView,
    setResultView,
  };
}
