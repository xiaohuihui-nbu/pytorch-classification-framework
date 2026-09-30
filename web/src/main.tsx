import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChevronRight, Cpu, Flower2 } from "lucide-react";
import React from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import JobsPage from "./pages/JobsPage";
import PredictPage from "./pages/PredictPage";
import ResultsPage from "./pages/ResultsPage";
import RunsPage from "./pages/RunsPage";
import TrainPage from "./pages/TrainPage";
import SettingsPage from "./SettingsPage";
import { nav } from "./shared";
import StatusBar from "./StatusBar";
import { useWorkbenchState } from "./useWorkbenchState";
import { WorkbenchContext } from "./WorkbenchContext";

import "./style.css";
import "./studio.css";
import "./themes.css";
import "./tasks.css";
import "./results.css";
import "./technology.css";

function App() {
  const state = useWorkbenchState();
  const {
    page,
    setPage,
    openResult,
    active,
    online,
    error,
    setError,
    settings,
    setSettings,
    scheduler,
    setScheduler,
    serviceStatus,
  } = state;
  return (
    <WorkbenchContext.Provider value={state}>
      <div className="app">
        <aside className="sidebar">
          <a
            className="brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setPage("train");
            }}
          >
            <span className="brand-icon">
              <Flower2 size={27} />
            </span>
            <span>
              Classification<small>STUDIO / 本地实验室</small>
            </span>
          </a>
          <div className="workspace-label">
            工作空间 <span>LOCAL</span>
          </div>
          <nav>
            {nav.map((item) => (
              <button
                key={item.id}
                className={page === item.id ? "active" : ""}
                onClick={() => setPage(item.id)}
              >
                <item.icon size={19} />
                {item.label}
                {item.id === "jobs" && active > 0 && <b>{active}</b>}
              </button>
            ))}
          </nav>
          <div className="sidebar-note">
            <Cpu size={22} />
            <strong>让实验，回归简单。</strong>
            <p>
              选择模型，准备数据。
              <br />
              其余交给训练工作台。
            </p>
          </div>
          <div className="connection">
            <span className={online ? "dot online" : "dot"} />
            {online ? "本地服务已连接" : "正在连接后端…"}
            <small>127.0.0.1 : 8000</small>
          </div>
        </aside>
        <main>
          <header>
            <div className="breadcrumb">
              工作空间 <ChevronRight size={14} />
              <strong>{nav.find((n) => n.id === page)?.label}</strong>
            </div>
            <span className="local-tag">
              <Cpu size={14} />
              本机多进程 · 训练 / 推理并行
            </span>
          </header>
          <div className="content">
            <div className="page-heading">
              <div>
                <p className="eyebrow">IMAGE CLASSIFICATION WORKSPACE</p>
                <h1>
                  {
                    {
                      train: "开始一次新的实验",
                      jobs: "每一步，都看得见",
                      results: "查看每一次任务的结果",
                      runs: "从结果中发现答案",
                      predict: "让模型看一张新图片",
                      settings: "让工作空间适合你",
                    }[page]
                  }
                </h1>
                <p className="subtitle">
                  {
                    {
                      train:
                        "从官方预训练模型出发，用自己的数据训练图像分类器。",
                      jobs: "集中管理任务，搜索、编辑或批量操作，点击任务进入独立结果页面。",
                      results: "查看任务进度、图片预测、可视化报告与执行日志。",
                      runs: "对比历史实验，查看训练曲线、混淆矩阵和独立评估报告。",
                      predict:
                        "选择训练好的模型，上传图片，即可查看类别与置信度。",
                      settings:
                        "统一管理任务并发、颜色主题与刷新偏好，所有页面同步生效。",
                    }[page]
                  }
                </p>
              </div>
              <div className="version">
                LOCAL STUDIO <b>02</b>
              </div>
            </div>
            {error && (
              <div role="alert" className="error">
                {error}
                <button aria-label="关闭错误提示" onClick={() => setError("")}>
                  ×
                </button>
              </div>
            )}
            <Routes>
              <Route path="/" element={<Navigate to="/train" replace />} />
              <Route path="/train" element={<TrainPage />} />
              <Route path="/jobs" element={<JobsPage />} />
              <Route path="/results/:id?" element={<ResultsPage />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route
                path="/settings"
                element={
                  <SettingsPage
                    settings={settings}
                    gpus={state.catalog.gpus}
                    online={online}
                    onSave={(value) => {
                      setSettings(value);
                      setScheduler((old) =>
                        old ? { ...old, limits: value.concurrency } : old,
                      );
                    }}
                  />
                }
              />
              <Route
                path="*"
                element={
                  <section className="panel">
                    <h2>页面不存在</h2>
                    <button
                      className="button primary"
                      onClick={() => setPage("train")}
                    >
                      返回训练工作台
                    </button>
                  </section>
                }
              />
            </Routes>
            <footer>
              <span>Classification Studio</span>
              <span>PyTorch 驱动 · 数据与模型保存在本机</span>
            </footer>
          </div>
        </main>
        <StatusBar
          online={online}
          scheduler={scheduler}
          status={serviceStatus}
          onSettings={() => setPage("settings")}
        />
      </div>
    </WorkbenchContext.Provider>
  );
}
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 2000 },
  },
});
// Preserve links generated by earlier versions (#results/id).
if (location.hash && !location.hash.startsWith("#/"))
  history.replaceState(null, "", "#/" + location.hash.slice(1));
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <App />
      </HashRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
