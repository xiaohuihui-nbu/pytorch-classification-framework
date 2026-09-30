import { Cpu, HardDrive, Settings2 } from "lucide-react";
import type { Scheduler, ServiceStatus } from "./api";

export default function StatusBar({
  online,
  scheduler,
  status,
  onSettings,
}: {
  online: boolean;
  scheduler: Scheduler | null;
  status: ServiceStatus | null;
  onSettings: () => void;
}) {
  const queue = scheduler
    ? Object.values(scheduler.queued).reduce((a, b) => a + b, 0)
    : 0;
  return (
    <div className="status-bar" role="region" aria-label="全局状态栏">
      <span
        className={`status-connection ${online ? "connected" : "disconnected"}`}
      >
        <i />
        {online ? "本地服务已连接" : "服务未连接"}
      </span>
      <div className="status-tasks">
        {(
          [
            ["train", "训练"],
            ["predict", "推理"],
            ["auxiliary", "评估"],
          ] as const
        ).map(([key, name]) => (
          <span key={key}>
            {name}{" "}
            <b>
              {online && scheduler
                ? `${scheduler.running[key]}/${scheduler.limits[key] === 0 ? "不限" : scheduler.limits[key]}`
                : "—"}
            </b>
          </span>
        ))}
        <span>
          排队 <b>{online ? queue : "—"}</b>
        </span>
      </div>
      <div className="status-system">
        <span>
          <Cpu size={13} />
          CPU{" "}
          {online && status ? `${Math.round(status.system.cpu_percent)}%` : "—"}
        </span>
        <span>
          <HardDrive size={13} />
          内存{" "}
          {online && status
            ? `${Math.round(status.system.memory_percent)}%`
            : "—"}
        </span>
      </div>
      <span className="status-updated">
        {online && status
          ? `更新于 ${new Date(status.server_time).toLocaleTimeString("zh-CN")}`
          : "等待重新连接"}
      </span>
      <button
        aria-label="打开全局设置"
        title="打开全局设置"
        onClick={onSettings}
      >
        <Settings2 size={15} />
      </button>
    </div>
  );
}
