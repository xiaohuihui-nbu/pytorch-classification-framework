import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { Job, ServiceStatus } from "./api";

export function useRealtime(selected: string, enabled: boolean) {
  const client = useQueryClient();
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    setConnected(false);
    let revision: number | undefined;
    let log = "";
    let lastSnapshot = Date.now();
    let staleAfter = 12000;
    const source = new EventSource(
      `/api/events${enabled && selected ? `?job_id=${encodeURIComponent(selected)}` : ""}`,
    );
    source.addEventListener("snapshot", (event) => {
      const packet = JSON.parse((event as MessageEvent).data) as {
        status: ServiceStatus;
        revision: number;
        detail?: Job & { deleted?: boolean };
        log?: { reset: boolean; text: string };
      };
      setConnected(true);
      lastSnapshot = Date.now();
      staleAfter = Math.max(
        12000,
        packet.status.settings.refresh_interval_seconds * 3000,
      );
      client.setQueryData(["status"], packet.status);
      if (packet.revision !== revision) {
        for (const key of ["jobs", "job-page", "result-page", "runs"])
          client.invalidateQueries({ queryKey: [key] });
        revision = packet.revision;
      }
      if (packet.log)
        log = (
          packet.log.reset ? packet.log.text : log + packet.log.text
        ).slice(-48000);
      if (packet.detail?.deleted) {
        client.removeQueries({ queryKey: ["job", selected] });
        client.invalidateQueries({ queryKey: ["jobs"] });
      } else if (packet.detail || packet.log)
        client.setQueryData<Job>(
          ["job", selected],
          (old) =>
            ({
              ...old,
              ...packet.detail,
              log: log || old?.log || "等待输出…",
            }) as Job,
        );
    });
    source.onerror = () => setConnected(false);
    const timer = setInterval(() => {
      if (Date.now() - lastSnapshot > staleAfter) setConnected(false);
    }, 3000);
    return () => {
      source.close();
      clearInterval(timer);
    };
  }, [selected, enabled, client]);
  return connected;
}
