import TaskManager from "../TaskManager";
import { useWorkbench } from "../WorkbenchContext";

export default function JobsPage() {
  const { catalog, jobs, runs, setSelected, refreshJobs, setPage, openResult } =
    useWorkbench();
  return (
    <div className="jobs-workspace">
      <TaskManager
        jobs={jobs}
        catalog={catalog}
        runs={runs}
        onSelect={setSelected}
        onOpen={openResult}
        onRefresh={refreshJobs}
        onNew={setPage}
      />
    </div>
  );
}
