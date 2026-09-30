import { createContext, useContext } from "react";
import type { useWorkbenchState } from "./useWorkbenchState";
export const WorkbenchContext = createContext<ReturnType<
  typeof useWorkbenchState
> | null>(null);
export function useWorkbench() {
  const state = useContext(WorkbenchContext);
  if (!state) throw new Error("Workbench provider missing");
  return state;
}
