import { apiGet } from "@/api/client";
import type { DRLControllerSummary, DRLDecision } from "@/types/drl";

export function fetchDrlController(): Promise<DRLControllerSummary> {
  return apiGet<DRLControllerSummary>("/drl/controller");
}

export function fetchDrlDecisions(): Promise<DRLDecision[]> {
  return apiGet<DRLDecision[]>("/drl/decisions");
}
