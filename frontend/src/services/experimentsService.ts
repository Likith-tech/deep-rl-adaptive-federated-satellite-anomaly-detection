import { apiGet } from "@/api/client";
import type { ExperimentDetail, ExperimentSummary } from "@/types/experiments";

export function fetchExperiments(): Promise<ExperimentSummary[]> {
  return apiGet<ExperimentSummary[]>("/experiments");
}

export function fetchExperiment(experimentId: string): Promise<ExperimentDetail> {
  return apiGet<ExperimentDetail>(`/experiments/${experimentId}`);
}

export function fetchModels(): Promise<ExperimentSummary[]> {
  return apiGet<ExperimentSummary[]>("/models");
}

export function fetchModel(modelId: string): Promise<ExperimentDetail> {
  return apiGet<ExperimentDetail>(`/models/${modelId}`);
}
