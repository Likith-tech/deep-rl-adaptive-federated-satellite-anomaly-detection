import { apiGet } from "@/api/client";
import type { HealthResponse } from "@/types/health";

export function fetchBackendHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health");
}
