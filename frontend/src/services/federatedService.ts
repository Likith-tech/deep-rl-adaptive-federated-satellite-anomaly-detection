import { apiGet } from "@/api/client";
import type { AdaptiveRuleConfig, FederatedOverview } from "@/types/federated";
import type { RoundMetric } from "@/types/experiments";

export function fetchFederatedOverview(): Promise<FederatedOverview> {
  return apiGet<FederatedOverview>("/federated/overview");
}

export function fetchFederatedRounds(experimentId: string): Promise<RoundMetric[]> {
  return apiGet<RoundMetric[]>(`/federated/rounds/${experimentId}`);
}

export function fetchAdaptiveControl(): Promise<AdaptiveRuleConfig[]> {
  return apiGet<AdaptiveRuleConfig[]>("/federated/adaptive-control");
}
