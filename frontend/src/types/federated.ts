import type { ExperimentSummary } from "@/types/experiments";

export interface FederatedOverview {
  total_methods_compared: number;
  best_mechanism: ExperimentSummary;
  fairest_mechanism: ExperimentSummary;
  mechanisms: ExperimentSummary[];
}

export interface AdaptiveRuleConfig {
  experiment_id: string;
  name: string;
  weights: Record<string, number>;
  test_f1: number;
  fairness_gap: number | null;
}
