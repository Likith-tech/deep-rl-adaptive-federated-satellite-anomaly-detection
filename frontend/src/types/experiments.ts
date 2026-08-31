export interface ExperimentSummary {
  id: string;
  name: string;
  phase: string;
  family: string;
  test_f1: number;
  test_roc_auc: number | null;
  fairness_gap: number | null;
  has_round_history: boolean;
  source_report: string;
}

export interface RoundMetric {
  round_number: number;
  val_loss: number | null;
  val_f1: number;
  val_roc_auc: number | null;
  aggregation_weights: Record<string, number> | null;
  client_staleness: Record<string, number> | null;
  action_name: string | null;
}

export interface ExperimentDetail extends ExperimentSummary {
  rounds: RoundMetric[];
}
