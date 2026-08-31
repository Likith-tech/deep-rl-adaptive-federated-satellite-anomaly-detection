export interface DRLControllerSummary {
  action_catalogue: string[];
  state_dimensions: number;
  episodes_trained: number;
  rounds_per_episode: number;
  total_env_steps: number;
  final_run_best_round: number;
  final_run_best_val_loss: number;
  test_f1: number;
  test_roc_auc: number;
  beat_best_rule_based_baseline: boolean;
}

export interface DRLDecision {
  round_number: number;
  action_name: string;
  weights: Record<string, number>;
  val_f1: number;
  val_loss: number;
}
