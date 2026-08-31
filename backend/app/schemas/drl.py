"""Schemas for the Phase 9B DQN controller API — real, already-committed
pretraining/final-run data, never re-run or re-derived by the backend."""

from __future__ import annotations

from pydantic import BaseModel


class DRLControllerSummary(BaseModel):
    action_catalogue: list[str]
    state_dimensions: int
    episodes_trained: int
    rounds_per_episode: int
    total_env_steps: int
    final_run_best_round: int
    final_run_best_val_loss: float
    test_f1: float
    test_roc_auc: float
    beat_best_rule_based_baseline: bool


class DRLDecision(BaseModel):
    round_number: int
    action_name: str
    weights: dict[str, float]
    val_f1: float
    val_loss: float
