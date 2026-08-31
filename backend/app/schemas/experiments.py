"""Schemas for the experiments/models registry API.

Every field here is sourced from an already-committed, real result —
see backend/app/services/results_registry.py for exactly which report
or JSON artifact each number comes from.
"""

from __future__ import annotations

from pydantic import BaseModel


class ExperimentSummary(BaseModel):
    id: str
    name: str
    phase: str
    family: str
    test_f1: float
    test_roc_auc: float | None = None
    fairness_gap: float | None = None
    has_round_history: bool
    source_report: str


class RoundMetric(BaseModel):
    round_number: int
    val_loss: float | None = None
    val_f1: float
    val_roc_auc: float | None = None
    aggregation_weights: dict[str, float] | None = None
    client_staleness: dict[str, int] | None = None
    action_name: str | None = None


class ExperimentDetail(ExperimentSummary):
    rounds: list[RoundMetric] = []
