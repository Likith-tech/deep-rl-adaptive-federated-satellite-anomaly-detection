"""Schemas for the federated-learning overview/adaptive-control API."""

from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.experiments import ExperimentSummary


class FederatedOverview(BaseModel):
    total_methods_compared: int
    best_mechanism: ExperimentSummary
    fairest_mechanism: ExperimentSummary
    mechanisms: list[ExperimentSummary]


class AdaptiveRuleConfig(BaseModel):
    experiment_id: str
    name: str
    weights: dict[str, float]
    test_f1: float
    fairness_gap: float | None = None
