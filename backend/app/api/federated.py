"""Federated-learning overview, round history, and adaptive-control
(rule weights) — real, already-committed data, read-only."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.experiments import RoundMetric
from backend.app.schemas.federated import AdaptiveRuleConfig, FederatedOverview
from backend.app.services import results_registry

router = APIRouter(prefix="/federated", tags=["federated"])


@router.get("/overview", response_model=FederatedOverview)
def get_overview() -> FederatedOverview:
    return results_registry.get_federated_overview()


@router.get("/rounds/{experiment_id}", response_model=list[RoundMetric])
def get_rounds(experiment_id: str) -> list[RoundMetric]:
    detail = results_registry.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")
    if not detail.has_round_history:
        raise HTTPException(status_code=404, detail=f"{experiment_id} has no round-by-round history")
    return detail.rounds


@router.get("/adaptive-control", response_model=list[AdaptiveRuleConfig])
def get_adaptive_control() -> list[AdaptiveRuleConfig]:
    return results_registry.list_adaptive_rule_configs()
