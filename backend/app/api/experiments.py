"""Experiments registry — every real, already-committed ML/FL/DRL
experiment this project has run, read from results/ and experiments/.
No training happens here; this API is read-only."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.experiments import ExperimentDetail, ExperimentSummary
from backend.app.services import results_registry

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentSummary])
def list_experiments() -> list[ExperimentSummary]:
    return results_registry.list_experiments()


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str) -> ExperimentDetail:
    detail = results_registry.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")
    return detail
