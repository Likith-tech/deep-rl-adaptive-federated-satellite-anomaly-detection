"""Model registry — reuses the experiments registry (each real
experiment produces one final trained model), shaped as "model versions"
for the frontend's Model Registry page."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.experiments import ExperimentDetail, ExperimentSummary
from backend.app.services import results_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ExperimentSummary])
def list_models() -> list[ExperimentSummary]:
    return results_registry.list_experiments()


@router.get("/{model_id}", response_model=ExperimentDetail)
def get_model(model_id: str) -> ExperimentDetail:
    detail = results_registry.get_experiment(model_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    return detail
