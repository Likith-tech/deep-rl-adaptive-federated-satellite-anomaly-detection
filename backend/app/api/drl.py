"""Phase 9B DQN controller summary + per-round decision log — real,
already-committed data, read-only. No DQN training happens here."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.drl import DRLControllerSummary, DRLDecision
from backend.app.services import results_registry

router = APIRouter(prefix="/drl", tags=["drl"])


@router.get("/controller", response_model=DRLControllerSummary)
def get_controller_summary() -> DRLControllerSummary:
    summary = results_registry.get_drl_controller_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="Phase 9B DQN run artifacts not found")
    return summary


@router.get("/decisions", response_model=list[DRLDecision])
def get_decisions() -> list[DRLDecision]:
    return results_registry.get_drl_decisions()
