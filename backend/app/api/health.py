"""Health check endpoint."""

from fastapi import APIRouter

from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])

SERVICE_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report that the OrbitShield backend is running."""
    return HealthResponse(status="ok", service="orbitshield-backend", version=SERVICE_VERSION)
