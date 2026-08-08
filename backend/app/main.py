"""OrbitShield backend entrypoint.

Phase 0 scope: application shell + health endpoint only. Domain APIs
(satellites, telemetry, threats, federated rounds, DRL decisions, etc.)
are added in later phases as the underlying systems are implemented.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.health import router as health_router
from backend.app.core.config import settings

app = FastAPI(
    title="OrbitShield API",
    description=(
        "Backend API for OrbitShield — the operational platform for the "
        "Deep Reinforcement Learning-Driven Adaptive Federated Framework "
        "for Satellite Network Anomaly Detection."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "OrbitShield API", "status": "running"}
