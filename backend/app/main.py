"""OrbitShield backend entrypoint.

Phase 16 scope: read-only APIs over this project's real, already-
committed ML/FL/DRL results (results/, experiments/) — experiments/
models registry, federated overview/rounds/adaptive-control, and the
Phase 9B DQN controller. No training happens through this API. Domain
APIs backed by data that doesn't exist yet (live satellite telemetry,
a real threat feed) are intentionally not added — those routes stay on
the frontend's "Awaiting live data" placeholder until a real backing
system exists.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.drl import router as drl_router
from backend.app.api.experiments import router as experiments_router
from backend.app.api.federated import router as federated_router
from backend.app.api.health import router as health_router
from backend.app.api.models import router as models_router
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
app.include_router(experiments_router)
app.include_router(models_router)
app.include_router(federated_router)
app.include_router(drl_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "OrbitShield API", "status": "running"}
