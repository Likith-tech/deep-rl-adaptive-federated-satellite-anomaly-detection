from __future__ import annotations

import pytest

from src.federated.adaptive import ClientSignals

NUM_SYNTHETIC_CLIENTS = 8


@pytest.fixture
def synthetic_signals() -> list[ClientSignals]:
    """8 synthetic clients with distinct, non-degenerate performance/
    data/resource/fairness values — never derived from real NSL-KDD
    data or any file on disk. `performance` increases with client index
    while `resource` DECREASES with it (deliberately anti-correlated),
    so tests can verify that different rule profiles genuinely favor
    different clients rather than coincidentally agreeing because both
    signals happen to rank clients identically."""
    return [
        ClientSignals(
            client_id=f"SAT-{i + 1:02d}",
            performance=0.50 + 0.05 * i,
            data=1000.0 * (i + 1),
            resource=0.30 + 0.07 * (NUM_SYNTHETIC_CLIENTS - 1 - i),
            fairness=1.0 + 0.10 * i,
        )
        for i in range(NUM_SYNTHETIC_CLIENTS)
    ]


@pytest.fixture
def degenerate_signals() -> list[ClientSignals]:
    """8 synthetic clients with IDENTICAL signals — exercises
    `normalize`'s degenerate (all-equal) fallback to 0.5."""
    return [
        ClientSignals(client_id=f"SAT-{i + 1:02d}", performance=0.9, data=5000.0, resource=0.5, fairness=2.0)
        for i in range(NUM_SYNTHETIC_CLIENTS)
    ]
