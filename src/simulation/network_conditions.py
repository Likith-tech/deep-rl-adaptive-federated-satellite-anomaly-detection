"""Generate SIMULATED per-satellite network/resource conditions.

Every value produced here is a controlled simulation parameter for a
reproducible FL testbed — not a measurement from real satellite
hardware or a real communication link. Ranges are supplied by
configuration (configs/satellite_simulation.yaml) and sampled uniformly
at random from a seeded generator, so the same seed + config always
reproduces the same conditions.
"""

from __future__ import annotations

import numpy as np

RESOURCE_FIELDS = (
    "bandwidth_mbps",
    "latency_ms",
    "compute_score",
    "availability_probability",
    "connectivity_quality",
)


def generate_network_conditions(
    client_ids: list[str],
    resource_ranges: dict[str, list[float]],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Sample SIMULATED resource conditions for each client, uniformly
    within the configured [min, max] range per field.

    resource_ranges: {"bandwidth_mbps": [5, 100], "latency_ms": [50, 500], ...}
    Returns: {client_id: {field_name: value, ...}, ...}
    """
    missing = [f for f in RESOURCE_FIELDS if f not in resource_ranges]
    if missing:
        raise ValueError(f"resource_ranges missing required fields: {missing}")
    for field in RESOURCE_FIELDS:
        low, high = resource_ranges[field]
        if low > high:
            raise ValueError(f"resource_ranges[{field!r}] has min > max: {resource_ranges[field]}")

    rng = np.random.default_rng(seed)
    conditions: dict[str, dict[str, float]] = {}
    for client_id in client_ids:
        conditions[client_id] = {
            field: float(rng.uniform(*resource_ranges[field])) for field in RESOURCE_FIELDS
        }
    return conditions
