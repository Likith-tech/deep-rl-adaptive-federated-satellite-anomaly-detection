from __future__ import annotations

import pytest

from src.simulation.network_conditions import RESOURCE_FIELDS, generate_network_conditions

RANGES = {
    "bandwidth_mbps": [5, 100],
    "latency_ms": [50, 500],
    "compute_score": [0.2, 1.0],
    "availability_probability": [0.6, 1.0],
    "connectivity_quality": [0.3, 1.0],
}


def test_generate_network_conditions_returns_all_clients():
    client_ids = ["SAT-01", "SAT-02", "SAT-03"]
    conditions = generate_network_conditions(client_ids, RANGES, seed=42)
    assert set(conditions.keys()) == set(client_ids)


def test_generate_network_conditions_values_within_configured_bounds():
    client_ids = [f"SAT-{i:02d}" for i in range(1, 9)]
    conditions = generate_network_conditions(client_ids, RANGES, seed=42)
    for client_id, values in conditions.items():
        for field in RESOURCE_FIELDS:
            low, high = RANGES[field]
            assert low <= values[field] <= high, f"{client_id}.{field}={values[field]} out of [{low},{high}]"


def test_generate_network_conditions_deterministic_with_same_seed():
    client_ids = ["SAT-01", "SAT-02"]
    a = generate_network_conditions(client_ids, RANGES, seed=42)
    b = generate_network_conditions(client_ids, RANGES, seed=42)
    assert a == b


def test_generate_network_conditions_different_seed_changes_values():
    client_ids = ["SAT-01", "SAT-02"]
    a = generate_network_conditions(client_ids, RANGES, seed=42)
    b = generate_network_conditions(client_ids, RANGES, seed=7)
    assert a != b


def test_generate_network_conditions_raises_on_missing_field():
    incomplete_ranges = {k: v for k, v in RANGES.items() if k != "latency_ms"}
    with pytest.raises(ValueError):
        generate_network_conditions(["SAT-01"], incomplete_ranges, seed=42)


def test_generate_network_conditions_raises_on_invalid_range():
    bad_ranges = dict(RANGES)
    bad_ranges["bandwidth_mbps"] = [100, 5]  # min > max
    with pytest.raises(ValueError):
        generate_network_conditions(["SAT-01"], bad_ranges, seed=42)
