from __future__ import annotations

import pytest

from src.simulation.satellite_client import SatelliteClient, make_client_ids


def test_satellite_client_initialization_with_valid_parameters():
    client = SatelliteClient(
        client_id="SAT-01",
        bandwidth_mbps=50.0,
        latency_ms=120.0,
        compute_score=0.7,
        availability_probability=0.9,
        connectivity_quality=0.8,
        data_size=1000,
    )
    assert client.client_id == "SAT-01"
    assert client.data_size == 1000
    assert client.bandwidth_mbps == 50.0


def test_satellite_client_data_size_defaults_to_zero():
    client = SatelliteClient(
        client_id="SAT-01",
        bandwidth_mbps=50.0,
        latency_ms=120.0,
        compute_score=0.7,
        availability_probability=0.9,
        connectivity_quality=0.8,
    )
    assert client.data_size == 0


def test_satellite_client_serialization_round_trip():
    client = SatelliteClient(
        client_id="SAT-02",
        bandwidth_mbps=42.5,
        latency_ms=200.0,
        compute_score=0.55,
        availability_probability=0.75,
        connectivity_quality=0.65,
        data_size=500,
    )
    data = client.to_dict()
    restored = SatelliteClient.from_dict(data)
    assert restored == client


def test_make_client_ids_default_prefix_and_padding():
    ids = make_client_ids(8)
    assert ids == [f"SAT-0{i}" for i in range(1, 9)]


def test_make_client_ids_double_digit_padding():
    ids = make_client_ids(10)
    assert ids[0] == "SAT-01"
    assert ids[-1] == "SAT-10"


def test_make_client_ids_custom_prefix():
    ids = make_client_ids(3, id_prefix="NODE")
    assert ids == ["NODE-01", "NODE-02", "NODE-03"]


def test_make_client_ids_rejects_invalid_count():
    with pytest.raises(ValueError):
        make_client_ids(0)
