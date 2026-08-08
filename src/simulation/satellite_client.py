"""Reusable representation of one simulated satellite client.

Every field on `SatelliteClient` other than `client_id` and `data_size`
is a SIMULATED operating condition — not a measurement from a real
satellite. NSL-KDD (the underlying data) is real terrestrial network
traffic; the "satellite" framing here is a controlled simulation layer
used to build a realistic multi-client, non-IID environment for the
later Federated Learning / Adaptive FL / DRL phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SatelliteClient:
    """Metadata for one simulated satellite client.

    client_id: e.g. "SAT-01".
    data_size: number of local training records assigned to this client
        (filled in by the partitioner; 0 until partitioning has run).
    bandwidth_mbps: SIMULATED uplink/downlink bandwidth, megabits/second.
    latency_ms: SIMULATED one-way communication latency, milliseconds.
    compute_score: SIMULATED normalized compute capacity, in [0, 1]
        (1.0 = most capable simulated client).
    availability_probability: SIMULATED probability this client is
        reachable/online during a given communication round, in [0, 1].
    connectivity_quality: SIMULATED link quality independent of
        availability (e.g. captures signal quality / packet loss when
        the link IS up), in [0, 1].
    """

    client_id: str
    bandwidth_mbps: float
    latency_ms: float
    compute_score: float
    availability_probability: float
    connectivity_quality: float
    data_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SatelliteClient":
        return cls(**data)


def make_client_ids(num_clients: int, id_prefix: str = "SAT") -> list[str]:
    """e.g. make_client_ids(8) -> ["SAT-01", ..., "SAT-08"]."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    width = max(2, len(str(num_clients)))
    return [f"{id_prefix}-{i:0{width}d}" for i in range(1, num_clients + 1)]
