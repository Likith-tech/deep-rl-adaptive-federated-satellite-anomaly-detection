"""ESA-ADB-specific static client signals for rule-based/DQN adaptive
aggregation (Phase 8/9B-style `src.federated.adaptive.ClientSignals`),
reusing that module's `performance`/`data`/`resource`/`fairness` shape
but built from ESA-ADB's REAL 8-client channel partition
(`src.satellite.esa_adb.clients.CLIENT_CHANNELS`) instead of Phase 4's
synthetic NSL-KDD satellite simulation.

Phase 4's simulated bandwidth/latency/compute/availability/connectivity
metadata (`src/federated/adaptive.py:build_resource_signal`,
`data/partitions/satellite_metadata.json`) is NSL-KDD-specific,
fabricated, and explicitly frozen/not reused for ESA-ADB. Rather than
inventing a second synthetic simulation for a real satellite telemetry
dataset, this module derives both signals from REAL, already-known
structural/statistical properties of each ESA-ADB client:

- resource: `1 / num_owned_channels`. Each client here IS a real
  satellite subsystem/group (`src.satellite.esa_adb.clients.
  CLIENT_SUBSYSTEM_GROUPS`); a client that owns more telemetry channels
  has more per-round local compute burden (more [X;M] columns feeding
  its own copy of the shared TemporalGRUAttention model, same
  architecture and cost per channel for every client). Fewer owned
  channels -> lower local compute burden -> higher resource
  desirability, so the raw signal is inverted (1/count) before the
  shared cross-client min-max `normalize()` in
  `src.federated.adaptive` rescales it. This is a real structural
  property of the approved partition, not a simulated one.
- fairness: binary Shannon entropy (bits) of the client's OWN local
  TRAIN label distribution (normal vs. anomaly), the direct ESA-ADB
  analogue of Phase 7/8's multi-category attack-distribution entropy
  (`src.simulation.non_iid_metrics.shannon_entropy`) — ESA-ADB is a
  binary anomaly-detection task, so "category diversity" collapses to
  "how close to a 50/50 normal/anomaly split is this client's own
  local data". A client whose local data is almost all-normal (entropy
  near 0) contributes less signal about anomalies per sample than one
  with a more balanced local mix; this rewards the latter, mirroring
  Phase 7/8's intent (don't let severely skewed/rare-signal clients get
  starved of aggregation weight) under ESA-ADB's actual label space.
  Computed ONCE from each client's TRAIN split only (never
  validation/test) — static for the whole experiment, exactly like
  Phase 8's category entropy.
"""

from __future__ import annotations

import math

import numpy as np

from src.satellite.esa_adb.clients import CLIENT_CHANNELS


def build_esa_adb_resource_signal(client_id: str) -> float:
    """Raw (pre cross-client-normalization) resource desirability for
    one client: `1 / num_owned_channels`. Higher is better (lighter
    local per-round compute burden), consistent with
    `src.federated.adaptive.build_resource_signal`'s "higher = better"
    convention."""
    num_channels = len(CLIENT_CHANNELS[client_id])
    if num_channels <= 0:
        raise ValueError(f"client {client_id!r} owns zero channels")
    return 1.0 / float(num_channels)


def binary_shannon_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (bits) of a binary 0/1 label array's empirical
    distribution. Returns 0.0 for a degenerate (all-one-class or empty)
    array — no signal, not undefined/NaN."""
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = float(np.asarray(labels).sum()) / n
    p0 = 1.0 - p1
    entropy = 0.0
    for p in (p0, p1):
        if p > 0.0:
            entropy -= p * math.log2(p)
    return entropy


def build_esa_adb_fairness_signals(client_train_labels: dict[str, np.ndarray]) -> dict[str, float]:
    """{client_id -> binary Shannon entropy of that client's own TRAIN
    label distribution}, computed once, train-split-only, exactly as
    described in this module's docstring."""
    return {cid: binary_shannon_entropy(labels) for cid, labels in client_train_labels.items()}


def build_esa_adb_resource_signals(client_ids: list[str]) -> dict[str, float]:
    return {cid: build_esa_adb_resource_signal(cid) for cid in client_ids}
