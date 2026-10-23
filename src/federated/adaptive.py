"""Phase 8 — transparent, deterministic (NOT reinforcement-learned)
client scoring for adaptive Federated Learning.

    client_score_k = w_perf * performance_k
                    + w_data * data_k
                    + w_resource * resource_k
                    + w_fair * fairness_k

Every component is min-max normalized to [0, 1] ACROSS THE PARTICIPATING
CLIENTS (so the four components are comparable regardless of their
native scale), then combined with configured weights (`w_perf` +
`w_data` + `w_resource` + `w_fair` = 1, see configs/adaptive_fl.yaml).
The resulting scores are renormalized to sum to 1 and used directly as
FedAvg aggregation weights (`src/federated/server.py:aggregate_with_weights`).

This is explicitly NOT reinforcement learning: there is no reward
signal, no policy network, no learned parameters here at all — every
number above is a fixed, documented arithmetic rule, chosen by
reasoning about what should matter (see docs/project-progress/
09-phase-8-rule-based-adaptive-fl.md), not fit to any objective.

Component rationale:
- performance: this round's local update's F1 on the GLOBAL validation
  set (evaluated by the caller, src/federated/adaptive_trainer.py, by
  loading each client's just-trained state_dict — the client and
  server code themselves are unmodified from Phase 6). Rewards updates
  that are currently pulling the shared model in a useful direction.
- data: client's local training sample count — same intuition as
  standard FedAvg (more data -> generally a more reliable local
  gradient estimate), just one of several signals here instead of the
  only one.
- resource: simulated Phase 4 satellite conditions (bandwidth, compute
  score, availability probability, connectivity quality — higher
  better; latency — lower better, inverted before normalizing).
  SIMULATED, not real satellite telemetry — see
  docs/project-progress/05-phase-4-satellite-simulation.md.
- fairness: Shannon entropy (src/simulation/non_iid_metrics.py, the
  same function used in Phase 7) of the client's own local
  attack-category distribution. A client with a more diverse local mix
  (or holding rare categories like r2l/u2r) gets a fairness boost even
  if its raw performance is mediocre — this is the deliberate
  counterweight against "just give high performers all the weight",
  which risks erasing rare-category clients from the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

RULE_COMPONENTS = ("performance", "data", "resource", "fairness")


@dataclass
class ClientSignals:
    """The raw, unnormalized per-client inputs to the scoring rule for
    one round. `performance` varies every round (recomputed from that
    round's local update); `data`, `resource`, and `fairness` are
    static for a given experiment (the partition and simulated
    conditions don't change round to round)."""

    client_id: str
    performance: float  # this round's local update's F1 on global validation
    data: float  # local training sample count
    resource: float  # combined simulated resource desirability, see build_resource_signal
    fairness: float  # Shannon entropy (bits) of local category distribution


def build_resource_signal(resource_metadata: dict) -> float:
    """Combine the 5 simulated Phase 4 resource fields into one raw
    "resource desirability" number (higher = better) by min-max
    normalizing each field ACROSS the fields' own natural scale is not
    meaningful here — instead this returns a simple, transparent
    average of the already-0-1-ish fields plus rescaled bandwidth/
    latency, deferring cross-CLIENT normalization to `normalize`
    (called on the result across all clients in a round)."""
    bandwidth = resource_metadata["bandwidth_mbps"]
    latency = resource_metadata["latency_ms"]
    compute = resource_metadata["compute_score"]
    availability = resource_metadata["availability_probability"]
    connectivity = resource_metadata["connectivity_quality"]
    # Latency is "lower is better" — invert via reciprocal before the
    # cross-client min-max normalization step handles final scaling.
    inverted_latency = 1.0 / latency if latency > 0 else 0.0
    return bandwidth + inverted_latency + compute + availability + connectivity


def normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a {client_id: raw_value} dict to [0, 1] across
    clients. If every client has the same value (no signal that round),
    returns 0.5 for all — neutral, not zero, so a degenerate round
    doesn't zero out a component's contribution unfairly."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {cid: 0.5 for cid in values}
    return {cid: (v - lo) / (hi - lo) for cid, v in values.items()}


def compute_client_scores(signals: list[ClientSignals], rule_weights: dict[str, float]) -> dict[str, float]:
    """Compute each client's raw (pre-renormalization) score for this
    round. `rule_weights` must have keys matching RULE_COMPONENTS and
    sum to 1 (validated by the caller / config, not re-validated here
    to keep this function a pure, simple calculation)."""
    performance = normalize({s.client_id: s.performance for s in signals})
    data = normalize({s.client_id: s.data for s in signals})
    resource = normalize({s.client_id: s.resource for s in signals})
    fairness = normalize({s.client_id: s.fairness for s in signals})

    scores = {}
    for s in signals:
        scores[s.client_id] = (
            rule_weights["performance"] * performance[s.client_id]
            + rule_weights["data"] * data[s.client_id]
            + rule_weights["resource"] * resource[s.client_id]
            + rule_weights["fairness"] * fairness[s.client_id]
        )
    return scores


def scores_to_aggregation_weights(scores: dict[str, float]) -> dict[str, float]:
    """Renormalize raw scores to sum to 1 so they can be used directly
    as FedAvg aggregation weights. Guards against the degenerate
    all-zero case (falls back to uniform weighting rather than
    dividing by zero) — every weight is guaranteed non-negative and
    the set always sums to 1."""
    total = sum(scores.values())
    if total <= 1e-12:
        n = len(scores)
        return {cid: 1.0 / n for cid in scores}
    return {cid: v / total for cid, v in scores.items()}


def select_top_k_clients(scores: dict[str, float], k: int) -> list[str]:
    """Optional client-SELECTION utility (as opposed to weighting):
    the top-k clients by score. Implemented and unit-tested for reuse
    by a later phase, but Phase 8's experiments use aggregation
    weighting (all 8 clients keep participating every round) as the
    primary, safer mechanism — hard selection risks silently dropping
    a client that holds a rare attack category, which is exactly the
    fairness failure mode this phase is designed to avoid. See
    docs/project-progress/09-phase-8-rule-based-adaptive-fl.md."""
    if k >= len(scores):
        return list(scores.keys())
    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return ranked[:k]
