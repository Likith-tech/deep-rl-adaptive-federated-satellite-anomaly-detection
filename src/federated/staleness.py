"""Phase 9 — staleness- and resource-aware Federated Learning.

This project's Phase 4 satellite simulation already assigns every
client a SIMULATED `availability_probability` (see
docs/project-progress/05-phase-4-satellite-simulation.md), but no phase
before this one ever used it to affect round-to-round PARTICIPATION —
Phase 6/8/9B all had every client fetch the current global model and
train against it every single round. This module is the first to make
availability actually gate whether a client receives the latest global
model each round, which is what makes "staleness" a real, structural
property of a run rather than a label with nothing behind it.

Mechanism (deliberately the SAME philosophy Phase 8 already committed
to: weighting, not selection — see `select_top_k_clients`'s docstring):
every client still trains and submits an update every round, so no
client's data is ever silently dropped. What varies is WHICH global
model state a client trains against:

    each round, for each client:
        roll a client-specific, seeded coin using availability_probability
        if it "syncs": cache this round's global state, origin_round = this round
        train locally against whatever is currently cached (possibly old)
        staleness = current_round - origin_round

A client with high availability syncs almost every round (staleness
~0). A client with low availability sometimes trains against a global
model that is several rounds behind — a real, measured staleness, not
an assumed one. The resulting per-client staleness is then converted to
an aggregation-weight discount, combined with the existing sample-count
weight, and applied through the UNMODIFIED
`FederatedServer.aggregate_with_weights` (Phase 8's real extension
point — confirmed generic, no server change required here either).

This is explicitly NOT reinforcement learning and NOT Phase 8's
performance/data/resource/fairness rule (though it composes with it —
see `run_staleness_federated_training`'s `combine_with_adaptive_scores`
flag): staleness weighting is a fixed, documented arithmetic rule
chosen by reasoning, exactly like every non-DQN phase before it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

STALENESS_MODES = ("none", "linear", "exponential")


@dataclass
class ParticipationDecision:
    """One client's real (not assumed) participation outcome for one
    round — whether it synced to the latest global model this round,
    which round's model it actually trained against (`origin_round`),
    and the resulting staleness."""

    client_id: str
    synced_this_round: bool
    origin_round: int
    staleness: int


def decide_participation(
    client_id: str,
    round_number: int,
    availability_probability: float,
    previous_origin_round: int,
    seed: int,
) -> ParticipationDecision:
    """Deterministic, seeded coin flip: does this client receive the
    latest global model this round? `origin_round` is an index into the
    trainer's history of global states: 0 = the shared initial model,
    k = the model produced by aggregating round k. At round
    `round_number`, the latest available state has index
    `round_number - 1`. Round 1 always syncs for every client (there is
    no older model to fall back to at the very start — matches every
    prior phase's shared-initialization convention). The RNG is seeded
    from (seed, client_id, round_number) so a rerun with the same seed
    reproduces the exact same participation pattern, independent of
    client iteration order."""
    latest_index = round_number - 1
    if round_number == 1:
        return ParticipationDecision(client_id, True, latest_index, 0)

    rng = random.Random(f"{seed}:{client_id}:{round_number}")
    synced = rng.random() < availability_probability
    origin_round = latest_index if synced else previous_origin_round
    staleness = latest_index - origin_round
    return ParticipationDecision(client_id, synced, origin_round, staleness)


def staleness_weight(staleness: int, mode: str, decay_rate: float = 0.5) -> float:
    """Convert a client's staleness (0 = trained against this exact
    round's global model) into a [0, 1] discount factor, `mode`:
    - "none": always 1.0 — the control/no-correction condition. Staleness
      is measured and reported but never discounted.
    - "linear": 1 / (1 + staleness) — a simple, transparent decay.
    - "exponential": decay_rate ** staleness — discounts more
      aggressively for a chosen `decay_rate` in (0, 1)."""
    if staleness < 0:
        raise ValueError(f"staleness must be >= 0, got {staleness}")
    if mode == "none":
        return 1.0
    if mode == "linear":
        return 1.0 / (1.0 + staleness)
    if mode == "exponential":
        if not (0.0 < decay_rate < 1.0):
            raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}")
        return decay_rate**staleness
    raise ValueError(f"unknown staleness mode {mode!r}, expected one of {STALENESS_MODES}")


def combine_with_sample_weights(
    sample_counts: dict[str, int], staleness_weights: dict[str, float]
) -> dict[str, float]:
    """Multiply each client's raw sample count by its staleness
    discount, then renormalize to sum to 1 so the result can be used
    directly as a `FederatedServer.aggregate_with_weights` argument —
    same renormalization discipline as
    `src/federated/adaptive.py:scores_to_aggregation_weights` (falls
    back to uniform in the degenerate all-zero case rather than
    dividing by zero)."""
    raw = {cid: sample_counts[cid] * staleness_weights[cid] for cid in sample_counts}
    total = sum(raw.values())
    if total <= 1e-12:
        n = len(raw)
        return {cid: 1.0 / n for cid in raw}
    return {cid: v / total for cid, v in raw.items()}
