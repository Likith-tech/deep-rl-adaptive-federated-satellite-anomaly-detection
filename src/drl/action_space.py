"""Phase 9B — the discrete 9-profile weighting catalogue (the approved
action space; see docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
section 7).

Each action is a NAMED, complete `(performance, data, resource,
fairness)` weight vector — the DQN picks WHICH rule to apply this
round, not a new weighting mechanism. Every profile is passed through
the exact same, unmodified Phase 8 pipeline
(`src.federated.adaptive.compute_client_scores` ->
`scores_to_aggregation_weights`) that already produces valid FedAvg
aggregation weights — this module adds no new aggregation logic of its
own, per the explicit instruction not to create a second, incompatible
FedAvg implementation.

This is a weighting-only action space: every action still involves all
8 clients — there is no notion of excluding a client here.
"""

from __future__ import annotations

from src.federated.adaptive import (
    RULE_COMPONENTS,
    ClientSignals,
    compute_client_scores,
    scores_to_aggregation_weights,
)

# Profiles 0-3 are Phase 8's own four named rules, reused verbatim
# (performance_only / resource_only / data_only / combined — the exact
# `combined` weights from configs/adaptive_fl.yaml). Profiles 4-8 are a
# uniform profile plus four additional blends spanning the simplex, so
# the DQN has a genuinely distinct, complete catalogue to choose among
# — not a re-derivation of a single rule under different names.
ACTION_PROFILES: tuple[dict, ...] = (
    {"name": "performance_only", "weights": {"performance": 1.00, "data": 0.00, "resource": 0.00, "fairness": 0.00}},
    {"name": "resource_only", "weights": {"performance": 0.00, "data": 0.00, "resource": 1.00, "fairness": 0.00}},
    {"name": "data_only", "weights": {"performance": 0.00, "data": 1.00, "resource": 0.00, "fairness": 0.00}},
    {"name": "combined", "weights": {"performance": 0.40, "data": 0.25, "resource": 0.15, "fairness": 0.20}},
    {"name": "uniform", "weights": {"performance": 0.25, "data": 0.25, "resource": 0.25, "fairness": 0.25}},
    {"name": "fairness_leaning", "weights": {"performance": 0.20, "data": 0.20, "resource": 0.20, "fairness": 0.40}},
    {"name": "performance_leaning", "weights": {"performance": 0.55, "data": 0.15, "resource": 0.10, "fairness": 0.20}},
    {"name": "resource_leaning", "weights": {"performance": 0.20, "data": 0.20, "resource": 0.45, "fairness": 0.15}},
    {"name": "balanced_performance_fairness", "weights": {"performance": 0.35, "data": 0.15, "resource": 0.10, "fairness": 0.40}},
)

NUM_ACTIONS = len(ACTION_PROFILES)


def _validate_profiles() -> None:
    if NUM_ACTIONS != 9:
        raise AssertionError(f"ACTION_PROFILES must contain exactly 9 entries, found {NUM_ACTIONS}")
    for profile in ACTION_PROFILES:
        weights = profile["weights"]
        if set(weights.keys()) != set(RULE_COMPONENTS):
            raise AssertionError(f"profile {profile['name']!r} has keys {set(weights.keys())}, expected {set(RULE_COMPONENTS)}")
        if any(w < 0 for w in weights.values()):
            raise AssertionError(f"profile {profile['name']!r} has a negative weight: {weights}")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise AssertionError(f"profile {profile['name']!r} weights sum to {sum(weights.values())}, expected 1.0")


_validate_profiles()


def profile_name(action: int) -> str:
    _check_action(action)
    return ACTION_PROFILES[action]["name"]


def profile_weights(action: int) -> dict[str, float]:
    _check_action(action)
    return dict(ACTION_PROFILES[action]["weights"])


def _check_action(action: int) -> None:
    if not (0 <= action < NUM_ACTIONS):
        raise ValueError(f"action must be in [0, {NUM_ACTIONS}), got {action}")


def action_to_aggregation_weights(action: int, signals: list[ClientSignals]) -> dict[str, float]:
    """Deterministic action -> per-client FedAvg aggregation-weight
    mapping. Reuses Phase 8's exact, unmodified scoring pipeline — this
    function only selects which rule-weight vector feeds it. Output is
    guaranteed non-negative and summing to 1 by
    `scores_to_aggregation_weights`'s own contract (Phase 8, unchanged)."""
    _check_action(action)
    rule_weights = ACTION_PROFILES[action]["weights"]
    scores = compute_client_scores(signals, rule_weights)
    return scores_to_aggregation_weights(scores)
