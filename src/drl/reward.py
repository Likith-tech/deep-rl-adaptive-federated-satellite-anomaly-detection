"""Phase 9B — the approved reward formula (see
docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
section 8, "Corrected 2026-08-15").

    r_t = delta_f1_t - lambda_fair * perf_spread_t
                      - lambda_stab * 1[action_t != action_{t-1}]

    delta_f1_t    = val_f1_t - val_f1_prev            (global validation set only)
    perf_spread_t = max_k(performance_k) - min_k(performance_k)

CRITICAL PRIVACY BOUNDARY: this module takes only plain floats and a
{client_id: float} dict of already-computed performance signals as
input. It performs NO file I/O of any kind, imports no data-loading
library, and never receives a client ID's raw dataset, a partition
path, or KDDTest+ — verified by a dedicated AST-based test
(tests/drl/test_reward.py), the same style already used for
src/federated/server.py's privacy boundary.

The true Phase 8 own-local-data fairness gap
(`max_k(local_F1_k) - min_k(local_F1_k)`, computed by
scripts/evaluate_adaptive_fl_results.py by reading every client's own
local training-partition file) is INTENTIONALLY NOT reproduced here —
it remains a POST-HOC-ONLY evaluation metric, computed once after
training, never read by this reward function or by the DQN.
"""

from __future__ import annotations


def compute_reward(
    val_f1: float,
    val_f1_prev: float | None,
    performance_signals: dict[str, float],
    action: int,
    previous_action: int | None,
    lambda_fair: float,
    lambda_stab: float,
) -> float:
    """Compute one round's scalar reward.

    First-round handling (explicit, not silent): `val_f1_prev is None`
    -> delta_f1 = 0.0 (no prior round to improve on yet).
    `previous_action is None` -> no stability penalty (there is no
    previous action to have switched away from).
    """
    delta_f1 = 0.0 if val_f1_prev is None else (val_f1 - val_f1_prev)

    if not performance_signals:
        raise ValueError("compute_reward requires at least one client performance signal")
    values = list(performance_signals.values())
    perf_spread = max(values) - min(values)

    switched = previous_action is not None and action != previous_action
    stability_penalty = 1.0 if switched else 0.0

    return delta_f1 - (lambda_fair * perf_spread) - (lambda_stab * stability_penalty)
