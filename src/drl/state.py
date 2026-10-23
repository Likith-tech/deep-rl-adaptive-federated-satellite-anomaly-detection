"""Phase 9B — the approved 44-dimensional state construction (see
docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
section 6/12).

Built ENTIRELY from signals Phase 8 already computes server-side
(`src.federated.adaptive.normalize`, applied to `ClientSignals`) — no
new information channel, no client-local data access. `fairness_k`
here is the SAME Shannon-entropy signal Phase 8 already uses in its
live scoring rule (a static, precomputed per-client scalar handed in as
part of `ClientSignals.fairness`) — never the post-hoc, parquet-reading
local-F1 fairness gap (see reward.py's docstring for that distinction).

Component order (fixed, deterministic — client IDs sorted
lexicographically, e.g. SAT-01..SAT-08):

    performance_k (8) + data_k (8) + resource_k (8) + fairness_k (8)
    + val_F1_t, val_F1_{t-1} (2)
    + round / num_rounds (1)
    + previous_action one-hot (NUM_ACTIONS = 9)
    = 44
"""

from __future__ import annotations

import numpy as np

from src.drl.action_space import NUM_ACTIONS
from src.federated.adaptive import ClientSignals, normalize

STATE_DIM = 8 * 4 + 2 + 1 + NUM_ACTIONS
assert STATE_DIM == 44, f"approved design requires state dimension 44, computed {STATE_DIM}"

REQUIRED_NUM_CLIENTS = 8


def build_state(
    signals: list[ClientSignals],
    val_f1: float,
    val_f1_prev: float | None,
    previous_action: int | None,
    round_number: int,
    num_rounds: int,
) -> np.ndarray:
    """Build the exact 44-dim state vector for one FL round.

    `val_f1_prev` and `previous_action` are `None` on the first round of
    an episode (there is no previous round yet) — this is NOT a missing
    value: the trend slot is filled with `val_f1` itself (zero implied
    trend) and the previous-action one-hot is left all-zero (no action
    has been taken yet), both explicit, documented conventions, not
    silent padding.

    Raises `ValueError` if `signals` does not contain exactly 8 clients
    — this function never silently pads or truncates.
    """
    if len(signals) != REQUIRED_NUM_CLIENTS:
        raise ValueError(f"build_state requires exactly {REQUIRED_NUM_CLIENTS} client signals, got {len(signals)}")

    performance = normalize({s.client_id: s.performance for s in signals})
    data = normalize({s.client_id: s.data for s in signals})
    resource = normalize({s.client_id: s.resource for s in signals})
    fairness = normalize({s.client_id: s.fairness for s in signals})

    ordered_ids = sorted(performance.keys())

    components: list[float] = []
    for block in (performance, data, resource, fairness):
        components.extend(block[cid] for cid in ordered_ids)

    components.append(float(val_f1))
    components.append(float(val_f1_prev) if val_f1_prev is not None else float(val_f1))
    components.append(float(round_number) / float(num_rounds))

    previous_action_one_hot = [0.0] * NUM_ACTIONS
    if previous_action is not None:
        if not (0 <= previous_action < NUM_ACTIONS):
            raise ValueError(f"previous_action must be in [0, {NUM_ACTIONS}) or None, got {previous_action}")
        previous_action_one_hot[previous_action] = 1.0
    components.extend(previous_action_one_hot)

    state = np.asarray(components, dtype=np.float64)

    if state.shape != (STATE_DIM,):
        raise AssertionError(f"state construction produced shape {state.shape}, expected ({STATE_DIM},)")
    if not np.all(np.isfinite(state)):
        raise AssertionError("state contains non-finite (NaN/inf) values")

    return state
