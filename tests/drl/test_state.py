from __future__ import annotations

import numpy as np
import pytest

from src.drl.state import STATE_DIM, build_state


def test_state_dimension_is_exactly_44(synthetic_signals):
    state = build_state(synthetic_signals, val_f1=0.9, val_f1_prev=0.88, previous_action=3, round_number=4, num_rounds=10)
    assert STATE_DIM == 44
    assert state.shape == (44,)


def test_state_component_ordering_is_deterministic(synthetic_signals):
    state_a = build_state(synthetic_signals, val_f1=0.9, val_f1_prev=0.88, previous_action=2, round_number=1, num_rounds=10)
    state_b = build_state(synthetic_signals, val_f1=0.9, val_f1_prev=0.88, previous_action=2, round_number=1, num_rounds=10)
    assert np.array_equal(state_a, state_b)

    # Shuffling the INPUT list order must not change the OUTPUT vector —
    # client ordering inside the state is fixed by sorted client_id, not
    # by input order.
    shuffled = list(reversed(synthetic_signals))
    state_shuffled = build_state(shuffled, val_f1=0.9, val_f1_prev=0.88, previous_action=2, round_number=1, num_rounds=10)
    assert np.array_equal(state_a, state_shuffled)


def test_state_values_are_finite(synthetic_signals, degenerate_signals):
    state = build_state(synthetic_signals, val_f1=0.9, val_f1_prev=0.88, previous_action=0, round_number=5, num_rounds=10)
    assert np.all(np.isfinite(state))

    # Degenerate (all-equal) signals must not produce NaN/inf — relies
    # on adaptive.normalize's documented 0.5 fallback.
    degenerate_state = build_state(degenerate_signals, val_f1=0.5, val_f1_prev=None, previous_action=None, round_number=1, num_rounds=10)
    assert np.all(np.isfinite(degenerate_state))


def test_first_round_state_has_no_previous_action_and_flat_trend(synthetic_signals):
    state = build_state(synthetic_signals, val_f1=0.7, val_f1_prev=None, previous_action=None, round_number=1, num_rounds=10)
    # val_F1_t and val_F1_{t-1} slots (indices 32, 33) both equal val_f1 (zero implied trend).
    assert state[32] == pytest.approx(0.7)
    assert state[33] == pytest.approx(0.7)
    # previous-action one-hot block (last 9 entries) is all-zero.
    assert np.all(state[-9:] == 0.0)


def test_previous_action_one_hot_is_set_correctly(synthetic_signals):
    state = build_state(synthetic_signals, val_f1=0.7, val_f1_prev=0.65, previous_action=4, round_number=2, num_rounds=10)
    one_hot = state[-9:]
    assert one_hot.sum() == 1.0
    assert one_hot[4] == 1.0


def test_build_state_rejects_wrong_number_of_clients(synthetic_signals):
    with pytest.raises(ValueError):
        build_state(synthetic_signals[:7], val_f1=0.7, val_f1_prev=0.6, previous_action=0, round_number=1, num_rounds=10)
    with pytest.raises(ValueError):
        build_state(synthetic_signals + [synthetic_signals[0]], val_f1=0.7, val_f1_prev=0.6, previous_action=0, round_number=1, num_rounds=10)


def test_build_state_rejects_invalid_previous_action(synthetic_signals):
    with pytest.raises(ValueError):
        build_state(synthetic_signals, val_f1=0.7, val_f1_prev=0.6, previous_action=9, round_number=1, num_rounds=10)
