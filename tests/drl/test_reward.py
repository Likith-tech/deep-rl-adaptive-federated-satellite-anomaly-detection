from __future__ import annotations

import pytest

from src.drl.reward import compute_reward

PERFORMANCE_SIGNALS = {"SAT-01": 0.80, "SAT-02": 0.95, "SAT-03": 0.85}  # spread = 0.15


def test_reward_matches_exact_mathematical_formula():
    val_f1, val_f1_prev = 0.90, 0.85
    lambda_fair, lambda_stab = 0.1, 0.05
    action, previous_action = 2, 2  # no switch

    reward = compute_reward(val_f1, val_f1_prev, PERFORMANCE_SIGNALS, action, previous_action, lambda_fair, lambda_stab)

    delta_f1 = val_f1 - val_f1_prev
    perf_spread = max(PERFORMANCE_SIGNALS.values()) - min(PERFORMANCE_SIGNALS.values())
    expected = delta_f1 - lambda_fair * perf_spread - lambda_stab * 0.0
    assert reward == pytest.approx(expected)
    assert reward == pytest.approx(0.05 - 0.1 * 0.15 - 0.0)


def test_first_round_reward_has_zero_delta_f1():
    reward = compute_reward(0.90, None, PERFORMANCE_SIGNALS, action=0, previous_action=None, lambda_fair=0.1, lambda_stab=0.05)
    perf_spread = max(PERFORMANCE_SIGNALS.values()) - min(PERFORMANCE_SIGNALS.values())
    # delta_f1 = 0 (no prior round) AND no stability penalty (no previous action to switch from)
    assert reward == pytest.approx(0.0 - 0.1 * perf_spread - 0.0)


def test_action_switch_penalty_applied_only_when_action_changes():
    kwargs = dict(val_f1=0.9, val_f1_prev=0.9, performance_signals=PERFORMANCE_SIGNALS, lambda_fair=0.0, lambda_stab=0.05)
    reward_same_action = compute_reward(action=3, previous_action=3, **kwargs)
    reward_switched_action = compute_reward(action=4, previous_action=3, **kwargs)

    assert reward_same_action == pytest.approx(0.0)  # delta_f1=0, no fairness penalty (lambda_fair=0), no switch
    assert reward_switched_action == pytest.approx(-0.05)  # only the stability penalty


def test_performance_spread_penalty_is_computed_correctly():
    signals = {"SAT-01": 0.5, "SAT-02": 0.5, "SAT-03": 0.9}  # spread = 0.4
    reward = compute_reward(0.7, 0.7, signals, action=1, previous_action=1, lambda_fair=0.2, lambda_stab=0.0)
    assert reward == pytest.approx(0.0 - 0.2 * 0.4 - 0.0)


def test_zero_performance_spread_when_all_clients_equal():
    signals = {"SAT-01": 0.7, "SAT-02": 0.7, "SAT-03": 0.7}
    reward = compute_reward(0.8, 0.7, signals, action=0, previous_action=0, lambda_fair=0.5, lambda_stab=0.5)
    assert reward == pytest.approx(0.1)  # delta_f1 only, spread and switch penalty both zero


def test_compute_reward_requires_at_least_one_performance_signal():
    with pytest.raises(ValueError):
        compute_reward(0.8, 0.7, {}, action=0, previous_action=0, lambda_fair=0.1, lambda_stab=0.05)
