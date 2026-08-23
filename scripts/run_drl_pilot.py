"""Phase 9B — PILOT verification run (NOT a training run, NOT a
performance experiment).

Run from the repository root:

    python scripts/run_drl_pilot.py

This script proves the DRL controller's basic mechanics are correct
using synthetic, clearly-labeled-as-synthetic 8-client signals — it
never loads NSL-KDD, never touches a client partition file, and never
trains the DQN. It exercises exactly the pilot-scope components
implemented this turn: config loading, state construction, the 9-action
weighting catalogue, action -> aggregation-weight conversion, reward
calculation, and one DQN forward pass + action selection.

Writes:
    experiments/drl_adaptive_fl/pilot/pilot_verification.json
    results/plots/drl_adaptive_fl_pilot/action_profile_weights.png
    results/reports/drl_adaptive_fl_pilot.md

See docs/project-progress/12-phase-9b-drl-pilot.md for the full write-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.drl.action_space import (  # noqa: E402
    ACTION_PROFILES,
    NUM_ACTIONS,
    action_to_aggregation_weights,
    profile_name,
)
from src.drl.config import load_drl_config  # noqa: E402
from src.drl.dqn_agent import DQNAgent  # noqa: E402
from src.drl.reward import compute_reward  # noqa: E402
from src.drl.state import STATE_DIM, build_state  # noqa: E402
from src.federated.adaptive import ClientSignals  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "drl_adaptive_fl" / "pilot"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "drl_adaptive_fl_pilot"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "drl_adaptive_fl_pilot.md"

NUM_CLIENTS = 8


def synthetic_client_signals() -> list[ClientSignals]:
    """SYNTHETIC signals only — 8 fabricated clients, not derived from
    any real dataset or partition file. Used solely to exercise the
    pilot components' mechanics."""
    rng = np.random.default_rng(42)
    signals = []
    for i in range(NUM_CLIENTS):
        signals.append(
            ClientSignals(
                client_id=f"SAT-{i + 1:02d}",
                performance=float(rng.uniform(0.6, 0.95)),
                data=float(rng.uniform(1000, 15000)),
                resource=float(rng.uniform(0.2, 1.0)),
                fairness=float(rng.uniform(0.5, 3.0)),
            )
        )
    return signals


def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_drl_config(REPO_ROOT / "configs" / "drl.yaml")
    print(f"Loaded configs/drl.yaml — state.dimension={config['state']['dimension']}, "
          f"action.count={config['action']['count']}, federation.num_clients={config['federation']['num_clients']}")

    signals = synthetic_client_signals()
    assert len(signals) == NUM_CLIENTS

    # 1. Action -> aggregation weight conversion, for every one of the 9 profiles.
    action_results = {}
    for action in range(NUM_ACTIONS):
        weights = action_to_aggregation_weights(action, signals)
        action_results[profile_name(action)] = weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert all(w >= 0.0 for w in weights.values())

    # 2. State construction (round 1: no previous action/trend; round 2: with them).
    state_round1 = build_state(signals, val_f1=0.70, val_f1_prev=None, previous_action=None, round_number=1, num_rounds=10)
    state_round2 = build_state(signals, val_f1=0.74, val_f1_prev=0.70, previous_action=3, round_number=2, num_rounds=10)
    assert state_round1.shape == (STATE_DIM,) == (44,)
    assert state_round2.shape == (44,)
    assert np.all(np.isfinite(state_round1)) and np.all(np.isfinite(state_round2))

    # 3. Reward calculation for the round-1 -> round-2 transition.
    performance_signals = {s.client_id: s.performance for s in signals}
    reward = compute_reward(
        val_f1=0.74, val_f1_prev=0.70, performance_signals=performance_signals,
        action=3, previous_action=None,
        lambda_fair=config["reward"]["lambda_fair"], lambda_stab=config["reward"]["lambda_stab"],
    )

    # 4. DQN forward pass + action selection (no training — weights are the
    #    untrained, seeded random initialization only).
    agent = DQNAgent(seed=config["dqn"]["seed"])
    q_values = agent.q_values(state_round2.astype(np.float32))
    greedy_action = agent.act(state_round2.astype(np.float32), epsilon=0.0, rng=np.random.default_rng(0))
    assert q_values.shape == (9,)
    assert 0 <= greedy_action < 9

    summary = {
        "pilot_scope_only": True,
        "no_training_performed": True,
        "no_kddtest_used": True,
        "no_client_parquet_read": True,
        "config_path": "configs/drl.yaml",
        "state_dimension": int(state_round2.shape[0]),
        "num_actions": NUM_ACTIONS,
        "action_profile_names": [profile_name(a) for a in range(NUM_ACTIONS)],
        "example_action_to_weights": {name: {cid: round(w, 6) for cid, w in w_dict.items()} for name, w_dict in action_results.items()},
        "example_reward_round1_to_round2": {
            "val_f1_prev": 0.70, "val_f1": 0.74, "action": 3, "previous_action": None,
            "lambda_fair": config["reward"]["lambda_fair"], "lambda_stab": config["reward"]["lambda_stab"],
            "reward": round(reward, 6),
        },
        "dqn_untrained_forward_pass": {
            "seed": config["dqn"]["seed"],
            "q_values": [round(float(v), 6) for v in q_values],
            "greedy_action": greedy_action,
            "greedy_action_name": profile_name(greedy_action),
            "note": "Untrained network — this action choice reflects random initialization only, NOT a learned policy. Not a performance claim.",
        },
    }
    (EXPERIMENTS_DIR / "pilot_verification.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {EXPERIMENTS_DIR / 'pilot_verification.json'}")

    # Plot: the 9 action profiles' (performance/data/resource/fairness)
    # weight vectors — the fixed, deterministic catalogue itself, not a
    # performance result.
    fig, ax = plt.subplots(figsize=(10, 5))
    components = ("performance", "data", "resource", "fairness")
    x = np.arange(NUM_ACTIONS)
    width = 0.2
    for i, comp in enumerate(components):
        values = [ACTION_PROFILES[a]["weights"][comp] for a in range(NUM_ACTIONS)]
        ax.bar(x + i * width, values, width, label=comp)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([profile_name(a) for a in range(NUM_ACTIONS)], rotation=30, ha="right")
    ax.set_ylabel("rule weight")
    ax.set_title("Phase 9B pilot — the 9 discrete action profiles (fixed catalogue, not a result)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "action_profile_weights.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'action_profile_weights.png'}")

    write_report(summary)
    print(f"Wrote {REPORT_PATH}")


def write_report(summary: dict) -> None:
    lines = [
        "# Phase 9B — DRL Pilot Verification Report",
        "",
        "**This is a PILOT MECHANICS verification, NOT a performance",
        "result.** No DQN training occurred. No NSL-KDD data was loaded.",
        "No claim is made here that DQN improves (or worsens) anomaly",
        "detection — see",
        "`docs/project-progress/12-phase-9b-drl-pilot.md` for the full",
        "write-up and `docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`",
        "for the approved design this pilot verifies.",
        "",
        "## What this verifies",
        "",
        "- `configs/drl.yaml` loads and validates against the approved",
        "  design invariants (state dimension 44, action count 9, 8",
        "  clients).",
        "- All 9 action profiles produce valid (non-negative, sum-to-1)",
        "  per-client aggregation weights over 8 SYNTHETIC clients — via",
        "  the exact, unmodified Phase 8 `compute_client_scores` /",
        "  `scores_to_aggregation_weights` pipeline.",
        "- State construction produces the exact 44-dimensional vector",
        "  for both a first-round (no previous action/trend) and a",
        "  second-round (with them) case.",
        "- Reward calculation for one example round transition.",
        "- The DQN's untrained forward pass produces a 9-value Q-vector",
        "  and a valid greedy action index — reflecting only its random",
        "  seeded initialization, not any learned behavior.",
        "",
        "## Action profile catalogue (fixed, deterministic — not a result)",
        "",
        "| Profile | performance | data | resource | fairness |",
        "|---|---:|---:|---:|---:|",
    ]
    for action_index in range(9):
        from src.drl.action_space import ACTION_PROFILES  # local import to avoid unused top-level import lint noise

        p = ACTION_PROFILES[action_index]
        w = p["weights"]
        lines.append(f"| {p['name']} | {w['performance']:.2f} | {w['data']:.2f} | {w['resource']:.2f} | {w['fairness']:.2f} |")

    lines += [
        "",
        "## Example reward computation",
        "",
        f"```json\n{json.dumps(summary['example_reward_round1_to_round2'], indent=2)}\n```",
        "",
        "## Untrained DQN forward pass (random init only — not a policy)",
        "",
        f"```json\n{json.dumps(summary['dqn_untrained_forward_pass'], indent=2)}\n```",
        "",
        "## What this does NOT show",
        "",
        "- Whether a trained DQN improves or worsens anomaly detection —",
        "  no training was run.",
        "- Any comparison against Phase 6 FedAvg or Phase 8 rule-based",
        "  adaptive FL results — those remain untouched, real, committed",
        "  numbers this pilot did not re-derive or approach.",
        "- Performance on real NSL-KDD data — every signal here is",
        "  synthetic.",
        "",
        "See `results/plots/drl_adaptive_fl_pilot/action_profile_weights.png`",
        "for the visualized action catalogue and",
        "`experiments/drl_adaptive_fl/pilot/pilot_verification.json` for the",
        "full machine-readable output of this script.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
