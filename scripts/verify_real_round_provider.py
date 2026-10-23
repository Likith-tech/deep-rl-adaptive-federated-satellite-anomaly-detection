"""Phase 9B — ONE real-data execution to validate
`RealNSLKDDRoundProvider` end-to-end, with granular timing.

Run from the repository root:

    python scripts/verify_real_round_provider.py

This is a VALIDATION CHECKPOINT, not a DQN experiment: it runs exactly
initialization -> round 1 (train + a fixed, non-learned "combined"
aggregation profile) -> round 2 (only to prove the round_provider's
"continue" branch, i.e. that it can train the NEXT round from the
newly-aggregated model). No DQN, no replay buffer, no multi-episode
loop, no KDDTest+ access.

Writes:
    experiments/drl_adaptive_fl/real_round_provider_verification/verification_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.drl.action_space import action_to_aggregation_weights, profile_name  # noqa: E402
from src.federated.drl_round_provider import build_real_round_provider  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "experiments" / "drl_adaptive_fl" / "real_round_provider_verification"

MODEL_CONFIG = {"type": "mlp", "input_dim": 121, "hidden_dimensions": [128, 64], "dropout": 0.3, "output_dim": 1}
TRAINING_CONFIG = {"local_epochs": 1, "batch_size": 128, "learning_rate": 0.001, "optimizer": "adam", "loss": "bce_with_logits", "seed": 42}
SEED = 42
NUM_ROUNDS = 2  # minimum to prove round_provider's "continue" branch once; NOT a DQN episode


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    overall_start = time.time()

    t0 = time.time()
    provider = build_real_round_provider(
        repo_root=REPO_ROOT, model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
        shared_init_seed=SEED, num_rounds=NUM_ROUNDS,
    )
    timings["initialization_seconds"] = time.time() - t0
    print(f"Provider initialized in {timings['initialization_seconds']:.3f}s "
          f"({len(provider.clients)} clients: {provider.client_ids})")

    t0 = time.time()
    round1_signals, initial_val_f1 = provider.initial_signals_provider()
    timings["round_1_client_training_and_eval_seconds"] = time.time() - t0
    print(f"Round 1 client training + per-client eval: {timings['round_1_client_training_and_eval_seconds']:.3f}s")
    print(f"Initial (untrained) global model validation F1: {initial_val_f1:.4f}")
    for s in round1_signals:
        print(f"  {s.client_id}: performance={s.performance:.4f} data={s.data:.0f} resource={s.resource:.4f} fairness={s.fairness:.4f}")

    action = 3  # "combined" — a FIXED, non-DQN profile, used only to exercise the interface
    weights = action_to_aggregation_weights(action, round1_signals)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in weights.values())
    print(f"Action {action} ({profile_name(action)}) -> weights: { {k: round(v, 4) for k, v in weights.items()} }")

    t0 = time.time()
    round2_signals, val_f1_after_round1 = provider.round_provider(1, weights, round1_signals)
    timings["round_1_aggregation_and_round_2_training_seconds"] = time.time() - t0
    print(f"Round 1 aggregation + round 2 client training: {timings['round_1_aggregation_and_round_2_training_seconds']:.3f}s")
    print(f"Validation F1 after round 1 aggregation: {val_f1_after_round1:.4f}")

    action_2 = 3
    weights_2 = action_to_aggregation_weights(action_2, round2_signals)
    t0 = time.time()
    round3_signals_placeholder, val_f1_after_round2 = provider.round_provider(2, weights_2, round2_signals)
    timings["round_2_aggregation_seconds"] = time.time() - t0
    print(f"Round 2 aggregation (episode end, no round 3 training): {timings['round_2_aggregation_seconds']:.3f}s")
    print(f"Validation F1 after round 2 aggregation: {val_f1_after_round2:.4f}")

    timings["total_seconds"] = time.time() - overall_start
    print(f"\nTotal wall-clock time: {timings['total_seconds']:.3f}s")

    summary = {
        "purpose": "ONE real-data validation execution of RealNSLKDDRoundProvider — NOT a DQN experiment, NOT a Phase 9B result.",
        "no_dqn_training": True,
        "no_kddtest_accessed": True,
        "num_rounds_executed": NUM_ROUNDS,
        "seed": SEED,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "client_ids": provider.client_ids,
        "initial_val_f1": initial_val_f1,
        "val_f1_after_round_1": val_f1_after_round1,
        "val_f1_after_round_2": val_f1_after_round2,
        "round_1_action": {"index": action, "name": profile_name(action), "weights": weights},
        "round_2_action": {"index": action_2, "name": profile_name(action_2), "weights": weights_2},
        "round_1_signals": [{"client_id": s.client_id, "performance": s.performance, "data": s.data, "resource": s.resource, "fairness": s.fairness} for s in round1_signals],
        "timings": timings,
    }
    (OUTPUT_DIR / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT_DIR / 'verification_summary.json'}")


if __name__ == "__main__":
    main()
