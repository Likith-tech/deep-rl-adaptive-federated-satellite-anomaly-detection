"""Phase 9B — THE FULL DQN ADAPTIVE FEDERATED LEARNING EXPERIMENT.

Run from the repository root:

    python scripts/run_phase9b_full_experiment.py

This is the real, full-budget run (NOT a pilot): multi-episode DQN
pretraining against the REAL 8-client NSL-KDD partition via the
UNMODIFIED Phase 6/8 `FederatedClient`/`FederatedServer`
(`src/federated/drl_round_provider.py`), followed by ONE frozen-policy
(epsilon=0), 10-round final run matching Phase 6/8's exact protocol,
model selection by validation loss ONLY, and exactly ONE final
KDDTest+ evaluation. KDDTest+ is never touched before that point.

DESIGN VALUES USED (all already approved; nothing invented here):
  - episodes = 50, rounds_per_episode = 15 — the LOWER bound of the
    approved range in configs/drl.yaml (`pretraining.episodes_min`,
    `pretraining.rounds_per_episode_min`), per explicit instruction to
    pick the lower/middle already-permitted value rather than stop for
    another design decision.
  - replay_warmup = dqn.batch_size (32) — the standard minimum
    (learn only once the buffer holds at least one full batch); not a
    new parameter, derived from the already-approved batch_size.
  - epsilon_decay_steps = episodes * rounds_per_episode (750) — the
    pilot config's `epsilon_decay_steps: 20` was itself explicitly
    scaled to the PILOT's 9-step budget; this run scales the same
    linear epsilon_start(1.0) -> epsilon_end(0.05) schedule to the
    REAL budget's step count. epsilon_start/epsilon_end themselves are
    unchanged from the approved values.
  - lambda_fair / lambda_stab, gamma, learning_rate, target_update_frequency,
    hidden_dimensions, replay_buffer_size: read directly from
    configs/drl.yaml, unchanged.

Writes (incrementally, so partial progress survives an interruption):
    experiments/drl_adaptive_fl/full_run/pretraining_log.jsonl   (one line per episode)
    experiments/drl_adaptive_fl/full_run/final_run.json
    experiments/drl_adaptive_fl/full_run/kddtest_evaluation.json
    experiments/drl_adaptive_fl/full_run/fairness_evaluation.json
    results/plots/drl_adaptive_fl/*.png
    results/reports/drl_adaptive_fl_results.md
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn-touching modules BEFORE torch/matplotlib — see
# the note in scripts/evaluate_local_models.py and
# scripts/evaluate_adaptive_fl_results.py about avoiding an
# intermittent access-violation/segfault crash on this Windows/
# Python 3.14 environment when the import order is reversed. Verified
# this turn: importing matplotlib/torch first before these modules DID
# segfault a smoke test; this order does not.
from src.drl.action_space import action_to_aggregation_weights, profile_name  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.federated.adaptive_trainer import _build_eval_model  # noqa: E402
from src.federated.drl_environment import DRLFederatedEnvironment  # noqa: E402
from src.federated.dqn_controller import DQNController  # noqa: E402
from src.federated.drl_round_provider import build_real_round_provider  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, run_validation  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "drl.yaml").read_text())
MODEL_CONFIG = CONFIG["model"]
TRAINING_CONFIG = CONFIG["federation"]  # local_epochs/batch_size/lr/optimizer/loss/seed
REWARD_CFG = CONFIG["reward"]
DQN_CFG = CONFIG["dqn"]
SHARED_INIT_SEED = CONFIG["initialization"]["shared_init_seed"]

EPISODES = 50           # pretraining.episodes_min (lower bound of the approved 50-80 range)
ROUNDS_PER_EPISODE = 15  # pretraining.rounds_per_episode_min (lower bound of the approved 15-20 range)
FINAL_RUN_ROUNDS = 10     # matches Phase 6/8's exact protocol for the final comparison run
REPLAY_WARMUP = DQN_CFG["batch_size"]
EPSILON_DECAY_STEPS = EPISODES * ROUNDS_PER_EPISODE

OUTPUT_DIR = REPO_ROOT / "experiments" / "drl_adaptive_fl" / "full_run"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "drl_adaptive_fl"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "drl_adaptive_fl_results.md"


def _evaluate_on_parquet(state_dict: dict, parquet_path: Path, feature_columns: list[str], device: torch.device):
    """Same pattern as scripts/evaluate_adaptive_fl_results.py's
    `evaluate_on_parquet` (Phase 8) — used here ONLY for the post-hoc
    fairness metric and the one final KDDTest+ evaluation, both AFTER
    the DQN policy is already frozen."""
    X, y = load_features_and_labels(parquet_path, feature_columns)
    loader = make_dataloader(X, y, batch_size=256, shuffle=False)
    model = _build_eval_model(MODEL_CONFIG, state_dict, device)
    criterion = nn.BCEWithLogitsLoss()
    return run_validation(model, loader, criterion, device)


def run_pretraining(controller: DQNController, log_file) -> None:
    for episode in range(1, EPISODES + 1):
        episode_start = time.time()
        provider = build_real_round_provider(
            repo_root=REPO_ROOT, model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
            shared_init_seed=SHARED_INIT_SEED, num_rounds=ROUNDS_PER_EPISODE,
        )
        env = DRLFederatedEnvironment(
            initial_signals_provider=provider.initial_signals_provider, round_provider=provider.round_provider,
            num_rounds=ROUNDS_PER_EPISODE, lambda_fair=REWARD_CFG["lambda_fair"], lambda_stab=REWARD_CFG["lambda_stab"],
        )
        state = env.reset()
        rounds_log = []
        total_reward = 0.0
        round_number = 1
        done = False
        while not done:
            performance_signals = {s.client_id: s.performance for s in env.current_signals}
            action = controller.choose_action(state)
            result = env.step(action)
            controller.remember(state, action, result.reward, result.observation, result.done)
            dqn_loss = controller.maybe_learn(replay_warmup=REPLAY_WARMUP)

            round_eval = provider.round_eval_history.get(round_number, {})
            perf_values = list(performance_signals.values())
            rounds_log.append({
                "round": round_number,
                "action": action, "action_name": profile_name(action),
                "weights": {k: round(v, 6) for k, v in result.info["weights"].items()},
                "reward": round(result.reward, 6),
                "val_f1": round(result.info["val_f1"], 6),
                "val_loss": round_eval.get("val_loss", None),
                "val_metrics": round_eval.get("val_metrics", None),
                "performance_spread": round(max(perf_values) - min(perf_values), 6),
                "epsilon": round(controller.epsilon(), 6),
                "dqn_loss": (round(dqn_loss, 6) if dqn_loss is not None else None),
                "replay_buffer_size": len(controller.buffer),
                "total_steps": controller.total_steps,
                "target_network_synced_this_step": controller.total_steps % controller.target_update_frequency == 0,
            })
            total_reward += result.reward
            state = result.observation
            done = result.done
            round_number += 1

        episode_record = {
            "episode": episode, "total_reward": round(total_reward, 6),
            "mean_reward": round(total_reward / len(rounds_log), 6),
            "rounds": rounds_log, "episode_seconds": round(time.time() - episode_start, 3),
        }
        log_file.write(json.dumps(episode_record) + "\n")
        log_file.flush()
        print(f"episode {episode:3d}/{EPISODES} | total_reward={total_reward:+.4f} | "
              f"mean_reward={episode_record['mean_reward']:+.4f} | epsilon={controller.epsilon():.4f} | "
              f"buffer={len(controller.buffer)} | {episode_record['episode_seconds']:.1f}s")


def run_final_frozen_evaluation(controller: DQNController) -> dict:
    """ONE frozen-policy (epsilon=0), FINAL_RUN_ROUNDS-round run on the
    real partition, matching Phase 6/8's exact protocol. Model
    selection by validation loss ONLY — KDDTest+ is not touched here."""
    provider = build_real_round_provider(
        repo_root=REPO_ROOT, model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
        shared_init_seed=SHARED_INIT_SEED, num_rounds=FINAL_RUN_ROUNDS,
    )
    env = DRLFederatedEnvironment(
        initial_signals_provider=provider.initial_signals_provider, round_provider=provider.round_provider,
        num_rounds=FINAL_RUN_ROUNDS, lambda_fair=REWARD_CFG["lambda_fair"], lambda_stab=REWARD_CFG["lambda_stab"],
    )
    zero_rng = np.random.default_rng(SHARED_INIT_SEED)
    state = env.reset()

    best_val_loss = float("inf")
    best_round = None
    best_state_dict = None
    rounds_log = []
    round_number = 1
    done = False
    while not done:
        action = controller.agent.act(state.astype(np.float32), epsilon=0.0, rng=zero_rng)  # frozen policy
        result = env.step(action)
        round_eval = provider.round_eval_history[round_number]
        rounds_log.append({
            "round": round_number, "action": action, "action_name": profile_name(action),
            "weights": {k: round(v, 6) for k, v in result.info["weights"].items()},
            "val_f1": round(result.info["val_f1"], 6), "val_loss": round_eval["val_loss"],
            "val_metrics": round_eval["val_metrics"],
        })
        if round_eval["val_loss"] < best_val_loss:
            best_val_loss = round_eval["val_loss"]
            best_round = round_number
            best_state_dict = {k: v.clone() for k, v in provider._last_aggregated_state.items()}
        state = result.observation
        done = result.done
        round_number += 1
        print(f"  final run round {round_number - 1}/{FINAL_RUN_ROUNDS} | action={action} ({profile_name(action)}) | "
              f"val_f1={round_eval['val_metrics']['f1']:.4f} | val_loss={round_eval['val_loss']:.4f}")

    return {
        "rounds": rounds_log, "best_round": best_round, "best_val_loss": best_val_loss,
        "best_state_dict": best_state_dict, "client_ids": provider.client_ids, "provider": provider,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    controller = DQNController(
        input_dim=DQN_CFG["input_dim"], output_dim=DQN_CFG["output_dim"], hidden_dimensions=tuple(DQN_CFG["hidden_dimensions"]),
        gamma=DQN_CFG["gamma"], learning_rate=DQN_CFG["learning_rate"], replay_buffer_size=DQN_CFG["replay_buffer_size"],
        batch_size=DQN_CFG["batch_size"], target_update_frequency=DQN_CFG["target_update_frequency"],
        epsilon_start=DQN_CFG["epsilon_start"], epsilon_end=DQN_CFG["epsilon_end"],
        epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=DQN_CFG["seed"],
    )

    print(f"=== Phase 9B FULL EXPERIMENT: {EPISODES} episodes x {ROUNDS_PER_EPISODE} rounds "
          f"({EPISODES * ROUNDS_PER_EPISODE} total env steps) ===")
    print(f"replay_warmup={REPLAY_WARMUP} epsilon_decay_steps={EPSILON_DECAY_STEPS}")

    log_path = OUTPUT_DIR / "pretraining_log.jsonl"
    with open(log_path, "w") as log_file:
        run_pretraining(controller, log_file)
    pretraining_seconds = time.time() - overall_start
    print(f"\nPretraining complete in {pretraining_seconds:.1f}s ({pretraining_seconds / 60:.1f} min). "
          f"Log: {log_path}")

    print(f"\n=== Final frozen-policy run ({FINAL_RUN_ROUNDS} rounds) ===")
    final_start = time.time()
    final_result = run_final_frozen_evaluation(controller)
    final_seconds = time.time() - final_start
    print(f"Final run complete in {final_seconds:.1f}s. Best round: {final_result['best_round']} "
          f"(val_loss={final_result['best_val_loss']:.4f})")

    (OUTPUT_DIR / "final_run.json").write_text(json.dumps(
        {"best_round": final_result["best_round"], "best_val_loss": final_result["best_val_loss"], "rounds": final_result["rounds"]},
        indent=2,
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_columns = json.loads((REPO_ROOT / "results" / "models" / "preprocessing" / "feature_columns.json").read_text())

    print("\n=== ONE final KDDTest+ evaluation (frozen best-round model) ===")
    kddtest_loss, kddtest_metrics = _evaluate_on_parquet(
        final_result["best_state_dict"], REPO_ROOT / "data" / "processed" / "test.parquet", feature_columns, device,
    )
    print(f"KDDTest+ : Acc={kddtest_metrics.accuracy:.4f} P={kddtest_metrics.precision:.4f} "
          f"R={kddtest_metrics.recall:.4f} F1={kddtest_metrics.f1:.4f} ROC-AUC={kddtest_metrics.roc_auc:.4f} "
          f"FPR={kddtest_metrics.false_positive_rate:.4f}")
    (OUTPUT_DIR / "kddtest_evaluation.json").write_text(json.dumps(
        {"val_loss_at_selection": final_result["best_val_loss"], "kddtest_loss": kddtest_loss, "kddtest_metrics": kddtest_metrics.to_dict()},
        indent=2,
    ))

    print("\n=== Post-hoc client fairness (Phase 8's own-local-data F1 gap metric) ===")
    partitions_dir = REPO_ROOT / "data" / "partitions"
    client_f1s = {}
    for cid in final_result["client_ids"]:
        _, client_metrics = _evaluate_on_parquet(final_result["best_state_dict"], partitions_dir / cid / "train.parquet", feature_columns, device)
        client_f1s[cid] = client_metrics.f1
        print(f"  {cid}: local F1 = {client_metrics.f1:.4f}")
    fairness = {"client_f1": client_f1s, "mean": float(np.mean(list(client_f1s.values()))), "min": float(np.min(list(client_f1s.values()))),
                "max": float(np.max(list(client_f1s.values()))), "gap": float(np.max(list(client_f1s.values())) - np.min(list(client_f1s.values())))}
    print(f"  fairness gap (max-min local F1): {fairness['gap']:.4f}")
    (OUTPUT_DIR / "fairness_evaluation.json").write_text(json.dumps(fairness, indent=2))

    total_seconds = time.time() - overall_start
    print(f"\n=== TOTAL wall-clock time: {total_seconds:.1f}s ({total_seconds / 60:.1f} min) ===")
    (OUTPUT_DIR / "timing.json").write_text(json.dumps(
        {"pretraining_seconds": pretraining_seconds, "final_run_seconds": final_seconds, "total_seconds": total_seconds}, indent=2,
    ))
    print("DONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUTPUT_DIR / "ERROR.txt").write_text(traceback.format_exc())
        raise
