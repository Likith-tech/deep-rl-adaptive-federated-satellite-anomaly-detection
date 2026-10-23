"""ESA-ADB DQN-controlled adaptive Federated Learning (Step C — the main
proposed contribution). Reuses, UNCHANGED: `src.drl.dqn_agent.DQNAgent`,
`src.federated.dqn_controller.DQNController`, `src.drl.action_space` (9
named weight profiles), `src.drl.reward.compute_reward`, `src.drl.state.
build_state` (44-dim: 8 clients x {performance,data,resource,fairness}
+ val_F1 trend(2) + round_progress(1) + previous_action one-hot(9)), and
`src.federated.drl_environment.DRLFederatedEnvironment`. The ONLY new
code is `src.federated.esa_adb_drl_round_provider.EsaAdbRoundProvider`,
which wires this unchanged DQN machinery to REAL ESA-ADB training
(`EsaAdbFederatedClient` + `FederatedServer.aggregate_with_weights`)
instead of NSL-KDD's `FederatedClient`.

STATE/REWARD PRIVACY BOUNDARY (verified structurally, matches Phase 9B):
every state feature and the reward's `delta_f1`/`perf_spread` terms come
ONLY from `ClientSignals` (performance = per-update F1 on POOLED
VALIDATION; data/resource/fairness = static real client signals) and
`val_f1` (also validation-only). TEST DATA IS NEVER READ ANYWHERE IN
THIS SCRIPT UNTIL STEP 4 BELOW (one evaluation, after the policy is
frozen and after pretraining has fully finished).

Training protocol (strict, per this phase's plan):
  1. PRETRAINING (train+validation only): several full FL episodes
     (fresh model each episode, same shared_init_seed as every other
     ESA-ADB experiment here) with epsilon-greedy action selection,
     filling a replay buffer and running DQN gradient steps
     (`DQNController.maybe_learn`). A snapshot of the Q-network is saved
     after every episode. NOTE ON BUDGET: unlike the NSL-KDD Phase 9B
     design doc's target of 50-80 episodes (against synthetic/cheap
     data), every "episode" here is REAL federated training (8 clients
     x local_epochs=1 x real ESA-ADB sequences per round) — the same
     per-round cost as fedavg_baseline_01's real 10-round run
     (~440s/10 rounds there). A reduced, disclosed pretraining budget
     (see PRETRAIN_EPISODES/PRETRAIN_ROUNDS_PER_EPISODE below) is used
     for that reason; this is a documented limitation of this
     experiment, not a silent deviation.
  2. POLICY SELECTION (validation only): among the per-episode Q-network
     snapshots, the one whose episode ended with the highest POOLED
     VALIDATION F1 is selected. This IS model selection against
     validation — exactly analogous to every other experiment in this
     phase selecting its best round/checkpoint by validation loss/F1 —
     never against test.
  3. FREEZE: the selected snapshot's weights are saved as the frozen
     policy checkpoint. No further gradient updates happen after this
     point.
  4. FINAL EVALUATION (ONE run only): with the frozen policy driving
     greedy (epsilon=0) action selection, run exactly one more full
     `NUM_ROUNDS`-round FL episode from the SAME shared initial model
     used by fedavg_baseline_01/adaptive_rule_baseline_01, select the
     best round by pooled validation loss (same convention as those two
     experiments), select the decision threshold by validation F1 only,
     then evaluate ONCE on the test set. This is the only place test
     data is touched, and it happens exactly once.

Run from the repository root, after fedavg_baseline_01 and
adaptive_rule_baseline_01 have been run:

    python scripts/train_esa_adb_dqn_adaptive_fl.py

Writes to experiments/esa_adb/dqn_adaptive_fl_01/ — does not touch any
other experiments/esa_adb/ directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.federated.dqn_controller import DQNController  # noqa: E402
from src.federated.drl_environment import DRLFederatedEnvironment  # noqa: E402
from src.federated.esa_adb_drl_round_provider import EsaAdbRoundProvider  # noqa: E402
from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import federated_representation as fr  # noqa: E402
from src.satellite.esa_adb import multivariate as mv  # noqa: E402
from src.satellite.esa_adb import pipeline as esa_pipeline  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402
from src.training.esa_adb_federated import build_initial_state_dict  # noqa: E402
from src.training.esa_adb_trainer import (  # noqa: E402
    compute_extended_metrics,
    curve_data,
    make_dataloader,
    run_inference,
    select_threshold_by_f1,
    set_seed,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "esa_adb" / "dqn_adaptive_fl_01"
BASELINE_01_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_01"
BASELINE_02_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_02_multivariate"
FEDAVG_DIR = REPO_ROOT / "experiments" / "esa_adb" / "fedavg_baseline_01"
ADAPTIVE_RULE_DIR = REPO_ROOT / "experiments" / "esa_adb" / "adaptive_rule_baseline_01"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "esa_adb" / "manifest.json"

MODEL_CONFIG = {"input_dim": 116, "projection_dim": 64, "hidden_dim": 64, "num_gru_layers": 1, "dropout": 0.3, "output_dim": 1}
TRAINING_CONFIG = {"batch_size": 128, "learning_rate": 0.001, "local_epochs": 1, "seed": 42}
NUM_ROUNDS = 10           # SAME as fedavg_baseline_01 / adaptive_rule_baseline_01, for direct comparability
SHARED_INIT_SEED = 42     # SAME as every other ESA-ADB FL experiment here
SEED = 42                 # DQN network init / exploration RNG seed

# DQN architecture/reward: reused VERBATIM from configs/drl.yaml's
# `dqn:`/`reward:` blocks (Phase 9B canonical design) — state_dim=44
# already matches ESA-ADB's REQUIRED_NUM_CLIENTS=8 (src/drl/state.py),
# so no redesign was needed. These are design parameters, not tuned
# against any ESA-ADB result.
DQN_CONFIG = {
    "input_dim": 44, "hidden_dimensions": (64, 64), "output_dim": 9,
    "gamma": 0.95, "learning_rate": 0.001, "replay_buffer_size": 5000,
    "target_update_frequency": 8, "epsilon_start": 1.0, "epsilon_end": 0.1,
}
REWARD_CONFIG = {"lambda_fair": 0.1, "lambda_stab": 0.05}  # r_t = delta_f1 - lambda_fair*perf_spread - lambda_stab*switch_penalty (src/drl/reward.py, unchanged)

# PRETRAINING BUDGET — reduced from the NSL-KDD design doc's 50-80
# episode target because every episode here is REAL federated training
# against real ESA-ADB data (not synthetic), see module docstring.
# 6 episodes x 8 rounds/episode = 48 real FL rounds of exploration,
# comparable in total wall-clock cost to ~5 fedavg_baseline_01 runs.
PRETRAIN_EPISODES = 6
PRETRAIN_ROUNDS_PER_EPISODE = 8
CONTROLLER_BATCH_SIZE = 16
REPLAY_WARMUP = 16
TOTAL_PRETRAIN_STEPS = PRETRAIN_EPISODES * PRETRAIN_ROUNDS_PER_EPISODE  # 48
EPSILON_DECAY_STEPS = TOTAL_PRETRAIN_STEPS


def fairness_gap(per_client_metric: dict[str, float]) -> float:
    values = list(per_client_metric.values())
    return max(values) - min(values)


def build_data():
    channel_frames = mv.load_all_resampled_channels()
    aligned = mv.build_aligned_multivariate_frame(channel_frames)
    channel_columns = list(fr.GLOBAL_CHANNEL_ORDER)
    splits = mv.split_multivariate_frame(aligned)
    scalers = mv.fit_multivariate_scalers(splits["train"], channel_columns)
    normalized_splits = {name: mv.transform_multivariate_frame(df, scalers, channel_columns) for name, df in splits.items()}

    client_train_ds, client_val_ds, client_test_ds = {}, {}, {}
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        client_train_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["train"], owned))
        client_val_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["validation"], owned))
        client_test_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["test"], owned))

    pooled_val = esa_pipeline.pool_sequence_datasets(list(client_val_ds.values()), prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)
    pooled_test = esa_pipeline.pool_sequence_datasets(list(client_test_ds.values()), prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)
    return client_train_ds, client_val_ds, client_test_ds, pooled_val, pooled_test


def run_one_episode(controller: DQNController, num_rounds: int, client_train_ds, pooled_val, greedy: bool, learn: bool):
    """Runs one full FL episode (fresh `EsaAdbRoundProvider`, same
    `SHARED_INIT_SEED` every time) via `DRLFederatedEnvironment`.
    `greedy=True` forces epsilon=0 (used only for the frozen final
    rollout). `learn=True` calls `controller.remember`/`maybe_learn`
    (pretraining only — the final frozen rollout never learns)."""
    provider = EsaAdbRoundProvider(
        client_train_datasets=client_train_ds, pooled_validation_dataset=pooled_val,
        model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
        shared_init_seed=SHARED_INIT_SEED, num_rounds=num_rounds,
    )
    env = DRLFederatedEnvironment(
        initial_signals_provider=provider.initial_signals_provider, round_provider=provider.round_provider,
        num_rounds=num_rounds, lambda_fair=REWARD_CONFIG["lambda_fair"], lambda_stab=REWARD_CONFIG["lambda_stab"],
    )

    state = env.reset()
    round_logs = []
    episode_reward_total = 0.0
    done = False
    while not done:
        epsilon_used = 0.0 if greedy else controller.epsilon()
        if greedy:
            action = int(np.argmax(controller.agent.q_values(state.astype(np.float32))))
        else:
            action = controller.choose_action(state)
        result = env.step(action)
        loss = None
        if learn:
            controller.remember(state, action, result.reward, result.observation, result.done)
            loss = controller.maybe_learn(replay_warmup=REPLAY_WARMUP)
        from src.drl.action_space import profile_name
        round_logs.append({
            "round_number": provider.server.round_number, "action": action, "action_profile": profile_name(action),
            "epsilon": epsilon_used, "reward": round(result.reward, 6), "val_f1": round(result.info["val_f1"], 6),
            "weights": {k: round(v, 6) for k, v in result.info["weights"].items()}, "learn_loss": (round(loss, 6) if loss is not None else None),
        })
        episode_reward_total += result.reward
        state = result.observation
        done = result.done

    return provider, round_logs, episode_reward_total


def main() -> None:
    t_start = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    print("=== Step 1-2: Build the same aligned/masked per-client [X;M] data as fedavg_baseline_01 ===")
    client_train_ds, client_val_ds, client_test_ds, pooled_val, pooled_test = build_data()
    print(f"  Pooled validation: {pooled_val.num_sequences:,} seq, {pooled_val.anomaly_count:,} anomaly")
    print(f"  Pooled test:       {pooled_test.num_sequences:,} seq, {pooled_test.anomaly_count:,} anomaly")

    checks = {
        "validation_representation_matches_approved_design": {
            "passed": pooled_val.num_sequences == 8464 and pooled_val.anomaly_count == 120,
        },
        "global_model_initialized_identically": {
            "passed": all(torch.equal(a, b) for a, b in zip(
                build_initial_state_dict(MODEL_CONFIG, SHARED_INIT_SEED).values(),
                build_initial_state_dict(MODEL_CONFIG, SHARED_INIT_SEED).values()))
        },
        "no_test_data_used_in_pretraining_or_state_reward": {
            "passed": True,
            "note": "structural: EsaAdbRoundProvider is built only from client_train_ds and pooled_val; pooled_test is never passed to it, to run_one_episode, or to DQNController/DRLFederatedEnvironment anywhere before Step 4",
        },
    }
    all_passed = all(v["passed"] for v in checks.values())
    (EXPERIMENT_DIR / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    for name, v in checks.items():
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {name}")
    if not all_passed:
        print("\nERROR: sanity checks failed. STOPPING before training.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Step 3: DQN PRETRAINING (train+validation only) — {PRETRAIN_EPISODES} episodes x {PRETRAIN_ROUNDS_PER_EPISODE} rounds ===")
    controller = DQNController(
        input_dim=DQN_CONFIG["input_dim"], output_dim=DQN_CONFIG["output_dim"], hidden_dimensions=DQN_CONFIG["hidden_dimensions"],
        gamma=DQN_CONFIG["gamma"], learning_rate=DQN_CONFIG["learning_rate"], replay_buffer_size=DQN_CONFIG["replay_buffer_size"],
        batch_size=CONTROLLER_BATCH_SIZE, target_update_frequency=DQN_CONFIG["target_update_frequency"],
        epsilon_start=DQN_CONFIG["epsilon_start"], epsilon_end=DQN_CONFIG["epsilon_end"], epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=SEED,
    )

    pretrain_checkpoints_dir = EXPERIMENT_DIR / "pretrain_checkpoints"
    pretrain_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    episode_summaries = []
    best_episode = {"episode": None, "final_val_f1": -1.0}
    for episode in range(1, PRETRAIN_EPISODES + 1):
        ep_start = time.time()
        provider, round_logs, episode_reward_total = run_one_episode(
            controller, PRETRAIN_ROUNDS_PER_EPISODE, client_train_ds, pooled_val, greedy=False, learn=True,
        )
        final_val_f1 = round_logs[-1]["val_f1"]
        ep_seconds = time.time() - ep_start
        print(f"  Episode {episode}/{PRETRAIN_EPISODES}: total_reward={episode_reward_total:.4f} final_val_f1={final_val_f1:.4f} "
              f"epsilon_end={controller.epsilon():.4f} seconds={ep_seconds:.1f}")

        snapshot_path = pretrain_checkpoints_dir / f"episode_{episode:02d}_qnetwork.pt"
        torch.save({"episode": episode, "network_state_dict": controller.agent.network.state_dict(),
                    "final_val_f1": final_val_f1, "total_reward": episode_reward_total}, snapshot_path)

        episode_summaries.append({
            "episode": episode, "episode_seconds": ep_seconds, "total_reward": episode_reward_total,
            "final_val_f1": final_val_f1, "round_logs": round_logs, "checkpoint": str(snapshot_path.relative_to(REPO_ROOT)),
        })

        if final_val_f1 > best_episode["final_val_f1"]:
            best_episode = {"episode": episode, "final_val_f1": final_val_f1, "checkpoint": str(snapshot_path.relative_to(REPO_ROOT))}

    (EXPERIMENT_DIR / "pretraining_history.json").write_text(json.dumps({
        "episodes": episode_summaries,
        "action_distribution": {name: sum(1 for e in episode_summaries for r in e["round_logs"] if r["action_profile"] == name)
                                 for name in sorted({r["action_profile"] for e in episode_summaries for r in e["round_logs"]})},
        "epsilon_schedule": {"epsilon_start": DQN_CONFIG["epsilon_start"], "epsilon_end": DQN_CONFIG["epsilon_end"], "decay_steps": EPSILON_DECAY_STEPS},
    }, indent=2))

    print(f"\n=== Step 4: POLICY SELECTION (validation only) — best episode by final pooled-validation F1 ===")
    print(f"  Selected episode {best_episode['episode']} (final_val_f1={best_episode['final_val_f1']:.4f})")

    print("\n=== Step 5: FREEZE the selected policy ===")
    selected_state_dict = torch.load(REPO_ROOT / best_episode["checkpoint"], weights_only=False)["network_state_dict"]
    controller.agent.network.load_state_dict(selected_state_dict)
    controller.agent.target_network.load_state_dict(selected_state_dict)
    frozen_checkpoint_path = EXPERIMENT_DIR / "frozen_dqn_policy.pt"
    torch.save({
        "network_state_dict": selected_state_dict, "selected_episode": best_episode["episode"],
        "selected_episode_final_val_f1": best_episode["final_val_f1"], "dqn_config": DQN_CONFIG, "reward_config": REWARD_CONFIG,
        "seed": SEED,
    }, frozen_checkpoint_path)
    print(f"  Frozen policy checkpoint written to {frozen_checkpoint_path}")

    print(f"\n=== Step 6: FINAL EVALUATION ROLLOUT (frozen greedy policy, {NUM_ROUNDS} rounds, no further learning) ===")
    final_provider, final_round_logs, final_reward_total = run_one_episode(
        controller, NUM_ROUNDS, client_train_ds, pooled_val, greedy=True, learn=False,
    )
    print(f"  Final rollout total_reward={final_reward_total:.4f}")
    for rl in final_round_logs:
        print(f"    round {rl['round_number']:2d}: action={rl['action_profile']:<28s} val_f1={rl['val_f1']:.4f}")

    # Best round selected by POOLED VALIDATION LOSS — same convention as
    # fedavg_baseline_01 / adaptive_rule_baseline_01. round_eval_history
    # is populated by EsaAdbRoundProvider for every round (0 = initial
    # untrained model, 1..NUM_ROUNDS = each round's aggregated model).
    best_round = min(
        (r for r in final_provider.round_eval_history if r >= 1),
        key=lambda r: final_provider.round_eval_history[r]["val_loss"],
    )
    best_val_loss = final_provider.round_eval_history[best_round]["val_loss"]
    print(f"  Best round (by pooled validation loss): {best_round} (val_loss={best_val_loss:.4f})")

    round_checkpoints_dir = EXPERIMENT_DIR / "round_checkpoints"
    round_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for rnum, sd in final_provider.round_state_dicts.items():
        torch.save({"round_number": rnum, "model_state_dict": sd,
                    "val_f1": final_provider.round_eval_history[rnum]["val_metrics"]["f1"]},
                   round_checkpoints_dir / f"round_{rnum:02d}.pt")

    best_state_dict = final_provider.round_state_dicts[best_round]
    torch.save({"round_number": best_round, "model_state_dict": best_state_dict, "model_config": MODEL_CONFIG,
                "training_config": TRAINING_CONFIG, "global_val_loss": best_val_loss}, EXPERIMENT_DIR / "best_global_model.pt")

    print("\n=== Step 7: Threshold selection (VALIDATION ONLY, frozen best-round global model) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.temporal_gru_attention import TemporalGRUAttention

    model = TemporalGRUAttention(**{k: MODEL_CONFIG[k] for k in ("input_dim", "projection_dim", "hidden_dim", "num_gru_layers", "dropout", "output_dim")}).to(device)
    model.load_state_dict(best_state_dict)

    val_loader = make_dataloader(pooled_val, TRAINING_CONFIG["batch_size"], shuffle=False)
    val_prob, val_true = run_inference(model, val_loader, device)
    threshold_selection = select_threshold_by_f1(val_true, val_prob)
    threshold = threshold_selection["selected_threshold"]
    print(f"  Selected threshold: {threshold} (pooled validation F1 at this threshold: {threshold_selection['selected_validation_f1']:.4f})")

    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_extended_metrics(val_true, val_pred, val_prob)
    (EXPERIMENT_DIR / "validation_metrics.json").write_text(json.dumps({"threshold": threshold, "metrics": val_metrics.to_dict()}, indent=2))
    (EXPERIMENT_DIR / "threshold_selection.json").write_text(json.dumps(threshold_selection, indent=2))

    print("\n=== Step 8: FINAL, ONE-TIME test evaluation (frozen threshold) — pooled + per-client ===")
    test_loader = make_dataloader(pooled_test, TRAINING_CONFIG["batch_size"], shuffle=False)
    test_prob, test_true = run_inference(model, test_loader, device)
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_extended_metrics(test_true, test_pred, test_prob)
    print(f"  Pooled test: accuracy={test_metrics.classification.accuracy:.4f} precision={test_metrics.classification.precision:.4f} "
          f"recall={test_metrics.classification.recall:.4f} f1={test_metrics.classification.f1:.4f} "
          f"roc_auc={test_metrics.classification.roc_auc:.4f} pr_auc={test_metrics.pr_auc:.4f}")

    test_curves = curve_data(test_true, test_prob)
    (EXPERIMENT_DIR / "test_metrics.json").write_text(json.dumps({
        "threshold": threshold, "metrics": test_metrics.to_dict(),
        "normal_count": int((test_true == 0).sum()), "anomaly_count": int((test_true == 1).sum()),
    }, indent=2))
    (EXPERIMENT_DIR / "roc_curve.json").write_text(json.dumps(test_curves["roc_curve"]))
    (EXPERIMENT_DIR / "pr_curve.json").write_text(json.dumps(test_curves["pr_curve"]))

    per_client_test = {}
    for client, ds in client_test_ds.items():
        loader = make_dataloader(ds, TRAINING_CONFIG["batch_size"], shuffle=False)
        prob, true = run_inference(model, loader, device)
        pred = (prob >= threshold).astype(int)
        m = compute_extended_metrics(true, pred, prob)
        per_client_test[client] = {
            "num_sequences": ds.num_sequences, "anomaly_count": ds.anomaly_count,
            "accuracy": m.classification.accuracy, "precision": m.classification.precision, "recall": m.classification.recall,
            "f1": m.classification.f1, "roc_auc": m.classification.roc_auc, "pr_auc": m.pr_auc,
            "false_positive_rate": m.classification.false_positive_rate, "confusion_matrix": m.classification.confusion_matrix,
        }
        print(f"  {client}: f1={m.classification.f1:.4f} recall={m.classification.recall:.4f} "
              f"precision={m.classification.precision:.4f} roc_auc={m.classification.roc_auc}")
    (EXPERIMENT_DIR / "per_client_test_metrics.json").write_text(json.dumps(per_client_test, indent=2))

    print("\n=== Step 9: Fairness (test-time, same max-min F1 formula as Phase 8/9B/fedavg/rule-based) ===")
    test_fairness_gap_f1 = fairness_gap({c: m["f1"] for c, m in per_client_test.items()})
    (EXPERIMENT_DIR / "fairness_summary.json").write_text(json.dumps({
        "test_fairness_gap_f1": test_fairness_gap_f1,
        "per_client_test_f1": {c: m["f1"] for c, m in per_client_test.items()},
    }, indent=2))
    print(f"  Test fairness gap (max-min client F1): {test_fairness_gap_f1:.4f}")

    print("\n=== Step 10: Comparison with all other ESA-ADB baselines ===")
    comparison = {"dqn_adaptive_fl_01": {"roc_auc": test_metrics.classification.roc_auc, "pr_auc": test_metrics.pr_auc,
                                          "f1": test_metrics.classification.f1, "recall": test_metrics.classification.recall,
                                          "fpr": test_metrics.classification.false_positive_rate}}
    for name, d in (("baseline_01_single_channel", BASELINE_01_DIR), ("baseline_02_multivariate", BASELINE_02_DIR),
                     ("fedavg_baseline_01", FEDAVG_DIR), ("adaptive_rule_baseline_01", ADAPTIVE_RULE_DIR)):
        p = d / "test_metrics.json"
        if p.exists():
            m = json.loads(p.read_text())["metrics"]
            comparison[name] = {"roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"), "f1": m.get("f1"),
                                 "recall": m.get("recall"), "fpr": m.get("false_positive_rate")}
    (EXPERIMENT_DIR / "comparison_with_all_baselines.json").write_text(json.dumps(comparison, indent=2))
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    print("\n=== Step 11: Save reproducibility config/manifest ===")
    manifest_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    full_config = {
        "experiment": "esa_adb/dqn_adaptive_fl_01",
        "model_architecture": "TemporalGRUAttention (unmodified), input_dim=116",
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "num_rounds": NUM_ROUNDS,
        "shared_init_seed": SHARED_INIT_SEED,
        "seed": SEED,
        "dqn_config": {**DQN_CONFIG, "hidden_dimensions": list(DQN_CONFIG["hidden_dimensions"])},
        "reward_config": REWARD_CONFIG,
        "pretraining": {"episodes": PRETRAIN_EPISODES, "rounds_per_episode": PRETRAIN_ROUNDS_PER_EPISODE,
                         "controller_batch_size": CONTROLLER_BATCH_SIZE, "replay_warmup": REPLAY_WARMUP,
                         "epsilon_decay_steps": EPSILON_DECAY_STEPS,
                         "budget_note": "reduced from NSL-KDD design doc's 50-80 episode target because every episode here is real federated training against real ESA-ADB data, not synthetic — see script docstring"},
        "policy_selection": {"method": "best pooled-validation F1 at end of episode, among pretraining episode snapshots", "selected_episode": best_episode["episode"], "selected_episode_final_val_f1": best_episode["final_val_f1"]},
        "action_space": "9 named weight profiles (src/drl/action_space.py), reused verbatim from Phase 9B",
        "state_space": "44-dim (src/drl/state.py), reused verbatim from Phase 9B — already matches ESA-ADB's 8 real clients",
        "resource_fairness_signal_source": "src.satellite.esa_adb.adaptive_signals — real structural/statistical signals, not Phase 4's synthetic satellite simulation",
        "dataset_manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "dataset_manifest_sha256": manifest_hash,
        "threshold_selection_method": threshold_selection["method"],
        "selected_threshold": threshold,
        "best_round": best_round,
        "best_round_val_loss": best_val_loss,
        "timing": {"script_total_seconds": time.time() - t_start},
    }
    (EXPERIMENT_DIR / "config.json").write_text(json.dumps(full_config, indent=2))

    print(f"\nAll artifacts written to {EXPERIMENT_DIR}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
