"""Phase 9B — TINY deterministic pilot rollout (NOT the full pretraining
budget, NOT a performance experiment).

Run from the repository root:

    python scripts/run_drl_pilot_rollout.py

Exercises the full state -> action -> environment.step -> reward ->
replay buffer -> learn() loop end-to-end, using `configs/drl.yaml`'s
`pilot:` block (3 episodes x 3 rounds) and entirely SYNTHETIC,
fabricated 8-client signals via a deterministic `round_provider` — no
real NSL-KDD data, no client partition file, no FederatedClient/
FederatedServer training is used. Proves the environment/controller
integration works and is reproducible; makes NO claim about DQN
performance.

Writes:
    experiments/drl_adaptive_fl/pilot_run/rollout_summary.json
    results/plots/drl_adaptive_fl_pilot/pilot_rollout_rewards.png
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

from src.drl.config import load_drl_config  # noqa: E402
from src.federated.adaptive import ClientSignals  # noqa: E402
from src.federated.dqn_controller import DQNController  # noqa: E402
from src.federated.drl_environment import DRLFederatedEnvironment  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "experiments" / "drl_adaptive_fl" / "pilot_run"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "drl_adaptive_fl_pilot"

NUM_CLIENTS = 8


def synthetic_signals(episode_seed: int, round_number: int) -> list[ClientSignals]:
    rng = np.random.default_rng(episode_seed * 1000 + round_number)
    return [
        ClientSignals(
            client_id=f"SAT-{i + 1:02d}",
            performance=float(rng.uniform(0.6, 0.95)),
            data=float(rng.uniform(1000, 10000)),
            resource=float(rng.uniform(0.2, 1.0)),
            fairness=float(rng.uniform(0.5, 3.0)),
        )
        for i in range(NUM_CLIENTS)
    ]


def make_round_provider(episode_seed: int):
    def round_provider(round_number, weights, signals):
        # Deterministic, synthetic function of (round_number, weights) only.
        weight_concentration = sum(w * w for w in weights.values())  # higher when weights are less uniform
        new_val_f1 = 0.70 + 0.01 * round_number - 0.02 * weight_concentration
        return synthetic_signals(episode_seed, round_number + 1), new_val_f1

    return round_provider


def make_initial_signals_provider(episode_seed: int):
    def provider():
        return synthetic_signals(episode_seed, 1), 0.70

    return provider


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_drl_config(REPO_ROOT / "configs" / "drl.yaml")
    pilot_cfg = config["pilot"]
    dqn_cfg = config["dqn"]
    reward_cfg = config["reward"]

    controller = DQNController(
        input_dim=dqn_cfg["input_dim"], output_dim=dqn_cfg["output_dim"],
        hidden_dimensions=tuple(dqn_cfg["hidden_dimensions"]), gamma=dqn_cfg["gamma"],
        learning_rate=dqn_cfg["learning_rate"], replay_buffer_size=dqn_cfg["replay_buffer_size"],
        batch_size=pilot_cfg["batch_size"], target_update_frequency=dqn_cfg["target_update_frequency"],
        epsilon_start=dqn_cfg["epsilon_start"], epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_steps=dqn_cfg["epsilon_decay_steps"], seed=pilot_cfg["seed"],
    )

    episode_logs = []
    all_rewards = []
    all_losses = []

    for episode in range(1, pilot_cfg["episodes"] + 1):
        episode_seed = pilot_cfg["seed"] + episode
        env = DRLFederatedEnvironment(
            initial_signals_provider=make_initial_signals_provider(episode_seed),
            round_provider=make_round_provider(episode_seed),
            num_rounds=pilot_cfg["rounds_per_episode"],
            lambda_fair=reward_cfg["lambda_fair"], lambda_stab=reward_cfg["lambda_stab"],
        )
        state = env.reset()
        episode_reward_total = 0.0
        round_logs = []

        done = False
        while not done:
            action = controller.choose_action(state)
            result = env.step(action)
            controller.remember(state, action, result.reward, result.observation, result.done)
            loss = controller.maybe_learn(replay_warmup=pilot_cfg["replay_warmup"])

            round_logs.append({"action": action, "reward": round(result.reward, 6), "val_f1": round(result.info["val_f1"], 6), "loss": (round(loss, 6) if loss is not None else None)})
            all_rewards.append(result.reward)
            if loss is not None:
                all_losses.append(loss)

            episode_reward_total += result.reward
            state = result.observation
            done = result.done

        episode_logs.append({"episode": episode, "episode_seed": episode_seed, "total_reward": round(episode_reward_total, 6), "rounds": round_logs})
        print(f"episode {episode}: total_reward={episode_reward_total:.4f}")

    summary = {
        "pilot_scope_only": True,
        "no_full_pretraining_performed": True,
        "no_real_dataset_used": True,
        "no_client_parquet_read": True,
        "config": {"episodes": pilot_cfg["episodes"], "rounds_per_episode": pilot_cfg["rounds_per_episode"], "replay_warmup": pilot_cfg["replay_warmup"], "batch_size": pilot_cfg["batch_size"], "seed": pilot_cfg["seed"]},
        "total_env_steps": len(all_rewards),
        "total_learn_steps": len(all_losses),
        "episodes": episode_logs,
    }
    (OUTPUT_DIR / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'rollout_summary.json'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(all_rewards, marker="o")
    axes[0].set_title("Per-step reward (tiny synthetic pilot rollout)")
    axes[0].set_xlabel("env step")
    axes[0].set_ylabel("reward")
    if all_losses:
        axes[1].plot(all_losses, marker="o", color="orange")
        axes[1].set_title("learn() loss (untrained-network pilot, not a result)")
        axes[1].set_xlabel("learn step")
        axes[1].set_ylabel("MSE loss")
    else:
        axes[1].set_title("no learn() steps ran (buffer below replay_warmup)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pilot_rollout_rewards.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'pilot_rollout_rewards.png'}")


if __name__ == "__main__":
    main()
