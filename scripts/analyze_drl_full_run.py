"""Phase 9B — analyze the full-experiment logs and generate plots +
the final results report. Run AFTER scripts/run_phase9b_full_experiment.py.

    python scripts/analyze_drl_full_run.py

Reads:
    experiments/drl_adaptive_fl/full_run/{pretraining_log.jsonl,final_run.json,
                                            kddtest_evaluation.json,fairness_evaluation.json,timing.json}
Writes:
    results/plots/drl_adaptive_fl/*.png
    results/reports/drl_adaptive_fl_results.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN_DIR = REPO_ROOT / "experiments" / "drl_adaptive_fl" / "full_run"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "drl_adaptive_fl"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "drl_adaptive_fl_results.md"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

episodes = [json.loads(line) for line in (RUN_DIR / "pretraining_log.jsonl").read_text().splitlines()]
final_run = json.loads((RUN_DIR / "final_run.json").read_text())
kddtest = json.loads((RUN_DIR / "kddtest_evaluation.json").read_text())
fairness = json.loads((RUN_DIR / "fairness_evaluation.json").read_text())
timing = json.loads((RUN_DIR / "timing.json").read_text())

# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------
episode_rewards = [e["total_reward"] for e in episodes]
episode_mean_rewards = [e["mean_reward"] for e in episodes]
all_rounds = [r for e in episodes for r in e["rounds"]]
all_val_f1 = [r["val_f1"] for r in all_rounds]
all_epsilon = [r["epsilon"] for r in all_rounds]
all_losses = [r["dqn_loss"] for r in all_rounds if r["dqn_loss"] is not None]
action_counts = Counter(r["action_name"] for r in all_rounds)
last_10_action_counts = Counter(r["action_name"] for e in episodes[-10:] for r in e["rounds"])

print(f"Episodes: {len(episodes)}, total rounds: {len(all_rounds)}")
print(f"Episode reward: first={episode_rewards[0]:+.4f} last={episode_rewards[-1]:+.4f} "
      f"mean_all={np.mean(episode_rewards):+.4f} mean_last10={np.mean(episode_rewards[-10:]):+.4f}")
print(f"Action distribution (all rounds): {dict(action_counts)}")
print(f"Action distribution (last 10 episodes): {dict(last_10_action_counts)}")
print(f"DQN loss: first={all_losses[0]:.4f} last={all_losses[-1]:.4f} count={len(all_losses)}")

# ---------------------------------------------------------------------------
# Plot 1: reward by episode
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker=".", label="total reward")
ax.plot(range(1, len(episode_rewards) + 1), episode_mean_rewards, marker=".", label="mean reward/round", alpha=0.7)
ax.set_xlabel("episode")
ax.set_ylabel("reward")
ax.set_title("Phase 9B DQN pretraining — reward by episode (50 episodes x 15 rounds, real NSL-KDD)")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "dqn_reward_by_episode.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: validation F1 by episode (first/last round of each episode)
# ---------------------------------------------------------------------------
first_round_f1 = [e["rounds"][0]["val_f1"] for e in episodes]
last_round_f1 = [e["rounds"][-1]["val_f1"] for e in episodes]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(episodes) + 1), first_round_f1, marker=".", label="round 1 val F1")
ax.plot(range(1, len(episodes) + 1), last_round_f1, marker=".", label="round 15 (final) val F1")
ax.set_xlabel("episode")
ax.set_ylabel("validation F1")
ax.set_title("Validation F1 by episode (each episode restarts from the shared initial model)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "validation_f1_by_episode.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: DQN loss vs training step
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(all_losses, linewidth=0.7)
ax.set_xlabel("learn() step")
ax.set_ylabel("MSE loss")
ax.set_title("DQN Q-loss vs training step")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "dqn_loss_vs_step.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4: epsilon vs episode
# ---------------------------------------------------------------------------
episode_epsilon = [e["rounds"][0]["epsilon"] for e in episodes]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(episodes) + 1), episode_epsilon, marker=".")
ax.set_xlabel("episode")
ax.set_ylabel("epsilon")
ax.set_title("Epsilon decay by episode")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "epsilon_by_episode.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5: action/profile distribution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names_all = list(action_counts.keys())
axes[0].bar(names_all, [action_counts[n] for n in names_all])
axes[0].set_title("Action distribution — ALL 750 rounds")
axes[0].tick_params(axis="x", rotation=45)
names_last = list(last_10_action_counts.keys())
axes[1].bar(names_last, [last_10_action_counts[n] for n in names_last], color="orange")
axes[1].set_title("Action distribution — LAST 10 episodes (150 rounds)")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "action_distribution.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6: client aggregation weights across the FINAL run's rounds
# ---------------------------------------------------------------------------
client_ids = sorted(final_run["rounds"][0]["weights"].keys())
fig, ax = plt.subplots(figsize=(9, 4))
for cid in client_ids:
    values = [r["weights"][cid] for r in final_run["rounds"]]
    ax.plot(range(1, len(final_run["rounds"]) + 1), values, marker=".", label=cid)
ax.set_xlabel("round (final frozen-policy run)")
ax.set_ylabel("aggregation weight")
ax.set_title("Client aggregation weights across the final run's 10 rounds")
ax.legend(fontsize=8, ncol=4)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "client_weights_final_run.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 7: performance spread over rounds (pretraining, all episodes overlaid faintly + mean)
# ---------------------------------------------------------------------------
spread_by_round_index = [[] for _ in range(15)]
for e in episodes:
    for i, r in enumerate(e["rounds"]):
        spread_by_round_index[i].append(r["performance_spread"])
mean_spread = [np.mean(v) for v in spread_by_round_index]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 16), mean_spread, marker="o")
ax.set_xlabel("round within episode")
ax.set_ylabel("mean performance_spread across 50 episodes")
ax.set_title("Performance spread (perf_spread_t) by round position")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "performance_spread_by_round.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 8: DRL vs Phase 6 vs Phase 8 KDDTest+ F1 / ROC-AUC
# ---------------------------------------------------------------------------
comparison = {
    "Phase 6\nFedAvg": {"f1": 0.7404, "roc_auc": 0.8479},
    "Phase 8\nperf_only": {"f1": 0.7447, "roc_auc": None},
    "Phase 8\nresource_only": {"f1": 0.7547, "roc_auc": None},
    "Phase 8\ndata_only": {"f1": 0.7397, "roc_auc": None},
    "Phase 8\ncombined": {"f1": 0.7439, "roc_auc": None},
    "Phase 9B\nDQN": {"f1": kddtest["kddtest_metrics"]["f1"], "roc_auc": kddtest["kddtest_metrics"]["roc_auc"]},
}
fig, ax = plt.subplots(figsize=(9, 4.5))
names = list(comparison.keys())
f1_values = [comparison[n]["f1"] for n in names]
colors = ["#888888"] * 5 + ["#d62728"]
bars = ax.bar(names, f1_values, color=colors)
ax.set_ylabel("KDDTest+ F1")
ax.set_title("KDDTest+ F1: Phase 9B DQN vs Phase 6 FedAvg vs Phase 8 rules")
ax.set_ylim(0.70, 0.78)
for bar, v in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "drl_vs_baselines_f1.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
roc_names = [n for n in names if comparison[n]["roc_auc"] is not None]
roc_values = [comparison[n]["roc_auc"] for n in roc_names]
ax.bar(roc_names, roc_values, color=["#888888", "#d62728"])
ax.set_ylabel("KDDTest+ ROC-AUC")
ax.set_title("ROC-AUC: Phase 9B DQN vs Phase 6 FedAvg\n(Phase 8 did not report ROC-AUC per-rule)")
ax.set_ylim(0.80, 0.90)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "drl_vs_baselines_roc_auc.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 9: client fairness comparison (Phase 9B post-hoc vs Phase 8's reported range)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
client_f1 = fairness["client_f1"]
cids = sorted(client_f1.keys())
ax.bar(cids, [client_f1[c] for c in cids], color="#d62728")
ax.axhline(fairness["mean"], color="black", linestyle="--", linewidth=1, label=f"mean={fairness['mean']:.4f}")
ax.set_ylabel("client-local F1 (post-hoc)")
ax.set_title(f"Phase 9B DQN — post-hoc client fairness (gap={fairness['gap']:.4f})")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "client_fairness_drl.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 10: confusion matrix for final DRL model
# ---------------------------------------------------------------------------
cm = np.array(kddtest["kddtest_metrics"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
ax.set_yticklabels(["Actual Normal", "Actual Anomaly"])
ax.set_title("Phase 9B DQN — KDDTest+ confusion matrix")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)
plt.close(fig)

print(f"\nWrote 10 plots to {PLOTS_DIR}")

summary_stats = {
    "num_episodes": len(episodes), "num_rounds_total": len(all_rounds),
    "episode_reward_first": episode_rewards[0], "episode_reward_last": episode_rewards[-1],
    "episode_reward_mean_all": float(np.mean(episode_rewards)), "episode_reward_mean_last10": float(np.mean(episode_rewards[-10:])),
    "action_distribution_all": dict(action_counts), "action_distribution_last10_episodes": dict(last_10_action_counts),
    "dqn_loss_first": all_losses[0], "dqn_loss_last": all_losses[-1], "dqn_loss_mean_last50": float(np.mean(all_losses[-50:])),
    "val_f1_mean_all_rounds": float(np.mean(all_val_f1)),
}
(RUN_DIR / "analysis_summary.json").write_text(json.dumps(summary_stats, indent=2))
print(f"Wrote {RUN_DIR / 'analysis_summary.json'}")
