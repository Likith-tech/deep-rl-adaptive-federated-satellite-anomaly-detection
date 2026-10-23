"""Phase 9 — run the staleness- and resource-aware FedAvg ablation study.

Run from the repository root (requires Phase 4's ORIGINAL partition —
data/partitions/, the same one Phase 6/8 used):

    python scripts/run_staleness_fl_experiments.py

For every staleness config in configs/staleness_fl.yaml (no_correction,
linear_decay, exponential_decay), runs the SAME FedAvg setup as Phase
6/8 (8 clients, 10 rounds, 1 local epoch, same partition, same seeds)
with the SAME real, seeded client-participation pattern (driven by each
client's Phase 4 `availability_probability` — see
src/federated/staleness.py) and ONLY the staleness-weighting response
to that pattern changed.

Writes, per config, under experiments/staleness_fl/<config_name>/:
    run_config.json, round_history.json, best_global_model.pt (gitignored),
    round_checkpoints/round_NN.pt (gitignored)

Phase 6/8/9B's own files (results/, experiments/adaptive_fl/,
experiments/drl_adaptive_fl/, data/partitions/) are never read for
writing and never modified — this script only READS the Phase 4/6
partition, and writes exclusively under experiments/staleness_fl/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.staleness_trainer import run_staleness_federated_training  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "staleness_fl"
CONFIG_PATH = REPO_ROOT / "configs" / "staleness_fl.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    model_config = config["model"]
    training_config = config["training"]
    num_rounds = config["federation"]["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]
    participation_seed = config["staleness"]["participation_seed"]

    manifest_path = Path(config["partitions"]["manifest"])
    if not (REPO_ROOT / manifest_path).exists():
        print(f"ERROR: {manifest_path} not found. Run Phase 4 first: "
              "python scripts/create_satellite_partitions.py", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads((REPO_ROOT / manifest_path).read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    satellite_metadata = json.loads((REPO_ROOT / config["partitions"]["satellite_metadata"]).read_text())
    client_availability = {
        cid: meta["availability_probability"] for cid, meta in satellite_metadata["clients"].items()
    }

    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    global_validation_parquet = PROCESSED_DIR / "validation.parquet"

    print(f"Staleness FL ablation study: clients={client_ids}, rounds={num_rounds}, "
          f"partition={config['partitions']['source']} (Phase 4/6/8 original, alpha={manifest['alpha']})")
    print(f"Client availability_probability (real Phase 4 simulated field): "
          f"{ {k: round(v, 3) for k, v in client_availability.items()} }")

    for config_name, staleness_cfg in config["staleness_configs"].items():
        mode = staleness_cfg["mode"]
        decay_rate = staleness_cfg["decay_rate"]

        run_dir = EXPERIMENT_ROOT / config_name
        print(f"\n{'=' * 70}\nStaleness config: {config_name} — mode={mode}, decay_rate={decay_rate}\n{'=' * 70}")

        start = time.time()
        result = run_staleness_federated_training(
            client_ids=client_ids,
            partitions_dir=PARTITIONS_DIR,
            global_validation_parquet=global_validation_parquet,
            feature_columns=feature_columns,
            model_config=model_config,
            training_config=training_config,
            num_rounds=num_rounds,
            shared_init_seed=shared_init_seed,
            checkpoint_dir=run_dir,
            client_availability=client_availability,
            staleness_mode=mode,
            staleness_decay_rate=decay_rate,
            participation_seed=participation_seed,
        )

        for r in result.round_history:
            n_stale = sum(1 for s in r.client_staleness.values() if s > 0)
            print(f"Round {r.round_number:2d}/{num_rounds} | val_loss={r.val_loss:.4f} "
                  f"val_f1={r.val_metrics['f1']:.4f} | clients_stale_this_round={n_stale}/{len(client_ids)}")
        print(f"Best round: {result.best_round} (val_loss={result.best_val_loss:.4f}, "
              f"val_f1={result.best_val_metrics.f1:.4f})")

        max_staleness_seen = max(
            s for r in result.round_history for s in r.client_staleness.values()
        )

        run_config = {
            "config_name": config_name,
            "staleness_mode": mode,
            "staleness_decay_rate": decay_rate,
            "staleness_rationale": staleness_cfg.get("rationale", "").strip(),
            "config": config,
            "client_ids": client_ids,
            "client_sample_counts": {cid: manifest["client_sample_counts"][cid] for cid in client_ids},
            "client_availability": client_availability,
            "num_rounds": num_rounds,
            "max_staleness_seen": max_staleness_seen,
            "best_round": result.best_round,
            "best_val_loss": result.best_val_loss,
            "best_val_metrics": result.best_val_metrics.to_dict(),
            "total_seconds": result.total_seconds,
            "wall_seconds": time.time() - start,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        print(f"Config '{config_name}' complete in {time.time() - start:.1f}s. Artifacts under {run_dir}")

    print(f"\nAll {len(config['staleness_configs'])} staleness configs complete. "
          "Run scripts/evaluate_staleness_fl_results.py next.")


if __name__ == "__main__":
    main()
