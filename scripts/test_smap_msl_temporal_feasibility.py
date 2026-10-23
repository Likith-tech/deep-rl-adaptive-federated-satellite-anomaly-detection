"""SMAP/MSL Track B (candidate) — temporal-sequence feasibility test.

Run from the repository root, after scripts/download_smap_msl.py:

    python scripts/test_smap_msl_temporal_feasibility.py

Proves — using REAL downloaded data and the EXISTING, UNMODIFIED
`TemporalGRUAttention` model class (src/models/temporal_gru_attention.py,
built in Phase 3) — that genuine sequence_length > 1 windows can be
built from real per-timestep SMAP/MSL telemetry and pushed through the
model's actual forward pass. This is a FEASIBILITY TEST, not training:
`model.eval()` + `torch.no_grad()` are used throughout; no gradient is
computed and no weight is ever updated.

This directly targets the exact limitation Phase 9A's own audit
disclosed for OPSSAT-AD: `dataset.csv` gave one fixed-width feature
vector per segment (forcing sequence_length=1), never a genuine
per-timestep sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.temporal_gru_attention import TemporalGRUAttention  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "smap_msl"

# (channel, feature_dim, sequence_length) — one SMAP channel, one MSL
# channel, and the Telemanom paper's own default sequence length (250)
# tried on a second SMAP channel, to prove more than one configuration works.
TEST_CASES = [("A-1", 25, 100), ("T-1", 25, 250), ("M-1", 55, 100)]


def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run scripts/download_smap_msl.py first.", file=sys.stderr)
        sys.exit(1)

    print("=== SMAP/MSL temporal-sequence feasibility test (forward pass only, no training) ===\n")
    all_ok = True
    for channel, expected_dim, seq_len in TEST_CASES:
        arr = np.load(RAW_DIR / "test" / f"{channel}.npy")
        if arr.shape[1] != expected_dim:
            print(f"UNEXPECTED: {channel} has dim {arr.shape[1]}, expected {expected_dim}")
            all_ok = False
            continue

        n_timesteps = arr.shape[0]
        n_possible_windows = n_timesteps - seq_len + 1
        batch = np.stack([arr[i:i + seq_len] for i in range(min(8, n_possible_windows))])
        x = torch.tensor(batch, dtype=torch.float32)

        model = TemporalGRUAttention(
            input_dim=expected_dim, projection_dim=16, hidden_dim=32,
            num_gru_layers=1, dropout=0.1, output_dim=1,
        )
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(x)

        ok = logits.shape == (x.shape[0], 1) and attn_weights.shape == (x.shape[0], seq_len)
        all_ok = all_ok and ok
        print(f"{channel} (dim={expected_dim}, seq_len={seq_len}): "
              f"input {tuple(x.shape)} -> logits {tuple(logits.shape)}, attn {tuple(attn_weights.shape)} "
              f"| {n_possible_windows:,} total possible windows in the full test file | "
              f"{'OK' if ok else 'SHAPE MISMATCH'}")

    print(f"\n{'ALL FEASIBILITY CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} "
          f"— no training occurred; this only proves the shapes/forward-pass are compatible.")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
