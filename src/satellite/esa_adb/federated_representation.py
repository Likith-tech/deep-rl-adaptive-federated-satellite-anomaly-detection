"""ESA-ADB Option A federated input representation — approved design
(see docs/research/datasets/esa_adb_federated_architecture_design.md
and docs/research/experiments/esa-adb/README.md Entry 6).

    Input = [X (58 normalized telemetry values); M (58 availability flags)]
    input_dim = 116

Isolated addition alongside multivariate.py: does NOT modify loader.py,
labeling.py, preprocessing.py, clients.py, pipeline.py, or
multivariate.py. Reuses multivariate.py's already-verified aligned,
per-channel-normalized global frame (built from the real timestamp
intersection of all 58 channels — see multivariate.py's own docstring
for the alignment finding) as the single source of real telemetry
values; this module only adds the per-client MASKING layer on top.

**Mask semantics (exact, tested):**

    Client OWNS channel c:      X[c] = real normalized value, M[c] = 1
    Client does NOT own c:      X[c] = 0.0 (placeholder),      M[c] = 0

The placeholder is always exactly 0.0 — but it is never confused with a
genuine normalized-zero reading from an OWNED channel, because M is an
explicit, separate signal (concatenated, never multiplied into X). A
gate-only `X * M` representation was explicitly rejected during the
architecture design phase for exactly this reason: after per-channel
StandardScaler normalization, 0.0 is a valid, meaningful value for the
owning client (it means "at the channel's own train-split mean"), so
collapsing it into the same bare-zero representation as "channel
absent" would destroy real information. Concatenation avoids this by
giving the model M as an unambiguous, independent input.

No telemetry is ever fabricated, interpolated, or borrowed from another
client: every non-zero X value in a client's frame is a real reading
from a channel that client's own real metadata
(clients.CLIENT_CHANNELS) says it owns.

**Global channel order:** GLOBAL_CHANNEL_ORDER is the same deterministic
58-channel ordering already used everywhere else in this package
(loader.target_yes_channels() — ascending numeric channel index). Every
client uses this EXACT order for both the X and M halves of its input;
no client-specific reordering is permitted.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch

from src.preprocessing.sequences import SequenceDataset, create_sequences
from src.satellite.esa_adb import clients as clients_mod
from src.satellite.esa_adb import loader
from src.satellite.esa_adb import preprocessing as prep
from src.satellite.esa_adb.multivariate import JOINT_LABEL_COLUMN

GLOBAL_CHANNEL_ORDER: tuple[str, ...] = tuple(loader.target_yes_channels())
assert len(GLOBAL_CHANNEL_ORDER) == 58, f"Expected 58 Target=YES channels, got {len(GLOBAL_CHANNEL_ORDER)}"

MASK_PREFIX = "mask_"


def mask_column(channel: str) -> str:
    return f"{MASK_PREFIX}{channel}"


def feature_columns() -> list[str]:
    """The 116 ordered feature column names: 58 X columns (real channel
    names, GLOBAL_CHANNEL_ORDER) followed by 58 M columns
    (mask_<channel>, same relative order) — matches the approved
    Input = [X; M] concatenation exactly."""
    return list(GLOBAL_CHANNEL_ORDER) + [mask_column(c) for c in GLOBAL_CHANNEL_ORDER]


def build_masked_client_frame(aligned_frame: pd.DataFrame, owned_channels: list[str]) -> pd.DataFrame:
    """Build one client's [X; M] frame from the already-aligned,
    already-normalized global frame (multivariate.py's
    build_aligned_multivariate_frame + transform_multivariate_frame
    output). `owned_channels` must be a subset of GLOBAL_CHANNEL_ORDER
    (clients.CLIENT_CHANNELS[<client>]).

    Real values are copied through UNCHANGED for owned channels;
    unowned channels get the fixed placeholder 0.0 with mask 0 — never
    a value read from `aligned_frame` for that channel (even though
    aligned_frame DOES contain that channel's real reading, since it is
    the shared, dataset-wide aligned frame — this function is exactly
    where "client does not get to see channels it doesn't own" is
    enforced)."""
    owned_set = set(owned_channels)
    unknown = owned_set - set(GLOBAL_CHANNEL_ORDER)
    if unknown:
        raise ValueError(f"owned_channels contains channels outside GLOBAL_CHANNEL_ORDER: {sorted(unknown)}")

    n = len(aligned_frame)
    data = {}
    for c in GLOBAL_CHANNEL_ORDER:
        if c in owned_set:
            data[c] = aligned_frame[c].to_numpy(dtype="float32")
            data[mask_column(c)] = np.ones(n, dtype="float32")
        else:
            data[c] = np.zeros(n, dtype="float32")
            data[mask_column(c)] = np.zeros(n, dtype="float32")

    out = pd.DataFrame(data, index=aligned_frame.index)
    out[JOINT_LABEL_COLUMN] = aligned_frame[JOINT_LABEL_COLUMN].to_numpy()
    return out


def build_client_sequences(client_frame: pd.DataFrame) -> SequenceDataset:
    """Reuses create_sequences UNCHANGED, feature_columns=the 116
    ordered [X; M] columns."""
    return create_sequences(
        client_frame, feature_columns=feature_columns(), record_label_column=JOINT_LABEL_COLUMN,
        sequence_length=prep.SEQUENCE_LENGTH, stride=prep.STRIDE, label_strategy=prep.LABEL_STRATEGY,
    )


def build_all_client_frames(aligned_split_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One masked [X; M] frame per real client (clients.CLIENT_CHANNELS),
    all built from the SAME aligned+normalized split frame."""
    return {
        client: build_masked_client_frame(aligned_split_frame, owned_channels)
        for client, owned_channels in clients_mod.CLIENT_CHANNELS.items()
    }


def build_pooled_validation_sequences(client_validation_frames: dict[str, pd.DataFrame]) -> SequenceDataset:
    """Global/pooled validation representation — FINAL DECISION (see
    docs/research/datasets/esa_adb_federated_architecture_design.md and
    the Entry 7 research-log discussion): the pooled validation set is
    the UNION of each client's OWN masked validation sequences (each
    client's real ownership mask, exactly as it trains with), never an
    "all-58-channels-available" hypothetical that no real client (and
    no real deployment scenario) would ever actually present to the
    model. See build_global_validation_all_available_HYPOTHETICAL below
    for the alternative that was investigated and rejected — kept here,
    unused by default, only for documentation/comparison purposes.
    """
    from src.satellite.esa_adb.pipeline import pool_sequence_datasets

    per_client_sequences = [build_client_sequences(df) for df in client_validation_frames.values()]
    return pool_sequence_datasets(per_client_sequences, prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)


def build_global_validation_all_available_HYPOTHETICAL(aligned_validation_frame: pd.DataFrame) -> SequenceDataset:
    """REJECTED alternative, computed only for documentation/comparison,
    never used as the actual global validation set. Represents "what if
    the aggregated model were evaluated with all 58 real channels
    simultaneously available" (M = all-1s) — this is NOT a scenario any
    real client, or any real deployment of this federated model, will
    ever encounter (it is exactly what Experiment 1A's centralized
    baseline already measures), so using it as "the" validation signal
    would silently re-test the centralized ceiling instead of genuinely
    evaluating the federated model's real-world performance. Not called
    by build_esa_adb_option_a.py's main pipeline."""
    full_mask_frame = build_masked_client_frame(aligned_validation_frame, list(GLOBAL_CHANNEL_ORDER))
    return build_client_sequences(full_mask_frame)


# ---------------------------------------------------------------------
# Model shape / FedAvg compatibility verification
# ---------------------------------------------------------------------


def instantiate_client_models(model_config: dict, num_clients: int = 8) -> list[torch.nn.Module]:
    """One TemporalGRUAttention instance per client, IDENTICAL config —
    used only to verify state_dict key/shape/parameter-count
    compatibility and FedAvg acceptance. No training, no forward pass
    over real data."""
    from src.models.temporal_gru_attention import TemporalGRUAttention

    return [
        TemporalGRUAttention(
            input_dim=model_config["input_dim"], projection_dim=model_config["projection_dim"],
            hidden_dim=model_config["hidden_dim"], num_gru_layers=model_config["num_gru_layers"],
            dropout=model_config["dropout"], output_dim=model_config["output_dim"],
        )
        for _ in range(num_clients)
    ]


def verify_state_dict_compatibility(models: list[torch.nn.Module]) -> dict:
    """Pairwise check across ALL pairs of client models: identical
    state_dict keys, identical tensor shapes per key, identical total
    parameter count. Returns a dict report; raises nothing (violations
    are reported, not asserted, so callers can decide how to react)."""
    state_dicts = [m.state_dict() for m in models]
    param_counts = [sum(p.numel() for p in m.parameters()) for m in models]

    violations = []
    for i, j in combinations(range(len(models)), 2):
        keys_i, keys_j = set(state_dicts[i].keys()), set(state_dicts[j].keys())
        if keys_i != keys_j:
            violations.append(f"client_{i} vs client_{j}: state_dict key mismatch")
            continue
        for k in keys_i:
            if state_dicts[i][k].shape != state_dicts[j][k].shape:
                violations.append(f"client_{i} vs client_{j}: shape mismatch at '{k}' "
                                   f"({state_dicts[i][k].shape} vs {state_dicts[j][k].shape})")
        if param_counts[i] != param_counts[j]:
            violations.append(f"client_{i} vs client_{j}: parameter count mismatch ({param_counts[i]} vs {param_counts[j]})")

    return {
        "num_models_compared": len(models),
        "num_pairs_checked": len(list(combinations(range(len(models)), 2))),
        "state_dict_keys": sorted(state_dicts[0].keys()) if state_dicts else [],
        "parameter_count_per_model": param_counts,
        "all_identical": len(violations) == 0,
        "violations": violations,
    }


def verify_fedavg_compatibility(models: list[torch.nn.Module], sample_counts: list[int]) -> dict:
    """Synthetic aggregation test: calls the REAL, unmodified
    src.federated.fedavg.federated_average on the client models'
    (randomly-initialized) state_dicts. Success means Option A's input
    representation change requires ZERO modification to the existing
    FedAvg code."""
    from src.federated.fedavg import federated_average

    state_dicts = [m.state_dict() for m in models]
    try:
        aggregated = federated_average(state_dicts, sample_counts)
        shape_matches_input = all(aggregated[k].shape == state_dicts[0][k].shape for k in aggregated)
        return {"succeeded": True, "error": None, "aggregated_shape_matches_client_shape": shape_matches_input}
    except Exception as e:  # noqa: BLE001 — deliberately broad, this is a compatibility probe
        return {"succeeded": False, "error": f"{type(e).__name__}: {e}", "aggregated_shape_matches_client_shape": None}
