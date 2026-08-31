"""Tests for src/satellite/esa_adb/federated_representation.py — Option A
([X; M], input_dim=116). Covers all 14 requested categories. Synthetic
data only for the pure-logic pieces; a couple of tests cross-check
against the real clients.py mapping (no download required, clients.py
is pure Python)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.satellite.esa_adb import clients as clients_mod
from src.satellite.esa_adb import federated_representation as fr
from src.satellite.esa_adb.multivariate import JOINT_LABEL_COLUMN


@pytest.fixture
def synthetic_aligned_frame() -> pd.DataFrame:
    """A small, fully-populated real-shaped global frame: all 58
    GLOBAL_CHANNEL_ORDER channels present, distinct deterministic
    values per channel so cross-contamination is detectable."""
    n = 20
    index = pd.date_range("2000-01-01", periods=n, freq="300s")
    data = {c: np.full(n, float(i + 1)) for i, c in enumerate(fr.GLOBAL_CHANNEL_ORDER)}  # channel i -> constant value i+1
    data[JOINT_LABEL_COLUMN] = np.array([0, 1] * (n // 2), dtype="int64")
    return pd.DataFrame(data, index=index)


# 1. deterministic 58-channel ordering
def test_global_channel_order_has_58_channels_and_is_sorted_numerically():
    assert len(fr.GLOBAL_CHANNEL_ORDER) == 58
    indices = [int(c.split("_")[1]) for c in fr.GLOBAL_CHANNEL_ORDER]
    assert indices == sorted(indices)


def test_global_channel_order_is_stable_across_imports():
    import importlib

    reloaded = importlib.reload(fr)
    assert reloaded.GLOBAL_CHANNEL_ORDER == fr.GLOBAL_CHANNEL_ORDER


# 2. correct client ownership (cross-checked against the real clients.py, not assumed)
def test_client_ownership_matches_real_clients_module():
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        assert set(owned).issubset(set(fr.GLOBAL_CHANNEL_ORDER))
    total_owned = sum(len(v) for v in clients_mod.CLIENT_CHANNELS.values())
    assert total_owned == 58


def test_real_client_channel_counts_are_3_6_7_8_8_8_9_9():
    """Documents the REAL counts (verified from clients.py directly),
    distinct from the counts stated in the task prompt for this turn —
    the prompt's numbers used the same multiset but different client
    labels; this test guards against silently trusting the wrong
    per-client assignment."""
    counts = sorted(len(v) for v in clients_mod.CLIENT_CHANNELS.values())
    assert counts == [3, 6, 7, 8, 8, 8, 9, 9]
    assert len(clients_mod.CLIENT_CHANNELS["client_3"]) == 3
    assert len(clients_mod.CLIENT_CHANNELS["client_1"]) == 6


# 3. correct mask generation / 4. real value preserved when mask=1 / 5. unavailable value masked when mask=0
def test_build_masked_client_frame_owned_channels_get_real_value_and_mask_1(synthetic_aligned_frame):
    owned = ["channel_12", "channel_41"]
    frame = fr.build_masked_client_frame(synthetic_aligned_frame, owned)

    idx_12 = fr.GLOBAL_CHANNEL_ORDER.index("channel_12")
    assert (frame["channel_12"] == float(idx_12 + 1)).all()
    assert (frame[fr.mask_column("channel_12")] == 1.0).all()


def test_build_masked_client_frame_unowned_channels_get_zero_and_mask_0(synthetic_aligned_frame):
    owned = ["channel_12"]
    frame = fr.build_masked_client_frame(synthetic_aligned_frame, owned)

    assert (frame["channel_41"] == 0.0).all()
    assert (frame[fr.mask_column("channel_41")] == 0.0).all()


# 6. zero-valued real telemetry distinguishable from unavailable telemetry
def test_real_zero_reading_stays_distinguishable_from_unavailable():
    """A channel whose REAL normalized reading happens to be exactly
    0.0 (e.g. at its own train-mean) must still show mask=1, never
    collapsing to the same (0.0, 0) pair as an unowned channel."""
    n = 5
    index = pd.date_range("2000-01-01", periods=n, freq="300s")
    data = {c: np.zeros(n) for c in fr.GLOBAL_CHANNEL_ORDER}  # every channel reads exactly 0.0
    data[JOINT_LABEL_COLUMN] = np.zeros(n, dtype="int64")
    aligned = pd.DataFrame(data, index=index)

    owned = ["channel_12"]
    frame = fr.build_masked_client_frame(aligned, owned)

    # channel_12 (owned): X=0.0 real value, M=1.
    assert (frame["channel_12"] == 0.0).all()
    assert (frame[fr.mask_column("channel_12")] == 1.0).all()
    # channel_41 (not owned): X=0.0 placeholder, M=0 — same X, DIFFERENT mask.
    assert (frame["channel_41"] == 0.0).all()
    assert (frame[fr.mask_column("channel_41")] == 0.0).all()
    # The two are only distinguishable via the mask column — verify that explicitly.
    assert not frame[fr.mask_column("channel_12")].equals(frame[fr.mask_column("channel_41")])


def test_build_masked_client_frame_rejects_channel_outside_global_order(synthetic_aligned_frame):
    with pytest.raises(ValueError, match="outside GLOBAL_CHANNEL_ORDER"):
        fr.build_masked_client_frame(synthetic_aligned_frame, ["channel_9999"])


# 7. temporal alignment (mask aligned with the same 50 timesteps as X)
def test_mask_and_value_columns_share_the_same_index(synthetic_aligned_frame):
    frame = fr.build_masked_client_frame(synthetic_aligned_frame, ["channel_12"])
    assert frame.index.equals(synthetic_aligned_frame.index)
    # No row is dropped or reordered by masking.
    assert len(frame) == len(synthetic_aligned_frame)


# 8. sequence shape [N, 50, 116]
def test_build_client_sequences_shape():
    n = 120
    index = pd.date_range("2000-01-01", periods=n, freq="300s")
    data = {c: np.random.default_rng(i).normal(size=n) for i, c in enumerate(fr.GLOBAL_CHANNEL_ORDER)}
    data[JOINT_LABEL_COLUMN] = np.zeros(n, dtype="int64")
    aligned = pd.DataFrame(data, index=index)

    frame = fr.build_masked_client_frame(aligned, list(fr.GLOBAL_CHANNEL_ORDER[:6]))  # client_1-shaped
    seq = fr.build_client_sequences(frame)

    assert seq.sequences.shape[1] == 50  # prep.SEQUENCE_LENGTH
    assert seq.sequences.shape[2] == 116


def test_feature_columns_are_58_x_then_58_mask_in_global_order():
    cols = fr.feature_columns()
    assert len(cols) == 116
    assert cols[:58] == list(fr.GLOBAL_CHANNEL_ORDER)
    assert cols[58:] == [fr.mask_column(c) for c in fr.GLOBAL_CHANNEL_ORDER]


# 9/10/11. identical model state_dict keys / shapes / parameter count
@pytest.fixture
def small_model_config() -> dict:
    return {"input_dim": 116, "projection_dim": 8, "hidden_dim": 8, "num_gru_layers": 1, "dropout": 0.0, "output_dim": 1}


def test_instantiated_client_models_have_identical_state_dicts(small_model_config):
    models = fr.instantiate_client_models(small_model_config, num_clients=8)
    report = fr.verify_state_dict_compatibility(models)
    assert report["all_identical"] is True
    assert report["violations"] == []
    assert len(set(report["parameter_count_per_model"])) == 1  # all 8 identical
    assert report["num_pairs_checked"] == 28  # C(8,2)


def test_verify_state_dict_compatibility_detects_a_real_mismatch(small_model_config):
    models = fr.instantiate_client_models(small_model_config, num_clients=3)
    mismatched_config = dict(small_model_config, input_dim=58)  # different input_dim -> different projection shape
    models.append(fr.instantiate_client_models(mismatched_config, num_clients=1)[0])

    report = fr.verify_state_dict_compatibility(models)
    assert report["all_identical"] is False
    assert len(report["violations"]) > 0


# 12. FedAvg compatibility
def test_fedavg_accepts_client_models_unmodified(small_model_config):
    models = fr.instantiate_client_models(small_model_config, num_clients=8)
    sample_counts = [100, 200, 50, 300, 150, 400, 250, 175]  # arbitrary, matches 8 models
    result = fr.verify_fedavg_compatibility(models, sample_counts)
    assert result["succeeded"] is True
    assert result["aggregated_shape_matches_client_shape"] is True
    assert result["error"] is None


# 13. deterministic client construction
def test_build_masked_client_frame_is_deterministic(synthetic_aligned_frame):
    owned = clients_mod.CLIENT_CHANNELS["client_4"]
    a = fr.build_masked_client_frame(synthetic_aligned_frame, owned)
    b = fr.build_masked_client_frame(synthetic_aligned_frame, owned)
    pd.testing.assert_frame_equal(a, b)


def test_build_all_client_frames_covers_all_eight_real_clients(synthetic_aligned_frame):
    frames = fr.build_all_client_frames(synthetic_aligned_frame)
    assert set(frames.keys()) == set(clients_mod.CLIENT_CHANNELS.keys())


# 14. no cross-client telemetry leakage
def test_no_cross_client_telemetry_leakage(synthetic_aligned_frame):
    """A client's frame must NEVER contain another client's owned
    channel's real value in a slot where THIS client doesn't own that
    channel — i.e. every unowned slot is exactly the fixed placeholder
    (0.0), never leaked real data from the shared aligned frame."""
    frames = fr.build_all_client_frames(synthetic_aligned_frame)
    for client, frame in frames.items():
        owned = set(clients_mod.CLIENT_CHANNELS[client])
        for c in fr.GLOBAL_CHANNEL_ORDER:
            if c not in owned:
                assert (frame[c] == 0.0).all(), f"{client} leaked a non-zero value for unowned channel {c}"
                assert (frame[fr.mask_column(c)] == 0.0).all()


def test_two_clients_owning_disjoint_channels_never_share_a_populated_slot(synthetic_aligned_frame):
    frames = fr.build_all_client_frames(synthetic_aligned_frame)
    client_a, client_b = "client_1", "client_2"  # real, disjoint channel sets
    owned_a = set(clients_mod.CLIENT_CHANNELS[client_a])
    owned_b = set(clients_mod.CLIENT_CHANNELS[client_b])
    assert owned_a.isdisjoint(owned_b)  # precondition: real partition is disjoint

    for c in owned_a:
        assert (frames[client_b][fr.mask_column(c)] == 0.0).all()
    for c in owned_b:
        assert (frames[client_a][fr.mask_column(c)] == 0.0).all()
