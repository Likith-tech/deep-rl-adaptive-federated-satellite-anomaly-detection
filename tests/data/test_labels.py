from __future__ import annotations

from src.preprocessing.labels import add_binary_label


def test_add_binary_label_preserves_original_and_adds_binary(synthetic_nsl_kdd_df):
    labeled = add_binary_label(synthetic_nsl_kdd_df, label_column="attack")

    assert "label_original" in labeled.columns
    assert "label_binary" in labeled.columns
    assert list(labeled["label_original"]) == list(synthetic_nsl_kdd_df["attack"])


def test_add_binary_label_maps_normal_to_zero_and_attacks_to_one(synthetic_nsl_kdd_df):
    labeled = add_binary_label(synthetic_nsl_kdd_df, label_column="attack")

    normal_rows = labeled[labeled["label_original"] == "normal"]
    attack_rows = labeled[labeled["label_original"] != "normal"]

    assert (normal_rows["label_binary"] == 0).all()
    assert (attack_rows["label_binary"] == 1).all()
