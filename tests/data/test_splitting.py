from __future__ import annotations

from src.preprocessing.labels import add_binary_label
from src.preprocessing.splitting import split_train_validation


def test_split_train_validation_is_reproducible_with_seed(synthetic_nsl_kdd_df):
    labeled = add_binary_label(synthetic_nsl_kdd_df, label_column="attack")

    train_a, val_a = split_train_validation(labeled, validation_size=0.5, random_seed=42)
    train_b, val_b = split_train_validation(labeled, validation_size=0.5, random_seed=42)

    assert list(train_a["label_original"]) == list(train_b["label_original"])
    assert list(val_a["label_original"]) == list(val_b["label_original"])


def test_split_train_validation_preserves_total_row_count(synthetic_nsl_kdd_df):
    labeled = add_binary_label(synthetic_nsl_kdd_df, label_column="attack")
    train, val = split_train_validation(labeled, validation_size=0.5, random_seed=42)

    assert len(train) + len(val) == len(labeled)


def test_split_train_validation_no_overlap(synthetic_nsl_kdd_df):
    labeled = add_binary_label(synthetic_nsl_kdd_df, label_column="attack")
    labeled = labeled.reset_index().rename(columns={"index": "row_id"})

    train, val = split_train_validation(labeled, validation_size=0.5, random_seed=42)

    assert set(train["row_id"]).isdisjoint(set(val["row_id"]))
