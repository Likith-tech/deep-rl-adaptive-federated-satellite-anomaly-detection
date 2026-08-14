from __future__ import annotations

import numpy as np

from src.satellite.loader import LABEL_COLUMN, NUMERIC_FEATURE_COLUMNS, split_by_author_train_flag
from src.satellite.preprocessing import (
    extract_labels,
    fit_preprocessing,
    stratified_train_validation_split,
    transform,
)


def test_stratified_split_is_deterministic_with_fixed_seed(synthetic_opssat_df):
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)

    split_a = stratified_train_validation_split(train_df, validation_size=0.2, seed=42)
    split_b = stratified_train_validation_split(train_df, validation_size=0.2, seed=42)

    train_a, val_a = split_a
    train_b, val_b = split_b
    assert list(train_a["segment"]) == list(train_b["segment"])
    assert list(val_a["segment"]) == list(val_b["segment"])


def test_stratified_split_preserves_label_balance_roughly(synthetic_opssat_df):
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)
    sub_train, val = stratified_train_validation_split(train_df, validation_size=0.25, seed=42)

    overall_rate = train_df[LABEL_COLUMN].mean()
    val_rate = val[LABEL_COLUMN].mean()
    # Stratified split should keep the anomaly rate close to the source rate.
    assert abs(overall_rate - val_rate) < 0.15


def test_split_has_no_overlap_between_train_and_validation(synthetic_opssat_df):
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)
    sub_train, val = stratified_train_validation_split(train_df, validation_size=0.2, seed=42)

    assert set(sub_train["segment"]).isdisjoint(set(val["segment"]))


def test_author_test_segments_never_appear_in_train_validation_split(synthetic_opssat_df):
    train_df, test_df = split_by_author_train_flag(synthetic_opssat_df)
    sub_train, val = stratified_train_validation_split(train_df, validation_size=0.2, seed=42)

    test_segment_ids = set(test_df["segment"])
    assert test_segment_ids.isdisjoint(set(sub_train["segment"]))
    assert test_segment_ids.isdisjoint(set(val["segment"]))


def test_scaler_is_fit_only_on_training_data(synthetic_opssat_df):
    """The scaler's learned mean must match the TRAINING subset's mean,
    not the full dataset's — proof that fit_preprocessing never sees
    validation/test rows."""
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)
    sub_train, val = stratified_train_validation_split(train_df, validation_size=0.3, seed=42)

    artifacts = fit_preprocessing(sub_train)

    expected_mean = sub_train[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64").mean(axis=0)
    assert np.allclose(artifacts.scaler.mean_, expected_mean)

    full_mean = synthetic_opssat_df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64").mean(axis=0)
    # The training-only mean should generally differ from the full-dataset mean
    # for at least one feature (guards against accidentally fitting on everything).
    assert not np.allclose(artifacts.scaler.mean_, full_mean)


def test_transform_produces_expected_feature_dimension(synthetic_opssat_df):
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)
    artifacts = fit_preprocessing(train_df)

    X = transform(train_df, artifacts)
    assert X.shape[0] == len(train_df)
    assert X.shape[1] == len(artifacts.feature_columns)
    assert X.shape[1] == len(NUMERIC_FEATURE_COLUMNS) + train_df["channel"].nunique()


def test_extract_labels_returns_binary_float_array(synthetic_opssat_df):
    y = extract_labels(synthetic_opssat_df)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert y.dtype == np.float32


def test_save_and_load_preprocessing_artifacts_roundtrip(tmp_path, synthetic_opssat_df):
    train_df, _ = split_by_author_train_flag(synthetic_opssat_df)
    artifacts = fit_preprocessing(train_df)
    artifacts.save(tmp_path / "artifacts")

    from src.satellite.preprocessing import OpsSatPreprocessingArtifacts

    loaded = OpsSatPreprocessingArtifacts.load(tmp_path / "artifacts")
    assert loaded.feature_columns == artifacts.feature_columns
    assert np.allclose(loaded.scaler.mean_, artifacts.scaler.mean_)

    X_original = transform(train_df, artifacts)
    X_loaded = transform(train_df, loaded)
    assert np.allclose(X_original, X_loaded)
