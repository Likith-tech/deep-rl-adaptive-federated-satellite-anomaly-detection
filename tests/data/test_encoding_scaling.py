from __future__ import annotations

import numpy as np

from src.preprocessing.encoding import apply_categorical_encoder, fit_categorical_encoder
from src.preprocessing.scaling import apply_scaler, fit_scaler


def test_categorical_encoder_fit_on_train_transforms_test_consistently(
    synthetic_nsl_kdd_df, categorical_columns
):
    train = synthetic_nsl_kdd_df.iloc[:4]
    test = synthetic_nsl_kdd_df.iloc[4:]

    encoder = fit_categorical_encoder(train, categorical_columns)
    train_encoded = apply_categorical_encoder(train, encoder, categorical_columns)
    test_encoded = apply_categorical_encoder(test, encoder, categorical_columns)

    # Same encoded columns regardless of split, even if test has different
    # category values than train (unknown categories are handled, not errors).
    assert list(train_encoded.columns) == list(test_encoded.columns)
    assert not any(c in categorical_columns for c in train_encoded.columns)


def test_scaler_fit_on_train_only(synthetic_nsl_kdd_df, numerical_columns):
    train = synthetic_nsl_kdd_df.iloc[:4]
    test = synthetic_nsl_kdd_df.iloc[4:]

    scaler = fit_scaler(train, numerical_columns)
    train_scaled = apply_scaler(train, scaler, numerical_columns)
    test_scaled = apply_scaler(test, scaler, numerical_columns)

    # Scaled train data should be approximately zero-mean (fit on itself).
    assert np.allclose(train_scaled[numerical_columns].mean(), 0, atol=1e-8)
    # Test data uses the SAME scaler params, so it is not necessarily zero-mean.
    assert scaler.mean_ is not None
    assert len(test_scaled) == len(test)
