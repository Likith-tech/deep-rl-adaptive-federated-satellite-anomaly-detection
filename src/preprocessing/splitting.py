"""Train / validation / test split strategy for NSL-KDD.

NSL-KDD connection records carry no timestamp or sequence-order field, so
there is no reliable temporal structure to preserve with a time-aware
split (unlike CICIDS2017's per-day captures). Instead this pipeline uses:

1. The dataset's own predefined test split (`KDDTest+.txt`) as the held-out
   TEST set. This is the standard NSL-KDD benchmark test set used across
   the literature — it deliberately includes attack types absent from
   training, which is valuable signal for anomaly detection and would be
   lost by re-shuffling everything together.
2. A stratified random split (fixed seed) of `KDDTrain+.txt` into TRAIN
   and VALIDATION, stratified on `label_binary` so both splits keep a
   representative normal/anomaly ratio.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_validation(
    train_df: pd.DataFrame,
    validation_size: float,
    random_seed: int,
    stratify_column: str = "label_binary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split, validation_split = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=random_seed,
        stratify=train_df[stratify_column],
        shuffle=True,
    )
    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)
