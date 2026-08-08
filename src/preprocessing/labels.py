"""Label processing: binary anomaly label + preserved original attack label."""

from __future__ import annotations

import pandas as pd

NORMAL_LABEL = "normal"


def add_binary_label(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Add `label_original` (verbatim attack name) and `label_binary`
    (0=normal, 1=anomaly), without removing the source column's information.
    """
    df = df.copy()
    df["label_original"] = df[label_column].astype(str)
    df["label_binary"] = (df["label_original"] != NORMAL_LABEL).astype(int)
    return df
