from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_labeled_df() -> pd.DataFrame:
    """A small synthetic dataset shaped like the real processed training
    data (category column + label_binary), never the real ~107k-row
    NSL-KDD data, so tests run in milliseconds."""
    rng = np.random.default_rng(0)
    n = 600
    # Imbalanced categories, similar spirit to real NSL-KDD (normal/dos
    # common, r2l/u2r rare) but small enough for fast tests.
    categories = rng.choice(
        ["normal", "dos", "probe", "r2l", "u2r"],
        size=n,
        p=[0.45, 0.35, 0.12, 0.06, 0.02],
    )
    label_binary = (categories != "normal").astype(int)
    df = pd.DataFrame(
        {
            "feature_0": rng.normal(size=n),
            "feature_1": rng.normal(size=n),
            "category": categories,
            "label_binary": label_binary,
        }
    )
    return df
