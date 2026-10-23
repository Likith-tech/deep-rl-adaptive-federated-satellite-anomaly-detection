"""Deterministic whole-feature-group splitting and overlap measurements."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_row_holdout(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded stratified-group folds; pick the fold nearest the row target.

    Only reciprocal integer fractions are supported. At 0.2 this constructs
    five folds and chooses the all-class fold nearest 20% by row count (ties:
    fold order). No model scores enter selection. Large groups prevent exact
    stratification; this is a local adaptation, not the paper's exact split.
    """
    folds = int(round(1 / fraction))
    if folds < 2 or not np.isclose(fraction, 1 / folds):
        raise ValueError('Group split requires a reciprocal integer fraction')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = set(frame.y)
    candidates = [(tr, te) for tr, te in splitter.split(frame, frame.y, frame.group)
                  if set(frame.iloc[tr].y) == labels and set(frame.iloc[te].y) == labels]
    if not candidates:
        raise ValueError('Cannot retain every class in train and holdout with whole groups')
    tr, te = min(candidates, key=lambda pair: abs(len(pair[1]) - fraction * len(frame)))
    train, test = frame.iloc[tr].copy(), frame.iloc[te].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def overlaps(frames: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for left, right in itertools.combinations(frames, 2):
        shared = set(frames[left]['group']) & set(frames[right]['group'])
        result[f'{left}__{right}'] = {
            'feature_groups': len(shared),
            'left_rows': int(frames[left]['group'].isin(shared).sum()),
            'right_rows': int(frames[right]['group'].isin(shared).sum()),
        }
    return result
