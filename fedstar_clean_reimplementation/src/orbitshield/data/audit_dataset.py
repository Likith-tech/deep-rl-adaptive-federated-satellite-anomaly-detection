"""Audit raw fields separately from input features and generated metadata."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .contracts import CLEAN, load_config, load_dataset, json_text


def profile(frame: pd.DataFrame, features: list[str], target: str | None, metadata: list[str]) -> dict:
    missing = frame.isna().sum()
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    counts = {} if target is None else {str(k): int(v) for k, v in frame[target].value_counts(dropna=False).items()}
    return {
        'rows': len(frame), 'raw_columns': len(frame.columns), 'column_names': frame.columns.tolist(),
        'dtypes': {c: str(t) for c, t in frame.dtypes.items()}, 'model_features': features,
        'numeric_features': numeric, 'categorical_features': [c for c in features if c not in numeric],
        'target': target, 'metadata_fields': metadata,
        'missing_counts': {c: int(v) for c, v in missing.items()},
        'missing_percent': {c: float(v / len(frame) * 100) for c, v in missing.items()},
        'nonfinite_counts': {c: int((~np.isfinite(frame[c].to_numpy(dtype=float))).sum()) for c in numeric},
        'duplicate_complete_rows': int(frame.duplicated().sum()),
        'duplicate_feature_rows': int(frame[features].duplicated().sum()) if features else None,
        'constant_features': [c for c in features if frame[c].nunique(dropna=False) == 1],
        'near_constant_features_99_5_percent': [c for c in features if frame[c].value_counts(dropna=False).iloc[0] / len(frame) >= .995],
        'label_counts': counts,
        'majority_fraction': max(counts.values()) / len(frame) if counts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only raw-data audit; prints JSON.')
    parser.add_argument('--dataset', choices=['nsl_kdd', 'stin', 'opssat_ad'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error('Specify --all or --dataset')
    names = ['nsl_kdd', 'stin', 'opssat_ad'] if args.all else [args.dataset]
    print(json_text({name: load_dataset(load_config(CLEAN / 'configs' / f'{name}.yaml')).audits for name in names}))


if __name__ == '__main__':
    main()
