"""Build data/processed/ton_iot/{train,validation,test}.parquet from the
real, downloaded TON_IoT Train_Test_Network.csv. Fits StandardScaler +
OneHotEncoder on train only. Run once before scripts/train_ton_iot_fedavg.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.satellite.ton_iot_loader import EXPECTED_ROWS, EXPECTED_TYPE_CLASSES, load_ton_iot_dataset  # noqa: E402
from src.satellite.ton_iot_preprocessing import fit_preprocessing, three_way_split, transform  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed" / "ton_iot"
SPLIT_SEED = 2024  # fixed dataset split seed — independent of FL training seeds


def main() -> None:
    df = load_ton_iot_dataset()
    print(f"Loaded {len(df):,} rows (expected {EXPECTED_ROWS:,})")
    assert len(df) == EXPECTED_ROWS, "Row count mismatch vs. audited value — re-check the source file"
    assert df["type"].nunique() == EXPECTED_TYPE_CLASSES, f"Expected {EXPECTED_TYPE_CLASSES} type classes, got {df['type'].nunique()}"

    train_df, val_df, test_df = three_way_split(df, seed=SPLIT_SEED)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    artifacts = fit_preprocessing(train_df)
    print(f"Fitted preprocessing on train only: {len(artifacts.feature_columns)} final feature columns")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        out = transform(split_df, artifacts)
        out.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        anomaly_rate = out["label_binary"].mean()
        print(f"  {name}: {len(out):,} rows, anomaly_rate={anomaly_rate:.4f}")

    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "source": "data/raw/ton_iot/train_test_network.csv",
        "split_seed": SPLIT_SEED,
        "feature_columns": artifacts.feature_columns,
        "num_features": len(artifacts.feature_columns),
        "rows": {"total": len(df), "train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    }, indent=2))
    print(f"Wrote artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
