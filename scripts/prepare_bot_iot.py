"""Build data/raw/bot_iot/bot_iot_sample_500k.parquet (if not already
present — it was built once during the feasibility audit, see
docs/research/experiments/ton-iot-bot-iot/README.md) and
data/processed/bot_iot/{train,validation,test}.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.satellite.bot_iot_loader import EXPECTED_CATEGORY_CLASSES, load_bot_iot_sample  # noqa: E402
from src.satellite.bot_iot_preprocessing import fit_preprocessing, three_way_split, transform  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed" / "bot_iot"
SPLIT_SEED = 2024


def main() -> None:
    df = load_bot_iot_sample()
    print(f"Loaded {len(df):,} rows (seeded 500k random sample of the 7,509,984-row full mirror)")
    assert df["category"].nunique() == EXPECTED_CATEGORY_CLASSES, (
        f"Expected {EXPECTED_CATEGORY_CLASSES} category classes, got {df['category'].nunique()}"
    )

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
        "source": "data/raw/bot_iot/bot_iot_sample_500k.parquet",
        "split_seed": SPLIT_SEED,
        "feature_columns": artifacts.feature_columns,
        "num_features": len(artifacts.feature_columns),
        "rows": {"total": len(df), "train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    }, indent=2))
    print(f"Wrote artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
