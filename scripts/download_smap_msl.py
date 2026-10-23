"""SMAP/MSL Track B (candidate) — explicit, reviewable download.

Run from the repository root:

    python scripts/download_smap_msl.py

This is NEVER called automatically by training/preprocessing code — a
one-time, human-triggered step, exactly like
scripts/download_opssat_ad.py was for Phase 9A.

Official source (verified in
docs/research/datasets/satellite_dataset_comparison.md): the raw
per-channel train/test `.npy` files and `labeled_anomalies.csv`
originate from NASA/JPL's own `khundman/telemanom` repository
(Apache 2.0), which documents the Kaggle mirror
`patrickfleith/nasa-anomaly-detection-dataset-smap-msl` as the current
download path (the original raw S3 link is no longer the repo's
documented instruction). Verified this session: the Kaggle dataset
API's download endpoint redirects to a signed, public
`storage.googleapis.com` URL with NO authentication required — this
script uses that same public endpoint directly, with no Kaggle API
key.

Downloads only the canonical dataset contents (82 train `.npy` files,
82 test `.npy` files, `labeled_anomalies.csv`) into
`data/raw/smap_msl/` (git-ignored, exactly like `data/raw/opssat_ad/`)
— NOT the bundled `2018-05-19_15.00.10/` directory, which is a
snapshot of the original authors' own trained Keras models and results
from their 2018 paper run, not part of the dataset itself.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "smap_msl"

# Verified this session (see docs/research/datasets/satellite_dataset_comparison.md):
# this exact endpoint 302-redirects to a public, unauthenticated, signed
# GCS URL for this specific public Kaggle dataset.
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
EXPECTED_ENTRY_PREFIXES = ("data/data/train/", "data/data/test/", "labeled_anomalies.csv")


def main() -> None:
    print(f"Fetching SMAP/MSL archive from {DOWNLOAD_URL} ...")
    req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "orbitshield-track-b/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 (fixed, documented, verified-public host)
        archive_bytes = resp.read()
    print(f"Downloaded {len(archive_bytes):,} bytes.")

    zf = zipfile.ZipFile(BytesIO(archive_bytes))
    names = zf.namelist()
    to_extract = [n for n in names if n.startswith(EXPECTED_ENTRY_PREFIXES) and not n.endswith("/")]
    if not to_extract:
        print("ERROR: expected train/test/labeled_anomalies entries not found in archive.", file=sys.stderr)
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(to_extract)} dataset files to {RAW_DIR} ...")
    for name in to_extract:
        data = zf.read(name)
        # Flatten "data/data/train/X.npy" -> "train/X.npy"; keep labeled_anomalies.csv at top level.
        rel = name.split("data/data/", 1)[-1] if "data/data/" in name else name
        dest = RAW_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    train_count = len(list((RAW_DIR / "train").glob("*.npy")))
    test_count = len(list((RAW_DIR / "test").glob("*.npy")))
    print(f"\nDone. {train_count} train channels, {test_count} test channels, "
          f"labeled_anomalies.csv written to {RAW_DIR}")
    print("Next: python scripts/inspect_smap_msl.py (produces the pre-training inspection report)")


if __name__ == "__main__":
    main()
