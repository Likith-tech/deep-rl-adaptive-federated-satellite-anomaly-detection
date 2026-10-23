"""Phase 9A — explicit, reviewable download of the OPSSAT-AD dataset.

Run from the repository root:

    python scripts/download_opssat_ad.py

This is NEVER called automatically by training/preprocessing code — it is
a one-time, human-triggered step, exactly like scripts/inspect_dataset.py
was for NSL-KDD in Phase 1.

Official source (verified in docs/research/datasets/spacecraft_dataset_search.md
and the Phase 9A pre-integration audit): Zenodo record
https://zenodo.org/records/12588358 ("OPSSAT-AD - anomaly detection dataset
for satellite telemetry", CC-BY-4.0). That record ID resolves (via Zenodo's
own versioning redirect) to the latest concept version — this script follows
that redirect via the Zenodo REST API rather than hardcoding a specific
version ID, so it keeps working if the authors publish a new version.

Downloads into data/raw/opssat_ad/ (git-ignored, exactly like data/raw/
already is for NSL-KDD). Writes every file in the official Zenodo record:
dataset.csv (18 pre-extracted features per segment — the file this phase's
preprocessing actually uses), segments.csv (raw telemetry, not used in this
phase, kept for provenance/future work), LICENSE, requirements.txt, and the
two example notebooks — nothing is selectively omitted, so the archive
matches the official release byte-for-byte.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "opssat_ad"

ZENODO_RECORD_API = "https://zenodo.org/api/records/12588358"
EXPECTED_LICENSE_ID = "cc-by-4.0"


def fetch_record_metadata() -> dict:
    req = urllib.request.Request(ZENODO_RECORD_API, headers={"User-Agent": "orbitshield-phase9a/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed, documented official host)
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(f"Fetching OPSSAT-AD record metadata from {ZENODO_RECORD_API} ...")
    record = fetch_record_metadata()

    license_id = (record.get("metadata", {}).get("license") or {}).get("id")
    if license_id != EXPECTED_LICENSE_ID:
        print(
            f"ERROR: expected license '{EXPECTED_LICENSE_ID}' but Zenodo reports "
            f"'{license_id}'. Stopping rather than downloading under an unverified "
            "license — re-check docs/research/datasets/spacecraft_dataset_search.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    files = record.get("files", [])
    if not files:
        print("ERROR: no files listed in the Zenodo record response.", file=sys.stderr)
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"License verified: {license_id}. Downloading {len(files)} files to {RAW_DIR} ...")

    for f in files:
        filename = f["key"]
        url = f["links"]["self"]
        expected_size = f["size"]
        dest = RAW_DIR / filename

        print(f"  {filename} ({expected_size:,} bytes) ...", end=" ")
        req = urllib.request.Request(url, headers={"User-Agent": "orbitshield-phase9a/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = resp.read()

        if len(data) != expected_size:
            print(f"SIZE MISMATCH (got {len(data)}, expected {expected_size})")
            sys.exit(1)

        dest.write_bytes(data)
        print("OK")

    print(f"\nDone. {len(files)} files written to {RAW_DIR}")
    print("Next: python scripts/inspect_opssat_ad.py (produces the pre-training inspection report)")


if __name__ == "__main__":
    main()
