"""ESA-ADB Track C (candidate) — explicit, reviewable, MINIMAL download.

Run from the repository root:

    python scripts/download_esa_adb.py [--mission 1|2|3|all]

Default: Mission 1 ONLY (3.776 GB compressed) — the larger of the two
officially-benchmarked missions (58 channels vs. Mission 2's 47; see
docs/research/datasets/satellite_dataset_comparison.md), sufficient
for the initial structural audit. Mission 3 is NOT part of the
official ESA-ADB benchmark at all (no `mission3_experiments.py`
exists in the benchmark repo) and is not downloaded by default.

This is NEVER called automatically by training/preprocessing code — a
one-time, human-triggered step, exactly like
scripts/download_opssat_ad.py (Phase 9A) and
scripts/download_smap_msl.py (Track B).

Official source (verified fresh this session, not assumed from a
prior report): Zenodo record https://zenodo.org/api/records/12528696
("ESA Anomaly Dataset", open access, version 1.0). File sizes and MD5
checksums are read directly from the live Zenodo API response — this
script verifies the downloaded bytes against the checksum Zenodo
itself reports, not a hardcoded value, so it keeps working if the
authors publish a new version.

Downloads into data/raw/esa_adb/ (git-ignored, exactly like
data/raw/opssat_ad/ and data/raw/smap_msl/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "esa_adb"

ZENODO_RECORD_API = "https://zenodo.org/api/records/12528696"


def fetch_record_metadata() -> dict:
    req = urllib.request.Request(ZENODO_RECORD_API, headers={"User-Agent": "orbitshield-track-c/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed, documented official host)
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="1", choices=["1", "2", "3", "all"],
                         help="Which mission ZIP(s) to download. Default: 1 only (minimal, per audit scope).")
    args = parser.parse_args()

    print(f"Fetching ESA-ADB record metadata from {ZENODO_RECORD_API} ...")
    record = fetch_record_metadata()
    files = record.get("files", [])
    if not files:
        print("ERROR: no files listed in the Zenodo record response.", file=sys.stderr)
        sys.exit(1)

    wanted_keys = {f"ESA-Mission{args.mission}.zip"} if args.mission != "all" else {f["key"] for f in files}
    to_download = [f for f in files if f["key"] in wanted_keys]
    if not to_download:
        print(f"ERROR: no matching files for --mission {args.mission}.", file=sys.stderr)
        sys.exit(1)

    total_expected = sum(f["size"] for f in to_download)
    print(f"Will download {len(to_download)} file(s), {total_expected:,} bytes "
          f"({total_expected / 1e9:.2f} GB) total.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in to_download:
        filename, url, expected_size = f["key"], f["links"]["self"], f["size"]
        expected_checksum = f.get("checksum", "")  # e.g. "md5:80750189..."
        dest = RAW_DIR / filename

        print(f"  {filename} ({expected_size:,} bytes) ...")
        req = urllib.request.Request(url, headers={"User-Agent": "orbitshield-track-c/1.0"})
        md5 = hashlib.md5()
        total_read = 0
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:  # noqa: S310
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                md5.update(chunk)
                total_read += len(chunk)

        if total_read != expected_size:
            print(f"    SIZE MISMATCH (got {total_read}, expected {expected_size})", file=sys.stderr)
            sys.exit(1)
        if expected_checksum.startswith("md5:") and md5.hexdigest() != expected_checksum.split(":", 1)[1]:
            print(f"    CHECKSUM MISMATCH (got {md5.hexdigest()}, expected {expected_checksum})", file=sys.stderr)
            sys.exit(1)
        print(f"    OK (size + MD5 checksum verified)")

    print(f"\nDone. {len(to_download)} file(s) written to {RAW_DIR}")
    print("Next: python scripts/inspect_esa_adb.py")


if __name__ == "__main__":
    main()
