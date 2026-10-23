"""ESA-ADB Track C (candidate) — Mission 1 structural audit.

Run from the repository root, after
`python scripts/download_esa_adb.py --mission 1`:

    python scripts/inspect_esa_adb.py

Reads ONLY data/raw/esa_adb/ExESA-Mission1.zip (real, downloaded data
— never fabricated). Extracts it (measuring real extracted size before
committing to it), inspects the actual real directory structure/files
inside, and writes results/reports/esa_adb_mission1_inspection.md.
Read-only audit — no training, no modification of any other artifact.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "esa_adb"
ZIP_PATH = RAW_DIR / "ESA-Mission1.zip"
EXTRACT_DIR = RAW_DIR / "Mission1"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "esa_adb_mission1_inspection.md"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found. Run scripts/download_esa_adb.py --mission 1 first.", file=sys.stderr)
        sys.exit(1)

    zf = zipfile.ZipFile(ZIP_PATH)
    infos = zf.infolist()
    total_uncompressed = sum(i.file_size for i in infos)
    print(f"Archive contains {len(infos)} entries, {human(total_uncompressed)} uncompressed "
          f"(compressed archive: {human(ZIP_PATH.stat().st_size)}).")

    top_level = sorted({i.filename.split("/")[0] for i in infos})
    print("Top-level entries:", top_level)

    # Extract everything -- real extracted size measured directly below,
    # not estimated from the compression ratio.
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {EXTRACT_DIR} ...")
    zf.extractall(EXTRACT_DIR)

    all_files = list(EXTRACT_DIR.rglob("*"))
    file_list = [p for p in all_files if p.is_file()]
    real_extracted_bytes = sum(p.stat().st_size for p in file_list)
    print(f"Real extracted size on disk: {human(real_extracted_bytes)} across {len(file_list)} files.")

    # Categorize files by extension / rough role.
    by_suffix = {}
    for p in file_list:
        by_suffix.setdefault(p.suffix or "(none)", []).append(p)

    lines = [
        "# ESA-ADB Mission 1 — Structural Audit (real, downloaded data)",
        "",
        "**AUDIT ONLY — no model was trained.**",
        "",
        f"- Compressed download: {human(ZIP_PATH.stat().st_size)} ({ZIP_PATH.stat().st_size:,} bytes)",
        f"- Real extracted size: {human(real_extracted_bytes)} ({real_extracted_bytes:,} bytes) across {len(file_list)} files",
        f"- Top-level archive entries: {top_level}",
        "",
        "## Files by extension",
        "",
    ]
    for suffix, files in sorted(by_suffix.items(), key=lambda kv: -len(kv[1])):
        total_size = sum(p.stat().st_size for p in files)
        lines.append(f"- `{suffix}`: {len(files)} files, {human(total_size)} total")
        for p in sorted(files)[:5]:
            lines.append(f"    - {p.relative_to(EXTRACT_DIR)} ({human(p.stat().st_size)})")
        if len(files) > 5:
            lines.append(f"    - ... and {len(files) - 5} more")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
