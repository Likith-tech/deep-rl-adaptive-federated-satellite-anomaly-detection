"""ESA-ADB Mission 1 — real data loader.

Loads ONLY the approved scope (see docs/research/experiments/esa-adb/README.md
Entry 2 and the user's "APPROVED DATASET SCOPE" instruction):

- Primary window: 2000-01-01 00:00:00 to 2000-12-31 23:59:59 (inclusive).
- The 58 Target=YES benchmark channels (never the 18 Target=NO channels).
- Real `Subsystem`/`Group`/`Physical Unit` metadata from channels.csv.

Does NOT modify src/satellite/loader.py (OPSSAT-AD) or src/data/loader.py
(NSL-KDD) — entirely separate code path, separate raw-data directory.
"""

from __future__ import annotations

import csv
import pickle
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MISSION_DIR = REPO_ROOT / "data" / "raw" / "esa_adb" / "Mission1" / "ESA-Mission1"
CHANNELS_CSV = MISSION_DIR / "channels.csv"
LABELS_CSV = MISSION_DIR / "labels.csv"
ANOMALY_TYPES_CSV = MISSION_DIR / "anomaly_types.csv"
CHANNELS_DIR = MISSION_DIR / "channels"

# Approved primary window (inclusive), per the user's "APPROVED DATASET
# SCOPE" instruction and the real busiest-12-month finding in
# results/reports/esa_adb_time_window_analysis.md.
WINDOW_START = pd.Timestamp("2000-01-01 00:00:00")
WINDOW_END = pd.Timestamp("2000-12-31 23:59:59.999999")

# Internal temporal train/validation/test split, entirely inside the
# approved window. All boundaries were verified directly against all
# 44 real in-window anomaly events with zero straddling before being
# fixed here — this is a real, checked leakage-safety property, not an
# assumption. The TEST portion (2000-10-01 to 2000-12-31) is exactly
# the user-approved "backup" window, reused here as the held-out
# temporal test split rather than as a separate, smaller pilot. TEST
# has never changed and must not change (see Entry 4).
#
# VALIDATION was originally 2000-09-01 to 2000-09-30 (September only).
# That window's real anomaly events (5 events, all Length=Point,
# single-channel subsystem_3) produced ZERO anomalous SEQUENCES under
# the "last"-timestep labeling rule — a genuine, investigated finding
# (see docs/research/experiments/esa-adb/README.md Entry 4), not a
# bug. Per the user-approved "Option B" candidate from that
# investigation, VALIDATION was widened to 2000-07-01 to 2000-09-30
# (Jul-Sep) to capture real, multi-channel anomaly events (id_22,
# id_23 in July; id_36 in August), giving 4/8 clients genuine
# validation-anomaly signal instead of 0/8. TRAIN was shortened to
# 2000-01-01 to 2000-06-30 accordingly. TEST is unaffected.
TRAIN_START = pd.Timestamp("2000-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2000-06-30 23:59:59.999999")
VALIDATION_START = pd.Timestamp("2000-07-01 00:00:00")
VALIDATION_END = pd.Timestamp("2000-09-30 23:59:59.999999")
TEST_START = pd.Timestamp("2000-10-01 00:00:00")
TEST_END = pd.Timestamp("2000-12-31 23:59:59.999999")


class EsaAdbDataNotFoundError(FileNotFoundError):
    """Raised when the real, downloaded Mission 1 archive is missing."""


def _require_mission_dir() -> None:
    if not MISSION_DIR.exists():
        raise EsaAdbDataNotFoundError(
            f"ESA-ADB Mission 1 data not found at {MISSION_DIR}.\n"
            "Download and extract it with:\n"
            "  python scripts/download_esa_adb.py --mission 1\n"
            "  python scripts/inspect_esa_adb.py\n"
            "See docs/research/datasets/esa_adb_dataset_audit.md for provenance."
        )


@dataclass(frozen=True)
class ChannelMeta:
    channel: str
    subsystem: str
    physical_unit: str
    group: str
    target: bool


def load_channel_metadata() -> dict[str, ChannelMeta]:
    """Load channels.csv (76 rows, all channels). Deterministic dict,
    insertion order == file order."""
    _require_mission_dir()
    with open(CHANNELS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["Channel"]: ChannelMeta(
            channel=r["Channel"],
            subsystem=r["Subsystem"],
            physical_unit=r["Physical Unit"],
            group=r["Group"],
            target=(r["Target"] == "YES"),
        )
        for r in rows
    }


def target_yes_channels() -> list[str]:
    """The 58 approved benchmark channels, sorted by numeric channel
    index for deterministic ordering."""
    meta = load_channel_metadata()
    channels = [c for c, m in meta.items() if m.target]
    return sorted(channels, key=lambda c: int(c.split("_")[1]))


def load_label_rows() -> list[dict[str, str]]:
    """Raw labels.csv rows (3,589 rows): ID, Channel, StartTime, EndTime.
    Each row is a per-CHANNEL anomaly-window mapping — never aggregated
    across channels here (that aggregation, used only for the earlier
    time-window audit, is NOT the rule used for per-timestep labeling;
    see labeling.py)."""
    _require_mission_dir()
    with open(LABELS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_anomaly_types() -> dict[str, dict[str, str]]:
    _require_mission_dir()
    with open(ANOMALY_TYPES_CSV, newline="") as f:
        return {r["ID"]: r for r in csv.DictReader(f)}


def load_channel_window(
    channel: str, window_start: pd.Timestamp = WINDOW_START, window_end: pd.Timestamp = WINDOW_END
) -> pd.Series:
    """Load ONE real channel's pickled telemetry series and return only
    the rows inside [window_start, window_end], sorted by timestamp,
    with duplicate timestamps removed (keep first — none were found in
    the channels sampled during the audit, but this is asserted, not
    assumed, at pipeline level; see pipeline.py's leakage/integrity
    checks). Deterministic: same input always produces the same output.
    """
    _require_mission_dir()
    zip_path = CHANNELS_DIR / f"{channel}.zip"
    if not zip_path.is_file():
        raise EsaAdbDataNotFoundError(f"Channel archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        df = pickle.loads(z.read(channel))

    series = df[channel]
    series = series[(series.index >= window_start) & (series.index <= window_end)]
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series.name = channel
    return series


def timestamp_to_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
