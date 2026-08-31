"""ESA-ADB Mission 1 — real anomaly timeline + candidate time-window
analysis (audit/design only, no training).

Run from the repository root, after Mission 1 has been downloaded and
extracted (scripts/download_esa_adb.py --mission 1,
scripts/inspect_esa_adb.py):

    python scripts/analyze_esa_adb_time_window.py

Reads ONLY the real, downloaded data/raw/esa_adb/Mission1/ESA-Mission1/
{channels,labels,anomaly_types}.csv files (metadata) and a small,
representative SAMPLE of real per-channel telemetry files (one per
subsystem, chosen deterministically, not all 58/76 — loading every
10-15M-row channel is unnecessary for a window-selection design
decision) to measure real row density inside each candidate window.

Writes:
    results/reports/esa_adb_time_window_analysis.md

This script performs NO training and does not touch any other
existing artifact.
"""

from __future__ import annotations

import csv
import pickle
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSION_DIR = REPO_ROOT / "data" / "raw" / "esa_adb" / "Mission1" / "ESA-Mission1"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "esa_adb_time_window_analysis.md"

DATA_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
DATA_END = datetime(2013, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# One representative channel per real subsystem (deterministically the
# first Target=YES channel listed for that subsystem in channels.csv),
# used to measure REAL row density within candidate windows without
# loading all 58 multi-hundred-MB channel files.
REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM: dict[str, str] = {}


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def month_range(start: datetime, end: datetime) -> list[str]:
    keys = []
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        keys.append(month_key(cur))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return keys


def load_metadata():
    with open(MISSION_DIR / "channels.csv") as f:
        channels = {r["Channel"]: r for r in csv.DictReader(f)}
    with open(MISSION_DIR / "labels.csv") as f:
        label_rows = list(csv.DictReader(f))
    with open(MISSION_DIR / "anomaly_types.csv") as f:
        types = {r["ID"]: r for r in csv.DictReader(f)}
    return channels, label_rows, types


def build_events(channels: dict, label_rows: list[dict], types: dict) -> list[dict]:
    """One real anomaly EVENT per unique ID, with its full real
    start/end (min/max across all its channel rows) and the real set
    of channels/subsystems it affects."""
    by_id = defaultdict(list)
    for r in label_rows:
        by_id[r["ID"]].append(r)

    events = []
    for eid, rows in by_id.items():
        starts = [parse_ts(r["StartTime"]) for r in rows]
        ends = [parse_ts(r["EndTime"]) for r in rows]
        chans = sorted({r["Channel"] for r in rows})
        subs = sorted({channels[c]["Subsystem"] for c in chans if c in channels})
        t = types.get(eid, {})
        events.append({
            "id": eid, "start": min(starts), "end": max(ends),
            "channels": chans, "subsystems": subs,
            "category": t.get("Category", ""), "dimensionality": t.get("Dimensionality", ""),
            "locality": t.get("Locality", ""), "length": t.get("Length", ""),
            "n_mappings": len(rows),
        })
    return events


def monthly_timeline(events: list[dict]) -> dict[str, dict]:
    timeline = defaultdict(lambda: {
        "events": 0, "mappings": 0, "category": Counter(), "dimensionality": Counter(),
        "locality": Counter(), "length": Counter(), "channels": set(), "subsystems": set(),
    })
    for e in events:
        # Attribute the event to every month it overlaps (an event can span months).
        for mk in month_range(e["start"], e["end"]):
            bucket = timeline[mk]
            bucket["events"] += 1
            bucket["mappings"] += e["n_mappings"]
            bucket["category"][e["category"]] += 1
            bucket["dimensionality"][e["dimensionality"]] += 1
            bucket["locality"][e["locality"]] += 1
            bucket["length"][e["length"]] += 1
            bucket["channels"].update(e["channels"])
            bucket["subsystems"].update(e["subsystems"])
    return timeline


def window_stats(events: list[dict], win_start: datetime, win_end: datetime) -> dict:
    overlapping = [e for e in events if e["start"] <= win_end and e["end"] >= win_start]
    channels, subsystems = set(), set()
    for e in overlapping:
        channels.update(e["channels"])
        subsystems.update(e["subsystems"])
    return {
        "window": (win_start.date().isoformat(), win_end.date().isoformat()),
        "n_events": len(overlapping),
        "n_channels_affected": len(channels),
        "n_subsystems_affected": len(subsystems),
        "categories": Counter(e["category"] for e in overlapping),
        "dimensionality": Counter(e["dimensionality"] for e in overlapping),
        "events_fully_inside": sum(1 for e in overlapping if e["start"] >= win_start and e["end"] <= win_end),
        "events_straddling_start": sum(1 for e in overlapping if e["start"] < win_start <= e["end"]),
        "events_straddling_end": sum(1 for e in overlapping if e["start"] <= win_end < e["end"]),
    }


def measure_real_row_density(channel: str, win_start: datetime, win_end: datetime) -> int:
    """Load ONE real channel's full pickled series and count how many
    real rows fall inside [win_start, win_end] — a real, measured
    density, not an average-rate estimate."""
    z = zipfile.ZipFile(MISSION_DIR / "channels" / f"{channel}.zip")
    df = pickle.loads(z.read(channel))
    mask = (df.index >= win_start.replace(tzinfo=None)) & (df.index <= win_end.replace(tzinfo=None))
    return int(mask.sum())


def main() -> None:
    if not MISSION_DIR.exists():
        print(f"ERROR: {MISSION_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    channels, label_rows, types = load_metadata()
    events = build_events(channels, label_rows, types)
    print(f"Loaded {len(events)} real anomaly events from {len(label_rows)} label rows.")

    timeline = monthly_timeline(events)
    all_months = month_range(DATA_START, DATA_END)

    # Pick one representative Target=YES channel per subsystem (first
    # alphabetically), deterministic, for real row-density measurement.
    by_subsystem_target = defaultdict(list)
    for cid, meta in channels.items():
        if meta["Target"] == "YES":
            by_subsystem_target[meta["Subsystem"]].append(cid)
    for sub, chans in by_subsystem_target.items():
        REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM[sub] = sorted(chans, key=lambda c: int(c.split("_")[1]))[0]
    print("Representative channels:", REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM)

    # Find the busiest 12-month, 6-month, 3-month, 1-month windows by
    # sliding month-by-month and counting real overlapping events.
    candidates = {}
    for length_months in (1, 3, 6, 12):
        best = None
        for i in range(0, len(all_months) - length_months + 1):
            win_start_key = all_months[i]
            win_end_key = all_months[min(i + length_months - 1, len(all_months) - 1)]
            y, m = map(int, win_start_key.split("-"))
            win_start = datetime(y, m, 1, tzinfo=timezone.utc)
            ey, em = map(int, win_end_key.split("-"))
            next_month = datetime(ey, em, 28, tzinfo=timezone.utc) + timedelta(days=4)
            win_end = next_month.replace(day=1) - timedelta(seconds=1)
            win_end = min(win_end, DATA_END)
            stats = window_stats(events, win_start, win_end)
            score = (stats["n_events"], stats["n_subsystems_affected"])
            if best is None or score > best[0]:
                best = (score, win_start, win_end, stats)
        candidates[length_months] = best

    lines = [
        "# ESA-ADB Mission 1 — Real Anomaly Timeline + Candidate Time-Window Analysis",
        "",
        "**DESIGN/AUDIT ONLY — no training.**",
        "",
        f"Real anomaly events analyzed: {len(events)}. Real label rows: {len(label_rows)}. "
        f"Full telemetry range: {DATA_START.date()} to {DATA_END.date()} ({len(all_months)} months).",
        "",
        "## Representative channels used for real row-density measurement (one Target=YES channel per subsystem)",
        "",
    ]
    for sub, c in REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM.items():
        lines.append(f"- {sub}: `{c}`")

    lines += ["", "## Busiest candidate window per length (by real overlapping-event count, ties broken by subsystem coverage)", ""]
    for length_months, (score, win_start, win_end, stats) in sorted(candidates.items()):
        lines += [
            f"### ~{length_months}-month candidate: {stats['window'][0]} to {stats['window'][1]}",
            "",
            f"- Real overlapping anomaly events: {stats['n_events']}",
            f"  - fully inside the window: {stats['events_fully_inside']}",
            f"  - straddling the window START (began before, still active at window start): {stats['events_straddling_start']}",
            f"  - straddling the window END (active at window end, continues after): {stats['events_straddling_end']}",
            f"- Channels affected: {stats['n_channels_affected']} / 58 Target=YES channels",
            f"- Subsystems affected: {stats['n_subsystems_affected']} / 4",
            f"- Category breakdown: {dict(stats['categories'])}",
            f"- Dimensionality breakdown: {dict(stats['dimensionality'])}",
            "",
        ]
        # Real row-density measurement for each representative channel in this window.
        lines.append("Real measured row counts (representative channels, this window):")
        total_rows_sample = 0
        for sub, c in REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM.items():
            n_rows = measure_real_row_density(c, win_start, win_end)
            total_rows_sample += n_rows
            lines.append(f"- `{c}` ({sub}): {n_rows:,} real rows in this window")
        lines.append(f"- **Sum across the {len(REPRESENTATIVE_CHANNELS_BY_SUBSYSTEM)} representative channels: {total_rows_sample:,} rows** "
                      f"(rough extrapolation x58/4 = ~{total_rows_sample * 58 // 4:,} rows if every Target=YES channel had similar "
                      f"density -- NOT a per-channel measurement for all 58, a scale estimate only).")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
