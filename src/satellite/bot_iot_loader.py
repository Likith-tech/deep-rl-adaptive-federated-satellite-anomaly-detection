"""Bot-IoT dataset track — reusable loader.

Mirrors src/satellite/ton_iot_loader.py's role. See
docs/research/experiments/ton-iot-bot-iot/README.md for the full
feasibility audit.

Source: the original UNSW Bot-IoT feature set (46 raw columns per the
FedSTAR paper's own description: pkSeqID, stime, flgs, flgs_number,
proto, proto_number, saddr, sport, daddr, dport, pkts, bytes, state,
state_number, ltime, seq, dur, mean, stddev, sum, min, max, spkts,
dpkts, sbytes, dbytes, rate, srate, drate, TnBPSrcIP, TnBPDstIP,
TnP_PSrcIP, TnP_PDstIP, TnP_PerProto, TnP_Per_Dport, AR_P_Proto_P_SrcIP,
AR_P_Proto_P_DstIP, N_IN_Conn_P_DstIP, N_IN_Conn_P_SrcIP,
AR_P_Proto_P_Sport, AR_P_Proto_P_Dport, Pkts_P_State_P_Protocol_P_DestIP,
Pkts_P_State_P_Protocol_P_SrcIP, attack, category, subcategory), was
obtained via a public, ungated HuggingFace mirror
(`masoltani/bot-iot`, `balanced_dataset.parquet`) after the canonical
UNSW distribution channel (research.unsw.edu.au -> UNSW
SharePoint/OneDrive folder) was confirmed to require an institutional
login this environment does not have (verified: HTTP 403 "Access
denied" without authentication, same failure mode as TON_IoT's
official channel).

This mirror is ALREADY numeric-encoded: the raw string/categorical
columns (flgs, proto, saddr, daddr, state, pkSeqID) were replaced by
their already-present `_number`/numeric equivalents and the
high-cardinality IP-address identifier columns (saddr, daddr) and the
row-sequence identifier (pkSeqID) were dropped by whoever built the
mirror — leaving 36 numeric feature columns + attack/category/
subcategory labels (39 total, vs. the paper's stated 46 raw columns).
This is a real, disclosed deviation: the FEATURE COUNT differs from
the paper's 46 (a ~10-feature reduction, all either duplicate-encoded
categorical columns or identifier columns, not information the
project's own NSL-KDD track keeps either — see ton_iot_loader.py's
`DROPPED_COLUMNS` for the same category of decision made explicitly
for TON_IoT here too). The full 5-class label structure (DDoS, DoS,
Reconnaissance, Theft, Normal) and the binary `attack` normal/attack
split were verified directly against the real file and exactly match
the paper's stated "5 categories" for Bot-IoT.

**Row-count deviation (disclosed):** the full mirror has 7,509,984
rows (a real subsample of the >73,000,000-row full Bot-IoT capture the
paper describes — UNSW's own public releases are themselves partial
extracts of the full capture, e.g. the well-known "5% extracted"
release). Training FedAvg with 5 seeds x 2 partitions on the full
7.5M-row mirror was not feasible in this environment's time budget, so
a SEEDED, UNSTRATIFIED random sample of 500,000 rows
(`data/raw/bot_iot/bot_iot_sample_500k.parquet`, built by
`scripts/prepare_bot_iot.py`, seed=2024) is used instead. Because the
sample is a simple random sample (not stratified/rebalanced), the
extreme real class imbalance is preserved by construction: 2,400/
500,000 = 0.48% normal in the sample vs. 35,800/7,509,984 = 0.48%
normal in the full mirror.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "bot_iot"
SAMPLE_FILENAME = "bot_iot_sample_500k.parquet"

LABEL_COLUMN = "attack"       # 0 = normal, 1 = attack (verified real)
CATEGORY_COLUMN = "category"  # 5-class (DDoS, DoS, Reconnaissance, Theft, Normal); partitioning-only

NUMERIC_FEATURE_COLUMNS = [
    "flgs_number", "proto_number", "sport", "dport", "pkts", "bytes", "state_number",
    "dur", "mean", "stddev", "sum", "min", "max", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate", "drate",
    "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP", "TnP_PerProto", "TnP_Per_Dport",
    "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP", "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP",
    "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
    "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP",
]

DROPPED_COLUMNS = ["stime", "ltime", "seq"]  # timestamps / sequence id — non-generalizable

EXPECTED_CATEGORY_CLASSES = 5


class BotIotDatasetNotFoundError(FileNotFoundError):
    """Raised when data/raw/bot_iot/bot_iot_sample_500k.parquet is missing."""


def load_bot_iot_sample(raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    dataset_path = raw_dir / SAMPLE_FILENAME
    if not dataset_path.is_file():
        raise BotIotDatasetNotFoundError(
            f"Bot-IoT sample parquet not found at {dataset_path}.\n"
            "Run scripts/prepare_bot_iot.py first — see "
            "docs/research/experiments/ton-iot-bot-iot/README.md for full provenance."
        )
    df = pd.read_parquet(dataset_path)

    expected_columns = set(NUMERIC_FEATURE_COLUMNS + DROPPED_COLUMNS + [LABEL_COLUMN, CATEGORY_COLUMN])
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"bot_iot_sample_500k.parquet is missing expected columns: {sorted(missing)}")

    return df
