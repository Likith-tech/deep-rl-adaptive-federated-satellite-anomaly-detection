"""TON_IoT dataset track — reusable loader.

Mirrors src/satellite/loader.py's role for OPSSAT-AD: discovers the raw
file, loads it, and exposes the (disclosed, documented) feature-column
decision used by this track. See
docs/research/experiments/ton-iot-bot-iot/README.md for the full
feasibility audit this loader's column choices are based on.

Source file: `train_test_network.csv`, the official UNSW ToN_IoT
"Train_Test_Network" release (44 columns: 42 raw fields + `label` +
`type`), obtained via a public, ungated HuggingFace mirror
(`codymlewis/TON_IoT_network`) after the canonical UNSW distribution
channels (research.unsw.edu.au -> UNSW SharePoint/OneDrive folders)
were confirmed to require an institutional login this environment does
not have (verified: both folder links return HTTP 403 "Access denied"
without authentication). Row count (211,043) and category count (10:
normal + 9 attack types) were verified directly against the downloaded
file and match the FedSTAR paper's own stated TON_IoT description
exactly (44 features, 211,043 instances, 10 categories).

Column disposition (all decided by direct inspection of real
cardinalities in the downloaded file, not assumed):
  - LABEL_COLUMN "label": binary, already 0/1, verified real (0=normal,
    1=attack) via direct value_counts.
  - TYPE_COLUMN "type": the 10-class attack-family label, used only as
    the category column for Dirichlet/IID client partitioning — never
    used as a training target (this track is binary anomaly detection).
  - NUMERIC_FEATURE_COLUMNS: 16 genuinely numeric fields.
  - CATEGORICAL_FEATURE_COLUMNS: 20 low-cardinality (2-13 unique
    values) string/flag fields, one-hot encoded.
  - DROPPED_COLUMNS: 6 high-cardinality, identifier-like fields
    (src_ip 51 unique, dst_ip 753, dns_query 726, http_uri 86,
    http_user_agent 35, weird_name 11-ish but free-text) that are
    non-generalizable host/content identifiers, not traffic-shape
    features — the same "drop non-generalizable identifiers" decision
    this project already applied to NSL-KDD in Phase 1
    (docs/research/dataset/feature_decisions.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "ton_iot"
DATASET_FILENAME = "train_test_network.csv"

LABEL_COLUMN = "label"          # 0 = normal, 1 = attack (verified real)
TYPE_COLUMN = "type"            # 10-class attack family; partitioning-only

NUMERIC_FEATURE_COLUMNS = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "proto", "service", "conn_state",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_version",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_addl", "weird_notice",
]

DROPPED_COLUMNS = ["src_ip", "dst_ip", "dns_query", "http_uri", "http_user_agent", "weird_name"]

EXPECTED_ROWS = 211_043
EXPECTED_TYPE_CLASSES = 10


class TonIotDatasetNotFoundError(FileNotFoundError):
    """Raised when data/raw/ton_iot/train_test_network.csv is missing."""


def load_ton_iot_dataset(raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    dataset_path = raw_dir / DATASET_FILENAME
    if not dataset_path.is_file():
        raise TonIotDatasetNotFoundError(
            f"TON_IoT train_test_network.csv not found at {dataset_path}.\n"
            "See docs/research/experiments/ton-iot-bot-iot/README.md for provenance "
            "and how it was obtained (public HuggingFace mirror of the official UNSW release)."
        )
    df = pd.read_csv(dataset_path)

    expected_columns = set(
        NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + DROPPED_COLUMNS + [LABEL_COLUMN, TYPE_COLUMN]
    )
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"train_test_network.csv is missing expected columns: {sorted(missing)}")

    return df
