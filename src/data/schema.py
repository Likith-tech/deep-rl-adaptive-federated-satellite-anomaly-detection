"""NSL-KDD dataset schema.

NSL-KDD ships as headerless CSV/TXT files. This module defines the
canonical 43-column schema (41 features + attack label + difficulty
score) as documented by the dataset's original publication, since the
raw files carry no header row.

Reference: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99
Data Set", IEEE CISDA 2009 (NSL-KDD feature definitions).
"""

from __future__ import annotations

# Column order exactly as it appears in KDDTrain+.txt / KDDTest+.txt.
FEATURE_COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

LABEL_COLUMN = "attack"
DIFFICULTY_COLUMN = "difficulty"

ALL_COLUMNS: list[str] = [*FEATURE_COLUMNS, LABEL_COLUMN, DIFFICULTY_COLUMN]

CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]

# Binary-valued columns encoded as 0/1 integers in the raw data.
BINARY_COLUMNS: list[str] = [
    "land",
    "logged_in",
    "is_host_login",
    "is_guest_login",
]

NUMERICAL_COLUMNS: list[str] = [
    col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS
]

# Attack -> category mapping, used only to preserve richer label
# information (label_original keeps the raw attack name; this mapping
# is not applied by default but is available for later multi-class work).
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos",
    "processtable": "dos", "udpstorm": "dos", "apache2": "dos", "worm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}

RAW_TRAIN_FILENAME = "KDDTrain+.txt"
RAW_TEST_FILENAME = "KDDTest+.txt"
