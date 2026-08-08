"""Shared fixtures for data/preprocessing tests: a tiny synthetic dataset
shaped exactly like NSL-KDD, so tests run in milliseconds and never
require the real ~22MB dataset to be present."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


@pytest.fixture
def synthetic_nsl_kdd_df() -> pd.DataFrame:
    rows = [
        # duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot,
        # num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root,
        # num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login,
        # is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
        # same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count,
        # dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        # dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
        # dst_host_rerror_rate, dst_host_srv_rerror_rate, attack, difficulty
        [0, "tcp", "http", "SF", 200, 300, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         5, 5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10, 10, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "normal", 20],
        [0, "tcp", "http", "SF", 210, 310, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         4, 4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 12, 12, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "normal", 19],
        [0, "udp", "private", "SF", 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         100, 1, 1.0, 1.0, 0.0, 0.0, 0.05, 0.1, 0.0, 255, 5, 0.02, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, "neptune", 19],
        [0, "tcp", "private", "S0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         120, 3, 1.0, 1.0, 0.0, 0.0, 0.02, 0.05, 0.0, 255, 10, 0.01, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, "neptune", 21],
        [1, "tcp", "ftp", "SF", 300, 400, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 8, 8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "normal", 18],
        [0, "icmp", "eco_i", "SF", 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         50, 50, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 255, 255, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "smurf", 15],
    ]
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


@pytest.fixture
def categorical_columns() -> list[str]:
    return CATEGORICAL_COLUMNS


@pytest.fixture
def numerical_columns() -> list[str]:
    return NUMERICAL_COLUMNS
