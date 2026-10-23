"""Synthetic ESA-ADB-shaped fixtures — never the real downloaded
data/raw/esa_adb/ archive for the pure-logic tests, so most of this
suite runs in milliseconds without requiring the 3.5GB Mission 1
download. A small number of tests in test_real_data_pipeline.py DO
require the real archive and are skipped automatically when it is
absent (see MISSION_DIR.exists() guard there)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_channel_series() -> pd.Series:
    """A small, regularly-sampled (60s) synthetic channel series over
    2000-01-01, values 0..99."""
    index = pd.date_range("2000-01-01 00:00:00", periods=100, freq="60s")
    return pd.Series(np.arange(100, dtype="float32"), index=index, name="channel_synth")


@pytest.fixture
def synthetic_label_rows() -> list[dict[str, str]]:
    """Two real-shaped labels.csv rows: one multivariate event
    affecting channel_synth AND channel_other, one point event."""
    return [
        {"ID": "id_1", "Channel": "channel_synth", "StartTime": "2000-01-01T00:10:00.000Z", "EndTime": "2000-01-01T00:20:00.000Z"},
        {"ID": "id_1", "Channel": "channel_other", "StartTime": "2000-01-01T00:10:00.000Z", "EndTime": "2000-01-01T00:20:00.000Z"},
        {"ID": "id_2", "Channel": "channel_synth", "StartTime": "2000-01-01T00:50:00.000Z", "EndTime": "2000-01-01T00:50:00.000Z"},
    ]
