"""Synthetic unit fixtures only; all writes stay in isolated temporary directories."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN / 'src'))
from orbitshield.data.contracts import sha256
from orbitshield.data.nsl_kdd_preprocessing import FEATURES, CATS
from orbitshield.data.stin_preprocessing import FEATURES as SF, LABELS, FILES
from orbitshield.data.opssat_ad_preprocessing import FEATURES as OF, SUMMARY, OBSERVATIONS


def config_file(folder, name, values):
    path = folder / (name + '.yaml')
    values.update(dataset=name, seed=42, validation_fraction=.2, missing_policy='reject', numeric_transform='standard')
    path.write_text(yaml.safe_dump(values), encoding='utf8')
    return path


@pytest.fixture
def nsl_config(tmp_path):
    for name, n, offset in [('train.txt', 60, 0), ('test.txt', 16, 100)]:
        rows = []
        for i in range(n):
            row = {c: float(i + offset + 1) for c in FEATURES}
            row.update(protocol_type=['tcp', 'udp'][i % 2], service=['http', 'smtp'][i % 2], flag='SF')
            rows.append([row[c] for c in FEATURES] + ['normal' if i % 2 else 'neptune', 1])
        pd.DataFrame(rows).to_csv(tmp_path / name, index=False, header=False)
    return config_file(tmp_path, 'nsl_kdd', {
        'raw_train': 'train.txt', 'raw_test': 'test.txt', 'categorical_columns': CATS,
        'split': 'official_group_validation'})


@pytest.fixture
def stin_config(tmp_path):
    directory = tmp_path / 'raw'
    directory.mkdir()
    tables = []
    for index in range(180):
        label = LABELS[index // 20]
        row = {'id': index, **{c: float(index + 1) for c in SF}, ' Label': label}
        tables.append(row)
    pool = pd.DataFrame(tables)
    entries = []
    for name in FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pool.iloc[:90] if name == 'SAT20.csv' else pool.iloc[90:] if name == 'TER20.csv' else pool
        frame.to_csv(path, index=False)
        entries.append({'path': name, 'sha256': sha256(path), 'size_bytes': path.stat().st_size})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': entries, 'commit': 'synthetic-unit-fixture', 'source': 'test-only'}), encoding='utf8')
    return config_file(tmp_path, 'stin', {'raw_dir': 'raw', 'manifest': 'manifest.json',
                                        'test_fraction': .2, 'classes': LABELS, 'split': 'grouped_80_20'})


@pytest.fixture
def ops_config(tmp_path):
    rows, observations = [], []
    for i in range(1, 31):
        rows.append({'segment': i, 'anomaly': i % 2, 'train': int(i <= 20), 'channel': 'test_channel',
                     **{c: float(i) for c in OF}, 'sampling': 1})
        for step in range(3):
            observations.append({'channel': 'test_channel', 'timestamp': f'2024-01-01T00:{i:02d}:{step:02d}Z',
                                 'value': float(i + step), 'label': 'anomaly', 'sampling': 1,
                                 'anomaly': i % 2, 'segment': i, 'train': int(i <= 20)})
    pd.DataFrame(rows)[SUMMARY].to_csv(tmp_path / 'dataset.csv', index=False)
    pd.DataFrame(observations)[OBSERVATIONS].to_csv(tmp_path / 'segments.csv', index=False)
    return config_file(tmp_path, 'opssat_ad', {'raw_dir': '.', 'features': OF, 'split': 'supplied_segments_validation'})


@pytest.fixture
def artifact(nsl_config, tmp_path):
    from orbitshield.data.common_preprocessing import prepare_dataset
    destination = tmp_path / 'artifact'
    prepare_dataset(nsl_config, destination)
    return destination


@pytest.fixture(autouse=True)
def forbid_production_writes(monkeypatch):
    from orbitshield.data import common_preprocessing as cp
    original = cp.write_new_artifact
    def guarded(output, *args, **kwargs):
        if Path(output).resolve().is_relative_to((CLEAN / 'data/processed').resolve()):
            pytest.fail('Tests must never write production artifacts')
        return original(output, *args, **kwargs)
    monkeypatch.setattr(cp, 'write_new_artifact', guarded)
