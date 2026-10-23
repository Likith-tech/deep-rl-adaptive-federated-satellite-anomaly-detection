"""Configuration, source provenance and canonical record contract."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

CLEAN = Path(__file__).resolve().parents[3]
REPO = CLEAN.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n'


def digest(value: object) -> str:
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf8'))
    if not isinstance(config, dict) or config.get('dataset') not in {'nsl_kdd', 'stin', 'opssat_ad'}:
        raise ValueError('Unknown dataset configuration')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('validation_fraction', 'test_fraction'):
        if key in config and not 0 < config[key] < 1:
            raise ValueError(f'{key} must be between zero and one')
    if config.get('numeric_transform') != 'standard' or config.get('missing_policy') != 'reject':
        raise ValueError('Supported policies: standard scaling and reject missing values')
    for key in ('raw_train', 'raw_test', 'raw_dir', 'manifest'):
        if key in config:
            config[key] = str((path.parent / config[key]).resolve())
    config['_config_path'] = str(path)
    config['_config_file_sha256'] = sha256(path)
    return config


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if frame.empty or frame[columns].isna().any().any() or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('Empty data or missing/non-finite numeric values')


def feature_groups(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    # Preserve historical NSL split; source hashes and dependency versions pin it.
    return pd.util.hash_pandas_object(frame[features], index=False).astype(str)


@dataclass
class Dataset:
    config: dict
    frames: dict[str, pd.DataFrame]
    numeric: list[str]
    categorical: list[str]
    label_mapping: dict[str, int]
    source_files: list[Path]
    audits: dict
    notes: list[str]
    trace: pd.DataFrame | None = None

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def load_dataset(config: dict) -> Dataset:
    from . import nsl_kdd_preprocessing, stin_preprocessing, opssat_ad_preprocessing
    return {'nsl_kdd': nsl_kdd_preprocessing.load,
            'stin': stin_preprocessing.load,
            'opssat_ad': opssat_ad_preprocessing.load}[config['dataset']](config)


def source_manifest(paths: list[Path]) -> dict:
    return {str(p.resolve()): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in paths}
