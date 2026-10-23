"""Versioned artifacts with independent source reconstruction and strict validation."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .contracts import (CLEAN, Dataset, digest, json_text, load_config, load_dataset,
                        sha256, source_manifest)
from .split_data import overlaps

VERSION = '2'


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inside_clean(path: Path) -> Path:
    path = Path(path).resolve()
    ensure(path.is_relative_to(CLEAN.resolve()), 'Outputs must be inside fedstar_clean_reimplementation')
    return path


def implementation_hashes() -> dict:
    return {str(p.relative_to(CLEAN)): sha256(p) for p in sorted((CLEAN / 'src').rglob('*.py'))}


def fit_transform_splits(data: Dataset) -> tuple[dict, ColumnTransformer, list[str]]:
    transformers = [('numeric', StandardScaler(), data.numeric)]
    if data.categorical:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), data.categorical))
    transformer = ColumnTransformer(transformers, verbose_feature_names_out=False)
    arrays = {}
    for split, frame in data.frames.items():
        x = frame[data.features]
        values = transformer.fit_transform(x) if split == 'train' else transformer.transform(x)
        arrays[split] = np.asarray(values, dtype=np.float32)
        ensure(np.isfinite(arrays[split]).all(), f'{split}: non-finite transformed values')
    return arrays, transformer, transformer.get_feature_names_out().tolist()


def transformer_state(transformer: ColumnTransformer) -> dict:
    scaler = transformer.named_transformers_['numeric']
    state = {'feature_names': transformer.get_feature_names_out().tolist(),
             'numeric_columns': list(transformer.transformers_[0][2]),
             'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist(),
             'scale': scaler.scale_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_)}
    if 'categorical' in transformer.named_transformers_:
        state['categories'] = [v.tolist() for v in transformer.named_transformers_['categorical'].categories_]
    return state


def counts(y: np.ndarray) -> dict:
    labels, sizes = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(labels, sizes)}


def records(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[['source_id', 'label_original', 'y', 'group']].reset_index(drop=True).astype(str)


def check_partitions(data: Dataset) -> dict:
    seen: set[str] = set()
    for split, frame in data.frames.items():
        ensure(not frame.empty, f'{split}: empty partition')
        ids = frame.source_id.tolist()
        ensure(len(ids) == len(set(ids)), f'{split}: duplicate source IDs')
        ensure(not (seen & set(ids)), f'{split}: cross-partition source ID overlap')
        seen.update(ids)
    measured = overlaps(data.frames)
    for pair, result in measured.items():
        allowed = data.config['dataset'] == 'nsl_kdd' and pair.endswith('__official_test')
        ensure(allowed or result['feature_groups'] == 0, f'{pair}: unexpected feature-group overlap')
    return measured


def write_new_artifact(output: Path, data: Dataset, arrays: dict, transformer: ColumnTransformer,
                       names: list[str], provenance: dict) -> None:
    output = inside_clean(output)
    # Never overwrite, including earlier v1 production artifacts or partial runs.
    output.mkdir(parents=True, exist_ok=False)
    schema = {'input_features': data.features, 'numeric': data.numeric,
              'categorical': data.categorical, 'feature_names': names,
              'label_mapping': data.label_mapping}
    (output / 'schema.json').write_text(json_text(schema), encoding='utf8')
    (output / 'resolved_config.json').write_text(json_text(data.config), encoding='utf8')
    (output / 'audit.json').write_text(json_text(data.audits), encoding='utf8')
    split_meta = {}
    for split, frame in data.frames.items():
        np.save(output / f'X_{split}.npy', arrays[split], allow_pickle=False)
        np.save(output / f'y_{split}.npy', frame.y.to_numpy(dtype=np.int64), allow_pickle=False)
        rec = records(frame)
        rec.to_csv(output / f'records_{split}.csv', index=False)
        split_meta[split] = {'rows': len(frame), 'label_counts': counts(frame.y.to_numpy()),
                             'ordered_source_id_sha256': digest(frame.source_id.tolist())}
    if data.trace is not None:
        data.trace.to_csv(output / 'observation_trace.csv', index=False)
    joblib.dump(transformer, output / 'preprocessor.joblib')
    meta = {**provenance, 'format_version': VERSION, 'dataset': data.config['dataset'],
            'seed': data.config['seed'], 'splits': split_meta, 'schema': schema,
            'fit_scope': 'train only', 'fit_source_id_sha256': digest(data.frames['train'].source_id.tolist()),
            'transformer_state': transformer_state(transformer), 'notes': data.notes,
            'overlap': check_partitions(data), 'total_rows': sum(len(f) for f in data.frames.values())}
    (output / 'metadata.json').write_text(json_text(meta), encoding='utf8')
    # Manifest detects ordinary corruption; source reconstruction below also detects
    # semantic corruption if a caller recomputes payload checksums.
    hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'artifact_hashes.json').write_text(json_text(hashes), encoding='utf8')


def validate_saved_artifact(output: Path, config_path: Path | None = None) -> dict:
    output = Path(output).resolve()
    hashes = json.loads((output / 'artifact_hashes.json').read_text(encoding='utf8'))
    ensure(set(hashes) == {p.name for p in output.iterdir() if p.is_file()} - {'artifact_hashes.json'},
           'Artifact file inventory mismatch')
    for name, expected in hashes.items():
        ensure(Path(name).name == name and sha256(output / name) == expected, f'Artifact hash mismatch: {name}')
    meta = json.loads((output / 'metadata.json').read_text(encoding='utf8'))
    ensure(meta['format_version'] == VERSION, 'Unsupported artifact version')
    resolved = json.loads((output / 'resolved_config.json').read_text(encoding='utf8'))
    current = load_config(config_path or Path(resolved['_config_path']))
    ensure(current == resolved and digest(current) == meta['config_sha256'], 'Preprocessing configuration changed')
    ensure(implementation_hashes() == meta['implementation_hashes'], 'Implementation changed; generate a new artifact')
    for path, item in meta['sources'].items():
        ensure(sha256(Path(path)) == item['sha256'], f'Source hash mismatch: {path}')
    # Reconstruct labels, source IDs and every split from the immutable raw sources.
    data = load_dataset(current)
    ensure(source_manifest(data.source_files) == meta['sources'], 'Source inventory mismatch')
    ensure(check_partitions(data) == meta['overlap'], 'Partition overlap metadata mismatch')
    ensure(data.label_mapping == meta['schema']['label_mapping'], 'Label mapping mismatch')
    schema = json.loads((output / 'schema.json').read_text(encoding='utf8'))
    ensure(schema == meta['schema'] and schema['input_features'] == data.features
           and schema['numeric'] == data.numeric and schema['categorical'] == data.categorical,
           'Feature schema mismatch')
    ensure(json.loads((output / 'audit.json').read_text(encoding='utf8')) == data.audits, 'Audit/source mismatch')
    # Fresh fit is only for verification and is never persisted or used to replace the artifact.
    expected_arrays, fresh, names = fit_transform_splits(data)
    transformer = joblib.load(output / 'preprocessor.joblib')  # only trusted locally generated files
    state = transformer_state(transformer)
    ensure(state == transformer_state(fresh) == meta['transformer_state'], 'Fitted transformation is not training-only')
    ensure(names == schema['feature_names'] == transformer.get_feature_names_out().tolist(), 'Feature order mismatch')
    ensure(set(meta['splits']) == set(data.frames), 'Unexpected/missing partition')
    ensure(meta['fit_source_id_sha256'] == digest(data.frames['train'].source_id.tolist()), 'Fitting source IDs mismatch')
    seen: set[str] = set()
    for split, raw in data.frames.items():
        x = np.load(output / f'X_{split}.npy', allow_pickle=False)
        y = np.load(output / f'y_{split}.npy', allow_pickle=False)
        saved = pd.read_csv(output / f'records_{split}.csv', dtype=str, keep_default_na=False)
        ensure(not saved.source_id.duplicated().any(), f'{split}: duplicate source IDs')
        ensure(not seen.intersection(saved.source_id), f'{split}: cross-partition ID overlap')
        seen.update(saved.source_id)
        ensure(saved.equals(records(raw)), f'{split}: source IDs/labels/groups or row order mismatch')
        ensure(x.ndim == 2 and x.shape == (len(raw), len(names)), f'{split}: feature dimension mismatch')
        ensure(np.isfinite(x).all(), f'{split}: non-finite features')
        ensure(y.ndim == 1 and np.array_equal(y, raw.y.to_numpy()), f'{split}: labels do not match sources')
        ensure(np.array_equal(x, expected_arrays[split]), f'{split}: feature values/order mismatch')
        ensure(np.array_equal(x, transformer.transform(raw[data.features]).astype(np.float32)),
               f'{split}: saved transformer mismatch')
        expected_meta = {'rows': len(raw), 'label_counts': counts(y),
                         'ordered_source_id_sha256': digest(raw.source_id.tolist())}
        ensure(meta['splits'][split] == expected_meta, f'{split}: metadata counts/distribution mismatch')
    ensure(len(seen) == meta['total_rows'], 'Total sample coverage mismatch')
    if data.trace is not None:
        expected_trace = data.trace.to_csv(index=False, lineterminator='\n')
        ensure((output / 'observation_trace.csv').read_text(encoding='utf8') == expected_trace,
               'Observation trace mismatch')
    return {'passed': True, 'dataset': meta['dataset'], 'artifact': str(output),
            'artifact_manifest_sha256': sha256(output / 'artifact_hashes.json'),
            'feature_dimension': len(names), 'splits': meta['splits'], 'overlap': meta['overlap'],
            'checks': ['source/config/code hashes', 'raw label and source alignment', 'feature order and values',
                       'within/across split ID uniqueness', 'source split reconstruction', 'train-only fresh-fit agreement',
                       'finite values and schema', 'metadata counts', 'declared overlap policy']}


def prepare_dataset(config_path: Path, output: Path | None = None) -> dict:
    config = load_config(config_path)
    data = load_dataset(config)
    check_partitions(data)
    sources = source_manifest(data.source_files)
    provenance = {'sources': sources, 'config_sha256': digest(config),
                  'implementation_hashes': implementation_hashes(),
                  'environment': {'python': platform.python_version(), **{
                      p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'PyYAML', 'joblib']}}}
    fingerprint = digest(provenance)[:16]
    output = output or CLEAN / 'data/processed' / config['dataset'] / f'v2-{fingerprint}'
    if Path(output).exists():
        # Identical completed artifacts can be reused read-only; incomplete ones fail validation.
        return validate_saved_artifact(output, config_path)
    arrays, transformer, names = fit_transform_splits(data)
    ensure(source_manifest(data.source_files) == sources, 'Raw source changed during preparation')
    write_new_artifact(output, data, arrays, transformer, names, provenance)
    return validate_saved_artifact(output, config_path)


def adapter_cli(name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CLEAN / 'configs' / f'{name}.yaml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json_text(prepare_dataset(args.config, args.output)))
