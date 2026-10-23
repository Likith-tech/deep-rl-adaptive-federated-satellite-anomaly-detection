"""Corruption tests refresh checksums deliberately to exercise semantic validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from orbitshield.data.common_preprocessing import validate_saved_artifact, prepare_dataset
from orbitshield.data.contracts import sha256, json_text


def reseal(folder):
    hashes = {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'}
    (folder / 'artifact_hashes.json').write_text(json_text(hashes), encoding='utf8')


@pytest.mark.parametrize('fault,match', [
    ('labels', 'labels do not match'), ('columns', 'feature values/order'),
    ('duplicate_id', 'duplicate source IDs'), ('wrong_id', 'source IDs/labels'),
    ('record_label', 'source IDs/labels'), ('nonfinite', 'non-finite features'),
    ('width', 'dimension'), ('overlap_id', 'ID overlap'),
    ('count', 'metadata counts'), ('distribution', 'metadata counts'),
    ('schema_order', 'Feature order'), ('source_hash', 'Source hash'),
    ('config', 'configuration changed')])
def test_reject_semantic_corruption(artifact, fault, match):
    if fault in ['labels', 'columns', 'nonfinite', 'width']:
        path = artifact / ('y_train.npy' if fault == 'labels' else 'X_train.npy')
        values = np.load(path)
        if fault == 'labels': values = 1 - values
        if fault == 'columns': values = values[:, ::-1]
        if fault == 'nonfinite': values[0, 0] = np.nan
        if fault == 'width': values = values[:, :-1]
        np.save(path, values)
    elif fault in ['duplicate_id', 'wrong_id', 'record_label', 'overlap_id']:
        path = artifact / ('records_validation.csv' if fault == 'overlap_id' else 'records_train.csv')
        values = pd.read_csv(path, dtype=str, keep_default_na=False)
        if fault == 'duplicate_id': values.loc[1, 'source_id'] = values.loc[0, 'source_id']
        if fault == 'wrong_id': values.loc[0, 'source_id'] = 'unknown-record'
        if fault == 'record_label': values.loc[0, 'label_original'] = 'fake-label'
        if fault == 'overlap_id':
            values.loc[0, 'source_id'] = pd.read_csv(artifact / 'records_train.csv').source_id.iloc[0]
        values.to_csv(path, index=False)
    elif fault == 'config':
        config = json.loads((artifact / 'resolved_config.json').read_text())
        Path(config['_config_path']).write_text(Path(config['_config_path']).read_text().replace('seed: 42', 'seed: 43'))
    else:
        path = artifact / 'metadata.json'
        meta = json.loads(path.read_text())
        if fault == 'count': meta['splits']['train']['rows'] += 1
        if fault == 'distribution': meta['splits']['train']['label_counts']['0'] += 1
        if fault == 'source_hash': next(iter(meta['sources'].values()))['sha256'] = '0' * 64
        if fault == 'schema_order':
            meta['schema']['feature_names'].reverse()
            (artifact / 'schema.json').write_text(json_text(meta['schema']))
        path.write_text(json_text(meta), encoding='utf8')
    reseal(artifact)
    with pytest.raises(ValueError, match=match):
        validate_saved_artifact(artifact)


def test_payload_hash_detects_unresealed_corruption(artifact):
    path = artifact / 'y_train.npy'
    np.save(path, 1 - np.load(path))
    with pytest.raises(ValueError, match='Artifact hash'):
        validate_saved_artifact(artifact)


def test_missing_artifact_file_rejected(artifact):
    # Rename only an isolated fixture, never a production file.
    (artifact / 'X_train.npy').rename(artifact / 'renamed.npy')
    with pytest.raises(ValueError, match='inventory'):
        validate_saved_artifact(artifact)


def test_existing_artifacts_are_reused_without_writes(artifact, nsl_config):
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in artifact.iterdir()}
    prepare_dataset(nsl_config, artifact)
    assert before == {p.name: (sha256(p), p.stat().st_mtime_ns) for p in artifact.iterdir()}


def test_wrong_fitted_statistics_rejected(artifact):
    import joblib
    path = artifact / 'preprocessor.joblib'
    t = joblib.load(path)
    t.named_transformers_['numeric'].mean_[0] += 1
    joblib.dump(t, path)
    reseal(artifact)
    with pytest.raises(ValueError, match='training-only'):
        validate_saved_artifact(artifact)


def test_actual_changed_source_rejected(artifact):
    meta = json.loads((artifact / 'metadata.json').read_text())
    path = Path(next(iter(meta['sources'])))
    path.write_text(path.read_text() + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='Source hash'):
        validate_saved_artifact(artifact)


def test_unexpected_feature_overlap_rejected(stin_config):
    from orbitshield.data.contracts import load_config, load_dataset
    from orbitshield.data.common_preprocessing import check_partitions
    data = load_dataset(load_config(stin_config))
    data.frames['test'].loc[0, 'group'] = data.frames['train']['group'].iloc[0]
    with pytest.raises(ValueError, match='unexpected feature-group overlap'):
        check_partitions(data)


def test_fitting_id_digest_rejected(artifact):
    path = artifact / 'metadata.json'
    meta = json.loads(path.read_text())
    meta['fit_source_id_sha256'] = 'wrong-training-membership'
    path.write_text(json_text(meta), encoding='utf8')
    reseal(artifact)
    with pytest.raises(ValueError, match='Fitting source IDs'):
        validate_saved_artifact(artifact)
