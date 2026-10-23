import json
import numpy as np
import pandas as pd
import pytest
from orbitshield.data.contracts import load_config, load_dataset, sha256
from orbitshield.data.common_preprocessing import prepare_dataset
from orbitshield.data.stin_preprocessing import LABELS, FEATURES
from orbitshield.data.split_data import grouped_row_holdout


def test_all_seven_files_verified_and_only_primary_pool_used(stin_config, tmp_path):
    data = load_dataset(load_config(stin_config))
    assert sum(len(f) for f in data.frames.values()) == 180
    assert len(data.numeric) == 30
    assert set(data.label_mapping) == set(LABELS)
    assert len(data.audits['files']) == 7
    assert all('host' not in v and 'testing set' not in v for f in data.frames.values() for v in f.source_id)
    result = prepare_dataset(stin_config, tmp_path / 'artifact')
    assert result['passed'] and result['feature_dimension'] == 30


def test_manifest_hash_failure(stin_config):
    path = stin_config.parent / 'raw/SAT20.csv'
    path.write_text(path.read_text() + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(load_config(stin_config))


def test_leading_label_header_and_no_benign_class(stin_config):
    data = load_dataset(load_config(stin_config))
    assert data.audits['files']['SAT20.csv']['original_header'][-1] == ' Label'
    assert 'id' not in data.features and 'Label' not in data.features
    assert set(data.label_mapping.values()) == set(range(9))


def test_duplicates_stay_together_and_split_repeats():
    frame = pd.DataFrame({'group': np.repeat([str(i) for i in range(50)], 2), 'y': np.repeat([i % 2 for i in range(50)], 2)})
    train, test = grouped_row_holdout(frame, .2, 2024)
    again, again_test = grouped_row_holdout(frame, .2, 2024)
    assert not set(train['group']) & set(test['group'])
    assert len(train) + len(test) == len(frame)
    assert train.equals(again) and test.equals(again_test)


@pytest.mark.parametrize('fault', ['predictor', 'label', 'schema'])
def test_invalid_stin_even_with_updated_manifest(stin_config, fault):
    path = stin_config.parent / 'raw/SAT20.csv'
    raw = pd.read_csv(path)
    if fault == 'predictor': raw.iloc[0, 1] = np.inf
    if fault == 'label': raw.loc[0, ' Label'] = 'invented-benign'
    if fault == 'schema': raw['extra_target'] = 1
    raw.to_csv(path, index=False)
    manifest_path = stin_config.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['sha256'] = sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError):
        load_dataset(load_config(stin_config))
