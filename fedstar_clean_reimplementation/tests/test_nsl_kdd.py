import json
import numpy as np
import pandas as pd
import pytest
from orbitshield.data.contracts import load_config, load_dataset
from orbitshield.data.common_preprocessing import prepare_dataset
from orbitshield.data.nsl_kdd_preprocessing import read_source


def test_labels_and_official_test_are_preserved(nsl_config, tmp_path):
    result = prepare_dataset(nsl_config, tmp_path / 'result')
    assert result['passed']
    assert result['splits']['official_test']['rows'] == 16
    data = load_dataset(load_config(nsl_config))
    assert set(data.frames['train'].source_id).isdisjoint(data.frames['validation'].source_id)
    assert not set(data.frames['train']['group']) & set(data.frames['validation']['group'])


@pytest.mark.parametrize('fault', ['width', 'label', 'nan', 'infinite'])
def test_bad_source_rejected(nsl_config, fault):
    path = nsl_config.parent / 'train.txt'
    raw = pd.read_csv(path, header=None)
    if fault == 'width': raw = raw.iloc[:, :-1]
    if fault == 'label': raw.iloc[0, 41] = 'not-a-known-attack'
    if fault == 'nan': raw.iloc[0, 0] = np.nan
    if fault == 'infinite': raw.iloc[0, 0] = np.inf
    raw.to_csv(path, header=False, index=False)
    with pytest.raises(ValueError):
        read_source(path, 'test_fixture')


def test_heldout_changes_do_not_change_fit(nsl_config, tmp_path):
    import joblib
    prepare_dataset(nsl_config, tmp_path / 'first')
    raw = pd.read_csv(nsl_config.parent / 'test.txt', header=None)
    raw.iloc[:, 0] = 999999
    raw.iloc[:, 2] = 'previously_unseen_service'
    raw.to_csv(nsl_config.parent / 'test.txt', header=False, index=False)
    prepare_dataset(nsl_config, tmp_path / 'second')
    first = joblib.load(tmp_path / 'first/preprocessor.joblib')
    second = joblib.load(tmp_path / 'second/preprocessor.joblib')
    np.testing.assert_array_equal(first.named_transformers_['numeric'].mean_, second.named_transformers_['numeric'].mean_)
    assert first.get_feature_names_out().tolist() == second.get_feature_names_out().tolist()
