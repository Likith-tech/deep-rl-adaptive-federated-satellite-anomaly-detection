import pandas as pd
import pytest
from orbitshield.data.contracts import load_config, load_dataset
from orbitshield.data.common_preprocessing import prepare_dataset


def test_offline_summary_contract_and_trace(ops_config, tmp_path):
    data = load_dataset(load_config(ops_config))
    assert len(data.features) == 19
    assert not {'anomaly', 'train', 'segment', 'channel'} & set(data.features)
    assert data.audits['temporal']['source_order_reversals'] == 0
    result = prepare_dataset(ops_config, tmp_path / 'artifact')
    assert result['passed'] and result['splits']['official_test']['rows'] == 10
    assert (tmp_path / 'artifact/observation_trace.csv').is_file()


@pytest.mark.parametrize('fault,match', [
    ('metadata', 'metadata mismatch'), ('ordering', 'ordering reversals'),
    ('duplicate_timestamp', 'duplicate channel'), ('extra_column', 'schema mismatch')])
def test_invalid_telemetry_rejected(ops_config, fault, match):
    path = ops_config.parent / 'segments.csv'
    frame = pd.read_csv(path)
    if fault == 'metadata': frame.loc[0, 'anomaly'] = 1 - frame.loc[0, 'anomaly']
    if fault == 'ordering': frame.loc[[0, 1], 'timestamp'] = frame.loc[[1, 0], 'timestamp'].to_numpy()
    if fault == 'duplicate_timestamp': frame.loc[1, 'timestamp'] = frame.loc[0, 'timestamp']
    if fault == 'extra_column': frame['target_copy'] = frame.anomaly
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_dataset(load_config(ops_config))
