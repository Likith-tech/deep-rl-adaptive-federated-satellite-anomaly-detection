"""Hash-pinned STIN nine-class attack-type preprocessing."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .contracts import Dataset, CLEAN, sha256, numeric_frame, feature_groups
from .audit_dataset import profile
from .split_data import grouped_row_holdout, overlaps

FEATURES = 'fl_dur fw_pk l_fw_pkt l_bw_pkt pkt_len_min pkt_len_max pkt_len_std fl_byt_s fl_iat_min bw_iat_tot bw_iat_min bw_psh_flag fw_urg_flag bw_urg_flag fw_hdr_len bw_pkt_s fin_cnt syn_cnt psh_cnt urg_cnt ece_cnt down_up_ratio fw_byt_blk_avg fw_pkt_blk_avg fw_blk_rate_avg bw_byt_blk_avg bw_pkt_blk_avg bw_blk_rate_avg fw_win_byt bw_win_byt'.split()
LABELS = sorted(['Backdoor', 'Botnet', 'LDAP_DDoS', 'MSSQL_DDoS', 'NetBIOS_DDoS',
                 'Portmap_DDoS', 'Syn_DDoS', 'UDP_DDoS', 'Web Attack'])
FILES = ['SAT20.csv', 'TER20.csv'] + ['training and testing set of FL/' + f for f in
         ['SAT20_host1.csv', 'SAT20_host2.csv', 'TER20_host1.csv', 'TER20_host2.csv', 'testing set.csv']]


def load(config: dict) -> Dataset:
    if config['split'] != 'grouped_80_20' or config['classes'] != LABELS:
        raise ValueError('Unsupported STIN split or class mapping')
    directory, manifest_path = Path(config['raw_dir']), Path(config['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    entries = {e['path']: e for e in manifest['files']}
    if set(entries) != set(FILES):
        raise ValueError('STIN manifest must describe exactly the seven release CSV files')
    tables, audits, hashes = {}, {}, {}
    mapping = {label: i for i, label in enumerate(LABELS)}
    for name in FILES:
        path = directory / name
        hashes[name] = sha256(path)
        if hashes[name].lower() != entries[name]['sha256'].lower():
            raise ValueError(f'STIN source hash mismatch: {name}')
        raw = pd.read_csv(path)
        original_header = raw.columns.tolist()
        raw.columns = raw.columns.str.strip()
        if raw.columns.tolist() != ['id'] + FEATURES + ['Label']:
            raise ValueError(f'STIN schema mismatch: {name}')
        numeric_frame(raw, ['id'] + FEATURES)
        if raw.Label.isna().any() or not set(raw.Label).issubset(LABELS):
            raise ValueError(f'Unknown/missing STIN class: {name}')
        audits[name] = profile(raw, FEATURES, 'Label', ['id'])
        audits[name].update(original_header=original_header, sha256=hashes[name], hash_matches_manifest=True)
        # Canonical numeric dtype makes e.g. CSV integer 1 and decimal 1.0
        # the same predictor group across separately inferred source files.
        raw[FEATURES] = raw[FEATURES].astype('float64')
        raw['label_original'] = raw.Label
        raw['y'] = raw.Label.map(mapping).astype('int64')
        raw['source_id'] = [f'stin:{name}:{i:09d}' for i in range(len(raw))]
        raw['group'] = feature_groups(raw, FEATURES)
        tables[name] = raw
    # Inspect overlap before deciding the sample pool. Host/test files are audits only.
    overlap = overlaps(tables)
    pool = pd.concat([tables['SAT20.csv'], tables['TER20.csv']], ignore_index=True)
    if set(pool.label_original) != set(LABELS):
        raise ValueError('STIN primary pool does not contain all nine classes')
    train_pool, test = grouped_row_holdout(pool, config['test_fraction'], config['seed'])
    train, validation = grouped_row_holdout(train_pool, config['validation_fraction'], config['seed'] + 1)
    frames = {'train': train, 'validation': validation, 'test': test}
    audits = {'files': audits, 'source_file_overlap': overlap, 'overlap': overlaps(frames),
              'primary_pool_rows': len(pool), 'training_pool_rows': len(train_pool),
              'test_fraction_achieved': len(test) / len(pool),
              'conflicting_label_feature_groups': int(pool.groupby('group').y.nunique().gt(1).sum()),
              'manifest_commit': manifest.get('commit'), 'manifest_source': manifest.get('source'),
              'normal_samples': 0,
              'source_combination': 'SAT20.csv + TER20.csv only; alternative host/test files are never appended'}
    return Dataset(config, frames, FEATURES, [], mapping,
                   [directory / name for name in FILES] + [manifest_path], audits, [
        'Nine-class attack-type classification. No benign class exists in these files.',
        'Professor approval is required if the intended capstone target is binary anomaly detection.',
        'Table 1 says 32 features; release has 30 predictors plus id and Label. ID and target excluded.',
        'Whole-feature groups preserve duplicates and approximate paper 80:20 by row count; validation is a local carve-out.',
        'Recorded upstream commit verified against local manifest hashes, not independently fetched from remote.'])


def prepare(config_path: Path | None = None, output: Path | None = None) -> dict:
    from .common_preprocessing import prepare_dataset
    return prepare_dataset(config_path or CLEAN / 'configs/stin.yaml', output)


if __name__ == '__main__':
    from .common_preprocessing import adapter_cli
    adapter_cli('stin')
