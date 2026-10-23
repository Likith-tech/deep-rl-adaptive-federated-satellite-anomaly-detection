"""Official NSL-KDD protocol with a group-safe training validation carve-out."""
from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .contracts import Dataset, CLEAN, numeric_frame, feature_groups
from .audit_dataset import profile
from .split_data import overlaps

FEATURES = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
CATS = ['protocol_type', 'service', 'flag']
NUMERIC = [c for c in FEATURES if c not in CATS]
ATTACKS = set('normal back land neptune pod smurf teardrop mailbomb processtable udpstorm apache2 worm satan ipsweep nmap portsweep mscan saint guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named buffer_overflow loadmodule rootkit perl sqlattack xterm ps'.split())


def read_source(path: Path, source: str) -> tuple[pd.DataFrame, dict]:
    with path.open(encoding='utf8', newline='') as stream:
        for line, cells in enumerate(csv.reader(stream, strict=True), 1):
            if len(cells) != 43:
                raise ValueError(f'{path.name}: line {line} has {len(cells)} fields, expected 43')
    raw = pd.read_csv(path, header=None, names=FEATURES + ['attack', 'difficulty'])
    numeric_frame(raw, NUMERIC + ['difficulty'])
    if raw[CATS + ['attack']].isna().any().any():
        raise ValueError('Missing categorical value or attack label')
    if not set(raw.attack).issubset(ATTACKS):
        raise ValueError('Unknown NSL attack label')
    if any(raw[c].astype(str).str.strip().eq('').any() for c in CATS):
        raise ValueError('Blank categorical value')
    audit = profile(raw, FEATURES, 'attack', ['difficulty'])
    audit['timestamps'] = 'not supplied; row order is not physical time'
    raw['label_original'] = raw.attack
    raw['y'] = raw.attack.ne('normal').astype('int64')
    raw['source_id'] = [f'nsl_kdd:{source}:{i:09d}' for i in range(len(raw))]
    raw['group'] = feature_groups(raw, FEATURES)
    return raw, audit


def load(config: dict) -> Dataset:
    if config['split'] != 'official_group_validation' or config['categorical_columns'] != CATS:
        raise ValueError('Unsupported NSL split or categorical schema')
    paths = [Path(config['raw_train']), Path(config['raw_test'])]
    pool, pa = read_source(paths[0], 'official_train')
    test, ta = read_source(paths[1], 'official_test')
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=config['validation_fraction'],
                                   random_state=config['seed']).split(pool, groups=pool['group']))
    frames = {'train': pool.iloc[tr].reset_index(drop=True),
              'validation': pool.iloc[va].reset_index(drop=True), 'official_test': test}
    audits = {'files': {str(paths[0]): pa, str(paths[1]): ta},
              'overlap': overlaps(frames),
              'official_pool_test_overlap': overlaps({'pool': pool, 'test': test}),
              'conflicting_label_feature_groups': int(pool.groupby('group').y.nunique().gt(1).sum())}
    return Dataset(config, frames, NUMERIC, CATS, {'normal': 0, 'attack': 1}, paths, audits, [
        'Binary NSL task and group validation are project adaptations.',
        'Official test preserved. Historical exact-feature overlap is allowed only with official_test and measured in audit.',
        'Numeric StandardScaler and categorical one-hot are local choices; all constants retained.'])


def prepare(config_path: Path | None = None, output: Path | None = None) -> dict:
    from .common_preprocessing import prepare_dataset
    return prepare_dataset(config_path or CLEAN / 'configs/nsl_kdd.yaml', output)


if __name__ == '__main__':
    from .common_preprocessing import adapter_cli
    adapter_cli('nsl_kdd')
