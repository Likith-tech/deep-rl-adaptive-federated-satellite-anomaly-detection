"""Offline OPS-SAT segment summaries; telemetry is validated and traceable."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .contracts import Dataset, CLEAN, numeric_frame, feature_groups
from .audit_dataset import profile
from .split_data import overlaps

FEATURES = 'sampling duration len mean var std kurtosis skew n_peaks smooth10_n_peaks smooth20_n_peaks diff_peaks diff2_peaks diff_var diff2_var gaps_squared len_weighted var_div_duration var_div_len'.split()
SUMMARY = ['segment', 'anomaly', 'train', 'channel'] + FEATURES
OBSERVATIONS = ['channel', 'timestamp', 'value', 'label', 'sampling', 'anomaly', 'segment', 'train']


def read_sources(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = pd.read_csv(directory / 'dataset.csv')
    obs = pd.read_csv(directory / 'segments.csv')
    if summary.columns.tolist() != SUMMARY or obs.columns.tolist() != OBSERVATIONS:
        raise ValueError('OPS-SAT exact schema mismatch')
    if summary.isna().any().any() or obs.isna().any().any():
        raise ValueError('OPS-SAT missing source value')
    numeric_frame(summary, [c for c in SUMMARY if c != 'channel'])
    numeric_frame(obs, ['value', 'sampling', 'anomaly', 'segment', 'train'])
    for frame in (summary, obs):
        for col in ('anomaly', 'train'):
            if not frame[col].isin([0, 1]).all():
                raise ValueError('OPS-SAT nonbinary label or split flag')
        for col in ('segment', 'sampling'):
            if not (frame[col].gt(0) & frame[col].mod(1).eq(0)).all():
                raise ValueError('OPS-SAT invalid segment/sampling')
        if frame.channel.astype(str).str.strip().eq('').any():
            raise ValueError('OPS-SAT blank channel')
    audit = {'files': {
        'dataset.csv': profile(summary, FEATURES, 'anomaly', ['segment', 'train', 'channel']),
        'segments.csv': profile(obs, ['value'], 'anomaly', ['channel', 'timestamp', 'label', 'sampling', 'segment', 'train'])}}
    obs['source_row'] = range(len(obs))
    obs.timestamp = pd.to_datetime(obs.timestamp, utc=True, errors='raise')
    if summary.segment.duplicated().any() or set(summary.segment) != set(obs.segment):
        raise ValueError('OPS-SAT segment join mismatch')
    if obs.duplicated(['channel', 'timestamp']).any():
        raise ValueError('OPS-SAT duplicate channel/timestamp')
    reversals = int(obs.groupby('segment', sort=False).timestamp.diff().dt.total_seconds().lt(0).sum())
    if reversals:
        raise ValueError(f'OPS-SAT source ordering reversals: {reversals}')
    joined = obs.merge(summary[['segment', 'anomaly', 'train', 'channel', 'sampling']],
                       on='segment', suffixes=('_obs', '_summary'), validate='many_to_one')
    mismatches = {c: int(joined[c + '_obs'].ne(joined[c + '_summary']).sum())
                  for c in ['anomaly', 'train', 'channel', 'sampling']}
    if any(mismatches.values()):
        raise ValueError(f'OPS-SAT cross-file metadata mismatch: {mismatches}')
    obs = obs.sort_values(['segment', 'timestamp'], kind='stable').reset_index(drop=True)
    obs['gap_seconds'] = obs.groupby('segment').timestamp.diff().dt.total_seconds().fillna(0)
    obs['is_gap'] = obs.gap_seconds.gt(obs.sampling)
    ranges = obs.groupby('segment').timestamp.agg(start='min', end='max')
    audit['temporal'] = {'observations': len(obs), 'channels': int(obs.channel.nunique()),
                         'source_order_reversals': reversals, 'gap_transitions': int(obs.is_gap.sum()),
                         'duplicate_channel_timestamp': 0, 'metadata_mismatches': mismatches,
                         'raw_label_counts': obs.label.value_counts().to_dict(),
                         'windowing': 'none; offline segment summaries'}
    summary = summary.join(ranges, on='segment')
    return summary, obs, audit


def load(config: dict) -> Dataset:
    if config['split'] != 'supplied_segments_validation' or config['features'] != FEATURES:
        raise ValueError('Unsupported OPS split or feature allowlist')
    directory = Path(config['raw_dir'])
    summary, obs, audit = read_sources(directory)
    summary['source_id'] = summary.segment.map(lambda s: f'opssat_ad:segment:{int(s):06d}')
    summary['label_original'] = summary.anomaly.astype(int).astype(str)
    summary['y'] = summary.anomaly.astype('int64')
    summary['group'] = feature_groups(summary, FEATURES)
    pool = summary[summary.train.eq(1)]
    train, validation = train_test_split(pool, test_size=config['validation_fraction'],
                                        random_state=config['seed'], stratify=pool.anomaly)
    frames = {'train': train.reset_index(drop=True), 'validation': validation.reset_index(drop=True),
              'official_test': summary[summary.train.eq(0)].reset_index(drop=True)}
    membership = {s: split for split, frame in frames.items() for s in frame.segment}
    # Same-channel intervals must not overlap across partitions.
    interval_overlaps = 0
    for _, group in summary.groupby('channel'):
        active = []
        for row in group.sort_values('start').itertuples():
            active = [p for p in active if p.end >= row.start]
            interval_overlaps += sum(membership[p.segment] != membership[row.segment] for p in active)
            active.append(row)
    if interval_overlaps:
        raise ValueError('OPS-SAT cross-partition interval overlap')
    obs['split'] = obs.segment.map(membership)
    audit['temporal']['cross_split_same_channel_interval_overlaps'] = interval_overlaps
    audit['supplied_split'] = {'train_segments': len(pool), 'test_segments': int(summary.train.eq(0).sum())}
    audit['overlap'] = overlaps(frames)
    return Dataset(config, frames, FEATURES, [], {'0': 0, '1': 1},
                   [directory / 'dataset.csv', directory / 'segments.csv'], audit, [
        '19 precomputed dataset.csv statistics selected as offline segment-summary inputs; not FedSTAR features.',
        'Numeric anomaly target; text label is metadata. Channels are not physical satellite nodes.',
        'Supplied segment split plus stratified training validation; not chronological holdout.',
        'Same-channel interval checks do not establish independence of related events across different channels.',
        'Exact summary extraction formulas and segment-construction provenance remain unverified.'],
        obs[['source_row', 'segment', 'channel', 'timestamp', 'gap_seconds', 'is_gap', 'anomaly', 'split']])


def prepare(config_path: Path | None = None, output: Path | None = None) -> dict:
    from .common_preprocessing import prepare_dataset
    return prepare_dataset(config_path or CLEAN / 'configs/opssat_ad.yaml', output)


if __name__ == '__main__':
    from .common_preprocessing import adapter_cli
    adapter_cli('opssat_ad')
