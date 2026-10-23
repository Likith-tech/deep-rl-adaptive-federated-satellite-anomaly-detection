"""Immutable artifact references and verifiable client assignments; never fit X.

The older full-tree code inventory predates downstream modules. This reader
requires every recorded code hash unchanged and explicitly permits only added
representation/federated modules. Historical validators/manifests stay strict.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from orbitshield.data.contracts import CLEAN, digest, json_text, load_config, sha256
from .partition import PartitionConfig, allocate, summarize

PARTITION_ROOT = CLEAN / 'data/client_partitions'


def check_code_inventory(recorded: dict, current: dict) -> None:
    if any(current.get(k) != v for k, v in recorded.items()):
        raise ValueError('Recorded implementation changed or missing')
    prefixes = ('src/orbitshield/representation/', 'src/orbitshield/federated/')
    if any(not k.replace('\\', '/').startswith(prefixes) for k in current.keys() - recorded.keys()):
        raise ValueError('Unrecorded code outside permitted downstream namespaces')


class TrainingArtifact:
    """Validate immutable bindings/labels/split IDs without fitting a transformer.

    Initial raw reconstruction is separately recorded by the milestone runner.
    Hashes bind this reader to that trusted artifact; hashes are not signatures.
    Held-out records are inspected only for identity isolation and consistency.
    Their labels and features never enter allocation.
    """
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        hashes = json.loads((self.path / 'artifact_hashes.json').read_text(encoding='utf8'))
        if set(hashes) != {p.name for p in self.path.iterdir() if p.is_file()} - {'artifact_hashes.json'}:
            raise ValueError('Artifact inventory mismatch')
        for name, expected in hashes.items():
            if Path(name).name != name or sha256(self.path / name) != expected:
                raise ValueError('Artifact hash mismatch: ' + name)
        self.meta = json.loads((self.path / 'metadata.json').read_text(encoding='utf8'))
        self.schema = json.loads((self.path / 'schema.json').read_text(encoding='utf8'))
        if self.meta['format_version'] != '2' or self.schema != self.meta['schema']:
            raise ValueError('Artifact schema/version mismatch')
        resolved = json.loads((self.path / 'resolved_config.json').read_text(encoding='utf8'))
        if load_config(Path(resolved['_config_path'])) != resolved or digest(resolved) != self.meta['config_sha256']:
            raise ValueError('Preprocessing configuration changed')
        current = {str(p.relative_to(CLEAN)): sha256(p) for p in sorted((CLEAN / 'src').rglob('*.py'))}
        check_code_inventory(self.meta['implementation_hashes'], current)
        for source, evidence in self.meta['sources'].items():
            if sha256(Path(source)) != evidence['sha256']:
                raise ValueError('Raw source changed: ' + source)
        all_ids = set()
        self.heldout_ids = set()
        for split, summary in self.meta['splits'].items():
            records = pd.read_csv(self.path / f'records_{split}.csv', dtype=str, keep_default_na=False)
            y = np.load(self.path / f'y_{split}.npy', allow_pickle=False)
            x = np.load(self.path / f'X_{split}.npy', mmap_mode='r', allow_pickle=False)
            if records.columns.tolist() != ['source_id', 'label_original', 'y', 'group']:
                raise ValueError('Record schema mismatch')
            ids = records.source_id.tolist()
            if len(set(ids)) != len(ids) or all_ids.intersection(ids):
                raise ValueError('Duplicate or cross-partition sample IDs')
            all_ids.update(ids)
            if (y.ndim != 1 or y.dtype.kind not in 'iu' or len(y) != len(ids)
                    or x.shape != (len(ids), len(self.schema['feature_names']))
                    or len(ids) != summary['rows']
                    or digest(ids) != summary['ordered_source_id_sha256']
                    or not np.array_equal(y, records.y.to_numpy(dtype=np.int64))):
                raise ValueError('Sample/label/feature alignment mismatch')
            labels, counts = np.unique(y, return_counts=True)
            if {str(int(c)): int(n) for c, n in zip(labels, counts)} != summary['label_counts']:
                raise ValueError('Label distribution mismatch')
            mapping = self.schema['label_mapping']
            original = records.label_original
            encoded = original.ne('normal').astype(np.int64) if self.meta['dataset'] == 'nsl_kdd' else original.map(mapping)
            if not np.array_equal(y, encoded.to_numpy()):
                raise ValueError('Original label encoding mismatch')
            if split == 'train':
                self.labels, self.records = y, records
            else:
                self.heldout_ids.update(ids)
        if 'train' not in self.meta['splits'] or digest(self.records.source_id.tolist()) != self.meta['fit_source_id_sha256']:
            raise ValueError('Training membership mismatch')
        self.binding = {'artifact': str(self.path), 'dataset': self.meta['dataset'], 'split': 'train',
                        'artifact_manifest_sha256': sha256(self.path / 'artifact_hashes.json'),
                        'ordered_training_ids_sha256': digest(self.records.source_id.tolist()),
                        'schema_sha256': sha256(self.path / 'schema.json'),
                        'label_mapping': self.schema['label_mapping'],
                        'mapping_convention': 'assignment[row] is client; row indexes unchanged train X/y/records'}

    def validate_assignment(self, owner: np.ndarray, config: PartitionConfig) -> dict:
        stats = summarize(self.labels, owner, config)
        # Integer mapping has one slot per source row. Validate IDs independently.
        assigned = [self.records.source_id.iloc[i] for i in np.flatnonzero((owner >= 0) & (owner < config.clients))]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(self.records.source_id):
            raise ValueError('Assignment coverage/uniqueness failure')
        if set(assigned) & self.heldout_ids:
            raise ValueError('Held-out leakage')
        for client in stats['clients']:
            rows = np.flatnonzero(owner == client['client'])
            client['ordered_source_ids_sha256'] = digest(self.records.source_id.iloc[rows].tolist())
        stats['integrity'] = {'training_coverage_exact': True, 'assignment_count': len(assigned),
                              'duplicate_assignments': 0, 'heldout_id_overlap': 0,
                              'features_copied_or_transformed': False}
        return stats


def write_partition(source: TrainingArtifact, config: PartitionConfig, output: Path) -> dict:
    owner = allocate(source.labels, config)
    stats = source.validate_assignment(owner, config)
    output = Path(output).resolve()
    # New output only. Avoid all existing artifact/report/experiment paths.
    if not output.is_relative_to(PARTITION_ROOT.resolve()):
        raise ValueError('Partition outputs must be under data/client_partitions')
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'assignment.npy', owner, allow_pickle=False)
    manifest = {'format_version': '1', 'algorithm_version': 'uniform-or-class-dirichlet-multinomial-v1',
                'source': source.binding, 'summary': stats,
                'assignment_sha256': sha256(output / 'assignment.npy'),
                'environment': {'numpy': np.__version__, 'rng': 'PCG64'},
                'code_hashes': {str(p.relative_to(CLEAN)): sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    (output / 'manifest.json').write_text(json_text(manifest), encoding='utf8')
    return manifest


def verify_partition(source: TrainingArtifact, folder: Path) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
    config = PartitionConfig(**manifest['summary']['config'])
    owner = np.load(folder / 'assignment.npy', allow_pickle=False)
    if manifest['format_version'] != '1' or manifest['source'] != source.binding:
        raise ValueError('Partition source binding mismatch')
    if sha256(folder / 'assignment.npy') != manifest['assignment_sha256']:
        raise ValueError('Partition hash mismatch')
    for name, expected in manifest['code_hashes'].items():
        if not name.replace('\\', '/').startswith('src/orbitshield/federated/') or sha256(CLEAN / name) != expected:
            raise ValueError('Partition implementation changed')
    if not np.array_equal(owner, allocate(source.labels, config)):
        raise ValueError('Partition does not reproduce from seed/configuration')
    if source.validate_assignment(owner, config) != manifest['summary']:
        raise ValueError('Partition statistics/ID alignment mismatch')
    return manifest['summary']
