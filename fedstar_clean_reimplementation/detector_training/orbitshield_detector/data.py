"""Verified immutable inputs. No preprocessing fit, transformation or resplitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from orbitshield.data.contracts import CLEAN, digest, sha256
from orbitshield.federated.artifacts import TrainingArtifact, verify_partition
from orbitshield.representation.artifacts import _state_digest
from .config import TASKS


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class RowDataset(Dataset):
    """Copy each selected row out of a read-only mmap; never expose writable source memory."""
    def __init__(self, features, labels, source_ids, split, indices=None):
        self.features, self.labels, self.source_ids, self.split = features, labels, tuple(source_ids), split
        if len(features) != len(labels) or len(labels) != len(source_ids):
            raise ValueError('Row alignment mismatch')
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)
        if (self.indices.ndim != 1 or self.indices.dtype.kind not in 'iu'
                or len(self.indices) == 0 or len(np.unique(self.indices)) != len(self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= len(labels))):
            raise ValueError('Invalid, duplicated or empty row selection')

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        x = np.array(self.features[row], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite input features')
        return torch.from_numpy(x), torch.tensor(int(self.labels[row]), dtype=torch.long), self.source_ids[row]


class ArtifactData:
    """Preflight may inspect every split for integrity, but never fits or evaluates a model.

    Training datasets expose only train/validation. Final test has a separate method.
    """
    def __init__(self, source_path, config, representation_path=None, partition_path=None):
        self.config = config
        self.source = TrainingArtifact(Path(source_path))
        self.path = self.source.path
        expected = TASKS[config.dataset]
        if (self.source.meta['dataset'] != config.dataset
                or self.source.schema['label_mapping'] != expected[1]
                or len(self.source.schema['feature_names']) != expected[0]
                or set(self.source.meta['splits']) != {'train', 'validation', expected[2]}):
            raise ValueError('Artifact does not match the approved task/schema/splits')
        for split in self.source.meta['splits']:
            x = np.load(self.path / f'X_{split}.npy', mmap_mode='r', allow_pickle=False)
            if x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError('Invalid transformed input dtype/values')
        self.representation_path = None
        self.projection_state = None
        self.representation_binding = None
        if config.representation in ('joint2', 'frozen2'):
            if representation_path is None:
                raise ValueError('2x2 modes require a bound representation artifact')
            self._verify_representation(Path(representation_path).resolve())
        elif representation_path is not None:
            raise ValueError('joint16 does not consume cached 2x2 representations')
        self.assignment = None
        self.partition_binding = None
        if partition_path is not None:
            folder = Path(partition_path).resolve()
            summary = verify_partition(self.source, folder)
            # Reject BEFORE any model/optimizer construction in the CLI.
            if summary['status'] != 'accepted' or not summary['minimum_size_check']['met']:
                raise ValueError('Rejected partition: minimum_size_not_met')
            if summary['config']['clients'] != 20:
                raise ValueError('Expected the approved 20-client protocol')
            self.assignment = np.load(folder / 'assignment.npy', allow_pickle=False)
            self.partition_binding = {'path': str(folder), 'manifest_sha256': sha256(folder / 'manifest.json'),
                                      'assignment_sha256': sha256(folder / 'assignment.npy')}

    def _verify_representation(self, folder):
        hashes = read_json(folder / 'artifact_hashes.json')
        expected_files = {'projection_state.pt', 'metadata.json'}
        for split in self.source.meta['splits']:
            expected_files.update({f'video_{split}.npy', f'y_{split}.npy', f'source_ids_{split}.json'})
        if set(hashes) != expected_files or set(hashes) != {p.name for p in folder.iterdir() if p.is_file()} - {'artifact_hashes.json'}:
            raise ValueError('Representation inventory mismatch')
        for name, value in hashes.items():
            if sha256(folder / name) != value:
                raise ValueError('Representation payload hash mismatch: ' + name)
        m = read_json(folder / 'metadata.json')
        if (m['format_version'] != '1' or m['dataset'] != self.config.dataset
                or m['status'] != 'untrained_project_adaptation_initialization_only'
                or m['paper_claim'] != 'not_exact_fedstar'
                or Path(m['input_artifact']).resolve() != self.path
                or m['input_manifest_sha256'] != sha256(self.path / 'artifact_hashes.json')
                or m['feature_schema_sha256'] != sha256(self.path / 'schema.json')
                or m['input_feature_schema'] != self.source.schema
                or m['input_feature_names_in_order'] != self.source.schema['feature_names']
                or m['input_feature_dimension'] != self.config.input_dim
                or set(m['splits']) != set(self.source.meta['splits'])):
            raise ValueError('Representation source/schema/status binding mismatch')
        if sha256(m['projection_config_path']) != m['projection_config_sha256']:
            raise ValueError('Representation configuration changed')
        expected_code = {name: sha256(CLEAN / 'src/orbitshield/representation' / name)
                         for name in ('projection.py', 'artifacts.py')}
        if expected_code != m['code_hashes']:
            raise ValueError('Representation implementation changed')
        keys = ('format_version', 'dataset', 'input_artifact', 'input_manifest_sha256',
                'feature_schema_sha256', 'projection_config_path', 'projection_config_sha256',
                'projection_state_sha256', 'code_hashes', 'torch_version')
        if digest({k: m[k] for k in keys})[:16] != m['artifact_id'] or folder.name != 'v1-' + m['artifact_id']:
            raise ValueError('Representation identity mismatch')
        from orbitshield.representation.projection import build_from_config
        original = build_from_config(Path(m['projection_config_path']), self.config.input_dim, allow_adaptation=True)
        settings = original.adaptation_settings
        if (settings.height, settings.width, settings.embedding_dim, settings.hidden_dim,
                settings.layernorm_eps, settings.layernorm_affine) != (2, 2, 16, 32, 1e-5, True):
            raise ValueError('Representation settings incompatible with detector recipe')
        from dataclasses import asdict
        if asdict(settings) != m['projection_settings']:
            raise ValueError('Representation metadata settings mismatch')
        if m['projection_state_file'] != 'projection_state.pt':
            raise ValueError('Unexpected projection state path')
        state = torch.load(folder / 'projection_state.pt', map_location='cpu', weights_only=True)
        if _state_digest(state) != m['projection_state_sha256']:
            raise ValueError('Representation state digest mismatch')
        original.load_state_dict(state, strict=True)
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Non-finite projection state')
        for split, summary in self.source.meta['splits'].items():
            ids = read_json(folder / f'source_ids_{split}.json')
            video = np.load(folder / f'video_{split}.npy', mmap_mode='r', allow_pickle=False)
            labels = np.load(folder / f'y_{split}.npy', allow_pickle=False)
            if (video.shape != (summary['rows'], 4, 64, 2, 2) or video.dtype != np.float32
                    or not np.isfinite(video).all() or labels.dtype != np.int64
                    or not np.array_equal(labels, np.load(self.path / f'y_{split}.npy', allow_pickle=False))
                    or digest(ids) != summary['ordered_source_id_sha256']):
                raise ValueError('Representation split/row/label mismatch')
        self.representation_path, self.projection_state = folder, state
        self.representation_binding = {'path': str(folder), 'manifest_sha256': sha256(folder / 'artifact_hashes.json'),
                                       'projection_seed': settings.seed, 'state_sha256': m['projection_state_sha256']}

    def binding(self):
        return {'source': self.source.binding, 'representation': self.representation_binding,
                'partition': self.partition_binding}

    def _rows(self, split, indices=None):
        # Recheck relevant bytes when opening a split after preflight.
        if sha256(self.path / 'artifact_hashes.json') != self.source.binding['artifact_manifest_sha256']:
            raise ValueError('Input manifest changed after preflight')
        hashes = read_json(self.path / 'artifact_hashes.json')
        for name in (f'X_{split}.npy', f'y_{split}.npy', f'records_{split}.csv'):
            if sha256(self.path / name) != hashes[name]:
                raise ValueError('Input changed after preflight')
        import pandas as pd
        ids = pd.read_csv(self.path / f'records_{split}.csv', dtype=str, keep_default_na=False).source_id.tolist()
        y = np.load(self.path / f'y_{split}.npy', mmap_mode='r', allow_pickle=False)
        feature_path = self.path / f'X_{split}.npy'
        if self.config.representation == 'frozen2':
            feature_path = self.representation_path / f'video_{split}.npy'
            if sha256(self.representation_path / 'artifact_hashes.json') != self.representation_binding['manifest_sha256']:
                raise ValueError('Representation manifest changed after preflight')
            rh = read_json(self.representation_path / 'artifact_hashes.json')
            if sha256(feature_path) != rh[feature_path.name]:
                raise ValueError('Cached video changed after preflight')
        return RowDataset(np.load(feature_path, mmap_mode='r', allow_pickle=False), y, ids, split, indices)

    def dataset(self, split, client=None):
        if split not in ('train', 'validation'):
            raise ValueError('Training interface cannot open the final test')
        indices = None
        if client is not None:
            if split != 'train' or self.assignment is None or type(client) is not int or not 0 <= client < 20:
                raise ValueError('Client selection requires a verified training partition')
            indices = np.flatnonzero(self.assignment == client)
        return self._rows(split, indices)

    def final_test(self):
        return self._rows(TASKS[self.config.dataset][2])
