"""Read-only bridge from validated tabular artifacts to sample-preserving batches."""
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
import hashlib
import json
import os
import shutil
import uuid
import numpy as np
import pandas as pd
import torch
import yaml
from orbitshield.data import common_preprocessing as cp
from .projection import (build_from_config, UnresolvedPaperSpecification)

REPRESENTATION_ROOT = cp.CLEAN / 'data/representations'
PSEUDO_VIDEO_FORMAT_VERSION = '1'


def scoped_code_hashes(recorded: dict, current: dict) -> dict:
    """Allow additive representation/federated code, never altered recorded code.

    Older v2 provenance hashed the whole src tree. An added downstream model
    cannot invalidate upstream data, but changed/removed data code still must.
    """
    if any(current.get(k) != v for k, v in recorded.items()):
        raise ValueError('Recorded implementation changed or missing')
    extra = set(current) - set(recorded)
    allowed = ('src/orbitshield/representation/', 'src/orbitshield/federated/')
    if any(not name.replace('\\', '/').startswith(allowed) for name in extra):
        raise ValueError('Unrecorded code outside additive representation namespace')
    return dict(recorded)


def validate_tabular_artifact(path: Path) -> dict:
    """Run UNMODIFIED v2 validator with strictly scoped legacy code inventory.

    No global monkeypatch and no artifact rewrite. Every source/config/array/
    label/fit check runs unchanged. Only permitted new downstream file names
    are outside the old tabular artifact's dependency inventory.
    """
    meta = json.loads((Path(path) / 'metadata.json').read_text(encoding='utf8'))
    current = cp.implementation_hashes()
    recorded = scoped_code_hashes(meta['implementation_hashes'], current)
    original = cp.validate_saved_artifact
    namespace = dict(original.__globals__, implementation_hashes=lambda: recorded)
    validator = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result = validator(path)
    result['additive_representation_code'] = {k: current[k] for k in current.keys() - recorded.keys()}
    result['compatibility_policy'] = 'all recorded code exact; only additional representation/federated modules permitted'
    return result


@dataclass(frozen=True)
class TabularBatch:
    features: torch.Tensor
    labels: torch.Tensor
    source_ids: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class RepresentedBatch:
    video: torch.Tensor
    labels: torch.Tensor
    source_ids: tuple[str, ...]
    split: str


class ArtifactBatches:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.validation = validate_tabular_artifact(self.path)
        self.feature_dimension = self.validation['feature_dimension']

    def batches(self, split: str, batch_size: int):
        if split not in self.validation['splits']:
            raise ValueError('Unknown partition')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        x = np.load(self.path / f'X_{split}.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(self.path / f'y_{split}.npy', mmap_mode='r', allow_pickle=False)
        ids = pd.read_csv(self.path / f'records_{split}.csv', dtype=str, keep_default_na=False).source_id
        for start in range(0, len(x), batch_size):
            stop = min(len(x), start + batch_size)
            yield TabularBatch(torch.from_numpy(np.array(x[start:stop], copy=True)),
                               torch.from_numpy(np.array(y[start:stop], copy=True)),
                               tuple(ids.iloc[start:stop]), split)


def represent_batch(model, batch: TabularBatch) -> RepresentedBatch:
    if batch.labels.ndim != 1 or len(batch.labels) != len(batch.features) or len(batch.source_ids) != len(batch.features):
        raise ValueError('Batch labels/source IDs are not aligned')
    # Forward sees ONLY X. Caller controls device and gradient scope explicitly.
    return RepresentedBatch(model(batch.features), batch.labels, batch.source_ids, batch.split)


def _file_hashes(folder: Path) -> dict[str, str]:
    return {p.name: cp.sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'}


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode('utf8'))
        h.update(str(tensor.dtype).encode('ascii'))
        h.update(json.dumps(list(tensor.shape)).encode('ascii'))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _label_counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def generate_pseudo_video_artifact(input_artifact: Path, projection_config: Path, *,
                                   output_root: Path = REPRESENTATION_ROOT,
                                   batch_size: int = 512) -> dict:
    """Generate immutable, batched pseudo-video tensors for every saved split.

    This applies the explicitly opted-in project adaptation at deterministic
    initialization. It does not fit on data or train the projection.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    source = ArtifactBatches(Path(input_artifact))
    schema_path = source.path / 'schema.json'
    schema = json.loads(schema_path.read_text(encoding='utf8'))
    feature_names = schema.get('feature_names')
    if not isinstance(feature_names, list) or len(feature_names) != source.feature_dimension:
        raise ValueError('Saved feature schema/order does not match validated input dimension')

    projection_config = Path(projection_config).resolve()
    config = yaml.safe_load(projection_config.read_text(encoding='utf8'))
    # Exact mode intentionally raises the paper-specification guard here.
    model = build_from_config(projection_config, source.feature_dimension, allow_adaptation=True).cpu().eval()
    settings = asdict(model.adaptation_settings)
    if model.reproduction_status != 'project_adaptation_not_exact_fedstar':
        raise ValueError('Unexpected projection provenance')
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    state_hash = _state_digest(state)
    input_manifest_hash = cp.sha256(source.path / 'artifact_hashes.json')
    source_meta = json.loads((source.path / 'metadata.json').read_text(encoding='utf8'))
    code_hashes = {
        'projection.py': cp.sha256(Path(__file__).with_name('projection.py')),
        'artifacts.py': cp.sha256(Path(__file__)),
    }
    identity = {
        'format_version': PSEUDO_VIDEO_FORMAT_VERSION,
        'dataset': source_meta['dataset'],
        'input_artifact': str(source.path),
        'input_manifest_sha256': input_manifest_hash,
        'feature_schema_sha256': cp.sha256(schema_path),
        'projection_config_path': str(projection_config),
        'projection_config_sha256': cp.sha256(projection_config),
        'projection_state_sha256': state_hash,
        'code_hashes': code_hashes,
        'torch_version': torch.__version__,
    }
    fingerprint = cp.digest(identity)[:16]
    root = Path(output_root).resolve()
    target = root / source_meta['dataset'] / f'v1-{fingerprint}'
    if target.exists():
        # Reuse only an artifact that fully validates against this same identity.
        return validate_pseudo_video_artifact(target, input_artifact=source.path,
                                              projection_config=projection_config,
                                              verify_values=True)

    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f'.{target.name}.{uuid.uuid4().hex}.tmp'
    stage.mkdir(exist_ok=False)
    try:
        torch.save(state, stage / 'projection_state.pt')
        split_info = {}
        height, width = settings['height'], settings['width']
        for split, input_summary in source.validation['splits'].items():
            rows = int(input_summary['rows'])
            tensor_path = stage / f'video_{split}.npy'
            video = np.lib.format.open_memmap(tensor_path, mode='w+', dtype=np.float32,
                                               shape=(rows, 4, 64, height, width))
            labels_out = np.lib.format.open_memmap(stage / f'y_{split}.npy', mode='w+', dtype=np.int64,
                                                    shape=(rows,))
            ids = []
            offset = 0
            with torch.inference_mode():
                for batch in source.batches(split, batch_size):
                    if batch.features.shape[1] != len(feature_names):
                        raise ValueError(f'{split}: feature width/order changed during batching')
                    represented = represent_batch(model, batch)
                    stop = offset + len(batch.features)
                    expected = (len(batch.features), 4, 64, height, width)
                    if tuple(represented.video.shape) != expected:
                        raise ValueError(f'{split}: output shape mismatch: {tuple(represented.video.shape)}')
                    if not torch.isfinite(represented.video).all():
                        raise ValueError(f'{split}: non-finite pseudo-video values')
                    if represented.split != split or represented.source_ids != batch.source_ids or not torch.equal(represented.labels, batch.labels):
                        raise ValueError(f'{split}: label/ID/split alignment failure')
                    video[offset:stop] = represented.video.cpu().numpy()
                    labels_out[offset:stop] = represented.labels.cpu().numpy()
                    ids.extend(represented.source_ids)
                    offset = stop
            if offset != rows or len(ids) != rows:
                raise ValueError(f'{split}: row coverage mismatch ({offset}/{rows})')
            video.flush()
            labels_out.flush()
            del video, labels_out
            ids_path = stage / f'source_ids_{split}.json'
            ids_path.write_text(cp.json_text(ids), encoding='utf8')
            split_info[split] = {
                'rows': rows,
                'input_shape': [rows, source.feature_dimension],
                'output_shape': [rows, 4, 64, height, width],
                'dtype': 'float32',
                'input_ordered_source_id_sha256': input_summary['ordered_source_id_sha256'],
                'output_ordered_source_id_sha256': cp.digest(ids),
                'labels_sha256': hashlib.sha256(np.load(stage / f'y_{split}.npy', mmap_mode='r').tobytes()).hexdigest(),
                'label_counts': _label_counts(np.load(stage / f'y_{split}.npy', mmap_mode='r')),
                'files': [tensor_path.name, f'y_{split}.npy', ids_path.name],
            }

        metadata = {
            **identity,
            'artifact_id': fingerprint,
            'dataset': source_meta['dataset'],
            'status': 'untrained_project_adaptation_initialization_only',
            'paper_claim': 'not_exact_fedstar',
            'input_artifact': str(source.path),
            'input_feature_dimension': source.feature_dimension,
            'input_feature_names_in_order': feature_names,
            'input_feature_schema': schema,
            'input_fit_scope': 'inherited preprocessor fit on train only; projection performs no fitting',
            'projection_config': config,
            'projection_config_path': str(projection_config),
            'projection_settings': settings,
            'projection_state_file': 'projection_state.pt',
            'batch_size': batch_size,
            'reproduction_tolerance': {'rtol': 1e-6, 'atol': 1e-6,
                                       'reason': 'float32 CPU linear kernels may vary slightly with inference batch size'},
            'splits': split_info,
            'environment': {'python': __import__('platform').python_version(), 'numpy': np.__version__,
                            'torch': torch.__version__, 'device': 'cpu'},
        }
        (stage / 'metadata.json').write_text(cp.json_text(metadata), encoding='utf8')
        (stage / 'artifact_hashes.json').write_text(cp.json_text(_file_hashes(stage)), encoding='utf8')
        # Atomic directory publication: target is new and never replaced.
        os.replace(stage, target)
    except Exception:
        # Only this invocation's uniquely named temporary folder is removed.
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_pseudo_video_artifact(target, input_artifact=source.path,
                                          projection_config=projection_config,
                                          verify_values=True)


def validate_pseudo_video_artifact(path: Path, *, input_artifact: Path | None = None,
                                   projection_config: Path | None = None,
                                   verify_values: bool = True, batch_size: int = 512) -> dict:
    """Verify payloads, source row/label bindings, shapes and optionally every value."""
    folder = Path(path).resolve()
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf8'))
    hashes = json.loads((folder / 'artifact_hashes.json').read_text(encoding='utf8'))
    actual_names = {p.name for p in folder.iterdir() if p.is_file()} - {'artifact_hashes.json'}
    if set(hashes) != actual_names:
        raise ValueError('Pseudo-video artifact file inventory mismatch')
    for name, expected_hash in hashes.items():
        if cp.sha256(folder / name) != expected_hash:
            raise ValueError(f'Pseudo-video payload hash mismatch: {name}')
    if metadata.get('format_version') != PSEUDO_VIDEO_FORMAT_VERSION or metadata.get('paper_claim') != 'not_exact_fedstar':
        raise ValueError('Unsupported pseudo-video format or misleading paper claim')
    if input_artifact is not None and Path(input_artifact).resolve() != Path(metadata['input_artifact']).resolve():
        raise ValueError('Input artifact path binding mismatch')
    source = ArtifactBatches(Path(metadata['input_artifact']))
    source_schema = json.loads((source.path / 'schema.json').read_text(encoding='utf8'))
    if cp.sha256(source.path / 'artifact_hashes.json') != metadata['input_manifest_sha256']:
        raise ValueError('Input artifact identity changed')
    if cp.sha256(source.path / 'schema.json') != metadata['feature_schema_sha256'] or source_schema != metadata['input_feature_schema']:
        raise ValueError('Input feature schema/order binding mismatch')
    if source.feature_dimension != metadata['input_feature_dimension'] or source_schema['feature_names'] != metadata['input_feature_names_in_order']:
        raise ValueError('Input feature dimension/order mismatch')
    config_path = Path(metadata['projection_config_path']).resolve()
    if cp.sha256(config_path) != metadata['projection_config_sha256']:
        raise ValueError('Recorded projection configuration changed')
    if projection_config is not None and cp.sha256(Path(projection_config).resolve()) != metadata['projection_config_sha256']:
        raise ValueError('Projection configuration identity mismatch')
    current_code_hashes = {
        'projection.py': cp.sha256(Path(__file__).with_name('projection.py')),
        'artifacts.py': cp.sha256(Path(__file__)),
    }
    if current_code_hashes != metadata['code_hashes']:
        raise ValueError('Pseudo-video implementation changed; create a new versioned artifact')
    identity_keys = ('format_version', 'dataset', 'input_artifact', 'input_manifest_sha256',
                     'feature_schema_sha256', 'projection_config_path', 'projection_config_sha256',
                     'projection_state_sha256', 'code_hashes', 'torch_version')
    if cp.digest({key: metadata[key] for key in identity_keys})[:16] != metadata['artifact_id']:
        raise ValueError('Pseudo-video artifact fingerprint mismatch')
    if folder.name != f"v1-{metadata['artifact_id']}":
        raise ValueError('Pseudo-video artifact directory/version mismatch')
    model = build_from_config(config_path, source.feature_dimension, allow_adaptation=True).cpu().eval()
    state = torch.load(folder / metadata['projection_state_file'], map_location='cpu', weights_only=True)
    if _state_digest(state) != metadata['projection_state_sha256']:
        raise ValueError('Saved projection state digest mismatch')
    model.load_state_dict(state, strict=True)
    # Respect the configuration path identity recorded in metadata. When the
    # caller omits it, resolve the original config relative to this repository.
    if model.adaptation_settings != model.adaptation_settings.__class__(**metadata['projection_settings']):
        raise ValueError('Projection settings differ from metadata')
    checked_rows = 0
    for split, summary in source.validation['splits'].items():
        saved_summary = metadata['splits'].get(split)
        if saved_summary is None or saved_summary['rows'] != summary['rows']:
            raise ValueError(f'{split}: split boundary mismatch')
        video = np.load(folder / f'video_{split}.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(folder / f'y_{split}.npy', mmap_mode='r', allow_pickle=False)
        ids = json.loads((folder / f'source_ids_{split}.json').read_text(encoding='utf8'))
        expected_shape = tuple(saved_summary['output_shape'])
        if video.shape != expected_shape or video.shape[0] != summary['rows'] or video.dtype != np.float32:
            raise ValueError(f'{split}: pseudo-video shape/dtype mismatch')
        if labels.shape != (summary['rows'],) or labels.dtype != np.int64 or not np.array_equal(labels, np.load(source.path / f'y_{split}.npy', mmap_mode='r', allow_pickle=False)):
            raise ValueError(f'{split}: label row alignment mismatch')
        if len(ids) != summary['rows'] or cp.digest(ids) != summary['ordered_source_id_sha256']:
            raise ValueError(f'{split}: source ID ordering mismatch')
        if not np.isfinite(video).all():
            raise ValueError(f'{split}: non-finite pseudo-video value')
        if hashlib.sha256(labels.tobytes()).hexdigest() != saved_summary['labels_sha256']:
            raise ValueError(f'{split}: label payload digest mismatch')
        if verify_values:
            offset = 0
            with torch.inference_mode():
                for batch in source.batches(split, batch_size):
                    output = model(batch.features)
                    stop = offset + len(batch.features)
                    if batch.source_ids != tuple(ids[offset:stop]) or not torch.equal(batch.labels, torch.from_numpy(np.array(labels[offset:stop], copy=True))):
                        raise ValueError(f'{split}: regenerated batch metadata alignment mismatch')
                    expected = output.cpu().numpy()
                    # BLAS may choose a different accumulation kernel for the
                    # caller's verification batch size. Confirm numerical
                    # equivalence while allowing only float32 round-off.
                    if not np.allclose(video[offset:stop], expected, rtol=1e-6, atol=1e-6):
                        delta = np.abs(video[offset:stop] - expected)
                        raise ValueError(f'{split}: saved tensor does not reproduce from input/model state '
                                         f'(max_abs_diff={float(delta.max())})')
                    offset = stop
            checked_rows += offset
    return {'passed': True, 'artifact': str(folder), 'artifact_id': metadata['artifact_id'],
            'dataset': metadata['dataset'], 'splits': metadata['splits'],
            'verified_rows': checked_rows if verify_values else 0,
            'all_values_recomputed': bool(verify_values), 'artifact_sha256': cp.sha256(folder / 'artifact_hashes.json')}
