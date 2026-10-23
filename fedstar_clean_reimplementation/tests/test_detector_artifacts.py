"""Read-only real-artifact integration; no model fitting or production writes."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'detector_training'))
from orbitshield_detector.config import DetectorConfig
from orbitshield_detector.data import ArtifactData
from orbitshield_detector.cli import safe_output


@pytest.mark.parametrize('dataset', ['nsl_kdd', 'opssat_ad', 'stin'])
def test_real_artifact_contract_all_three_modes_and_no_heldout_training(dataset, monkeypatch):
    from orbitshield.data import common_preprocessing as cp
    def forbidden(*args, **kwargs): raise AssertionError('No fit/transform is permitted in detector loading')
    monkeypatch.setattr(cp, 'fit_transform_splits', forbidden)
    import joblib
    monkeypatch.setattr(joblib, 'load', forbidden)
    source = json.loads((ROOT / 'reports/review_manifest.json').read_text())['artifacts'][dataset]
    representation = next((ROOT / 'data/representations' / dataset).glob('v1-*'))
    partition = next((ROOT / 'data/client_partitions' / dataset / 'iid').glob('seed-2024-*'))
    original_load = np.load
    for mode in ('joint16', 'joint2', 'frozen2'):
        data = ArtifactData(source, DetectorConfig(dataset, mode),
                            representation if mode != 'joint16' else None, partition)
        assert data.source.schema['label_mapping'] == DetectorConfig(dataset, mode).resolved()['label_mapping']
        # Preflight inspected test for integrity. Subsequent training access may not open it.
        def guarded(path, *args, **kwargs):
            assert 'test' not in Path(path).name
            return original_load(path, *args, **kwargs)
        with monkeypatch.context() as context:
            context.setattr(np, 'load', guarded)
            train = data.dataset('train', 0)
            x, label, identity = train[0]
            index = np.flatnonzero(data.assignment == 0)[0]
            assert identity == data.source.records.source_id.iloc[index]
            assert label.item() == data.source.labels[index]
            assert x.shape == ((4, 64, 2, 2) if mode == 'frozen2' else (data.config.input_dim,))
            assert label.dtype == torch.long
            with pytest.raises(ValueError, match='final test'): data.dataset('test')
            with pytest.raises(ValueError): data.dataset('validation', 0)


def test_actual_opssat_seed_2025_rejected():
    source = json.loads((ROOT / 'reports/review_manifest.json').read_text())['artifacts']['opssat_ad']
    partition = next((ROOT / 'data/client_partitions/opssat_ad/dirichlet').glob('seed-2025-*'))
    manifest = json.loads((partition / 'manifest.json').read_text())
    assert [c['client'] for c in manifest['summary']['clients'] if c['samples'] == 0] == [10, 11]
    with pytest.raises(ValueError, match='minimum_size_not_met'):
        ArtifactData(source, DetectorConfig('opssat_ad'), partition_path=partition)


def test_output_safety_never_selects_existing_artifacts():
    for path in (ROOT, ROOT / 'data', ROOT / 'reports', ROOT / 'runs/detector', ROOT / 'runs/detector/../../../data'):
        with pytest.raises(ValueError): safe_output(path)


@pytest.fixture
def synthetic_ops_artifacts(ops_config, tmp_path):
    from orbitshield.data.common_preprocessing import prepare_dataset
    from orbitshield.representation.artifacts import generate_pseudo_video_artifact
    from orbitshield_detector.reproducibility import configure
    configure(2024)
    source = tmp_path / 'source'
    prepare_dataset(ops_config, source)
    result = generate_pseudo_video_artifact(source, ROOT / 'configs/projection_adaptation_example.yaml',
        output_root=tmp_path / 'representation', batch_size=4)
    return source, Path(result['artifact'])


def test_resealed_representation_id_corruption_is_rejected(synthetic_ops_artifacts):
    from orbitshield.data.contracts import sha256
    source, representation = synthetic_ops_artifacts
    ids_path = representation / 'source_ids_train.json'
    ids = json.loads(ids_path.read_text())
    ids[0], ids[1] = ids[1], ids[0]
    ids_path.write_text(json.dumps(ids), encoding='utf8')
    manifest = representation / 'artifact_hashes.json'
    hashes = json.loads(manifest.read_text())
    hashes[ids_path.name] = sha256(ids_path)
    manifest.write_text(json.dumps(hashes), encoding='utf8')
    with pytest.raises(ValueError, match='split/row/label'):
        ArtifactData(source, DetectorConfig('opssat_ad', 'frozen2'), representation)


def test_frozen_training_never_changes_projection_artifacts(synthetic_ops_artifacts):
    from orbitshield.data.contracts import sha256
    from orbitshield_detector.model import Detector
    from orbitshield_detector.config import TrainConfig
    from orbitshield_detector.trainer import train_epoch, loader, optimizer_for
    source, representation = synthetic_ops_artifacts
    before = {str(p): sha256(p) for folder in (source, representation) for p in folder.iterdir() if p.is_file()}
    config = DetectorConfig('opssat_ad', 'frozen2')
    data = ArtifactData(source, config, representation)
    model = Detector(config)
    train_epoch(model, loader(data.dataset('train'), 8, 2024, True), optimizer_for(model, TrainConfig()), 'cpu')
    assert all(sha256(p) == expected for p, expected in before.items())
