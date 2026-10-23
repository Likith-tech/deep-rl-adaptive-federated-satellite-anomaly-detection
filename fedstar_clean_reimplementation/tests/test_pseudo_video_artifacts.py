import json
import numpy as np
import pytest
from pathlib import Path

from orbitshield.data.common_preprocessing import prepare_dataset
from orbitshield.data.contracts import CLEAN, sha256
from orbitshield.representation.artifacts import (
    generate_pseudo_video_artifact, validate_pseudo_video_artifact)
from orbitshield.representation.projection import UnresolvedPaperSpecification


def test_versioned_batched_artifact_roundtrip_alignment_and_no_input_mutation(nsl_config, tmp_path):
    input_artifact = tmp_path / 'tabular'
    prepare_dataset(nsl_config, input_artifact)
    before = {p.name: sha256(p) for p in input_artifact.iterdir()}
    config = CLEAN / 'configs/projection_adaptation_example.yaml'
    output_root = tmp_path / 'representations'

    result = generate_pseudo_video_artifact(input_artifact, config, output_root=output_root, batch_size=7)
    assert result['passed'] and result['all_values_recomputed']
    folder = output_root / 'nsl_kdd' / Path(result['artifact']).name
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf8'))
    assert metadata['status'] == 'untrained_project_adaptation_initialization_only'
    assert metadata['paper_claim'] == 'not_exact_fedstar'
    width = np.load(input_artifact / 'X_train.npy', mmap_mode='r').shape[1]
    assert metadata['input_feature_dimension'] == width
    assert metadata['splits']['train']['output_shape'] == [len(np.load(input_artifact / 'X_train.npy')), 4, 64, 2, 2]

    for split in metadata['splits']:
        videos = np.load(folder / f'video_{split}.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(folder / f'y_{split}.npy', allow_pickle=False)
        ids = json.loads((folder / f'source_ids_{split}.json').read_text(encoding='utf8'))
        assert videos.dtype == np.float32 and np.isfinite(videos).all()
        assert np.array_equal(labels, np.load(input_artifact / f'y_{split}.npy', allow_pickle=False))
        source_records = __import__('pandas').read_csv(input_artifact / f'records_{split}.csv')
        assert ids == source_records.source_id.tolist()
        assert videos.shape[0] == len(ids) == len(labels)

    # A different batching schedule produces the same artifact identity and bytes.
    repeated = generate_pseudo_video_artifact(input_artifact, config, output_root=output_root, batch_size=11)
    assert repeated['artifact'] == result['artifact']
    assert {p.name: sha256(p) for p in input_artifact.iterdir()} == before
    assert validate_pseudo_video_artifact(folder, input_artifact=input_artifact, verify_values=True, batch_size=13)['verified_rows'] == sum(
        v['rows'] for v in metadata['splits'].values())

    # Corruption is detected, and the existing version is never silently repaired.
    video_path = folder / 'video_train.npy'
    videos = np.load(video_path, mmap_mode='r+', allow_pickle=False)
    videos[0, 0, 0, 0, 0] += 1
    videos.flush()
    with pytest.raises(ValueError, match='payload hash mismatch'):
        validate_pseudo_video_artifact(folder, verify_values=False)
    with pytest.raises(ValueError, match='payload hash mismatch'):
        generate_pseudo_video_artifact(input_artifact, config, output_root=output_root, batch_size=7)


def test_exact_configuration_fails_closed_before_tensor_output(artifact, tmp_path):
    with pytest.raises(UnresolvedPaperSpecification):
        generate_pseudo_video_artifact(artifact, CLEAN / 'configs/projection_exact.yaml',
                                      output_root=tmp_path / 'representations')
    assert not (tmp_path / 'representations').exists()
