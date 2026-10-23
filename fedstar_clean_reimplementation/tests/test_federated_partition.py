"""Coverage, randomization, leakage and immutable-artifact contracts."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from orbitshield.data.contracts import CLEAN, sha256
from orbitshield.data.common_preprocessing import prepare_dataset
from orbitshield.federated.partition import PartitionConfig, allocate, summarize
from orbitshield.federated.artifacts import TrainingArtifact, check_code_inventory, write_partition, verify_partition


@pytest.mark.parametrize('mode', ['iid', 'dirichlet'])
def test_exact_coverage_determinism_and_seed_change(mode):
    y = np.arange(1003, dtype=np.int64) % 9
    config = PartitionConfig(mode, 2024)
    state = np.random.get_state()
    owner = allocate(y, config)
    assert len(owner) == len(y) and set(owner) == set(range(20))
    assert np.array_equal(owner, allocate(y, config))
    assert not np.array_equal(owner, allocate(y, replace(config, seed=2025)))
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))
    rows = np.concatenate([np.flatnonzero(owner == i) for i in range(20)])
    assert np.array_equal(np.sort(rows), np.arange(len(y)))
    assert len(np.unique(rows)) == len(y)
    if mode == 'iid':
        counts = np.bincount(owner, minlength=20)
        assert counts.max()-counts.min() == 1
        assert sorted(counts) == [50]*17 + [51]*3


@pytest.mark.parametrize('n', [0, 1, 7, 19, 20, 21])
@pytest.mark.parametrize('mode', ['iid', 'dirichlet'])
def test_small_rare_single_class_and_empty_clients_are_explicit(n, mode):
    y = np.zeros(n, dtype=np.int64)
    if n: y[-1] = 99  # arbitrary rare code; no requirement for every class per client
    config = PartitionConfig(mode, 2024)
    owner = allocate(y, config)
    result = summarize(y, owner, config)
    assert len(result['clients']) == 20
    assert sum(c['samples'] for c in result['clients']) == n
    assert result['minimum_size_check']['feasible_by_total'] == (n >= 20)
    assert result['minimum_size_check']['attempts'] == 1
    assert result['minimum_size_check']['met'] == (result['sample_count_statistics']['minimum'] >= 1)
    if n < 20: assert result['status'] == 'minimum_size_not_met'
    relaxed = summarize(y, owner, replace(config, minimum_client_size=0))
    assert relaxed['status'] == 'accepted'
    assert np.array_equal(owner, allocate(y, replace(config, minimum_client_size=0)))


def test_minimum_check_never_rebalances_or_retries():
    y = np.arange(300, dtype=np.int64) % 3
    config = PartitionConfig('dirichlet', 2024, minimum_client_size=100)
    owner = allocate(y, config)
    result = summarize(y, owner, config)
    assert not result['minimum_size_check']['met']
    assert not result['minimum_size_check']['feasible_by_total']
    assert np.array_equal(owner, allocate(y, replace(config, minimum_client_size=0)))
    with pytest.raises(ValueError, match='single-attempt'):
        replace(config, retry_policy='until_success')


def test_iid_is_uniform_and_dirichlet_label_skew_is_measured():
    y = np.repeat(np.arange(9), 2000)
    iid = summarize(y, allocate(y, PartitionConfig('iid', 2024)), PartitionConfig('iid', 2024))
    skew = summarize(y, allocate(y, PartitionConfig('dirichlet', 2024)), PartitionConfig('dirichlet', 2024))
    assert iid['mean_nonempty_client_total_variation'] < .06
    assert skew['mean_nonempty_client_total_variation'] > .2
    assert all(c['samples'] == 900 for c in iid['clients'])
    assert iid['global_class_counts'] == skew['global_class_counts']
    assert sum(c['class_counts']['8'] for c in skew['clients']) == 2000


@pytest.mark.parametrize('bad', [np.ones((2,2),dtype=int), np.array([0.,1.]), np.array(['0'])])
def test_invalid_labels_rejected(bad):
    with pytest.raises(ValueError, match='integer'):
        allocate(bad, PartitionConfig('iid', 2024))


@pytest.mark.parametrize('change', [{'eta': 0}, {'eta': float('nan')}, {'seed': -1},
                                   {'clients': 0}, {'minimum_client_size': -1}, {'mode': 'unknown'}])
def test_bad_configuration(change):
    with pytest.raises(ValueError):
        replace(PartitionConfig('iid', 2024), **change)


@pytest.mark.parametrize('fixture_name', ['nsl_config', 'stin_config', 'ops_config'])
@pytest.mark.parametrize('mode', ['iid', 'dirichlet'])
def test_all_adapters_training_only_alignment_and_no_fitting(fixture_name, mode, request, tmp_path, monkeypatch):
    from orbitshield.data import common_preprocessing as cp
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    config = request.getfixturevalue(fixture_name)
    folder = tmp_path / 'partition_source'
    prepare_dataset(config, folder)
    before = {p.name: sha256(p) for p in folder.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail('Partitioning must never fit or load a preprocessing object')
    monkeypatch.setattr(cp, 'fit_transform_splits', forbidden)
    monkeypatch.setattr(StandardScaler, 'fit', forbidden)
    monkeypatch.setattr(OneHotEncoder, 'fit', forbidden)
    monkeypatch.setattr(cp.joblib, 'load', forbidden)
    source = TrainingArtifact(folder)
    settings = PartitionConfig(mode, 2024)
    owner = allocate(source.labels, settings)
    stats = source.validate_assignment(owner, settings)
    assert stats['integrity']['heldout_id_overlap'] == 0
    assert stats['training_samples'] == len(source.labels)
    for client in range(20):
        rows = np.flatnonzero(owner == client)
        local = source.records.iloc[rows]
        assert not set(local.source_id) & source.heldout_ids
        assert np.array_equal(local.y.astype(int), source.labels[rows])
        assert stats['clients'][client]['samples'] == len(rows)
        for label, count in stats['clients'][client]['class_counts'].items():
            assert count == np.count_nonzero(source.labels[rows] == int(label))
    if fixture_name == 'stin_config':
        assert len(source.schema['label_mapping']) == 9
        assert 'normal' not in source.schema['label_mapping']
    assert before == {p.name: sha256(p) for p in folder.iterdir()}
    # Writer is restricted to the new output namespace; redirect only that root.
    import orbitshield.federated.artifacts as module
    monkeypatch.setattr(module, 'PARTITION_ROOT', tmp_path / 'data/client_partitions')
    destination = tmp_path / 'data/client_partitions/demo'
    manifest = write_partition(source, settings, destination)
    assert manifest['summary'] == stats
    assert verify_partition(source, destination) == stats
    with pytest.raises(FileExistsError):
        write_partition(source, settings, destination)
    corrupt = owner.copy()
    corrupt[0] = (int(corrupt[0]) + 1) % 20
    np.save(destination / 'assignment.npy', corrupt)
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_partition(source, destination)
    # Reseal only an isolated partition fixture to test deterministic semantic checks.
    manifest['assignment_sha256'] = sha256(destination / 'assignment.npy')
    (destination / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError, match='reproduce'):
        verify_partition(source, destination)


@pytest.mark.parametrize('fault', ['missing', 'out_of_range', 'float'])
def test_invalid_assignment_cannot_claim_coverage(fault):
    y = np.arange(41, dtype=np.int64) % 2
    config = PartitionConfig('iid', 2024)
    owner = allocate(y, config)
    if fault == 'missing': owner = owner[:-1]
    if fault == 'out_of_range': owner[0] = 20
    if fault == 'float': owner = owner.astype(float)
    with pytest.raises(ValueError, match='assignment'):
        summarize(y, owner, config)


def test_code_scope_accepts_only_additive_downstream_files():
    recorded = {'src/orbitshield/data/core.py': 'old'}
    check_code_inventory(recorded, {**recorded, 'src/orbitshield/federated/partition.py': 'new'})
    with pytest.raises(ValueError, match='changed'):
        check_code_inventory(recorded, {next(iter(recorded)): 'edited'})
    with pytest.raises(ValueError, match='outside'):
        check_code_inventory(recorded, {**recorded, 'src/orbitshield/data/new.py': 'new'})


def test_existing_protected_files_and_generated_reports_unchanged():
    baseline = json.loads((CLEAN / 'logs/partition-baseline.json').read_text(encoding='utf8'))
    # These paths are intentionally extended by the pseudo-video stage: the
    # README documents the new CLI and the representation bridge now accepts
    # the already-added federated namespace. All other historical baseline
    # files remain byte-protected.
    authorized_task_updates = {
        (CLEAN / 'README.md').resolve(),
        (CLEAN / 'src/orbitshield/representation/artifacts.py').resolve(),
    }
    assert all(Path(p).is_file() and sha256(Path(p)) == value.lower()
               for p, value in baseline['files'].items()
               if Path(p).resolve() not in authorized_task_updates)
    evidence = json.loads((CLEAN / 'reports/evidence_manifest.json').read_text(encoding='utf8'))
    assert len(evidence['reports']) == 16
    assert all(sha256(CLEAN / 'reports' / p) == h for p, h in evidence['reports'].items())
