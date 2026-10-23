"""Synthetic optimization only; never trains on production datasets."""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detector_training'))
from orbitshield_detector.config import DetectorConfig, TrainConfig
from orbitshield_detector.data import RowDataset
from orbitshield_detector.model import Detector
from orbitshield_detector.metrics import classification_metrics, better
from orbitshield_detector.checkpoint import cpu_state, read_checkpoint, save_checkpoint
from orbitshield_detector.reproducibility import configure
from orbitshield_detector.trainer import loader, optimizer_for, score, train_epoch, train_centralized
from orbitshield_detector.fedavg import aggregate, train_fedavg
from orbitshield_detector.evaluate import evaluate_final


def rows(split, n=5):
    rng = np.random.default_rng(17)
    return RowDataset(rng.normal(size=(n, 4, 64, 2, 2)).astype(np.float32),
        np.arange(n, dtype=np.int64) % 2, [f'{split}:{i}' for i in range(n)], split)


def new_model():
    configure(2024)
    return Detector(DetectorConfig('opssat_ad', 'frozen2'))


def test_metric_oracles_missing_classes_and_ties():
    result = classification_metrics(np.array([0, 0, 1, 1]),
        np.array([[.9, .1], [.4, .6], [.2, .8], [.7, .3]]), 2)
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert result['macro_f1'] == result['accuracy'] == .5
    assert result['roc_auc'] == .75
    assert result['average_precision'] == pytest.approx(5/6)
    absent = classification_metrics(np.zeros(3, dtype=np.int64), np.tile([.7, .3], (3, 1)), 2)
    assert absent['average_precision'] is None and absent['roc_auc'] is None
    assert absent['per_class']['1']['recall'] is None
    nine = classification_metrics(np.array([0, 1]), np.eye(9)[[0, 1]], 9)
    assert nine['macro_f1'] == pytest.approx(2/9)
    assert 'roc_auc' not in nine and len(nine['confusion_matrix']) == 9
    a = {'macro_f1': .5, 'loss': .8}
    assert not better(a, a, 'macro_f1')
    assert better({**a, 'loss': .7}, a, 'macro_f1')
    with pytest.raises(ValueError): better({'average_precision': None, 'loss': 1}, None, 'average_precision')


def test_validation_is_readonly_and_training_rejects_heldout():
    model = new_model()
    settings = TrainConfig(batch_size=2)
    optimizer = optimizer_for(model, settings)
    before = cpu_state(model)
    metrics, predictions = score(model, loader(rows('validation'), 2, 0), 'cpu')
    assert len(predictions['source_ids']) == 5 and np.isfinite(metrics['loss'])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not optimizer.state
    with pytest.raises(ValueError, match='train split'):
        train_epoch(model, loader(rows('validation'), 2, 0), optimizer, 'cpu')
    with pytest.raises(ValueError, match='Wrong split'):
        score(model, loader(rows('official_test'), 2, 0), 'cpu')
    original = rows('train')
    assert sum(len(y) for _, y, _ in loader(original, 2, 0, True)) == 5  # no dropped singleton
    train_epoch(model, loader(original, 2, 0, True), optimizer, 'cpu')
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_boundary_resume_is_identical_and_incompatible_contract_rejected(tmp_path):
    settings = TrainConfig(batch_size=2, epochs=2, patience=10)
    contract = {'mode': 'centralized', 'test_fixture': True}
    full = new_model()
    final = train_centralized(full, rows('train'), rows('validation'), settings, contract, tmp_path / 'full')
    first = new_model()
    train_centralized(first, rows('train'), rows('validation'), settings, contract, tmp_path / 'part', stop_after=1)
    resumed = new_model()
    result = train_centralized(resumed, rows('train'), rows('validation'), settings, contract,
        tmp_path / 'resumed', resume=tmp_path / 'part/epoch-0001.pt')
    assert result['history'] == final['history']
    assert result['best_step'] == final['best_step']
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    with pytest.raises(ValueError, match='contract mismatch'):
        read_checkpoint(tmp_path / 'full/epoch-0002.pt', {'different': True})
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / 'full/epoch-0002.pt', full, None, contract, result)


def test_final_test_requires_completed_selection_and_never_optimizes(tmp_path):
    model = new_model()
    settings = TrainConfig(batch_size=2, epochs=2)
    contract = {'mode': 'centralized', 'synthetic': True}
    train_centralized(model, rows('train'), rows('validation'), settings, contract, tmp_path / 'run')
    class TestSource:
        calls = 0
        def final_test(self):
            self.calls += 1
            return rows('official_test')
    data = TestSource()
    with pytest.raises(ValueError, match='not yet complete'):
        evaluate_final(model, data, settings, contract, tmp_path / 'run/epoch-0001.pt', tmp_path / 'too_early')
    assert data.calls == 0 and not (tmp_path / 'too_early').exists()
    result = evaluate_final(model, data, settings, contract, tmp_path / 'run/epoch-0002.pt', tmp_path / 'evaluation')
    assert data.calls == 1 and np.isfinite(result['loss'])
    assert (tmp_path / 'evaluation/evaluation.json').is_file()
    with pytest.raises(FileExistsError):
        evaluate_final(model, data, settings, contract, tmp_path / 'run/epoch-0002.pt', tmp_path / 'evaluation')


def test_weighted_aggregation_oracle_and_invalid_clients():
    result = aggregate([{'w': torch.tensor([1., 3.])}, {'w': torch.tensor([5., 7.])}], [1, 3])
    torch.testing.assert_close(result['w'], torch.tensor([4., 6.]))
    with pytest.raises(ValueError): aggregate([{'w': torch.ones(1)}], [0])
    with pytest.raises(ValueError): aggregate([{'a': torch.ones(1)}, {'b': torch.ones(1)}], [1, 1])
    with pytest.raises(ValueError): aggregate([{'w': torch.tensor([float('nan')])}], [1])


class TinyDetector(nn.Module):
    """Small optimizer/round oracle; architecture is tested separately."""
    def __init__(self):
        super().__init__()
        self.config = DetectorConfig('opssat_ad', 'frozen2')
        self.linear = nn.Linear(4 * 64 * 2 * 2, 2)
    def forward(self, x): return self.linear(x.flatten(1))


def test_fedavg_round_resume_and_identical_client_starts(tmp_path, monkeypatch):
    from orbitshield_detector import fedavg
    configure(2024)
    settings = TrainConfig(batch_size=2, rounds=2, local_epochs=1)
    contract = {'mode': 'fedavg', 'synthetic': True}
    clients = [rows('train', 2) for _ in range(20)]
    captured = []
    original = fedavg.train_epoch
    def capture(model, *args):
        captured.append(cpu_state(model))
        return original(model, *args)
    monkeypatch.setattr(fedavg, 'train_epoch', capture)
    full = TinyDetector()
    full_progress = train_fedavg(full, clients, rows('validation'), settings, contract, tmp_path / 'all')
    assert all(torch.equal(captured[0]['linear.weight'], c['linear.weight']) for c in captured[:20])
    configure(2024)
    first = TinyDetector()
    train_fedavg(first, clients, rows('validation'), settings, contract, tmp_path / 'one', stop_after=1)
    configure(999)
    second = TinyDetector()
    result = train_fedavg(second, clients, rows('validation'), settings, contract, tmp_path / 'two',
                         resume=tmp_path / 'one/round-0001.pt')
    assert result['history'] == full_progress['history']
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in full.state_dict().items())
    with pytest.raises(ValueError, match='20 accepted'):
        train_fedavg(second, clients[:19], rows('validation'), settings, contract, tmp_path / 'invalid')


def test_row_dataset_copy_and_invalid_index_contract():
    original = rows('train')
    x, _, _ = original[0]
    x.fill_(100)
    assert not np.all(original.features[0] == 100)
    for invalid in ([], [0, 0], [-1], [5], [0.5]):
        with pytest.raises(ValueError):
            RowDataset(original.features, original.labels, original.source_ids, 'train', invalid)
