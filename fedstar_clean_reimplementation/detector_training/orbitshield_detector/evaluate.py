"""Explicit final-test evaluation of the validation-selected model."""
import json
from pathlib import Path
import torch
from .checkpoint import read_checkpoint
from .trainer import loader, score


def evaluate_final(model, data, settings, contract, checkpoint_path, output):
    checkpoint = read_checkpoint(checkpoint_path, contract)
    progress = checkpoint['progress']
    if progress['best_model'] is None or progress['best_metrics'] is None:
        raise ValueError('No validation-selected checkpoint')
    limit = settings.rounds if contract['mode'] == 'fedavg' else settings.epochs
    early_stopped = contract['mode'] == 'centralized' and progress['bad_steps'] >= settings.patience
    if progress['completed'] < limit and not early_stopped:
        raise ValueError('Training budget/model selection is not yet complete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.load_state_dict(progress['best_model'], strict=True)
    model.to(settings.device)
    metrics, predictions = score(model, loader(data.final_test(), settings.batch_size, 0),
                                  settings.device, final_test=True)
    from orbitshield.data.contracts import sha256
    result = {'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': sha256(checkpoint_path),
              'selected_step': progress['best_step'], 'selection_validation': progress['best_metrics'],
              'contract': contract, 'test_metrics': metrics, 'predictions': predictions}
    with (output / 'evaluation.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return metrics
