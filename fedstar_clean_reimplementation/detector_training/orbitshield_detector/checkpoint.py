"""Immutable boundary checkpoints; exact configuration/provenance matching on resume."""
import os
from pathlib import Path
import uuid
import torch
from .reproducibility import capture_rng, restore_rng


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, model, optimizer, contract, progress):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Checkpoint is immutable: ' + str(path))
    payload = {'format_version': 1, 'contract': contract, 'model': cpu_state(model),
               'optimizer': optimizer.state_dict() if optimizer is not None else None,
               'rng': capture_rng(), 'progress': progress}
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        # Windows rename fails if the target exists; never replace an old checkpoint.
        temporary.rename(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path, contract):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('contract') != contract:
        raise ValueError('Checkpoint configuration/data/code/environment contract mismatch')
    return checkpoint


def resume_checkpoint(path, model, optimizer, contract):
    state = read_checkpoint(path, contract)
    model.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        if state['optimizer'] is None:
            raise ValueError('Missing optimizer state')
        optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state['progress']
