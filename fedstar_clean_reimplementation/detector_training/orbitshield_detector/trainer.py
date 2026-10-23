"""Reusable local detector training; no controller and no final-test access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import cpu_state, resume_checkpoint, save_checkpoint
from .config import TASKS
from .metrics import better, classification_metrics


def loader(dataset, batch_size, seed, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)


def derived_seed(seed, *coordinates):
    encoded = json.dumps([seed, *coordinates]).encode('ascii')
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little') % (2**63)


def optimizer_for(model, settings):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=settings.learning_rate, weight_decay=settings.weight_decay,
        betas=(settings.beta1, settings.beta2), eps=settings.epsilon, foreach=False)


def train_epoch(model, batches, optimizer, device):
    if batches.dataset.split != 'train':
        raise ValueError('Optimizer updates require the train split')
    model.train()
    loss_sum, count = 0.0, 0
    for x, y, _ in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Non-finite gradients')
        optimizer.step()
        loss_sum += float(loss.detach()) * len(y)
        count += len(y)
    if not count:
        raise ValueError('Empty training client')
    return loss_sum / count


def score(model, batches, device, *, final_test=False):
    allowed = ('test', 'official_test') if final_test else ('validation',)
    if batches.dataset.split not in allowed:
        raise ValueError('Wrong split for validation/final evaluation')
    model.eval()
    losses, count, labels, probabilities, source_ids = 0.0, 0, [], [], []
    with torch.inference_mode():
        for x, y, ids in batches:
            logits = model(x.to(device))
            losses += float(torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum'))
            count += len(y)
            labels.extend(y.tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
            source_ids.extend(ids)
    metrics = classification_metrics(np.asarray(labels, dtype=np.int64), np.asarray(probabilities), model.config.classes)
    metrics['loss'] = losses / count
    return metrics, {'source_ids': source_ids, 'labels': labels, 'probabilities': probabilities}


def initial_progress():
    return {'completed': 0, 'best_metrics': None, 'best_step': None, 'best_model': None,
            'bad_steps': 0, 'history': []}


def record_validation(progress, model, metrics, step, metric, training_loss):
    improved = better(metrics, progress['best_metrics'], metric)
    progress['history'].append({'step': step, 'training_loss': training_loss, 'validation': metrics})
    progress['completed'] = step
    if improved:
        progress.update(best_metrics=metrics, best_step=step, best_model=cpu_state(model), bad_steps=0)
    else:
        progress['bad_steps'] += 1


def train_centralized(model, train, validation, settings, contract, output, *, resume=None, stop_after=None):
    """Checkpoints at complete epochs; stop_after is for interruption/smoke tests only."""
    if train.split != 'train' or validation.split != 'validation':
        raise ValueError('Expected train and validation datasets')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(settings.device)
    optimizer = optimizer_for(model, settings)
    progress = resume_checkpoint(resume, model, optimizer, contract) if resume else initial_progress()
    metric = TASKS[model.config.dataset][3]
    for epoch in range(progress['completed'] + 1, settings.epochs + 1):
        if progress['bad_steps'] >= settings.patience:
            break
        batches = loader(train, settings.batch_size, derived_seed(model.config.seed, 'central', epoch), True)
        train_loss = train_epoch(model, batches, optimizer, settings.device)
        result, _ = score(model, loader(validation, settings.batch_size, 0), settings.device)
        record_validation(progress, model, result, epoch, metric, train_loss)
        save_checkpoint(output / f'epoch-{epoch:04d}.pt', model, optimizer, contract, progress)
        if stop_after is not None and epoch >= stop_after:
            break
    return progress
