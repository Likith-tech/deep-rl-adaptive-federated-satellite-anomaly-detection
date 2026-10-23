"""Controller-free, volume-weighted FedAvg; explicitly not full FedSTAR."""
from pathlib import Path
import torch
from .checkpoint import cpu_state, resume_checkpoint, save_checkpoint
from .config import TASKS
from .trainer import (derived_seed, initial_progress, loader, optimizer_for,
                      record_validation, score, train_epoch)


def aggregate(states, sizes):
    if not states or len(states) != len(sizes) or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError('Aggregation requires positive sample counts')
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError('Incompatible client parameter names')
    result = {}
    for key, first in states[0].items():
        values = [state[key] for state in states]
        if any(v.shape != first.shape or v.dtype != first.dtype or not torch.isfinite(v).all() for v in values):
            raise ValueError('Invalid client parameters')
        if first.is_floating_point():
            # Fixed client order, CPU float64 accumulation, original dtype on output.
            result[key] = sum(v.cpu().double() * (n / sum(sizes)) for v, n in zip(values, sizes)).to(first.dtype)
        else:
            if any(not torch.equal(first, v) for v in values):
                raise ValueError('Conflicting non-floating buffers')
            result[key] = first.clone()
    return result


def train_fedavg(model, clients, validation, settings, contract, output, *, resume=None, stop_after=None):
    if len(clients) != 20 or any(c.split != 'train' or len(c) == 0 for c in clients):
        raise ValueError('FedAvg requires 20 accepted nonempty training clients')
    if validation.split != 'validation':
        raise ValueError('Expected validation split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(settings.device)
    progress = resume_checkpoint(resume, model, None, contract) if resume else initial_progress()
    for round_id in range(progress['completed'] + 1, settings.rounds + 1):
        global_state = cpu_state(model)
        states, sizes, local_losses = [], [], []
        for client_id, data in enumerate(clients):
            model.load_state_dict(global_state, strict=True)
            # Reset moments each round/client. Same global initialization for every client.
            optimizer = optimizer_for(model, settings)
            for epoch in range(1, settings.local_epochs + 1):
                batch_seed = derived_seed(model.config.seed, 'federated', round_id, client_id, epoch)
                local_loss = train_epoch(model, loader(data, settings.batch_size, batch_seed, True),
                                         optimizer, settings.device)
            states.append(cpu_state(model))
            sizes.append(len(data))
            local_losses.append(local_loss)
        model.load_state_dict(aggregate(states, sizes), strict=True)
        result, _ = score(model, loader(validation, settings.batch_size, 0), settings.device)
        record_validation(progress, model, result, round_id, TASKS[model.config.dataset][3],
                          sum(n * loss for n, loss in zip(sizes, local_losses)) / sum(sizes))
        save_checkpoint(output / f'round-{round_id:04d}.pt', model, None, contract, progress)
        if stop_after is not None and round_id >= stop_after:
            break
    return progress
