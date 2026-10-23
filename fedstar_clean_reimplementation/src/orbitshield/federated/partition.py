"""Explicit, single-attempt IID and label-Dirichlet allocations.

IID is a uniform permutation cut into near-equal chunks, not class quotas.
Dirichlet draws one probability vector per observed class and then multinomial
integer counts. No size balancing, parameter changes or rejection sampling.
Minimum size is an acceptance check, never an allocation intervention.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PartitionConfig:
    mode: str
    seed: int
    clients: int = 20
    eta: float = 0.5
    minimum_client_size: int = 1
    retry_policy: str = 'none'

    def __post_init__(self):
        if self.mode not in {'iid', 'dirichlet'}:
            raise ValueError('mode must be iid or dirichlet')
        if type(self.clients) is not int or not 1 <= self.clients <= 32767:
            raise ValueError('clients must be in [1,32767]')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(self.minimum_client_size) is not int or self.minimum_client_size < 0:
            raise ValueError('minimum_client_size must be a nonnegative integer')
        if type(self.eta) not in {int, float} or not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError('eta must be finite and positive')
        if self.retry_policy != 'none':
            raise ValueError('Only explicit single-attempt retry_policy=none is supported')


def allocate(labels: np.ndarray, config: PartitionConfig) -> np.ndarray:
    """Return client number for each *training* row; supports empty inputs.

    Labels must already be the artifact's integer encoding. Sorting class codes
    and PCG64 random generation are local reproducibility choices.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu':
        raise ValueError('Expected one-dimensional integer encoded labels')
    rng = np.random.Generator(np.random.PCG64(config.seed))
    owner = np.full(len(labels), -1, dtype=np.int16)
    if config.mode == 'iid':
        for client, rows in enumerate(np.array_split(rng.permutation(len(labels)), config.clients)):
            owner[rows] = client
    else:
        for label in np.unique(labels):
            rows = rng.permutation(np.flatnonzero(labels == label))
            probabilities = rng.dirichlet(np.full(config.clients, config.eta))
            sizes = rng.multinomial(len(rows), probabilities)
            for client, chunk in enumerate(np.split(rows, np.cumsum(sizes)[:-1])):
                owner[chunk] = client
    return owner


def summarize(labels: np.ndarray, owner: np.ndarray, config: PartitionConfig) -> dict:
    labels, owner = np.asarray(labels), np.asarray(owner)
    if (owner.shape != labels.shape or owner.dtype.kind not in 'iu'
            or np.any(owner < 0) or np.any(owner >= config.clients)):
        raise ValueError('Invalid or incomplete assignment')
    classes, totals = np.unique(labels, return_counts=True)
    global_p = totals / len(labels) if len(labels) else np.zeros(len(classes))
    clients = []
    for client in range(config.clients):
        local = labels[owner == client]
        counts = np.array([np.count_nonzero(local == c) for c in classes])
        p = counts / len(local) if len(local) else np.zeros(len(classes))
        clients.append({'client': client, 'samples': len(local),
                        'class_counts': {str(int(c)): int(n) for c, n in zip(classes, counts)},
                        'class_proportions': {str(int(c)): float(n) for c, n in zip(classes, p)},
                        'total_variation_from_global': float(np.abs(p-global_p).sum()/2) if len(local) else None})
    sizes = np.array([v['samples'] for v in clients])
    met = bool(np.all(sizes >= config.minimum_client_size))
    nonempty_tv = [v['total_variation_from_global'] for v in clients if v['samples']]
    return {'config': asdict(config), 'training_samples': len(labels),
            'global_class_counts': {str(int(c)): int(n) for c, n in zip(classes, totals)},
            'global_class_proportions': {str(int(c)): float(p) for c, p in zip(classes, global_p)},
            'clients': clients,
            'sample_count_statistics': {'minimum': int(sizes.min()), 'maximum': int(sizes.max()),
                                        'mean': float(sizes.mean()), 'std_population': float(sizes.std()),
                                        'empty_clients': int(np.count_nonzero(sizes == 0))},
            'mean_nonempty_client_total_variation': float(np.mean(nonempty_tv)) if nonempty_tv else None,
            'minimum_size_check': {'required': config.minimum_client_size, 'met': met,
                                   'feasible_by_total': len(labels) >= config.clients * config.minimum_client_size,
                                   'attempts': 1, 'retry_policy': 'none'},
            'status': 'accepted' if met else 'minimum_size_not_met'}
