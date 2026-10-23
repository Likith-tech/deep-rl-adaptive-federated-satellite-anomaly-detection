"""Equation (1) order is supported; concrete architectural choices are NOT exact FedSTAR."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import torch
from torch import nn
import yaml


class UnresolvedPaperSpecification(ValueError):
    pass


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


class EquationOne(nn.Module):
    """Compose explicit components: stack_t(MLP_t(LN(Linear(x))) + E_pos).

    No guessed dimensions or MLP defaults. Callers supply all components. This
    verifies the algebra/layout, not fidelity of supplied modules to the paper.
    Generators must return (B,64,H,W); positional addition cannot include B/T.
    """
    def __init__(self, input_dim: int, embedding: nn.Linear, norm: nn.LayerNorm,
                 generators: list[nn.Module], positional: nn.Parameter,
                 spatial_shape: tuple[int, int]):
        super().__init__()
        positive_integer(input_dim, 'input_dim')
        if len(spatial_shape) != 2:
            raise ValueError('spatial_shape must contain H,W')
        for v in spatial_shape:
            positive_integer(v, 'spatial dimension')
        if not isinstance(embedding, nn.Linear) or embedding.in_features != input_dim or embedding.bias is None:
            raise ValueError('Embedding must be biased Linear(D,E)')
        if not isinstance(norm, nn.LayerNorm) or norm.normalized_shape != (embedding.out_features,):
            raise ValueError('LayerNorm must normalize the per-sample embedding axis E')
        if len(generators) != 4 or len({id(g) for g in generators}) != 4:
            raise ValueError('Exactly four independent view generators are required')
        parameter_sets = [{id(p) for p in g.parameters()} for g in generators]
        if any(parameter_sets[i] & parameter_sets[j] for i in range(4) for j in range(i)):
            raise ValueError('View generators must not share parameters')
        if not isinstance(positional, nn.Parameter) or not positional.requires_grad:
            raise ValueError('Positional encoding must be a learnable Parameter')
        target = (64, *spatial_shape)
        if positional.ndim > 3 or torch.broadcast_shapes(tuple(positional.shape), target) != target:
            raise ValueError('E_pos must broadcast to C,H,W only, never across samples/views')
        if not all(p.requires_grad for p in embedding.parameters()):
            raise ValueError('Embedding must be trainable')
        self.input_dim = input_dim
        self.embedding = embedding
        self.norm = norm
        self.generators = nn.ModuleList(generators)
        self.positional = positional
        self.spatial_shape = tuple(spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] < 1:
            raise ValueError(f'Input must be nonempty (B,{self.input_dim}); explicitly include batch size 1')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Input must be floating point and finite')
        if x.device != self.embedding.weight.device or x.dtype != self.embedding.weight.dtype:
            raise ValueError('Input device/dtype must match the projection')
        embedded = self.norm(self.embedding(x))
        views = []
        for generator in self.generators:
            view = generator(embedded)
            if tuple(view.shape) != (len(x), 64, *self.spatial_shape):
                raise ValueError('View generator must return (B,64,H,W) without mixing samples')
            views.append(view + self.positional)
        result = torch.stack(views, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError('Non-finite representation (check inputs and parameters)')
        return result


@dataclass(frozen=True)
class AdaptationSettings:
    """Every architectural choice is explicit; these are project assumptions."""
    embedding_dim: int
    height: int
    width: int
    hidden_dim: int
    activation: str
    layernorm_eps: float
    layernorm_affine: bool
    positional_encoding: str
    initialization: str
    seed: int
    views: int
    channels: int

    def __post_init__(self):
        for name in ['embedding_dim', 'height', 'width', 'hidden_dim']:
            positive_integer(getattr(self, name), name)
        if self.views != 4 or self.channels != 64:
            raise ValueError('Paper-supported contract requires T=4,C=64')
        if self.activation != 'gelu' or self.positional_encoding != 'learned_shared_full_map':
            raise ValueError('This adaptation implements GELU and a shared full-map learnable E_pos only')
        if self.initialization != 'pytorch_linear_default_pos_zero':
            raise ValueError('Unsupported initialization policy')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0,2**63)')
        if type(self.layernorm_affine) is not bool or not math.isfinite(self.layernorm_eps) or self.layernorm_eps <= 0:
            raise ValueError('Invalid LayerNorm settings')


class AdaptedViewMLP(nn.Module):
    def __init__(self, settings: AdaptationSettings):
        super().__init__()
        self.spatial_shape = (settings.height, settings.width)
        self.layers = nn.Sequential(nn.Linear(settings.embedding_dim, settings.hidden_dim),
                                    nn.GELU(), nn.Linear(settings.hidden_dim, 64 * settings.height * settings.width))

    def forward(self, x):
        # A learned output, not a reshape/padding of original dataset features.
        return self.layers(x).reshape(x.shape[0], 64, *self.spatial_shape)


def build_adaptation(input_dim: int, settings: AdaptationSettings) -> EquationOne:
    """CPU initialization with local RNG scope. No data fitting or global seed reset."""
    positive_integer(input_dim, 'input_dim')
    with torch.random.fork_rng(devices=[]):
        # CPU generator only; avoid changing CUDA RNG states on CUDA hosts.
        torch.random.default_generator.manual_seed(settings.seed)
        model = EquationOne(
            input_dim, nn.Linear(input_dim, settings.embedding_dim),
            nn.LayerNorm(settings.embedding_dim, eps=settings.layernorm_eps,
                         elementwise_affine=settings.layernorm_affine),
            [AdaptedViewMLP(settings) for _ in range(4)],
            nn.Parameter(torch.zeros(64, settings.height, settings.width)),
            (settings.height, settings.width))
    model.reproduction_status = 'project_adaptation_not_exact_fedstar'
    model.adaptation_settings = settings
    return model


def build_from_config(path: Path, input_dim: int, *, allow_adaptation: bool = False) -> EquationOne:
    config = yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if not isinstance(config, dict):
        raise ValueError('Expected projection configuration mapping')
    if config.get('mode') == 'exact':
        # Even manually filling dimensions is not evidence of paper fidelity.
        raise UnresolvedPaperSpecification(
            'Exact FedSTAR blocked: embedding width, H/W, view MLP, LayerNorm settings, '
            'E_pos shape/initialization and prose/equation ordering conflict need author evidence')
    if config.get('mode') != 'project_adaptation' or not allow_adaptation:
        raise ValueError('Project adaptation requires explicit allow_adaptation=True')
    if set(config) != {'mode', 'purpose', 'settings'} or config['purpose'] != 'contract_demonstration_not_trained':
        raise ValueError('Unexpected projection configuration or purpose')
    return build_adaptation(input_dim, AdaptationSettings(**config['settings']))
