"""Versioned project recipe. Unspecified paper values are never called exact."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import math
import yaml

TASKS = {
    'nsl_kdd': (122, {'normal': 0, 'attack': 1}, 'official_test', 'macro_f1'),
    'opssat_ad': (19, {'0': 0, '1': 1}, 'official_test', 'average_precision'),
    'stin': (30, {name: i for i, name in enumerate([
        'Backdoor', 'Botnet', 'LDAP_DDoS', 'MSSQL_DDoS', 'NetBIOS_DDoS',
        'Portmap_DDoS', 'Syn_DDoS', 'UDP_DDoS', 'Web Attack'])}, 'test', 'macro_f1'),
}

# These are the complete fixed choices of recipe v1, not recovered author settings.
ARCHITECTURE = {
    'recipe': 'orbitshield_detector_adaptation_v1', 'paper_claim': 'not_exact_fedstar',
    'embedding_dim': 16, 'view_hidden_dim': 32, 'views': 4, 'view_channels': 64,
    'view_activation': 'gelu', 'position': 'shared_full_map_zero_initialized_after_mlp',
    'encoder_channels': [64, 64], 'encoder_kernels': [3, 3],
    'encoder_strides': [1, 2], 'encoder_padding': [1, 1], 'groupnorm_groups': 8,
    'encoder_activation': 'silu', 'translator_channels': 256, 'translator_blocks': 4,
    'view_folding': 'view_major_T_times_C', 'lka_kernels': [5, 7, 1],
    'lka_dilations': [1, 3, 1], 'lka_padding': [2, 9, 0], 'padding_mode': 'zeros',
    'gsta_wiring': 'pre_channel_LN_residual_LKA_then_pre_channel_LN_residual_MLP',
    'ffn_expansion': 4, 'ffn_activation': 'gelu', 'stochastic_depth': 0.0,
    'layer_scale': None, 'classifier_hidden': 256, 'classifier_dropout': 0.1,
    'classifier_order': 'spatial_mean_linear_LN_GELU_dropout_linear',
    'normalization_eps': 1e-5, 'normalization_affine': True, 'bias': True,
    'initialization': 'pytorch_module_defaults_position_zero',
}


@dataclass(frozen=True)
class DetectorConfig:
    dataset: str
    representation: str = 'joint16'
    seed: int = 2024

    def __post_init__(self):
        if self.dataset not in TASKS or self.representation not in {'joint16', 'joint2', 'frozen2'}:
            raise ValueError('Unknown dataset or representation; exact mode is unavailable')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an unsigned 32-bit integer')

    @property
    def input_dim(self): return TASKS[self.dataset][0]

    @property
    def classes(self): return len(TASKS[self.dataset][1])

    @property
    def spatial_size(self): return 16 if self.representation == 'joint16' else 2

    def resolved(self):
        return {**asdict(self), 'architecture': deepcopy(ARCHITECTURE), 'input_dim': self.input_dim,
                'classes': self.classes, 'label_mapping': TASKS[self.dataset][1],
                'spatial_size': self.spatial_size,
                'projection_initialization': 'fresh_seed' if self.representation == 'joint16' else 'bound_artifact_state'}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 50
    patience: int = 10
    rounds: int = 50
    local_epochs: int = 10
    threads: int = 1
    device: str = 'cpu'

    def __post_init__(self):
        for key in ('batch_size', 'epochs', 'patience', 'rounds', 'local_epochs', 'threads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ('learning_rate', 'epsilon'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('Invalid ' + key)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight_decay')
        if not all(math.isfinite(v) and 0 <= v < 1 for v in (self.beta1, self.beta2)):
            raise ValueError('Invalid AdamW betas')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')


def load_config(path: Path, dataset: str, seed: int = 2024):
    config = yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if set(config) != {'representation', 'recipe', 'training'} or config['recipe'] != ARCHITECTURE['recipe']:
        raise ValueError('Unknown detector recipe/configuration')
    return DetectorConfig(dataset, config['representation'], seed), TrainConfig(**config['training'])
