"""Eqs. (2)-(6) components inside an explicitly selected gSTA adaptation."""
import torch
from torch import nn
from orbitshield.representation.projection import AdaptationSettings, build_adaptation
from .config import DetectorConfig


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each spatial location, never across samples."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvSC(nn.Sequential):
    def __init__(self, stride):
        super().__init__(nn.Conv2d(64, 64, 3, stride=stride, padding=1, bias=True),
                         nn.GroupNorm(8, 64, eps=1e-5, affine=True), nn.SiLU())


class LKA(nn.Module):
    """Eq. (3) attention cascade; Eq. (4) sigmoid(Attn) * F.

    5x5 DW, dilation 3 and pointwise convolution are paper-specified.
    7x7 second kernel and zero padding are project choices.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dilated = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return torch.sigmoid(self.pw(self.dilated(self.dw(x)))) * x


class AdaptedGSTA(nn.Module):
    """Chosen dual residual/pre-LN wiring; not verified author gSTA code."""
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = ChannelLayerNorm(256), ChannelLayerNorm(256)
        self.attention = LKA()
        self.ffn = nn.Sequential(nn.Conv2d(256, 1024, 1), nn.GELU(), nn.Conv2d(1024, 256, 1))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ClassificationHead(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256, eps=1e-5),
                                    nn.GELU(), nn.Dropout(0.1), nn.Linear(256, classes))

    def forward(self, x):
        return self.layers(x.mean(dim=(-2, -1)))


class Detector(nn.Module):
    def __init__(self, config: DetectorConfig, projection_state=None):
        super().__init__()
        self.config = config
        self.reproduction_status = 'project_adaptation_not_exact_fedstar'
        h = config.spatial_size
        self.projection = None
        if config.representation != 'frozen2':
            settings = AdaptationSettings(16, h, h, 32, 'gelu', 1e-5, True,
                'learned_shared_full_map', 'pytorch_linear_default_pos_zero', config.seed, 4, 64)
            self.projection = build_adaptation(config.input_dim, settings)
            if config.representation == 'joint2':
                if projection_state is None:
                    raise ValueError('joint2 requires the verified saved initialization')
                self.projection.load_state_dict(projection_state, strict=True)
            elif projection_state is not None:
                raise ValueError('joint16 uses fresh initialization; cannot load the 2x2 artifact')
        elif projection_state is not None:
            raise ValueError('frozen2 consumes cached videos, not projection parameters')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = nn.Sequential(ConvSC(1), ConvSC(2))
            self.translator = nn.Sequential(*[AdaptedGSTA() for _ in range(4)])
            self.classifier = ClassificationHead(config.classes)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Expected finite floating-point input')
        if self.projection is not None:
            x = self.projection(x)
        h = self.config.spatial_size
        if x.ndim != 5 or tuple(x.shape[1:]) != (4, 64, h, h) or len(x) == 0:
            raise ValueError('Expected nonempty (B,4,64,H,W) video')
        b = len(x)
        spatial = self.encoder(x.reshape(b * 4, 64, h, h))
        # Contiguous view-major concatenation mixes only views belonging to one sample.
        folded = spatial.reshape(b, 256, *spatial.shape[-2:])
        logits = self.classifier(self.translator(folded))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite detector logits')
        return logits

    def classifier_parameters(self):
        return self.classifier.named_parameters()
