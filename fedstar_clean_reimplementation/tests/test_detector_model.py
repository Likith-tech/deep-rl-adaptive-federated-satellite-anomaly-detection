"""Numerical operation tests, gradients and representation-mode separation."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detector_training'))
from orbitshield_detector.config import DetectorConfig, TrainConfig, load_config
from orbitshield_detector.model import Detector, LKA, ClassificationHead, ConvSC, AdaptedGSTA
from orbitshield_detector.reproducibility import configure
from orbitshield.representation.projection import AdaptationSettings, build_adaptation


@pytest.fixture(autouse=True)
def deterministic_detector():
    configure(2024)


def initial_projection(dimension):
    return build_adaptation(dimension, AdaptationSettings(16, 2, 2, 32, 'gelu', 1e-5, True,
        'learned_shared_full_map', 'pytorch_linear_default_pos_zero', 2024, 4, 64))


@pytest.mark.parametrize('dataset', ['nsl_kdd', 'stin', 'opssat_ad'])
@pytest.mark.parametrize('mode', ['joint16', 'joint2', 'frozen2'])
def test_all_tasks_modes_singleton_forward_and_gradients(dataset, mode):
    config = DetectorConfig(dataset, mode)
    projection = initial_projection(config.input_dim)
    model = Detector(config, projection.state_dict() if mode == 'joint2' else None)
    x = torch.randn(1, config.input_dim)
    if mode == 'frozen2':
        with torch.no_grad(): x = projection(x).detach()
    result = model(x)
    assert result.shape == (1, config.classes) and torch.isfinite(result).all()
    F.cross_entropy(result, torch.tensor([config.classes - 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if mode == 'frozen2':
        assert model.projection is None
        assert all(p.grad is None for p in projection.parameters())
    else:
        for p in model.projection.parameters():
            assert torch.count_nonzero(p.grad) > 0
        before = model.projection.positional.detach().clone()
        torch.optim.SGD(model.parameters(), lr=.01).step()
        assert not torch.equal(before, model.projection.positional)


def test_lka_against_explicit_convolutions_and_sigmoid_product():
    layer = LKA(3).double()
    x = torch.randn(2, 3, 8, 8, dtype=torch.double)
    first = F.conv2d(x, layer.dw.weight, layer.dw.bias, padding=2, groups=3)
    second = F.conv2d(first, layer.dilated.weight, layer.dilated.bias, padding=9, dilation=3, groups=3)
    attn = F.conv2d(second, layer.pw.weight, layer.pw.bias)
    torch.testing.assert_close(layer(x), torch.sigmoid(attn) * x, rtol=0, atol=0)
    assert layer.dw.groups == layer.dilated.groups == 3


def test_small_spatial_map_dilated_kernel_uses_only_center():
    layer = LKA(2)
    x = torch.randn(1, 2, 2, 2)
    out = layer(x)
    with torch.no_grad():
        layer.dilated.weight[:, :, 0, 0].add_(1000)
    assert torch.equal(out, layer(x))
    large = torch.randn(1, 2, 16, 16)
    out = layer(large)
    with torch.no_grad():
        layer.dilated.weight[:, :, 2, 3].add_(10)
    assert not torch.equal(out, layer(large))


def test_convsc_and_classifier_equation_order():
    block = ConvSC(2).double()
    x = torch.randn(2, 64, 16, 16, dtype=torch.double)
    conv = F.conv2d(x, block[0].weight, block[0].bias, stride=2, padding=1)
    expected = F.silu(F.group_norm(conv, 8, block[1].weight, block[1].bias, 1e-5))
    torch.testing.assert_close(block(x), expected)
    head = ClassificationHead(9).double().eval()
    z = torch.randn(2, 256, 3, 4, dtype=torch.double)
    a, ln, _, dropout, b = head.layers
    hidden = F.linear(z.mean((-1, -2)), a.weight, a.bias)
    manual = F.linear(F.gelu(F.layer_norm(hidden, (256,), ln.weight, ln.bias, 1e-5)), b.weight, b.bias)
    torch.testing.assert_close(head(z), manual)
    assert dropout.p == .1


def test_gsta_explicit_residual_wiring():
    block = AdaptedGSTA().eval()
    x = torch.randn(2, 256, 2, 2)
    first = x + block.attention(block.norm1(x))
    torch.testing.assert_close(block(x), first + block.ffn(block.norm2(first)))


def test_no_sample_mixing_and_initialization_rng_isolation():
    config = DetectorConfig('opssat_ad', 'joint16')
    rng = torch.get_rng_state().clone()
    model = Detector(config).eval()
    assert torch.equal(rng, torch.get_rng_state())
    again = Detector(config).eval()
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())
    x = torch.randn(3, 19)
    output = model(x)
    torch.testing.assert_close(output, torch.cat([model(row[None]) for row in x]), atol=1e-6, rtol=1e-5)
    x[1] += 3
    torch.testing.assert_close(model(x)[[0, 2]], output[[0, 2]], atol=0, rtol=0)
    assert len(list(model.classifier_parameters())) == 6


@pytest.mark.parametrize('fault', ['exact', 'unknown_dataset', 'missing_state', 'bad_config', 'nan_input'])
def test_invalid_configuration_and_input(fault):
    with pytest.raises((ValueError, TypeError)):
        if fault == 'exact': DetectorConfig('stin', 'exact')
        elif fault == 'unknown_dataset': DetectorConfig('fake')
        elif fault == 'missing_state': Detector(DetectorConfig('stin', 'joint2'))
        elif fault == 'bad_config': TrainConfig(learning_rate=float('nan'))
        else: Detector(DetectorConfig('stin'))(torch.full((1, 30), float('nan')))


def test_profiles_resolve_approved_recipe():
    root = Path(__file__).resolve().parents[1]
    for mode in ('joint16', 'joint2', 'frozen2'):
        model, training = load_config(root / f'configs/detector/{mode}.yaml', 'stin')
        assert model.classes == 9 and training.batch_size == 64
        assert model.resolved()['architecture']['paper_claim'] == 'not_exact_fedstar'


def test_view_folding_preserves_sample_and_view_channel_order():
    model = Detector(DetectorConfig('opssat_ad', 'frozen2')).eval()
    model.encoder = nn.Identity()
    captured = []
    handle = model.translator.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    x = torch.arange(2 * 4 * 64 * 2 * 2, dtype=torch.float32).reshape(2, 4, 64, 2, 2)
    with torch.no_grad(): model(x)
    handle.remove()
    for sample in range(2):
        for view in range(4):
            assert torch.equal(captured[0][sample, view*64:(view+1)*64], x[sample, view])


def test_frozen_and_joint2_match_only_at_shared_initialization():
    projection = initial_projection(19)
    frozen = Detector(DetectorConfig('opssat_ad', 'frozen2')).eval()
    joint = Detector(DetectorConfig('opssat_ad', 'joint2'), projection.state_dict()).eval()
    x = torch.randn(2, 19)
    with torch.no_grad():
        cached = projection(x)
        torch.testing.assert_close(frozen(cached), joint(x), rtol=0, atol=0)
    before = {k: v.detach().clone() for k, v in projection.state_dict().items()}
    optimizer = torch.optim.SGD(joint.parameters(), lr=.01)
    F.cross_entropy(joint(x), torch.tensor([0, 1])).backward()
    optimizer.step()
    assert any(not torch.equal(v, joint.projection.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, projection.state_dict()[k]) for k, v in before.items())
