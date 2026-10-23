"""Mathematical and isolation contracts, not a claim to know missing paper settings."""
from dataclasses import replace
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from orbitshield.data.contracts import CLEAN
from orbitshield.data.common_preprocessing import prepare_dataset
from orbitshield.representation.projection import (
    EquationOne, AdaptationSettings, build_adaptation, build_from_config, UnresolvedPaperSpecification)
from orbitshield.representation.artifacts import ArtifactBatches, represent_batch, scoped_code_hashes


def settings():
    return AdaptationSettings(8, 2, 3, 12, 'gelu', 1e-5, True,
                              'learned_shared_full_map', 'pytorch_linear_default_pos_zero', 2024, 4, 64)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
    torch.set_num_threads(threads)


@pytest.mark.parametrize('dimension', [122, 30, 19])
@pytest.mark.parametrize('batch_size', [1, 2, 7])
def test_four_view_shape_and_independent_generators(dimension, batch_size):
    model = build_adaptation(dimension, settings())
    x = torch.arange(batch_size * dimension, dtype=torch.float32).reshape(batch_size, dimension) / 100
    out = model(x)
    assert out.shape == (batch_size, 4, 64, 2, 3)
    assert model.norm.normalized_shape == (8,)
    assert not torch.equal(out[:, 0], out[:, 1])
    ids = [set(id(p) for p in view.parameters()) for view in model.generators]
    assert all(not ids[i] & ids[j] for i in range(4) for j in range(i))


def test_equation_order_against_manual_formula_and_position_delta():
    model = build_adaptation(19, settings()).double()
    with torch.no_grad():
        model.positional.copy_(torch.arange(64 * 2 * 3).reshape(64, 2, 3) / 100)
        model.norm.weight.copy_(torch.linspace(.5, 1.5, 8))
        model.norm.bias.copy_(torch.linspace(-.2, .2, 8))
    x = torch.linspace(-2, 3, 38, dtype=torch.float64).reshape(2, 19)
    z = x @ model.embedding.weight.T + model.embedding.bias
    normalized = (z - z.mean(-1, keepdim=True)) / torch.sqrt(z.var(-1, keepdim=True, unbiased=False) + 1e-5)
    normalized = normalized * model.norm.weight + model.norm.bias
    expected = []
    for view in model.generators:
        first, _, last = view.layers
        hidden = normalized @ first.weight.T + first.bias
        hidden = .5 * hidden * (1 + torch.erf(hidden / 2**.5))
        pixels = (hidden @ last.weight.T + last.bias).reshape(2, 64, 2, 3)
        expected.append(pixels + model.positional)
    actual = model(x)
    torch.testing.assert_close(actual, torch.stack(expected, 1), atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        model.positional.add_(2)
    torch.testing.assert_close(model(x) - actual, torch.full_like(actual, 2), atol=1e-12, rtol=1e-12)


def test_no_cross_sample_mixing_or_batch_statistic_dependence():
    model = build_adaptation(30, settings())
    x = torch.linspace(-3, 2, 150).reshape(5, 30)
    actual = model(x)
    singles = torch.cat([model(row[None]) for row in x])
    torch.testing.assert_close(actual, singles)
    order = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(model(x[order]), actual[order])
    changed = x.clone()
    changed[1] = changed[1] * 100 + 5
    torch.testing.assert_close(model(changed)[[0, 2, 3, 4]], actual[[0, 2, 3, 4]], atol=0, rtol=0)
    single = x[:1].clone().requires_grad_()
    mixed = torch.cat([single, x[1:]])
    grad = torch.autograd.grad(model(mixed)[1].square().sum(), single)[0]
    assert torch.count_nonzero(grad) == 0


def test_reproducibility_rng_isolation_and_checkpoint_roundtrip():
    before = torch.random.get_rng_state().clone()
    first = build_adaptation(122, settings())
    assert torch.equal(torch.random.get_rng_state(), before)
    second = build_adaptation(122, settings())
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    third = build_adaptation(122, replace(settings(), seed=2025))
    assert not torch.equal(first.embedding.weight, third.embedding.weight)
    stream = io.BytesIO()
    torch.save(first.state_dict(), stream)
    stream.seek(0)
    third.load_state_dict(torch.load(stream, weights_only=True))
    x = torch.ones(2, 122)
    assert torch.equal(first(x), third(x))


@pytest.mark.parametrize('fault', ['one_dim', 'width', 'three_dim', 'empty', 'integer', 'nan', 'inf', 'dtype'])
def test_invalid_inputs_rejected(fault):
    model = build_adaptation(19, settings())
    x = torch.ones(2, 19)
    if fault == 'one_dim': x = x[0]
    if fault == 'width': x = x[:, :-1]
    if fault == 'three_dim': x = x[None]
    if fault == 'empty': x = x[:0]
    if fault == 'integer': x = x.long()
    if fault == 'dtype': x = x.double()
    if fault == 'nan': x[0, 0] = float('nan')
    if fault == 'inf': x[0, 0] = float('inf')
    with pytest.raises(ValueError):
        model(x)


def test_invalid_components_and_nonfinite_parameter_rejected():
    model = build_adaptation(19, settings())
    with pytest.raises(ValueError, match='independent'):
        EquationOne(19, model.embedding, model.norm, [model.generators[0]] * 4, model.positional, (2, 3))
    with pytest.raises(ValueError, match='E_pos'):
        EquationOne(19, model.embedding, model.norm, list(model.generators), nn.Parameter(torch.zeros(1, 64, 2, 3)), (2, 3))
    with torch.no_grad():
        model.positional.fill_(float('inf'))
    with pytest.raises(ValueError, match='Non-finite'):
        model(torch.ones(1, 19))


def test_backpropagation_and_training_only_update():
    model = build_adaptation(30, settings())
    train = torch.linspace(-1, 2, 90).reshape(3, 30).requires_grad_()
    heldout = torch.ones(2, 30, requires_grad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = model.embedding.weight.detach().clone()
    model(train).square().mean().backward()
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        assert torch.count_nonzero(value.grad) > 0, name
    assert train.grad is not None and heldout.grad is None
    optimizer.step()
    assert not torch.equal(model.embedding.weight, before)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        model(heldout)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable; CPU-only installed PyTorch')
def test_cuda_forward_backward_and_determinism():
    model = build_adaptation(19, settings()).cuda()
    x = torch.linspace(-1, 2, 38, device='cuda').reshape(2, 19)
    out = model(x)
    assert out.is_cuda and out.shape == (2, 4, 64, 2, 3)
    assert torch.equal(out, model(x))
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_exact_config_blocked_and_adaptation_explicit():
    with pytest.raises(UnresolvedPaperSpecification):
        build_from_config(CLEAN / 'configs/projection_exact.yaml', 122)
    path = CLEAN / 'configs/projection_adaptation_example.yaml'
    with pytest.raises(ValueError, match='allow_adaptation'):
        build_from_config(path, 30)
    model = build_from_config(path, 30, allow_adaptation=True)
    assert model(torch.ones(1, 30)).shape == (1, 4, 64, 2, 2)
    assert model.reproduction_status == 'project_adaptation_not_exact_fedstar'


@pytest.mark.parametrize('field,value', [('views', 3), ('channels', 32), ('height', 0),
                                       ('embedding_dim', None), ('layernorm_eps', float('nan'))])
def test_bad_configuration(field, value):
    with pytest.raises((ValueError, TypeError)):
        replace(settings(), **{field: value})


@pytest.mark.parametrize('fixture_name', ['nsl_config', 'stin_config', 'ops_config'])
def test_adapter_integration_preserves_all_partition_labels_and_ids(fixture_name, request, tmp_path):
    config = request.getfixturevalue(fixture_name)
    path = tmp_path / 'integration_artifact'
    prepare_dataset(config, path)
    source = ArtifactBatches(path)
    model = build_adaptation(source.feature_dimension, settings())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    all_ids = set()
    for split in source.validation['splits']:
        labels, ids = [], []
        for batch in source.batches(split, 7):
            before_x = batch.features.clone()
            with torch.no_grad():
                result = represent_batch(model, batch)
            assert result.split == split and result.source_ids == batch.source_ids
            assert result.labels is batch.labels and torch.equal(batch.features, before_x)
            assert result.video.shape == (len(batch.features), 4, 64, 2, 3)
            ids.extend(result.source_ids)
            labels.extend(result.labels.tolist())
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
        np.testing.assert_array_equal(labels, np.load(path / f'y_{split}.npy'))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_additive_compatibility_never_accepts_changed_preprocessing():
    recorded = {'src/orbitshield/data/common_preprocessing.py': 'original'}
    added = {'src/orbitshield/representation/projection.py': 'new'}
    assert scoped_code_hashes(recorded, dict(recorded, **added)) == recorded
    with pytest.raises(ValueError, match='changed'):
        scoped_code_hashes(recorded, {**recorded, next(iter(recorded)): 'changed', **added})
    with pytest.raises(ValueError, match='outside'):
        scoped_code_hashes(recorded, {**recorded, 'src/orbitshield/data/inject.py': 'new'})
