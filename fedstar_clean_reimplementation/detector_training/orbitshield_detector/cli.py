"""Read-only preflight by default; training and final evaluation require explicit flags."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import numpy as np
import torch
from orbitshield.data.contracts import CLEAN, sha256
from .config import load_config
from .data import ArtifactData
from .model import Detector
from .reproducibility import configure


def make_contract(config, settings, data, mode):
    package = Path(__file__).resolve().parent
    code = {str(p.relative_to(CLEAN)): sha256(p) for p in sorted(package.glob('*.py'))}
    for name in ('train_detector.py', 'evaluate_detector.py'):
        code[name] = sha256(CLEAN / name)
    return {'model': config.resolved(), 'training': asdict(settings), 'data': data.binding(),
            'mode': mode, 'code_hashes': code, 'environment': {
                'python': platform.python_version(), 'torch': str(torch.__version__),
                'numpy': np.__version__, 'platform': platform.platform(),
                'device': settings.device, 'threads': settings.threads,
                'cuda_build': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if settings.device == 'cuda' else None},
            'loss': 'unweighted_cross_entropy_logits', 'scheduler': None,
            'class_weighting': None, 'sampling': 'uniform_train_shuffle_no_drop_last',
            'adamw_decay': 'all_trainable_parameters', 'local_optimizer_state': 'reset_each_round_client',
            'fedavg_accumulation': 'fixed_client_order_float64_cpu',
            'selection': 'max_task_metric_then_min_validation_loss_then_earliest',
            'federated_early_stopping': False, 'controller': None}


def safe_output(path):
    output = Path(path).resolve()
    root = (CLEAN / 'runs/detector').resolve()
    if not output.is_relative_to(root) or output == root or output.exists():
        raise ValueError('Choose a new child directory under runs/detector')
    return output


def main(final=False):
    p = argparse.ArgumentParser(description='Explicit adapted detector; exact FedSTAR unavailable')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['nsl_kdd', 'opssat_ad', 'stin'], required=True)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--representation-artifact', type=Path)
    p.add_argument('--partition', type=Path)
    p.add_argument('--mode', choices=['centralized', 'fedavg'], default='centralized')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', type=Path)
    if final:
        p.add_argument('--checkpoint', type=Path, required=True)
        p.add_argument('--enable-final-test', action='store_true')
    else:
        p.add_argument('--enable-training', action='store_true')
    args = p.parse_args()
    config, settings = load_config(args.config, args.dataset, args.seed)
    if (args.mode == 'fedavg') != (args.partition is not None):
        p.error('FedAvg requires a partition; centralized mode must not specify one')
    configure(config.seed, settings.threads, settings.device)
    data = ArtifactData(args.source, config, args.representation_artifact, args.partition)
    model = Detector(config, data.projection_state if config.representation == 'joint2' else None)
    contract = make_contract(config, settings, data, args.mode)
    if not final and not args.enable_training:
        if args.resume is not None or args.output is not None:
            p.error('Preflight does not accept output/resume; enable training explicitly')
        print(json.dumps({'status': 'preflight_passed_no_training', 'contract': contract,
                          'parameters': sum(v.numel() for v in model.parameters())}, indent=2))
        return
    if final and not args.enable_final_test:
        p.error('Final test requires --enable-final-test after model selection is locked')
    if args.output is None:
        p.error('--output required for execution')
    output = safe_output(args.output)
    if final:
        if args.resume is not None:
            p.error('Evaluation does not resume optimization')
        from .evaluate import evaluate_final
        result = evaluate_final(model, data, settings, contract, args.checkpoint, output)
    else:
        if args.mode == 'centralized':
            from .trainer import train_centralized
            progress = train_centralized(model, data.dataset('train'), data.dataset('validation'),
                                        settings, contract, output, resume=args.resume)
        else:
            from .fedavg import train_fedavg
            progress = train_fedavg(model, [data.dataset('train', i) for i in range(20)],
                                   data.dataset('validation'), settings, contract, output, resume=args.resume)
        result = {'contract': contract, 'completed': progress['completed'],
                  'best_step': progress['best_step'], 'validation': progress['best_metrics'],
                  'history': progress['history'], 'test_evaluated': False}
        with (output / 'run.json').open('x', encoding='utf8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
