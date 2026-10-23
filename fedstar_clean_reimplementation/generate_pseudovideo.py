"""Generate or validate immutable pseudo-video representation artifacts."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from orbitshield.representation.artifacts import (
    REPRESENTATION_ROOT, generate_pseudo_video_artifact, validate_pseudo_video_artifact)
from orbitshield.representation.projection import UnresolvedPaperSpecification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='create all split tensors from validated tabular artifacts')
    generate.add_argument('--dataset', choices=['all', 'nsl_kdd', 'stin', 'opssat_ad'], default='all')
    generate.add_argument('--allow-adaptation', action='store_true',
                          help='required explicit opt-in; outputs use untrained project-selected settings')
    generate.add_argument('--config', type=Path, default=ROOT / 'configs/projection_adaptation_example.yaml')
    generate.add_argument('--batch-size', type=int, default=512)
    generate.add_argument('--output-root', type=Path, default=REPRESENTATION_ROOT)
    verify = commands.add_parser('verify', help='verify all payloads and recompute every tensor value')
    verify.add_argument('--artifact', type=Path, required=True)
    verify.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    try:
        if args.command == 'generate':
            if not args.allow_adaptation:
                parser.error('generate requires explicit --allow-adaptation')
            index = json.loads((ROOT / 'reports/review_manifest.json').read_text(encoding='utf8'))
            names = list(index['artifacts']) if args.dataset == 'all' else [args.dataset]
            output = {}
            for name in names:
                print(f'Generating {name} from validated tabular artifact', flush=True)
                output[name] = generate_pseudo_video_artifact(
                    Path(index['artifacts'][name]), args.config, output_root=args.output_root,
                    batch_size=args.batch_size)
        else:
            output = validate_pseudo_video_artifact(args.artifact, verify_values=True,
                                                    batch_size=args.batch_size)
        print(json.dumps(output, indent=2, sort_keys=True))
    except UnresolvedPaperSpecification as exc:
        print(f'Exact FedSTAR projection is unavailable: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
