"""Create or verify training-only partitions. Run from the repository root.

Create refuses to overwrite completed/partial output folders and the audit.
Verify is read-only and rechecks every mapping, original payload and baseline.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from orbitshield.data.contracts import digest, json_text, sha256
from orbitshield.federated.partition import PartitionConfig, allocate
from orbitshield.federated.artifacts import TrainingArtifact, write_partition, verify_partition


def preservation() -> dict:
    baseline = json.loads((ROOT / 'logs/partition-baseline.json').read_text(encoding='utf8'))
    changed = [p for p, h in baseline['files'].items() if not Path(p).is_file() or sha256(Path(p)) != h.lower()]
    if changed:
        raise ValueError('Pre-existing protected files changed: ' + repr(changed))
    evidence = json.loads((ROOT / 'reports/evidence_manifest.json').read_text(encoding='utf8'))
    for name, expected in evidence['reports'].items():
        if sha256(ROOT / 'reports' / name) != expected:
            raise ValueError('Historical generated report changed: ' + name)
    payloads = 0
    for name, item in evidence['artifacts'].items():
        path = Path(item['artifact'])
        if sha256(path / 'artifact_hashes.json') != item['artifact_manifest_sha256']:
            raise ValueError('Historical artifact manifest changed: ' + name)
        hashes = json.loads((path / 'artifact_hashes.json').read_text(encoding='utf8'))
        for file, h in hashes.items():
            if sha256(path / file) != h:
                raise ValueError('Historical payload changed: ' + file)
            payloads += 1
    result = subprocess.run(['git', 'diff', '--binary'], cwd=ROOT.parent, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--short'], cwd=ROOT.parent, capture_output=True, text=True)
    # Compare stdout separately because combined shell stderr ordering is unstable.
    return {'preexisting_files_unchanged': len(baseline['files']),
            'historical_report_hashes_passed': len(evidence['reports']),
            'artifact_payload_hashes_passed': payloads,
            'git_diff_sha256': digest(result.stdout), 'git_status_sha256': digest(status.stdout),
            'git_exit_codes': [result.returncode, status.returncode]}


def report_text(results: list, preserved: dict, test_record: Path) -> str:
    test = json.loads(test_record.read_text(encoding='utf8'))
    test_log = test_record.with_suffix('.log')
    lines = ['# Federated partition audit', '', 'Date: 2026-10-12. Partitioning only; no detector training or DRL experiment was run.', '',
             '## Paper evidence and implementation choices', '',
             'Verified against the original 21-page PDF, SHA256 `f38148c15a8410d1038f45f55463bd39d48d42093bb33f267e585ec6a264de42`.',
             'Section 4.2.2, p.12 specifies 20 clients, stochastic uniform IID equal-size sampling, and label-skewed Dirichlet eta=0.5. Section 4.3.4, p.15 lists experiment seeds 2024–2028. Page 12 reports seed-2024 minimum sizes of 6,149 STIN, 8,550 TON_IoT and 3,148 Bot-IoT; these are observed results, not prescribed minimum thresholds or a retry algorithm.', '',
             'NSL-KDD and OPS-SAT-AD are project extensions; the paper benchmarks STIN, TON_IoT and Bot-IoT. STIN here remains nine-class attack-type classification, with no benign class. Existing validation carve-outs and upstream split policies are retained, so even STIN does not use the authors\' exact training pool.', '',
             '| Detail | Paper evidence | Local implementation |', '|---|---|---|',
             '| Client count / skew | 20 / eta=0.5, Section 4.2.2 p.12 | Configured directly |',
             '| IID | Stochastic uniform sampling; equal-size clients | PCG64 uniform permutation, array_split; earlier client numbers receive the remainder, maximum size gap 1. No class quotas. Class proportions agree in expectation, not exactly for every client. |',
             '| Non-IID | Label-skewed Dirichlet | Sorted observed class codes; shuffle class row indices; independent symmetric Dirichlet(0.5) vector over 20 clients; multinomial integer counts; split shuffled indices. No balancing, resampling or parameter adjustment. Integer allocation and RNG are local choices. |',
             '| Minimum size | Reports observed minima for seed 2024 | Local minimum=1, checked after one draw; retry_policy=none. Failed candidates remain explicitly marked minimum_size_not_met and must not be treated as accepted training allocations. Empty clients permitted in diagnostic mappings; no rows discarded. |',
             '| Seeds | Seed 2024 check, repetitions 2024–2028 | Separate manifests for every dataset/mode/seed; same seed does not establish the authors\' RNG sequence. |', '',
             'Code: `src/orbitshield/federated/partition.py` and `artifacts.py`. Configuration: `configs/federated_partition.yaml`. These implement the documented partitioning stage with declared local choices; exact FedSTAR reproduction is not claimed.', '',
             '## Artifact and isolation contract', '',
             'Each folder below `data/client_partitions/<dataset>/<mode>/seed-<seed>-<configuration digest>/` contains a compact int16 `assignment.npy` and `manifest.json`. Mapping row i refers only to row i of the source artifact\'s X_train/y_train/records_train. Per-client row indices are `flatnonzero(assignment == client)`. IDs, original labels, groups/provenance and feature order are obtained unchanged from that source; no large feature array is copied. All 20 client IDs exist in the manifest even if empty. Source path, artifact-manifest/schema hashes, ordered training-ID digest, full global label mapping, code hashes, RNG version and configuration are bound in the manifest.', '',
             'The reader hashes every payload and raw source, checks source configuration/code bindings, integer labels against records and original label encoding, dimensions, split uniqueness and training membership. It does not load the fitted preprocessor or call fit/transform. Held-out records are inspected only for consistency and disjoint IDs; their labels/features never influence allocation. Historical source-reconstruction tests remain separate from this reader. This is trusted-artifact integrity verification, not an independent reconstruction of preprocessing.', '',
             'Historical validators hash the entire src tree. A new scoped reader permits only additive representation/federated modules while requiring every recorded file byte-identical. Existing validators, representation bridge, YAMLs and historical manifests were not weakened or resealed. Old strict CLIs may reject legacy artifacts after downstream additions; use this milestone\'s verify command for its explicit compatibility policy.', '',
             'All samples are retained. Existing duplicate predictor vectors may be assigned to different clients: clients are simulated record allocations, not physical hosts or independence guarantees. The original NSL official-test predictor overlap, STIN conflicting-label groups and OPS event/provenance limitations remain as documented in earlier audits. ID isolation does not resolve these upstream limitations.', '',
             '## Measured results', '',
             'TV is total variation distance: 0.5 * sum over global classes of |client proportion - training-global proportion|. Mean TV averages nonempty clients equally; empty clients have null TV. Standard deviation uses ddof=0. These describe this allocation only; no detection accuracy was measured.', '',
             '| Dataset | Mode | Seed | Training N | Min | Max | Mean | SD | Empty | Mean TV | Minimum met |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        s, c = r['summary'], r['summary']['config']
        n = s['sample_count_statistics']
        tv = s['mean_nonempty_client_total_variation']
        lines.append(f"| {r['dataset']} | {c['mode']} | {c['seed']} | {s['training_samples']} | {n['minimum']} | {n['maximum']} | {n['mean']:.2f} | {n['std_population']:.2f} | {n['empty_clients']} | {tv:.5f} | {s['minimum_size_check']['met']} |")
    failures = [r for r in results if r['summary']['status'] != 'accepted']
    lines += ['', f"Minimum-size outcome: {len(results)-len(failures)} accepted mappings; {len(failures)} diagnostic mapping(s) failed the local minimum. No retries occurred."]
    for r in failures:
        c, sizes = r['summary']['config'], r['summary']['sample_count_statistics']
        lines += [f"{r['dataset']} {c['mode']} seed {c['seed']}: {sizes['empty_clients']} empty clients, minimum {sizes['minimum']}; marked minimum_size_not_met. This mapping is retained for audit and is not accepted for client training under minimum={c['minimum_client_size']}."]
    lines += ['', 'Per-client counts and proportions for **all five seeds** are stored in each manifest and `logs/partition-results.json`. Seed-2024 class counts are expanded below. A missing local class is valid; the global output-label mapping is preserved.', '']
    for dataset in sorted({r['dataset'] for r in results}):
        subset = [r for r in results if r['dataset'] == dataset and r['summary']['config']['seed'] == 2024]
        mapping = subset[0]['source']['label_mapping']
        lines += [f'### {dataset}', '', 'Source label mapping: `' + json.dumps(mapping, sort_keys=True) + '`.', '',
                  'Global training counts: `' + json.dumps(subset[0]['summary']['global_class_counts'], sort_keys=True) + '`.',
                  'Global training proportions: `' + json.dumps(subset[0]['summary']['global_class_proportions'], sort_keys=True) + '`.', '']
        classes = list(subset[0]['summary']['global_class_counts'])
        lines += ['| Mode | Client | N | ' + ' | '.join(classes) + ' | TV |',
                  '|---|---:|---:|' + '---:|' * (len(classes)+1)]
        for r in subset:
            for c in r['summary']['clients']:
                tv = c['total_variation_from_global']
                tv_text = f'{tv:.5f}' if tv is not None else 'null (empty)'
                lines.append(f"| {r['summary']['config']['mode']} | {c['client']} | {c['samples']} | " + ' | '.join(str(c['class_counts'][v]) for v in classes) + f" | {tv_text} |")
        lines.append('')
    lines += ['## Verification and preservation', '',
              f'Test runner command: `.\\.venv\\Scripts\\python.exe -B fedstar_clean_reimplementation/run_tests.py`; record `{test_record.relative_to(ROOT)}`; exit {test["exit_code"]}. Exact output:', '', '```text', test_log.read_text(encoding='utf8').strip(), '```', '',
              'The complete existing suite ran unchanged alongside the new tests. New tests cover row coverage/uniqueness, seed determinism and changes, uneven sizes, rare/single/empty classes, minimum acceptance without retries, measured IID/skew differences, all three adapter label formats, held-out ID isolation, no fitting or preprocessor loading, immutable source hashes, persisted mapping verification and deliberately resealed mapping corruption. The production evidence pass also checks all original artifacts/reports and all 30 partitions.', '',
              'Preflight: `logs/partition-preflight.json` records the initial preservation check and successful source-reconstruction validation of all three real artifacts. That historical verification refits training transforms only for comparison; the partition reader/allocation never fits them. The first test attempt hit Windows sandbox temporary-directory permission errors; a SHA256 letter-case comparison in the new preservation test was also corrected. Both attempts are retained in separate logs.', '',
              'Every production mapping was regenerated in memory and compared exactly to its saved mapping; each was also compared to seed+1 and changed. Source binding, per-client class counts, ordered ID digests, complete training coverage, zero duplicate assignments and zero held-out ID overlap passed. In degenerate tiny datasets a changed seed need not change an allocation; tests do not promise otherwise.', '',
              'Preservation result: `' + json.dumps(preserved, sort_keys=True) + '`.',
              'The 237-file task baseline includes existing clean code/configs/docs/reports/tests/processed/root files/logs/backups, all artifact-recorded raw sources and the paper. All baseline bytes are checked; this scope does not claim to hash unrelated repository files or tool/temp caches. No existing file was edited. Git baseline and final outputs are compared separately because the clean directory is already untracked.', '',
              'Commands from repository root:', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe -B fedstar_clean_reimplementation/partition_clients.py create --test-record fedstar_clean_reimplementation/logs/' + test_record.name,
              '.\\.venv\\Scripts\\python.exe -B fedstar_clean_reimplementation/partition_clients.py verify',
              '.\\.venv\\Scripts\\python.exe -B fedstar_clean_reimplementation/check_preservation.py verify --audit-backup fedstar_clean_reimplementation/backups/before_projection_20261012/reports/FEDSTAR_IMPLEMENTATION_AUDIT.md', '```', '',
              'Create refuses existing output paths. Verify performs no writes. Tests and production outputs use distinct directories. No prior preprocessing artifact or report was regenerated.', '',
              '## Limits and next steps', '',
              'This does not recover the authors\' exact partition algorithm, random generator, training rows or STIN minimum of 6,149. It does not require each client to contain every class. No physical satellite/host interpretation is established. The prior missing architecture specifications and positional-encoding conflict remain unresolved. GPU correctness remains subject to the existing CUDA test outcome; partitioning itself runs on CPU.', '',
              'Review any minimum_size_not_met manifest before a future training task. A new retry or balancing policy would require a separately documented configuration and new manifests; never silently replace these results. Obtain professor approval for the existing dataset/task adaptations and author clarification before making stronger reproduction claims.', '',
              'New files: `src/orbitshield/federated/{__init__,partition,artifacts}.py`, `configs/federated_partition.yaml`, `tests/test_federated_partition.py`, `partition_clients.py`, this report, `logs/partition-baseline.json`, `logs/partition-preflight.json`, `logs/partition-results.json`, `logs/partition-final-scope.json`, new isolated test logs, and 30 new partition folders (60 files). No existing files were modified.', '',
              'Scope note: `reports/PREPROCESSING_PAPER_VS_IMPLEMENTATION.md` appeared after this task\'s baseline. No command or implementation in this task created or edited it. It was left untouched and is recorded separately as an unattributed concurrent addition, not claimed as this milestone\'s deliverable. The final scope record distinguishes that path from task-created files.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'verify'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/federated_partition.yaml')
    parser.add_argument('--test-record', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf8'))
    if set(config) != {'clients','modes','eta','seeds','minimum_client_size','retry_policy'}:
        raise ValueError('Unexpected partition configuration keys')
    if config['clients'] != 20 or config['eta'] != .5 or config['modes'] != ['iid','dirichlet']:
        raise ValueError('Milestone requires 20 clients, eta=0.5, IID and Dirichlet')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    preserved = preservation()
    results_path = ROOT / 'logs/partition-results.json'
    report_path = ROOT / 'reports/FEDERATED_PARTITION_AUDIT.md'
    if args.command == 'create':
        if results_path.exists() or report_path.exists():
            raise FileExistsError('Partition evidence/report already exists; use verify')
        if args.test_record is None or json.loads(args.test_record.read_text())['exit_code'] != 0:
            raise ValueError('Provide a passing complete-suite --test-record')
    prior = json.loads(results_path.read_text()) if args.command == 'verify' else None
    index = json.loads((ROOT / 'reports/review_manifest.json').read_text())
    results = []
    for dataset, path in index['artifacts'].items():
        source = TrainingArtifact(Path(path))
        for mode in config['modes']:
            for seed in config['seeds']:
                settings = PartitionConfig(mode, seed, config['clients'], config['eta'],
                                           config['minimum_client_size'], config['retry_policy'])
                folder = ROOT / 'data/client_partitions' / dataset / mode / f'seed-{seed}-{digest(settings.__dict__)[:8]}'
                if args.command == 'create':
                    write_partition(source, settings, folder)
                summary = verify_partition(source, folder)
                owner = np.load(folder / 'assignment.npy', allow_pickle=False)
                changed = not np.array_equal(owner, allocate(source.labels, replace(settings, seed=seed+1)))
                if not changed:
                    raise ValueError('Production seed-change check unexpectedly unchanged')
                result = {'dataset': dataset, 'folder': str(folder.relative_to(ROOT)),
                          'source': source.binding, 'summary': summary,
                          'manifest_sha256': sha256(folder / 'manifest.json'),
                          'same_seed_exact': True, 'seed_plus_one_changed': changed}
                results.append(result)
                print(dataset, mode, seed, summary['sample_count_statistics'], summary['status'], flush=True)
    preserved = preservation()
    evidence = {'configuration': config, 'configuration_sha256': sha256(args.config),
                'results': results, 'preservation': preserved}
    if prior is not None:
        if evidence != prior:
            raise ValueError('Saved partition evidence does not reproduce')
        print('PASS: all 30 partitions reproduce; ' + json.dumps(preserved), flush=True)
    else:
        results_path.write_text(json_text(evidence), encoding='utf8')
        report_path.write_text(report_text(results, preserved, args.test_record.resolve()), encoding='utf8')
        print('Created ' + str(report_path), flush=True)


if __name__ == '__main__':
    main()
