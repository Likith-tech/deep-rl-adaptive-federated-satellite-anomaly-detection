"""Generate review evidence only after every selected artifact passes validation."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contracts import CLEAN, digest, json_text, sha256
from .common_preprocessing import validate_saved_artifact, inside_clean


def backup_file(path: Path) -> None:
    path = inside_clean(path)
    if path.is_file():
        destination = CLEAN / 'backups/generated' / sha256(path) / path.relative_to(CLEAN)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def write_text(path: Path, content: str) -> None:
    path = inside_clean(path)
    backup_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf8')


def table(headers: list[str], rows: list[list]) -> str:
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join(
        '| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |\n' for row in rows)


def generate(index_path: Path = CLEAN / 'reports/review_manifest.json') -> dict:
    index = json.loads(index_path.read_text(encoding='utf8'))
    # Complete all checks before writing any review output.
    results = {name: validate_saved_artifact(Path(path)) for name, path in index['artifacts'].items()}
    metas = {name: json.loads((Path(path) / 'metadata.json').read_text(encoding='utf8'))
             for name, path in index['artifacts'].items()}
    audits = {name: json.loads((Path(path) / 'audit.json').read_text(encoding='utf8'))
              for name, path in index['artifacts'].items()}
    report = CLEAN / 'reports'
    dimensions, splits, classes, missing, types = [], [], [], [], []
    for name, meta in metas.items():
        dimensions.append([name, meta['total_rows'], len(meta['schema']['input_features']),
                           len(meta['schema']['feature_names']), results[name]['artifact_manifest_sha256']])
        types.append([name, len(meta['schema']['numeric']), len(meta['schema']['categorical']),
                      'target/IDs excluded'])
        inverse = {v: k for k, v in meta['schema']['label_mapping'].items()}
        for split, values in meta['splits'].items():
            splits.append([name, split, values['rows'], values['ordered_source_id_sha256']])
            classes.extend([name, split, inverse[int(label)], count]
                           for label, count in values['label_counts'].items())
        rows = []
        for file, audit in audits[name]['files'].items():
            row = [file, audit['rows'], audit['raw_columns'], len(audit['numeric_features']),
                   len(audit['categorical_features']), audit['duplicate_complete_rows'],
                   audit['duplicate_feature_rows'], sum(audit['missing_counts'].values())]
            rows.append(row)
            missing.append([name, file, sum(audit['missing_counts'].values()),
                            sum(audit['nonfinite_counts'].values()), 'reject invalid source before fitting'])
        audit_md = '# ' + name + ' source audit\n\n' + table(
            ['Source', 'Rows', 'Raw columns', 'Numeric inputs', 'Categorical inputs',
             'Complete duplicates', 'Feature duplicates', 'Missing cells'], rows)
        audit_md += '\n' + '\n'.join('- ' + s for s in meta['notes'])
        audit_md += '\n\nComputed overlap and protocol details:\n\n\x60\x60\x60json\n' + json_text(
            {k: v for k, v in audits[name].items() if k != 'files'}) + '\x60\x60\x60\n'
        write_text(report / 'dataset_audit' / f'{name}.json', json_text(audits[name]))
        write_text(report / 'dataset_audit' / f'{name}.md', audit_md)
    write_text(report / 'dataset_comparison.md', '# Current preprocessing dimensions\n\n' + table(
        ['Dataset', 'Unique source records retained', 'Raw input features', 'Processed width', 'Artifact manifest SHA256'], dimensions))
    write_text(report / 'preprocessing/splits.md', '# Actual partitions\n\n' + table(
        ['Dataset', 'Split', 'Rows/segments', 'Ordered source ID SHA256'], splits))
    write_text(report / 'preprocessing/class_distribution.md', '# Actual class counts\n\n' + table(
        ['Dataset', 'Split', 'Class', 'Count'], classes))
    write_text(report / 'preprocessing/missing_values.md', '# Source missingness\n\n' + table(
        ['Dataset', 'File', 'Missing cells', 'Nonfinite predictor cells', 'Policy'], missing))
    write_text(report / 'preprocessing/feature_types.md', '# Predictor types (targets and metadata excluded)\n\n' + table(
        ['Dataset', 'Numeric', 'Categorical', 'Exclusions'], types))
    write_text(report / 'preprocessing/integrity_and_leakage.json', json_text(results))
    write_text(report / 'preprocessing/integrity_and_leakage.md',
               '# Executed artifact validation\n\nAll listed artifacts passed source reconstruction, saved-array comparison and training-only refit checks.\n\n' +
               table(['Dataset', 'Passed', 'Width', 'Checks'], [[n, r['passed'], r['feature_dimension'], ', '.join(r['checks'])] for n, r in results.items()]) +
               '\nNSL official-test feature overlap is preserved, not removed: ' + json_text(results['nsl_kdd']['overlap']) +
               '\nOPS checks same-channel segment intervals; related events across different channels remain an independence limitation.\n')
    deviations = [
        ['Benchmark scope', 'Sec.4.1/Table 1 pp.11-12: STIN, TON_IoT, Bot-IoT', 'STIN plus NSL/OPS extensions', 'Adaptation'],
        ['Train/test', 'Sec.4.2.1 p.12: 80:20', 'STIN whole-group approximate 80:20; official NSL and supplied OPS protocols', 'STIN ratio intent; other datasets adapted'],
        ['Validation', 'No separate fraction specified', '20% training-pool carve-out', 'Adaptation'],
        ['Raw scaling/encoding', 'Raw StandardScaler/one-hot recipe not established', 'Train-only StandardScaler; NSL nominal one-hot; reject invalid raw data', 'Local policy'],
        ['STIN labels/dimension', 'Table 1: 9 attack categories, 32 features', '9 attack classes, 30 predictors; ID/Label excluded', 'Feature-count terminology unresolved'],
        ['OPS inputs', 'OPS-SAT absent', '19 existing offline segment-summary columns', 'Adaptation; source extraction formulas unverified'],
        ['Projection', 'Eq.(1) p.5: MLP_t(LN(x W_e + b_e)) + E_pos; T=4,C=64', 'Tabular matrix only; H/W, embedding/view dimensions unresolved', 'Deferred; no fabricated reshape'],
        ['Clients', 'Sec.4.2.2 p.12: N=20, Dirichlet eta=.5', 'No client partition or controller in this milestone', 'Deferred'],
    ]
    write_text(report / 'preprocessing/paper_deviations.md', '# Paper versus implemented preprocessing\n\n' + table(
        ['Topic', 'Paper evidence', 'Implemented contract', 'Status'], deviations))
    write_text(report / 'preprocessing/workflow.md', '# Preprocessing workflow\n\n\x60\x60\x60mermaid\nflowchart LR\n A[Read and hash raw files] --> B[Validate schemas and labels]\n B --> C[Audit duplicate groups]\n C --> D[Split records or whole groups]\n D --> E[Fit on training only]\n E --> F[Save versioned arrays and provenance]\n F --> G[Reconstruct sources and verify artifacts]\n G --> H[Generate review reports]\n\x60\x60\x60\n')
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, (name, meta) in zip(axes, metas.items()):
        mapping = meta['schema']['label_mapping']
        labels = list(mapping)
        parts = list(meta['splits'])
        for i, split in enumerate(parts):
            values = [meta['splits'][split]['label_counts'].get(str(mapping[label]), 0) for label in labels]
            positions = [j + (i - 1) * .25 for j in range(len(labels))]
            ax.bar(positions, values, width=.25, label=split)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha='right')
        ax.set_ylabel('Segments' if name == 'opssat_ad' else 'Records')
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    target = report / 'figures/class_distribution.png'
    backup_file(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    evidence = {'artifacts': results, 'generated_from': 'validated current artifacts',
                'reports': {str(p.relative_to(report)): sha256(p) for directory in
                            ['dataset_audit', 'preprocessing', 'figures'] for p in (report / directory).rglob('*') if p.is_file()}}
    evidence['reports']['dataset_comparison.md'] = sha256(report / 'dataset_comparison.md')
    write_text(report / 'evidence_manifest.json', json_text(evidence))
    return results


if __name__ == '__main__':
    print(json_text(generate()))
