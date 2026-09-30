#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — E03 serial/prefetch ABBA, nunca submissão."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_benchmark import benchmark_setup, select_studies, PARTITION_SHA
from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_probe22 import PARENT_SHA


def adapt_raptor(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and ast.unparse(n.target) == '(study_index, study_uid)')
    prep = loop.body[0]
    if not isinstance(prep, ast.Try) or 'build_study(' not in ast.get_source_segment(source, prep):
        raise ValueError('Raptor loop drift')
    lines = source.splitlines(keepends=True)
    replacement = (
        '        with _e03_studies(test_ids, series, tsdir, reader, recipe_index) as prepared:\n'
        '            for study_index, study_uid, (volume, mask) in prepared:\n'
        + ''.join('    ' + line if line.strip() else line for line in lines[prep.end_lineno:loop.end_lineno]))
    source = ''.join(lines[:loop.lineno-1]) + replacement + ''.join(lines[loop.end_lineno:])
    source = replace_once(source, 'def find_test_root():\n',
        "def find_test_root():\n    return os.environ['H43_BENCHMARK_ROOT']\n")
    source = replace_once(source, '    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)',
        '    np.savez(f"/kaggle/working/e03_run{E03_RUN_INDEX}_raw.npz",\n'
        '             ids=np.asarray(test_ids), arm_probs=np.stack(arm_probs), ranks=ranks)\n'
        '    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)')
    ast.parse(source)
    return source


def build(raw, selected, prefetch, runtime, study_count=12):
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('Confirmed parent SHA drift')
    if study_count not in (12, 36) or len(selected) != study_count or len(set(selected)) != study_count:
        raise ValueError('E03 requires the selected 12 or 36 distinct training studies')
    parent = json.loads(raw)
    node = next(n for n in ast.parse(parent['cells'][48]['source']).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in n.targets))
    original = ast.literal_eval(node.value)
    raptor = adapt_raptor(original)
    def cell(source, kind='code'):
        return {'cell_type': kind, 'metadata': {}, 'source': source,
                **({'outputs': [], 'execution_count': None} if kind == 'code' else {})}
    cells = [cell('# E03 — Raptor serial vs one-ahead prefetch\n\n'
        f'Runtime-only on {study_count} V01 training studies. Never submit this notebook.\n'
        'No AUC. Parent models unchanged. Source: maverickss26/rsna-knee-0941-restructured.\n'
        + parent['cells'][0]['source'], 'markdown'),
        cell(parent['cells'][3]['source']), cell(benchmark_setup(selected)),
        cell(parent['cells'][4]['source']), cell(prefetch), cell(raptor),
        cell(runtime), cell('e03_benchmark()\n')]
    for c in cells:
        if c['cell_type'] == 'code':
            ast.parse(c['source'])
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False, 'accelerator': 'nvidiaTeslaT4'},
            'e03': {'parent_sha256': PARENT_SHA, 'partition_sha256': PARTITION_SHA,
                'raptor_original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'raptor_adapted_sha256': hashlib.sha256(raptor.encode()).hexdigest(),
                'prefetch_sha256': hashlib.sha256(prefetch.encode()).hexdigest(),
                'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
                'studies': selected, 'sequence': ['serial', 'prefetch', 'prefetch', 'serial'],
                'purpose': 'raptor_efficiency_only_no_auc_no_submission'}}, 'cells': cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    partition = Path('data/processed/validation_weak_v1/manifest.json').read_bytes()
    if hashlib.sha256(partition).hexdigest() != PARTITION_SHA:
        raise ValueError('Frozen partition drift')
    train = {r['StudyInstanceUID'] for r in json.loads(partition)['splits']['train']}
    prior = json.loads(Path('reports/avance_av004_benchmark_v1/h43_benchmark_selection.json').read_text())['ids']
    if not set(prior) <= train:
        raise ValueError('Benchmark IDs outside frozen train')
    with Path('data/raw/train_series.csv').open() as f:
        counts = Counter(r['StudyInstanceUID'] for r in csv.DictReader(f))
    selected = select_studies(prior, counts, 12)
    result = build(Path('reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb').read_bytes(),
        selected, Path('scripts/ordered_prefetch.py').read_text(), Path('scripts/h43_e03_runtime.py').read_text())
    body = json.dumps(result, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse overwrite of different artifact')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f:
            f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
        'studies': len(selected), 'series': sum(counts[u] for u in selected),
        'series_counts': [counts[u] for u in selected]}, indent=2))


if __name__ == '__main__':
    main()
