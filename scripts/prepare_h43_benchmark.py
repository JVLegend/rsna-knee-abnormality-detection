#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — benchmark sem labels/score, separado da submissão."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_h43_parent import replace_once

PARTITION_SHA = '365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df'
PILOT_SHA = 'a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306'
BENCH_ROOT = '/kaggle/working/h43_benchmark_input'


def select_studies(ids, counts, n=36):
    if len(set(ids)) != len(ids) or not 2 <= n <= len(ids):
        raise ValueError('Invalid/duplicate IDs or benchmark size')
    if any(counts.get(uid, 0) <= 0 for uid in ids):
        raise ValueError('Missing series metadata')
    order = sorted(ids, key=lambda uid: (counts[uid], hashlib.sha256(('2026:' + uid).encode()).hexdigest()))
    # Estratificação sistemática na quantidade de séries, inclui extremos.
    indices = [round(i * (len(order) - 1) / (n - 1)) for i in range(n)]
    return [order[i] for i in indices]


def benchmark_setup(selected):
    return f'''# Apenas artefatos de benchmark no working; input original não é alterado.
import csv as _bcsv, json as _bjson, os as _bos
from pathlib import Path as _BPath
_B_IDS = {selected!r}
_B_ROOT = _BPath({BENCH_ROOT!r})
_B_SOURCE = next(p for p in (
    _BPath('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    _BPath('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train_series.csv').is_file())
_B_ROOT.mkdir(exist_ok=False)
(_B_ROOT / 'test_series').mkdir()
for _uid in _B_IDS:
    _src = _B_SOURCE / 'train_series' / _uid
    if not _src.is_dir():
        raise RuntimeError('Missing benchmark study: ' + _uid)
    (_B_ROOT / 'test_series' / _uid).symlink_to(_src, target_is_directory=True)
with (_B_ROOT / 'test.csv').open('w', newline='') as _f:
    _w = _bcsv.writer(_f); _w.writerow(['StudyInstanceUID']); _w.writerows([u] for u in _B_IDS)
with (_B_SOURCE / 'train_series.csv').open() as _f:
    _r = _bcsv.DictReader(_f); _fields = _r.fieldnames
    _series = [r for r in _r if r['StudyInstanceUID'] in set(_B_IDS)]
with (_B_ROOT / 'test_series.csv').open('w', newline='') as _f:
    _w = _bcsv.DictWriter(_f, fieldnames=_fields); _w.writeheader(); _w.writerows(_series)
with (_B_SOURCE / 'sample_submission.csv').open() as _f:
    _fields = next(_bcsv.reader(_f))
with (_B_ROOT / 'sample_submission.csv').open('w', newline='') as _f:
    _w = _bcsv.writer(_f); _w.writerow(_fields)
    _w.writerows([u] + [0.5] * (len(_fields)-1) for u in _B_IDS)
for _name in ['train.csv', 'train_series.csv']:
    (_B_ROOT / _name).symlink_to(_B_SOURCE / _name)
_bos.environ['H43_BENCHMARK_ROOT'] = str(_B_ROOT)
H43_RECEIPT['purpose'] = 'runtime_benchmark_only_no_auc_no_submission'
H43_RECEIPT['benchmark_studies'] = len(_B_IDS)
H43_RECEIPT['benchmark_series'] = len(_series)
_BPath('/kaggle/working/h43_benchmark_selection.json').write_text(_bjson.dumps(
    {{'ids': _B_IDS, 'series': len(_series), 'uses_labels_for_selection': False,
      'partition': 'V01 train only', 'test_root': str(_B_ROOT)}}, indent=2))
print('H43_BENCHMARK_INPUT', len(_B_IDS), len(_series), flush=True)
'''


def adapt_notebook(raw, selected):
    if hashlib.sha256(raw).hexdigest() != PILOT_SHA:
        raise ValueError('Pilot build drift')
    notebook = json.loads(raw)
    cells = notebook['cells']
    # Offsets incluem a célula de integridade inserida antes da config original.
    for index in [6, 7, 8, 9, 10, 12, 14]:
        cells[index]['source'] = '# EDA disabled for runtime benchmark; no model change.\n'
    cells[23]['source'] = replace_once(cells[23]['source'], 'def find_root():\n',
        "def find_root():\n    return Path(os.environ['H43_BENCHMARK_ROOT'])\n")
    cells[39]['source'] = replace_once(cells[39]['source'],
        "COMP = _find_dir('rsna-knee-abnormality-detection')",
        "COMP = Path(os.environ['H43_BENCHMARK_ROOT'])")
    source = cells[48]['source']
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in n.targets))
    raptor = replace_once(ast.literal_eval(node.value), 'def find_test_root():\n',
                          "def find_test_root():\n    return os.environ['H43_BENCHMARK_ROOT']\n")
    cells[48]['source'] = '_KE_SRC = ' + repr(raptor) + '\n'
    cells[49]['source'] = replace_once(cells[49]['source'],
        'competition_root=rt.base.find_competition_root(),',
        f'competition_root=Path({BENCH_ROOT!r}),')
    cells[-1]['source'] = replace_once(cells[-1]['source'],
        "'/kaggle/working/submission.csv'", "'/kaggle/working/benchmark_predictions.csv'")
    cells[-1]['source'] = cells[-1]['source'].replace(
        'ready for runtime/coverage review, not auto-submitted', 'BENCHMARK ONLY, never submit this kernel')
    # Setup após preflight, antes de qualquer descoberta de root da fonte.
    cells.insert(4, {'cell_type': 'code', 'metadata': {}, 'source': benchmark_setup(selected),
                     'outputs': [], 'execution_count': None})
    measured = []
    starts = {38: 'dino', 40: 'a5', 46: 'rad', 50: 'raptor_coat_fusion'}
    ends = {38: 'dino', 44: 'a5', 46: 'rad', 50: 'raptor_coat_fusion'}
    def timing_cell(source):
        return {'cell_type': 'code', 'metadata': {}, 'source': source,
                'outputs': [], 'execution_count': None}
    for i, cell in enumerate(cells):
        if i in starts:
            measured.append(timing_cell("_H43_STAGE_START = __import__('time').monotonic()\n"))
        measured.append(cell)
        if i in ends:
            measured.append(timing_cell(
                f"H43_RECEIPT.setdefault('timings_seconds', {{}})[{ends[i]!r}] = __import__('time').monotonic() - _H43_STAGE_START\n"
                "Path('/kaggle/working/h43_benchmark_timings.json').write_text(json.dumps(H43_RECEIPT['timings_seconds'], indent=2))\n"
                "print('H43_BENCHMARK_TIMINGS', H43_RECEIPT['timings_seconds'], flush=True)\n"))
    cells = notebook['cells'] = measured
    for cell in cells:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata']['h43_benchmark'] = {'studies': len(selected), 'purpose': 'runtime_only',
                                            'parent_build_sha256': PILOT_SHA,
                                            'partition_sha256': PARTITION_SHA}
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    partition = Path('data/processed/validation_weak_v1/manifest.json').read_bytes()
    if hashlib.sha256(partition).hexdigest() != PARTITION_SHA:
        raise ValueError('Frozen partition changed')
    ids = [r['StudyInstanceUID'] for r in json.loads(partition)['splits']['train']]
    with Path('data/raw/train_series.csv').open() as handle:
        counts = Counter(r['StudyInstanceUID'] for r in csv.DictReader(handle))
    selected = select_studies(ids, counts)
    raw = Path('reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb').read_bytes()
    notebook = adapt_notebook(raw, selected)
    notebook['metadata']['h43_benchmark']['series_counts'] = [counts[u] for u in selected]
    body = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse overwrite of different build')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as handle:
            handle.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **notebook['metadata']['h43_benchmark']}, indent=2))


if __name__ == '__main__':
    main()
