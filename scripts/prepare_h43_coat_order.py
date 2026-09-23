"""#RSNA #Kaggle #Pesquisa — ABBA36 CoAt: conclusão livre vs ordem fixa."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.h43_coat_order import patch_coat
from scripts.prepare_h43_benchmark import PARTITION_SHA

STACK_SHA = '7e4fce5475b117e7301f2590f9edbcbd4c881fa7b0305dd2e5386e630a6e4f6d'
RUN = r'''import os, sys, subprocess, time, json, hashlib
from pathlib import Path
files = H43_RECEIPT['preflight']['files']
runtime_candidates = [Path(r['path']) for r in files.values()
                      if Path(r['path']).name == 'coatnet_resgated_ep10_top3_inference.py']
wheel_candidates = [Path(r['path']) for r in files.values() if r['path'].endswith('.whl')]
assert len(runtime_candidates) == len(wheel_candidates) == 1
original = runtime_candidates[0]
assert h43_sha(original) == COAT_SOURCE_SHA
assert H43_RECEIPT['benchmark_studies'] == 36 and H43_RECEIPT['benchmark_series'] == 205
envd = Path('/kaggle/working/coat_order_env')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--quiet',
                '--target', str(envd), str(wheel_candidates[0])], check=True)
env = dict(os.environ)
env['PYTHONPATH'] = f'{envd}:{original.parent}:' + env.get('PYTHONPATH', '')
records = []
for index, mode in enumerate(['completion', 'ordered', 'ordered', 'completion']):
    directory = Path(f'/kaggle/working/coat_order_run{index}')
    directory.mkdir(exist_ok=False)
    source = patch_coat(original.read_text(), mode)
    runtime = directory / 'coat_order_runtime.py'
    runtime.write_text(source)
    out = directory / 'coat_diagnostic.csv'
    start = time.monotonic()
    proc = subprocess.run([sys.executable, str(runtime),
        '--competition-root', os.environ['H43_BENCHMARK_ROOT'],
        '--artifact-root', str(original.parent), '--output', str(out),
        '--gpu-batch-studies', '2', '--backbone-micro-images', '8'],
        env=env, cwd=directory, capture_output=True, text=True)
    (directory / 'worker.log').write_text(proc.stdout + '\n' + proc.stderr)
    if proc.returncode: raise RuntimeError(f'CoAt run {index} failed: {proc.stderr[-1500:]}')
    r = json.loads((directory / 'coat_resgated_ep10_top3_submission_receipt.json').read_text())
    assert r['models'] == 3 and r['studies'] == 36 and r['series'] == 205
    assert r['fallback_studies'] == 0 and not r['failures']
    assert all(p['returncode'] == 0 for p in r['processes'])
    record = {'index': index, 'mode': mode, 'seconds': time.monotonic()-start,
        'source_sha256': h43_sha(runtime), 'csv_sha256': h43_sha(out),
        'prediction_sha256': r['predictions_sha256']}
    records.append(record)
    print('COAT_ORDER_RUN', record, flush=True)
Path('/kaggle/working/coat_order_abba_receipt.json').write_text(json.dumps({
    'status': 'COAT_ORDER_ABBA_COMPLETE_NOT_SUBMITTED', 'records': records,
    'original_source_sha256': COAT_SOURCE_SHA,
    'purpose': 'training36_batch_order_diagnostic_only_no_auc_no_submission'}, indent=2))
'''


def build(raw, helper, original_source):
    if hashlib.sha256(raw).hexdigest() != STACK_SHA: raise ValueError('Audited stack drift')
    for mode in ['completion', 'ordered']: patch_coat(original_source, mode)
    old = json.loads(raw)
    def cell(source, kind='code'):
        return dict(cell_type=kind, metadata={}, source=source,
                    **({'execution_count': None, 'outputs': []} if kind == 'code' else {}))
    cells = [cell('# CoAt36 — ABBA completion/ordered\n\nNever submit: V01 train only, no AUC.\n'
                  'Original CoAt: mattiaangeli/rsna-knee-coat-resgated-ep10-top3.\n', 'markdown'),
             old['cells'][3], old['cells'][4], cell(helper), cell(RUN)]
    for c in cells:
        if c['cell_type'] == 'code': ast.parse(c['source'])
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'coat_order': {'stack_sha256': STACK_SHA, 'partition_sha256': PARTITION_SHA,
                       'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
                       'sequence': ['completion', 'ordered', 'ordered', 'completion'],
                       'backend_changed': False, 'model_changed': False, 'submission': False}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    partition = Path('data/processed/validation_weak_v1/manifest.json').read_bytes()
    if hashlib.sha256(partition).hexdigest() != PARTITION_SHA: raise ValueError('Partition drift')
    selection = json.loads(Path('reports/avance_av012_serial_v1/h43_benchmark_selection.json').read_text())
    train = {r['StudyInstanceUID'] for r in json.loads(partition)['splits']['train']}
    if len(selection['ids']) != 36 or selection['series'] != 205 or not set(selection['ids']) <= train:
        raise ValueError('Sample outside frozen train')
    n = build(Path('reports/avance_av012_build/h43_deterministic36_serial_v1.ipynb').read_bytes(),
              Path('scripts/h43_coat_order.py').read_text(),
              Path('reports/avance_av013_sources/coat/coatnet_resgated_ep10_top3_inference.py').read_text())
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
