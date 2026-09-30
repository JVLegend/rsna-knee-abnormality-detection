"""#RSNA #Kaggle #Pesquisa — Raptor36 ABBA determinístico, somente diagnóstico."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h43_e03 import build as build_abba
from scripts.prepare_h43_benchmark import PARTITION_SHA

SETUP = """import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
"""
FLAGS = """# Aplicado após a definição Raptor, antes de qualquer forward Raptor.
import random
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(2026)
np.random.seed(2026)
random.seed(2026)
from pathlib import Path
import json
Path('/kaggle/working/raptor_determinism_environment.json').write_text(json.dumps({
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'cudnn_benchmark': torch.backends.cudnn.benchmark,
    'cudnn_deterministic': torch.backends.cudnn.deterministic,
    'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
    'cudnn_tf32': torch.backends.cudnn.allow_tf32,
    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    'device': torch.cuda.get_device_name(0), 'seed': 2026,
    'purpose': 'same_worker_36_study_abba_no_submission'}, indent=2))
"""


def build(raw, ids, prefetch, runtime):
    runtime = runtime.replace('12 training studies, one worker/session;', '36 training studies, one worker/session;')
    n = build_abba(raw, ids, prefetch, runtime, study_count=36)
    n['cells'][1]['source'] = SETUP + n['cells'][1]['source']
    n['cells'][-1]['source'] = FLAGS + n['cells'][-1]['source']
    n['metadata']['raptor_determinism'] = {
        'protocol': 'ABBA36_fixed_backend_v1', 'changed_from_parent': 'backend determinism flags only',
        'same_worker_control': True, 'cause_confirmed': False,
        'purpose': 'diagnostic_not_submission', 'studies': 36,
        'limitations': 'Within-worker parity does not prove cross-worker determinism or identify autotuner causally.'}
    for c in n['cells']:
        if c['cell_type'] == 'code': ast.parse(c['source'])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    partition = Path('data/processed/validation_weak_v1/manifest.json').read_bytes()
    if hashlib.sha256(partition).hexdigest() != PARTITION_SHA: raise ValueError('Frozen partition drift')
    train = {r['StudyInstanceUID'] for r in json.loads(partition)['splits']['train']}
    selection = json.loads(Path('reports/avance_av009_serial_v1/h43_benchmark_selection.json').read_text())
    if selection['series'] != 205 or not set(selection['ids']) <= train: raise ValueError('Sample drift')
    n = build(Path('reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb').read_bytes(),
              selection['ids'], Path('scripts/ordered_prefetch.py').read_text(), Path('scripts/h43_e03_runtime.py').read_text())
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
