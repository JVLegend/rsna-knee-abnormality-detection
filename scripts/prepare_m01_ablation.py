"""#RSNA #Kaggle #Pesquisa — build a private, frozen-feature head ablation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE_BUILD = Path('reports/avance_av021_v02/baseline_v1.py')
BASE_SHA = '0049a9b0dd59285a621eab8fd54af776a85b68a5d4220bbbf6d4a7a4bf19ef85'
FEATURE_SHA = 'd5ad2044ce6991959183a00c24fc3a0168ca98356df0222438f6023610d63d87'
FEATURE_RAW_SHA = '395832fb64f3c3a9f6219c92a175612298d8f37d6419370259c2586a492bb7f7'
BASE_CONTRACT = '1167612d930940ee3e2c1d3ec795620de3e0fde89cad82ea0dd4514f8fb07234'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def literal(source, name):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name)


def assemble():
    if digest(BASE_BUILD) != BASE_SHA:
        raise ValueError('Audited baseline source changed')
    audit = json.loads(Path('reports/avance_av021_v02/baseline_audit_v1.json').read_text())
    if audit['status'] != 'PASSED_V02_BASELINE_AUDIT' or audit['build_sha256'] != BASE_SHA:
        raise ValueError('Baseline not approved')
    original = BASE_BUILD.read_text()
    payload = literal(original, 'V02_BASELINE')
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != BASE_CONTRACT:
        raise ValueError('Baseline contract changed')
    runtime = Path('scripts/m01_ablation_runtime.py').read_text()
    spec = {'name': 'M01_pooling_v1', 'architectures': ['shared', 'mean', 'target'],
            'baseline_build_sha256': BASE_SHA, 'baseline_contract': BASE_CONTRACT,
            'feature_sha256': FEATURE_SHA, 'feature_fingerprint': FEATURE_RAW_SHA,
            'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
            'decision': 'beat_shared_both_seeds_by_2e-6_then_lowest_mean_tie_mean',
            'confirmation_evaluated': False, 'submission_eligible': False}
    wanted = {'sha', 'StudyAttention', 'metrics', 'atomic_save', 'train_seed'}
    selected = [ast.get_source_segment(original, n) for n in ast.parse(original).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if len(selected) != len(wanted):
        raise ValueError('Ambiguous baseline definitions')
    imports = '''import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
'''
    source = (imports + '\nV02_BASELINE = ' + repr(payload) + '\nM01 = ' + repr(spec)
              + '\n' + '\n\n'.join(selected) + '\nSharedAttention = StudyAttention\n'
              + runtime)
    ast.parse(source)
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = assemble()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(source)
    print(json.dumps({'path': str(args.output), 'sha256': digest(args.output)}))
