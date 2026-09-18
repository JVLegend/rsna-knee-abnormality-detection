"""#RSNA #Kaggle #Pesquisa — freeze G01 geometry experiment before evaluation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_m01_ablation import BASE_BUILD, BASE_SHA, BASE_CONTRACT, FEATURE_SHA, FEATURE_RAW_SHA, digest, literal


def assemble():
    if digest(BASE_BUILD) != BASE_SHA:
        raise ValueError('Audited V02 source changed')
    original = BASE_BUILD.read_text()
    payload = literal(original, 'V02_BASELINE')
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != BASE_CONTRACT:
        raise ValueError('V02 contract changed')
    cache = Path('reports/avance_av021_v02/cache_train_dev_v1.json')
    if digest(cache) != '129389226d6a270ca86690bfca509c7da4a8c31cca1b5d05896378da55112211':
        raise ValueError('Audited cache manifest changed')
    records = json.loads(cache.read_text())['splits']
    counts = {r['StudyInstanceUID']+'/'+s['series_uid']: s['slices_recorded']
              for rows in records.values() for r in rows for s in r['series']}
    geometry = Path('src/rsna_knee_baseline/physical_triplets.py').read_text()
    runtime = Path('scripts/g01_ablation_runtime.py').read_text()
    spec = {'name': 'G01_physical_adjacency_v1',
            'arms': ['control', 'physical_quartiles', 'physical_adjacent'],
            'feature_sha256': FEATURE_SHA, 'feature_fingerprint': FEATURE_RAW_SHA,
            'baseline_contract': BASE_CONTRACT, 'baseline_build_sha256': BASE_SHA,
            'geometry_sha256': hashlib.sha256(geometry.encode()).hexdigest(),
            'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
            'series_counts': counts, 'confirmation_evaluated': False, 'submission_eligible': False,
            'decision': 'beat_control_both_seeds_by_2e-6_then_mean_tie_quartiles'}
    names = {'sha', 'normalize_cache_compatible', 'StudyAttention', 'metrics', 'atomic_save', 'train_seed', 'discover_inputs'}
    selected = [ast.get_source_segment(original, n) for n in ast.parse(original).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(selected) != len(names): raise ValueError('Ambiguous audited definitions')
    imports = '''import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
'''
    source = (imports + '\nV02_BASELINE = '+repr(payload)+'\nG01 = '+repr(spec)
              + '\nCACHE_SOURCE = '+repr(literal(original, 'CACHE_SOURCE'))+'\n'
              + '\n\n'.join(selected)+'\n'+geometry+'\n'+runtime)
    ast.parse(source)
    return source


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();source = assemble();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(source)
    print(json.dumps({'path':str(a.output),'sha256':digest(a.output)}))
