"""#RSNA #Kaggle #Pesquisa — train-only resolution compatibility/cost pilot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_v04_confirmation import V03_BUILD, V03_SHA, literal, digest
from scripts.prepare_v05_validation import build as validation_build
from scripts.prepare_r02_training import FEATURE_SHA

V05 = Path('data/processed/validation_weak_v5/manifest.json')
V05_SHA = '9bb462aee209c132af51073851525e885545c76627fe542c0248361673e6de84'


def select_training(rows, count=20):
    if count < 2 or len(rows) < count or len({r['StudyInstanceUID'] for r in rows}) != len(rows):
        raise ValueError('Invalid training pilot population')
    selected = sorted(rows, key=lambda r: hashlib.sha256(('G04:20260921:'+r['StudyInstanceUID']).encode()).hexdigest())[:count]
    return [{'StudyInstanceUID': r['StudyInstanceUID'], 'series': r['series']} for r in selected]


def assemble():
    if digest(V05) != V05_SHA or digest(V03_BUILD) != V03_SHA:
        raise ValueError('Source drift')
    v05 = json.loads(V05.read_text())
    if v05 != validation_build() or not v05['class_gate_passed']:
        raise ValueError('Validation contract drift/coverage gate')
    source = V03_BUILD.read_text(); v03 = literal(source, 'V03')
    if v05['splits']['train'] != v03['manifest']['train']:
        raise ValueError('Training changed')
    rows = select_training(v05['splits']['train'])
    receipt = Path('reports/avance_av024_v03_v1/v03_receipt.json')
    r = json.loads(receipt.read_text())
    from scripts.prepare_v04_confirmation import R02_ROOT, R02_RECEIPT_SHA
    rp = R02_ROOT/'r02_receipt.json'
    if digest(rp) != R02_RECEIPT_SHA or digest(receipt) != json.loads(rp.read_text())['spec']['baseline_receipt_sha256']:
        raise ValueError('Receipt drift')
    first = Path('reports/avance_av023_g01_v1/g01_geometry.json')
    extra = Path('reports/avance_av024_v03_v1/v03_geometry.json')
    if digest(first) != v03['geometry_sha256'] or digest(extra) != r['geometry_sha256']:
        raise ValueError('Geometry drift')
    geom = {(g['study'], g['series']): {'pixel_sha256': g['arms']['physical_adjacent']['pixel_sha256'],
             'selected': g['arms']['physical_adjacent']} for g in json.loads(first.read_text())['series']}
    geom.update({(g['study'], g['series']): {'pixel_sha256': g['pixel_sha256'], 'selected': g['selected']}
                 for g in json.loads(extra.read_text())})
    expected = [geom[row['StudyInstanceUID'], s['series_uid']] for row in rows for s in row['series']]
    runtime = Path('scripts/g04_preflight_runtime.py').read_text()
    spec = {'name': 'G04_train_only_resolution_preflight_v1', 'rows': rows, 'expected_geometry': expected,
            'v05_sha256': V05_SHA, 'v03_build_sha256': V03_SHA, 'feature_sha256': FEATURE_SHA,
            'model_sha256': '1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050',
            'config_sha256': '1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
            'resolutions': [224, 336], 'batch_studies': 2, 'atol': 1e-4, 'rtol': 1e-5,
            'guard_seconds': 360, 'timeout_seconds': 480, 'confirmation_evaluated': False,
            'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest()}
    wanted = {'sha', 'normalize_cache_compatible', 'physical_plan', 'discover_inputs'}
    definitions = [ast.get_source_segment(source, n) for n in ast.parse(source).body
                   if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(definitions) != len(wanted): raise ValueError('Helper inventory drift')
    imports = "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nos.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\nimport hashlib,json,time\nfrom pathlib import Path\nimport numpy as np\nimport torch\n"
    result = imports+'\nG04 = '+repr(spec)+'\nCACHE_SOURCE = '+repr(literal(source, 'CACHE_SOURCE'))+'\n'+'\n\n'.join(definitions)+'\n'+runtime
    ast.parse(result)
    if len(result.encode()) >= 1_000_000: raise ValueError('Source budget exceeded')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    source = assemble(); a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: f.write(source)
    print(json.dumps({'sha256': digest(a.output), 'bytes': len(source.encode())}))
