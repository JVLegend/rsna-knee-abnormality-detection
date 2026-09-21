"""#RSNA #Kaggle #Pesquisa — freeze the one-shot reserved comparison before inference."""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.prepare_m01_ablation import digest, literal
from scripts.prepare_v03_scale import V01, V01_SHA, PLANES, TARGETS

V03_BUILD=Path('reports/avance_av024_v03/v03_v1.py')
V03_SHA='3ae4654579db73f31779962b8ea25a620be67ec257522973ce0c36947e57088b'
R02_ROOT=Path('reports/avance_av026_r02_v1')
R02_AUDIT=Path('reports/avance_av026_r02/audit_v1.json')
R02_AUDIT_SHA='e6208422e628ccc13c6164c4cd3fb6c6f3e690db408e0ff52549c2051fb7d172'
R02_RECEIPT_SHA='1b863e5ab81a28ff8b8c84f0b15c8e71b5336e3ac0b88c7fb3bdc7ecf1adb7a8'
CHECKPOINTS=[
    ('control',2026,8,'e2875906f644c76ee1a6c5b8b24af08050a3e4a59b448a647e29daaece7a654d'),
    ('control',42,14,'5a099c18e7248ab498f8d135dc08c9962ca2f7043062598467045751d552f8d8'),
    ('paired_consistency',2026,8,'92aa32ae8ab84214a7dea9850cad03e378b78842c51614659522f3665b5c408b'),
    ('paired_consistency',42,14,'70c41c23451945638eefc7d75aea4667c4137b73bca8e71f1315e9b003fa8972'),
]


def reserved_rows(old, expanded, series):
    reserved=old['splits']['confirmation'];used=expanded['train']+expanded['development']
    ids=[r['StudyInstanceUID'] for r in reserved];groups={r['report_hash'] for r in reserved}
    if len(ids)!=len(set(ids)) or set(ids)&{r['StudyInstanceUID'] for r in used} or groups&{r['report_hash'] for r in used}:
        raise ValueError('Reserved split/group overlap')
    if [(r['StudyInstanceUID'],r['report_hash']) for r in reserved]!=[(r['StudyInstanceUID'],r['report_hash']) for r in expanded['reserved']]:
        raise ValueError('Reserved identity changed')
    by_id=defaultdict(list);seen=set()
    for s in series:
        key=(s['StudyInstanceUID'],s['SeriesInstanceUID'])
        if key in seen:raise ValueError('Duplicate series metadata')
        seen.add(key);by_id[key[0]].append(s)
    rows=[]
    for r in reserved:
        labels=[float(r['labels'][t]) for t in TARGETS]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in labels):raise ValueError('Invalid frozen labels')
        frozen={s['plane']:s['series_uid'] for s in r['series']}
        if len(r['series'])!=3 or set(frozen)!=set(PLANES):raise ValueError('Frozen plane coverage')
        chosen=[]
        for plane in PLANES:
            options=[s for s in by_id[r['StudyInstanceUID']] if s['Anatomical_Plane']==plane]
            if not options:raise ValueError('Missing plane')
            options.sort(key=lambda s:(-int(s['Fluid_Sensitive']),-int(s['Fat_Suppression']),s['SeriesInstanceUID']))
            uid=options[0]['SeriesInstanceUID']
            if uid!=frozen[plane]:raise ValueError('Series policy differs from frozen V01')
            chosen.append({'plane':plane,'series_uid':uid})
        rows.append({'StudyInstanceUID':r['StudyInstanceUID'],'report_hash':r['report_hash'],'labels':labels,'series':chosen})
    return rows


def assemble():
    if digest(V01)!=V01_SHA or digest(V03_BUILD)!=V03_SHA or digest(R02_AUDIT)!=R02_AUDIT_SHA:
        raise ValueError('Frozen source/audit drift')
    receipt_path=R02_ROOT/'r02_receipt.json'
    if digest(receipt_path)!=R02_RECEIPT_SHA:raise ValueError('Receipt changed')
    audit=json.loads(R02_AUDIT.read_text());receipt=json.loads(receipt_path.read_text())
    if audit['status']!='PASSED_R02_TRAINING_AUDIT' or audit['selected_reference']!='paired_consistency':
        raise ValueError('Candidate not approved in development')
    source=V03_BUILD.read_text();v03=literal(source,'V03');m=v03['manifest'];old=json.loads(V01.read_text())
    for path,sha in m['sources'].items():
        if digest(Path(path))!=sha:raise ValueError('Metadata source drift')
    with Path('data/raw/train_series.csv').open(newline='') as f:series=list(csv.DictReader(f))
    rows=reserved_rows(old,m,series)
    if len(rows)!=150:raise ValueError('Reserved count changed')
    g01=Path('reports/avance_av023_g01_v1/g01_geometry.json')
    extra=Path('reports/avance_av024_v03_v1/v03_geometry.json')
    v03_receipt_path=Path('reports/avance_av024_v03_v1/v03_receipt.json')
    if digest(v03_receipt_path)!=receipt['spec']['baseline_receipt_sha256']:raise ValueError('V03 receipt drift')
    v03_receipt=json.loads(v03_receipt_path.read_text())
    if digest(g01)!=v03['geometry_sha256'] or digest(extra)!=v03_receipt['geometry_sha256']:
        raise ValueError('Training geometry drift')
    first=json.loads(g01.read_text())['series'];new=json.loads(extra.read_text())
    old_series={(r['StudyInstanceUID'],s['plane']):s['series_uid'] for r in old['splits']['train']+old['splits']['development'] for s in r['series']}
    expected_old=[(r['StudyInstanceUID'],old_series[r['StudyInstanceUID'],p],p) for r in m['train'][:299]+m['development'] for p in PLANES]
    if [(s['study'],s['series'],s['plane']) for s in first]!=expected_old:raise ValueError('G01 order drift')
    expected_new=[(r['StudyInstanceUID'],s['series_uid'],s['plane']) for r in m['train'][299:] for s in r['series']]
    if [(s['study'],s['series'],s['plane']) for s in new]!=expected_new:raise ValueError('V03 geometry identity drift')
    forbidden=sorted({s['arms']['physical_adjacent']['pixel_sha256'] for s in first}|{s['pixel_sha256'] for s in new})
    checkpoints=[]
    for arm,seed,epoch,sha in CHECKPOINTS:
        p=R02_ROOT/arm/f'v02_seed{seed}_best.pt'
        s=next(s for s in receipt['seeds'] if (s['arm'],s['seed'])==(arm,seed))
        if digest(p)!=sha or s['best_checkpoint_sha256']!=sha or s['best_epoch']!=epoch:raise ValueError('Checkpoint drift')
        checkpoints.append({'arm':arm,'seed':seed,'epoch':epoch,'sha256':sha,'fingerprint':s['fingerprint']})
    runtime=Path('scripts/v04_confirmation_runtime.py').read_text()
    spec={'name':'V04_reserved_confirmation_v1','rows':rows,'targets':TARGETS,'planes':PLANES,
          'v01_sha256':V01_SHA,'v03_build_sha256':V03_SHA,'r02_audit_sha256':R02_AUDIT_SHA,
          'r02_receipt_sha256':R02_RECEIPT_SHA,'checkpoints':checkpoints,'forbidden_pixel_hashes':forbidden,
          'forbidden_series_count':len(first)+len(new),'g01_geometry_sha256':digest(g01),'v03_geometry_sha256':digest(extra),
          'model_sha256':'1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050',
          'config_sha256':'1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
          'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest(),
          'bootstrap_replicates':5000,'bootstrap_seed':20260921,'margin':2e-6,'max_target_regression':.01,
          'timeout_seconds':1200,'guard_seconds':1050,'submission_eligible':False}
    wanted={'sha','normalize_cache_compatible','StudyAttention','discover_inputs','physical_plan'}
    definitions=[ast.get_source_segment(source,n) for n in ast.parse(source).body
                 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    if len(definitions)!=len(wanted):raise ValueError('Frozen helper inventory')
    imports="import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nos.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\nimport hashlib,json,time\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\n"
    result=imports+'\nV04 = '+repr(spec)+'\nCACHE_SOURCE = '+repr(literal(source,'CACHE_SOURCE'))+'\n'+'\n\n'.join(definitions)+'\n'+runtime
    ast.parse(result)
    if len(result.encode())>=1_000_000:raise ValueError('Source exceeds Kaggle limit')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=assemble();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(source)
    print(json.dumps({'path':str(a.output),'sha256':digest(a.output),'bytes':len(source.encode())}))
