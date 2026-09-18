"""#RSNA #Kaggle #Pesquisa — expand train groups without touching reserved splits."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.freeze_weak_validation import TARGETS, KEY, group_hash, rows_by_id, digest, freeze
from scripts.prepare_m01_ablation import literal, BASE_BUILD, BASE_SHA

PLANES = ['Sagittal','Coronal','Axial']
V01 = Path('data/processed/validation_weak_v1/manifest.json')
V01_SHA = '365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df'
G01_BUILD = Path('reports/avance_av023_g01/g01_v2.py')
G01_SHA = '986a3ffbb16bc3cb489e320496cb93346703e32e616ff70e32c39883dc23fbc3'
FEATURE_SHA = 'd39efed5caa5d24e3d983c335b1f25b0db2fe98e8bdf1701407fef65b9feaad7'
GEOMETRY_SHA = '349b57f47cc3f6cddddcb9ebc16b90c8f110c149501b4a0c53e507c27e45d3fa'


def select_pool(train, teacher, series, splits, target=1000):
    existing=splits['train'];reserved=splits['development']+splits['confirmation']
    reserved_ids={r[KEY] for r in reserved};reserved_groups={r['report_hash'] for r in reserved}
    old_ids={r[KEY] for r in existing};old_groups={r['report_hash'] for r in existing}
    if old_ids & reserved_ids or old_groups & reserved_groups:raise ValueError('Frozen splits overlap')
    if target<=len(existing):raise ValueError('Scale target must exceed original train')
    hashes={}
    for uid,row in train.items():
        try:hashes[uid]=group_hash(row['Report'])
        except ValueError:pass
    gold={uid for uid,row in train.items() if any(row[t].strip() for t in TARGETS)}
    gold_groups={hashes[uid] for uid in gold if uid in hashes}
    grouped=defaultdict(list);seen=set()
    for s in series:
        key=(s[KEY],s['SeriesInstanceUID'])
        if key in seen:raise ValueError('Duplicate series metadata')
        seen.add(key);grouped[s[KEY]].append(s)
    def compact(uid):
        labels=[float(teacher[uid][t]) for t in TARGETS]
        if any(not math.isfinite(x) or not 0<=x<=1 for x in labels):raise ValueError('Invalid labels')
        chosen=[]
        for plane in PLANES:
            options=[s for s in grouped[uid] if s['Anatomical_Plane']==plane]
            if not options:raise ValueError('Missing plane')
            options.sort(key=lambda s:(-int(s['Fluid_Sensitive']),-int(s['Fat_Suppression']),s['SeriesInstanceUID']))
            chosen.append({'plane':plane,'series_uid':options[0]['SeriesInstanceUID']})
        return {KEY:uid,'report_hash':hashes[uid],'labels':labels,'series':chosen}
    eligible=defaultdict(list);excluded=Counter()
    for uid in sorted(train):
        if uid in reserved_ids or hashes.get(uid) in reserved_groups:reason='reserved_id_or_group'
        elif uid in gold or hashes.get(uid) in gold_groups:reason='gold_id_or_group'
        elif uid not in hashes:reason='blank_report'
        elif uid not in teacher:reason='missing_teacher'
        else:
            try:
                row=compact(uid);eligible[row['report_hash']].append(row);continue
            except (ValueError,KeyError,TypeError):reason='invalid_labels_or_series'
        excluded[reason]+=1
    candidates={r[KEY]:r for group in eligible.values() for r in group}
    original=[]
    for r in existing:
        uid=r[KEY]
        if uid not in candidates or candidates[uid]['report_hash']!=r['report_hash']:
            raise ValueError('Original train not eligible')
        if candidates[uid]['labels']!=[r['labels'][t] for t in TARGETS]:raise ValueError('Teacher changed')
        original.append(candidates[uid])
    additional=[]
    order=sorted(eligible,key=lambda g:(g not in old_groups,hashlib.sha256(('20260918:'+g).encode()).hexdigest()))
    for group in order:
        if group not in old_groups and len(original)+len(additional)>=target:break
        additional.extend(r for r in eligible[group] if r[KEY] not in old_ids)
    if len(original)+len(additional)<target:raise ValueError('Insufficient eligible training pool')
    return {'train':original+additional,'original_count':len(original),
            'eligible_count':len(candidates),'excluded_counts':dict(excluded)}


def manifest(output):
    if digest(V01)!=V01_SHA:raise ValueError('V01 changed')
    old=json.loads(V01.read_text())
    for name in ['train','teacher']:
        if digest(Path(old['sources'][name]['path']))!=old['sources'][name]['sha256']:raise ValueError('Input drift')
    train_path=Path(old['sources']['train']['path']);teacher_path=Path(old['sources']['teacher']['path'])
    series_path=Path('data/raw/train_series.csv')
    with series_path.open(newline='') as f:series=list(csv.DictReader(f))
    result=select_pool(rows_by_id(train_path),rows_by_id(teacher_path),series,old['splits'])
    result.update(format='V03_scale1000_v1',selection_seed=20260918,targets=TARGETS,
                  development=[{KEY:r[KEY],'report_hash':r['report_hash'],'labels':[r['labels'][t] for t in TARGETS]}
                               for r in old['splits']['development']],
                  reserved=[{KEY:r[KEY],'report_hash':r['report_hash']} for r in old['splits']['confirmation']],
                  sources={str(p):digest(p) for p in [V01,train_path,teacher_path,series_path]},
                  confirmation_evaluated=False,submission_eligible=False)
    if result['original_count']!=299 or len(result['development'])!=250 or len(result['reserved'])!=150:
        raise ValueError('Frozen split size drift')
    freeze(output,result)
    print(json.dumps({'train':len(result['train']),'extra':len(result['train'])-299,
                      'eligible':result['eligible_count'],'excluded':result['excluded_counts'],'sha256':digest(output)}))


def assemble(manifest_path):
    data=json.loads(manifest_path.read_text())
    for p,h in data['sources'].items():
        if digest(Path(p))!=h:raise ValueError('Manifest source drift')
    if digest(BASE_BUILD)!=BASE_SHA or digest(G01_BUILD)!=G01_SHA:raise ValueError('Audited source changed')
    if digest(V01)!=V01_SHA:raise ValueError('Frozen split changed')
    old=json.loads(V01.read_text())
    with Path('data/raw/train_series.csv').open(newline='') as f:series=list(csv.DictReader(f))
    selected=select_pool(rows_by_id(Path(old['sources']['train']['path'])),
                         rows_by_id(Path(old['sources']['teacher']['path'])),series,old['splits'])
    if data['train']!=selected['train'] or data['original_count']!=299:
        raise ValueError('Frozen training selection changed')
    for key,split in [('development','development'),('reserved','confirmation')]:
        if [(r[KEY],r['report_hash']) for r in data[key]]!=[(r[KEY],r['report_hash']) for r in old['splits'][split]]:
            raise ValueError('Reserved identity changed')
    if any(r['labels']!=[oldr['labels'][t] for t in TARGETS] for r,oldr in zip(data['development'],old['splits']['development'])):
        raise ValueError('Development teacher changed')
    audit=json.loads(Path('reports/avance_av023_g01/g01_audit_v1.json').read_text())
    if audit['status']!='PASSED_G01_AUDIT' or audit['selected_reference']!='physical_adjacent':raise ValueError('Adjacent reference not approved')
    baseline=BASE_BUILD.read_text();g01=G01_BUILD.read_text();original=literal(baseline,'V02_BASELINE')
    spec=dict(original['spec'],timeout_seconds=1200)
    v02={'spec':spec,'splits':{'development':data['development']},'targets':TARGETS}
    runtime=Path('scripts/v03_scale_runtime.py').read_text()
    v03={'manifest':data,'manifest_sha256':digest(manifest_path),'feature_sha256':FEATURE_SHA,
         'geometry_sha256':GEOMETRY_SHA,'g01_build_sha256':G01_SHA,'arms':['control','expanded'],
         'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest()}
    wanted={'sha','normalize_cache_compatible','StudyAttention','metrics','atomic_save','train_seed','discover_inputs'}
    selected=[ast.get_source_segment(baseline,n) for n in ast.parse(baseline).body
              if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    geometry=next(ast.get_source_segment(g01,n) for n in ast.parse(g01).body if isinstance(n,ast.FunctionDef) and n.name=='physical_plan')
    imports='''import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
'''
    source=(imports+'\nV02_BASELINE = '+repr(v02)+'\nV03 = '+repr(v03)+'\nCACHE_SOURCE = '
            +repr(literal(baseline,'CACHE_SOURCE'))+'\n'+'\n\n'.join(selected)+'\n'+geometry+'\n'+runtime)
    ast.parse(source)
    if len(source.encode())>=1_000_000:raise ValueError('Kaggle source limit')
    return source


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['manifest','build'])
    p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='manifest':manifest(a.output)
    else:
        source=assemble(a.manifest);a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:f.write(source)
        print(json.dumps({'sha256':digest(a.output),'bytes':len(source.encode())}))
