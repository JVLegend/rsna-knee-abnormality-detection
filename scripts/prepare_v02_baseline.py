"""#RSNA #Kaggle #Pesquisa — frozen train/dev cache gate and private V02 build."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.freeze_weak_validation import digest,freeze,TARGETS
from scripts.preflight_v02_cache import MANIFEST_SHA,PLANES,check_image

SPEC={'name':'V02_frozen_dino_attention_v1','seeds':[2026,42],'epochs':20,
      'batch_size':4,'learning_rate':.001,'weight_decay':.0001,
      'encoder_seed':2026,'encoder':'official_dinov2_small_frozen_fp32_cls384',
      'head':'shared_attention384_64_1_classifier384_12',
      'primary_metric':'mean_soft_BCE_on_250_development_studies',
      'selection':'minimum_development_soft_BCE; earliest_epoch_on_exact_tie',
      'comparator':'constant_per_target_mean_of_299_training_soft_labels',
      'labels':'unchanged_teacher_including_0.5','auc_measured':False,
      'confirmation_used':False,'submission_eligible':False,'timeout_seconds':1800}


def validate(payload):
    if payload['status']!='PASSED_BASELINE_CACHE' or payload['manifest_sha256']!=MANIFEST_SHA:
        raise ValueError('Approved frozen cache required')
    if payload['confirmation_pixels_read']!=0:raise ValueError('Confirmation access forbidden')
    ids=set();groups=set();hashes=set()
    for split,count in [('train',299),('development',250)]:
        rows=payload['splits'][split]
        if len(rows)!=count:raise ValueError('Coverage drift')
        new_ids={s['StudyInstanceUID'] for s in rows};new_groups={s['report_hash'] for s in rows}
        if len(new_ids)!=count or ids&new_ids or groups&new_groups:raise ValueError('Split overlap')
        current=set()
        for s in rows:
            if len(s['labels'])!=12 or any(not 0<=v<=1 for v in s['labels']):raise ValueError('Invalid teacher labels')
            if [r['plane'] for r in s['series']]!=list(PLANES):raise ValueError('Plane order drift')
            for r in s['series']:
                if len(r['selected_files'])!=3 or any(Path(f).name!=f for f in r['selected_files']):
                    raise ValueError('Invalid selected filenames')
                current.add(r['image_sha256'])
        if hashes&current:raise ValueError('Cross-split identical pixels')
        ids|=new_ids;groups|=new_groups;hashes|=current
    return payload['splits']


def audit(output):
    manifest=Path('data/processed/validation_weak_v1/manifest.json')
    if digest(manifest)!=MANIFEST_SHA:raise ValueError('Frozen split drift')
    m=json.loads(manifest.read_text())
    for source in m['sources'].values():
        if digest(Path(source['path']))!=source['sha256']:raise ValueError('Frozen input drift')
    seen_ids=set();seen_groups=set()
    for rows in m['splits'].values():
        ids={s['StudyInstanceUID'] for s in rows};groups={s['report_hash'] for s in rows}
        if len(ids)!=len(rows) or ids&seen_ids or groups&seen_groups:raise ValueError('Reserved split overlap')
        seen_ids|=ids;seen_groups|=groups
    index_path=Path(m['sources']['cache_index']['path']);index=json.loads(index_path.read_text())
    if (index['size'],index['channels'],index['quantiles'])!=(224,3,[.25,.5,.75]):raise ValueError('Cache recipe drift')
    records={(r['study_uid'],r['series_uid']):r for r in index['records']}
    if len(records)!=len(index['records']):raise ValueError('Duplicate index series')
    splits={};root=index_path.parent.resolve()
    for split in ['train','development']:
        result=[]
        for study in m['splits'][split]:
            series=[]
            for s in sorted(study['series'],key=lambda r:PLANES.index(r['plane'])):
                path=(root/s['array_path']).resolve()
                if not path.is_relative_to(root):raise ValueError('Cache path outside root')
                record=records[(study['StudyInstanceUID'],s['series_uid'])]
                if record['anatomical_plane']!=s['plane'] or record['n_slices']!=s['slices_recorded']:
                    raise ValueError('Series metadata drift')
                series.append(dict(s,**check_image(path,s,record),
                                   selected_files=[x['file'] for x in record['selected']]))
            result.append({'StudyInstanceUID':study['StudyInstanceUID'],'report_hash':study['report_hash'],
                           'labels':[study['labels'][t] for t in TARGETS],'series':series})
            if len(result)%25==0:print('cache',split,len(result),'studies',flush=True)
        splits[split]=result
    out={'status':'PASSED_BASELINE_CACHE','manifest_sha256':MANIFEST_SHA,'splits':splits,
         'targets':TARGETS,'confirmation_pixels_read':0,'development_pixels_read':750,
         'train_pixels_read':897,'spec':SPEC}
    validate(out);freeze(output,out)
    print(json.dumps({'status':out['status'],'studies':549,'series':1647,'sha256':digest(output)}))


def assemble(cache):
    payload=json.loads(cache.read_text());splits=validate(payload)
    pilot_audit=json.loads(Path('reports/avance_av020_v02/pilot_audit_v3.json').read_text())
    if (pilot_audit['status']!='PASSED_V02_PILOT_AUDIT' or
        pilot_audit['build_sha256']!='212fb597d5d9152c0c2cd061483ad939fb0d03beeb636cad89b7be9c2d0bbb18' or
        digest(Path('scripts/v02_pilot_runtime.py'))!='e2ee4fcfe6ecab1b1831cc326c11b676222aa7964b283abfa356806a684a1dc0'):
        raise ValueError('Audited pilot implementation required')
    manifest=Path('data/processed/validation_weak_v1/manifest.json')
    if digest(manifest)!=MANIFEST_SHA:raise ValueError('Frozen split drift')
    original=json.loads(manifest.read_text())
    for split,rows in splits.items():
        if [r['StudyInstanceUID'] for r in rows]!=[r['StudyInstanceUID'] for r in original['splits'][split]]:
            raise ValueError('Order drift')
        for row,source in zip(rows,original['splits'][split]):
            if row['labels']!=[source['labels'][t] for t in TARGETS] or row['report_hash']!=source['report_hash']:
                raise ValueError('Teacher/group drift')
    # Preserve the exact audited pilot functions; do not execute its main.
    pilot=Path('scripts/v02_pilot_runtime.py').read_text()
    common=pilot.split('\ndef main():',1)[0]
    helper=Path('scripts/build_dicom_25d_features.py').read_text()
    runtime=Path('scripts/v02_baseline_runtime.py').read_text()
    compact={name:[{'StudyInstanceUID':r['StudyInstanceUID'],'report_hash':r['report_hash'],
                   'labels':r['labels'],'series':[{k:s[k] for k in ['series_uid','plane','selected_files','image_sha256']}
                                                 for s in r['series']]} for r in rows] for name,rows in splits.items()}
    data={'spec':SPEC,'splits':compact,'targets':TARGETS,'manifest_sha256':MANIFEST_SHA,
          'cache_sha256':digest(cache),'pilot_source_sha256':hashlib.sha256(pilot.encode()).hexdigest(),
          'helper_sha256':hashlib.sha256(helper.encode()).hexdigest(),
          'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest()}
    body='# Private V02 training baseline, never submit.\nV02_BASELINE = '+repr(data)+'\nCACHE_SOURCE = '+repr(helper)+'\n'+common+'\n'+runtime
    ast.parse(body)
    return body


def build(cache,output):
    body=assemble(cache)
    if output.exists():raise ValueError('Refuse build overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(body)
    print(json.dumps({'build':str(output),'sha256':digest(output),'studies':549,'seeds':SPEC['seeds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','build'])
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='audit':audit(a.output)
    else:build(a.cache,a.output)
