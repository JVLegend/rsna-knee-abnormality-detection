"""#RSNA #Kaggle #Pesquisa — assemble private train-only engineering pilot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.preflight_v02_cache import MANIFEST_SHA, PLANES
from scripts.freeze_weak_validation import digest

CONFIG_SHA='1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'


def validate(audit):
    if audit.get('status')!='PASSED_TRAIN_CACHE_PIXELS' or audit.get('manifest_sha256')!=MANIFEST_SHA:
        raise ValueError('Approved V01 cache audit required')
    if (audit.get('train_studies'),audit.get('series'))!=(299,897):raise ValueError('Training coverage drift')
    if audit.get('development_pixels_read')!=0 or audit.get('confirmation_pixels_read')!=0:
        raise ValueError('Reserved split read by pilot')
    pilot=audit['pilot'];train={r['StudyInstanceUID']:r for r in audit['studies']}
    if len(train)!=299 or len(audit['studies'])!=299 or sum(len(r['series']) for r in train.values())!=897:
        raise ValueError('Training record inventory incomplete')
    if len(pilot)!=12 or len({r['StudyInstanceUID'] for r in pilot})!=12:raise ValueError('Pilot IDs drift')
    for r in pilot:
        if train.get(r['StudyInstanceUID'])!=r:raise ValueError('Pilot not exact train subset')
        if [s['plane'] for s in r['series']]!=list(PLANES):raise ValueError('Pilot plane order drift')
        if len(r['labels'])!=12 or any(not 0<=v<=1 for v in r['labels']):raise ValueError('Invalid teacher labels')
    return pilot


def build(audit,helper,runtime):
    pilot=validate(audit)
    payload={'manifest_sha256':MANIFEST_SHA,'config_sha256':CONFIG_SHA,'pilot':pilot,
        'input_contract':{'image_size':224,'channels':'quantiles_0.25_0.50_0.75',
            'plane_order':list(PLANES),'normalization':'uint8/255 then ImageNet mean/std',
            'encoder':'official DINOv2-S/14 frozen fp32 CLS, no competition head',
            'head':'shared study attention 384->64->1, classifier384->12',
            'loss':'BCEWithLogits unmodified soft teacher including 0.5',
            'pilot_only':True,'validation_used':False},
        'helper_sha256':hashlib.sha256(helper.encode()).hexdigest(),
        'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest()}
    source='# Private generated train-only pilot; never submit.\nV02_PILOT = '+repr(payload)+'\n'
    source+='CACHE_SOURCE = '+repr(helper)+'\n'+runtime
    ast.parse(source)
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.audit.read_text())
    manifest=Path('data/processed/validation_weak_v1/manifest.json')
    if digest(manifest)!=MANIFEST_SHA:raise ValueError('Frozen manifest drift')
    m=json.loads(manifest.read_text());root=Path(m['sources']['cache_index']['path']).parent
    train={s['StudyInstanceUID']:s for s in m['splits']['train']}
    for s in validate(audit):
        original=train[s['StudyInstanceUID']]
        from scripts.freeze_weak_validation import TARGETS
        if s['labels']!=[original['labels'][t] for t in TARGETS]:raise ValueError('Teacher labels drift')
        for r in s['series']:
            path=(root/r['array_path']).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path)!=r['npz_sha256']:raise ValueError('Pilot cache changed')
    body=build(audit,Path('scripts/build_dicom_25d_features.py').read_text(),Path('scripts/v02_pilot_runtime.py').read_text())
    if a.output.exists() and a.output.read_text()!=body:raise ValueError('Refuse different build overwrite')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not a.output.exists():
        with a.output.open('x') as f:f.write(body)
    print(json.dumps({'path':str(a.output),'sha256':digest(a.output),'studies':12,'series':36,'submission_eligible':False},indent=2))


if __name__=='__main__':main()
