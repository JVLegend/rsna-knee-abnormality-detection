"""#RSNA #Kaggle #Pesquisa — audit train-only compatibility, not model accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.prepare_g04_preflight import assemble, digest, literal, FEATURE_SHA


def check_arrays(a224, a336, reference, studies=20):
    for a in (a224, a336, reference):
        if a.shape != (studies, 3, 384) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid pilot features')
    if not np.allclose(a224, reference, atol=1e-4, rtol=1e-5):
        raise ValueError('224 baseline parity failed')
    if np.array_equal(a224, a336):
        raise ValueError('336 features identical to 224; check resolution execution')
    return float(np.abs(a224-reference).max())


def summarize_timings(timings, studies=20):
    if len(timings) != studies//2 or studies%2:
        raise ValueError('Pilot batch coverage drift')
    rows=[]
    for i,t in enumerate(timings):
        if t['begin'] != i*2 or t['studies'] != 2 or t['order'] != ([224,336] if i%2==0 else [336,224]):
            raise ValueError('Pilot ordering drift')
        values=[t['decode_and_both_resizes_seconds'], t['inference_seconds']['224'], t['inference_seconds']['336']]
        if not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Invalid timing')
        peaks=[t['peak_allocated_bytes'][str(s)] for s in [224,336]]
        if any(not isinstance(p,int) or not 0<p<14*1024**3 for p in peaks):
            raise ValueError('Insufficient VRAM margin')
        rows.append(np.array(values)/2)
    per=np.stack(rows)
    # Both sizes on all 1300 overestimates the planned 224 reuse, but cold I/O can still be slower.
    planning=1.5*1300*float(per.sum(1).max())+420
    return {'mean_decode_both_resizes_seconds_per_study':float(per[:,0].mean()),
            'mean_forward224_seconds_per_study':float(per[:,1].mean()),
            'mean_forward336_seconds_per_study':float(per[:,2].mean()),
            'forward336_over224':float(per[:,2].sum()/per[:,1].sum()),
            'max_measured_total_seconds_per_study':float(per.sum(1).max()),
            'peak_allocated_bytes':{str(s):max(t['peak_allocated_bytes'][str(s)] for t in timings) for s in [224,336]},
            'planning_1300_both_resolutions_seconds':planning,
            'planning_warning':'20 training cases only; cold I/O and larger jobs can exceed this extrapolation. Not a runtime guarantee.'}


def assess(directory, build, output):
    if output.exists(): raise FileExistsError(output)
    source=build.read_text()
    if source!=assemble(): raise ValueError('Frozen source drift')
    spec=literal(source,'G04'); contract=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    receipt_path=directory/'g04_receipt.json'; r=json.loads(receipt_path.read_text())
    if (r['status']!='COMPLETE_G04_PREFLIGHT_NOT_MODEL_VALIDATION' or r['spec']!=spec or r['contract_hash']!=contract
        or r['gpu_names']!=['Tesla T4']*2 or any(r[k] for k in ['training_performed','head_predictions_generated','confirmation_evaluated','submission_eligible'])):
        raise ValueError('Receipt scope drift')
    if digest(directory/'g04_features.npz')!=r['feature_sha256'] or digest(directory/'g04_geometry.json')!=r['geometry_sha256']:
        raise ValueError('Output SHA drift')
    ids=[x['StudyInstanceUID'] for x in spec['rows']]
    with np.load(directory/'g04_features.npz',allow_pickle=False) as a:
        if a['ids'].tolist()!=ids or str(a['contract_hash'])!=contract: raise ValueError('Feature identities drift')
        a224=a['features224'].copy(); a336=a['features336'].copy()
    archive=Path('reports/avance_av024_v03_v1/v03_features.npz')
    if digest(archive)!=FEATURE_SHA: raise ValueError('Reference SHA drift')
    with np.load(archive,allow_pickle=False) as a:
        train_ids=a['train_ids'].tolist()
        reference=a['train'][[train_ids.index(uid) for uid in ids]]
    delta=check_arrays(a224,a336,reference,len(ids))
    if abs(delta-r['max_224_feature_delta'])>1e-12: raise ValueError('Parity metric drift')
    geometry=json.loads((directory/'g04_geometry.json').read_text())
    expected=[(row['StudyInstanceUID'],s['series_uid'],s['plane']) for row in spec['rows'] for s in row['series']]
    if len(geometry)!=len(expected): raise ValueError('Geometry coverage drift')
    for g,identity,p in zip(geometry,expected,spec['expected_geometry']):
        if ((g['study'],g['series'],g['plane'])!=identity or g['selected']!=p['selected']
            or g['pixel_sha256']['224']!=p['pixel_sha256'] or len(g['pixel_sha256']['336'])!=64):
            raise ValueError('Geometry/pixel baseline drift')
    result={'status':'PASSED_G04_PREFLIGHT_NOT_MODEL_VALIDATION','build_sha256':digest(build),
            'receipt_sha256':digest(receipt_path),'training_studies':len(ids),'max_224_feature_delta':delta,
            'seconds':r['seconds'],'timing':summarize_timings(r['timings'],len(ids)),
            'confirmation_evaluated':False,'development_evaluated':False,'submission_eligible':False,
            'limitations':['No accuracy test, labels or head predictions.',
                           '336 extraction not independently reproduced; provenance/shape/finitude only.',
                           'Native field of view unchanged; not a physical-mm crop experiment.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','build','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); assess(a.directory,a.build,a.output)
