"""#RSNA #Kaggle #Pesquisa — independent V03 feature/selection/checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_v03_scale import assemble,digest,literal
from scripts.assess_v02_baseline import replay,loss


def assess(directory,build,manifest,baseline,output):
    source=build.read_text()
    if source!=assemble(manifest):raise ValueError('Build/manifest drift')
    spec=literal(source,'V03');m=spec['manifest'];contract=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    r=json.loads((directory/'v03_receipt.json').read_text());count=len(m['train'])
    if (r['status']!='COMPLETE_V03_NOT_SUBMISSION' or r['contract_hash']!=contract or r['spec']!=spec
        or r['confirmation_evaluated'] or r['submission_eligible'] or r['gpu_names']!=['Tesla T4']*2
        or r['training_studies']!=count or r['extra_studies']!=count-299 or r['development_studies']!=250):
        raise ValueError('Receipt scope drift')
    fp=directory/'v03_features.npz';gp=directory/'v03_geometry.json'
    if digest(fp)!=r['feature_sha256'] or digest(gp)!=r['geometry_sha256']:raise ValueError('Artifact drift')
    with np.load(fp,allow_pickle=False) as a:
        train=a['train'];dev=a['development']
        if (a['train_ids'].tolist()!=[x['StudyInstanceUID'] for x in m['train']]
            or a['development_ids'].tolist()!=[x['StudyInstanceUID'] for x in m['development']]
            or str(a['contract_hash'])!=contract):raise ValueError('Feature IDs drift')
    if train.shape!=(count,3,384) or dev.shape!=(250,3,384) or any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in [train,dev]):
        raise ValueError('Invalid features')
    prior_path=baseline/'g01_features.npz';prior_geo=baseline/'g01_geometry.json'
    if digest(prior_path)!=spec['feature_sha256'] or digest(prior_geo)!=spec['geometry_sha256']:raise ValueError('Reference changed')
    with np.load(prior_path,allow_pickle=False) as a:
        if not np.array_equal(a['physical_adjacent'][:299],train[:299]) or not np.array_equal(a['physical_adjacent'][299:],dev):
            raise ValueError('Original feature bytes changed')
    dev_hashes={s['arms']['physical_adjacent']['pixel_sha256'] for s in json.loads(prior_geo.read_text())['series'][299*3:]}
    geometry=json.loads(gp.read_text())
    if len(geometry)!=(count-299)*3:raise ValueError('Extra coverage drift')
    for i,g in enumerate(geometry):
        row=m['train'][299+i//3];series=row['series'][i%3];n=g['n_slices'];center=round(.5*(n-1))
        indices=[max(0,min(n-1,center+d)) for d in [-1,0,1]];positions=np.asarray(g['positions_mm'])
        if ((g['study'],g['series'],g['plane'])!=(row['StudyInstanceUID'],series['series_uid'],series['plane'])
            or g['pixel_sha256'] in dev_hashes or g['selected']['indices']!=indices
            or positions.shape!=(n,) or not np.isfinite(positions).all()
            or (n>1 and np.diff(positions).min()<=1e-4)
            or not np.array_equal(g['selected']['positions_mm'],positions[indices])):
            raise ValueError('Geometry/split gate failed')
    labels=np.asarray([row['labels'] for row in m['development']],dtype=np.float32);results=[]
    baseline_receipt=json.loads((baseline/'g01_receipt.json').read_text())
    if [(s['arm'],s['seed']) for s in r['seeds']]!=[(a,s) for a in ['control','expanded'] for s in [2026,42]]:
        raise ValueError('Run matrix drift')
    for s in r['seeds']:
        arm,seed=s['arm'],s['seed'];scores=[h['development']['mean_soft_bce'] for h in s['history']]
        if (s['epochs']!=20 or [h['epoch'] for h in s['history']]!=list(range(1,21))
            or not np.isfinite(scores).all() or s['training_studies']!=(299 if arm=='control' else count)):
            raise ValueError('History/count drift')
        best=int(np.argmin(scores))+1
        if best!=s['best_epoch']:raise ValueError('Selection drift')
        fingerprint=hashlib.sha256(f"{contract}:{r['feature_sha256']}:{arm}:{seed}".encode()).hexdigest()
        if s['fingerprint']!=fingerprint:raise ValueError('Fingerprint drift')
        folder=directory/arm;pred=folder/f'v02_seed{seed}_development.npz'
        if digest(pred)!=s['prediction_sha256']:raise ValueError('Prediction drift')
        with np.load(pred,allow_pickle=False) as a:
            if a['ids'].tolist()!=[x['StudyInstanceUID'] for x in m['development']]:raise ValueError('Prediction IDs drift')
            predictions={k:a[k+'_logits'].copy() for k in ['best','last']}
        deltas={}
        for kind,epoch in [('best',best),('last',20)]:
            cp=folder/f'v02_seed{seed}_{kind}.pt'
            if digest(cp)!=s[kind+'_checkpoint_sha256']:raise ValueError('Checkpoint SHA drift')
            state=torch.load(cp,map_location='cpu',weights_only=True)
            if (state['seed'],state['epoch'],state['fingerprint'])!=(seed,epoch,fingerprint):raise ValueError('Checkpoint contract drift')
            if kind=='last' and (state['history']!=s['history'] or not state['optimizer']['state']):raise ValueError('Resume state missing')
            expected=replay(dev,state['model']);observed=predictions[kind]
            if observed.shape!=labels.shape or not np.allclose(expected,observed,atol=1e-4,rtol=1e-5):raise ValueError('Head replay failed')
            deltas[kind]=float(np.max(np.abs(expected-observed)))
            if abs(float(loss(observed,labels).mean())-scores[epoch-1])>2e-6:raise ValueError('Metric replay failed')
        if arm=='control':
            previous=next(v for v in baseline_receipt['seeds'] if v['arm']=='physical_adjacent' and v['seed']==seed)
            pp=baseline/'physical_adjacent'/f'v02_seed{seed}_development.npz'
            if digest(pp)!=previous['prediction_sha256'] or best!=previous['best_epoch']:raise ValueError('Control reference drift')
            with np.load(pp,allow_pickle=False) as a:
                if any(not np.allclose(predictions[k],a[k+'_logits'],atol=1e-4,rtol=1e-5) for k in predictions):raise ValueError('Control did not reproduce')
        per=loss(predictions['best'],labels).mean(axis=0)
        if not np.allclose(per,s['best_metrics']['per_target_soft_bce'],atol=2e-6,rtol=0):raise ValueError('Target loss drift')
        results.append({'arm':arm,'seed':seed,'training_studies':s['training_studies'],'selected_epoch':best,
                        'mean_soft_bce':float(per.mean()),'last_soft_bce':float(loss(predictions['last'],labels).mean()),
                        'per_target_soft_bce':dict(zip(m['targets'],per.tolist())),'replay_max_delta':deltas})
    values={(s['arm'],s['seed']):s['mean_soft_bce'] for s in results}
    promote=all(values['expanded',seed]<values['control',seed]-2e-6 for seed in [2026,42])
    out={'status':'PASSED_V03_AUDIT','build_sha256':digest(build),'results':results,
         'selected_reference':'expanded' if promote else 'control','confirmation_evaluated':False,'submission_eligible':False,
         'seconds':r['seconds'],'feature_seconds':r['feature_seconds'],
         'limitations':['Weak-label agreement, not clinical or Kaggle performance.','Expanded data also increases training steps.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','build','manifest','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assess(a.directory,a.build,a.manifest,a.baseline,a.output)
