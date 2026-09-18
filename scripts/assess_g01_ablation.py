"""#RSNA #Kaggle #Pesquisa — independent geometry/feature/head/metric audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_g01_ablation import assemble
from scripts.prepare_m01_ablation import literal, digest
from scripts.assess_v02_baseline import replay, loss


def decide(rows):
    scores={(r['arm'],r['seed']):r['mean_soft_bce'] for r in rows}
    arms=['control','physical_quartiles','physical_adjacent']
    if set(scores)!={(a,s) for a in arms for s in [2026,42]} or len(rows)!=6:
        raise ValueError('Incomplete result matrix')
    if not np.isfinite(list(scores.values())).all():raise ValueError('Nonfinite results')
    eligible=[a for a in arms[1:] if all(scores[a,s]<scores['control',s]-2e-6 for s in [2026,42])]
    return min(eligible,key=lambda a:(sum(scores[a,s] for s in [2026,42]),a!='physical_quartiles')) if eligible else 'control'


def assess(directory,build,baseline,output):
    source=build.read_text()
    if source!=assemble():raise ValueError('Source drift')
    spec=literal(source,'G01');payload=literal(source,'V02_BASELINE')
    contract=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    r=json.loads((directory/'g01_receipt.json').read_text())
    if (r['status']!='COMPLETE_G01_NOT_SUBMISSION' or r['spec']!=spec or r['contract_hash']!=contract
        or r['confirmation_evaluated'] or r['submission_eligible'] or r['gpu_names']!=['Tesla T4']*2):
        raise ValueError('Receipt drift')
    rows=payload['splits']['train']+payload['splits']['development'];ids=[row['StudyInstanceUID'] for row in rows]
    fp=directory/'g01_features.npz';gp=directory/'g01_geometry.json'
    if digest(fp)!=r['feature_sha256'] or digest(gp)!=r['geometry_sha256']:raise ValueError('Artifact SHA drift')
    with np.load(fp,allow_pickle=False) as a:
        if a['ids'].tolist()!=ids or str(a['contract_hash'])!=contract:raise ValueError('Feature identities changed')
        features={arm:a[arm].copy() for arm in spec['arms']}
    if any(x.shape!=(549,3,384) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Invalid features')
    original=baseline/'v02_baseline_features.npz'
    if digest(original)!=spec['feature_sha256']:raise ValueError('Baseline archive drift')
    with np.load(original,allow_pickle=False) as a:
        if not np.array_equal(a['features'],features['control']):raise ValueError('Control feature drift')
    geometry=json.loads(gp.read_text())
    if geometry['contract_hash']!=contract or len(geometry['series'])!=1647:raise ValueError('Geometry coverage changed')
    summary={a:{'changed_pixel_series':0,'gaps_mm':[]} for a in spec['arms'][1:]}
    for i,g in enumerate(geometry['series']):
        row=rows[i//3];s=row['series'][i%3]
        if (g['study'],g['series'],g['plane'],g['original_files'],g['original_pixel_sha256'])!=(
            row['StudyInstanceUID'],s['series_uid'],s['plane'],s['selected_files'],s['image_sha256']):
            raise ValueError('Geometry identity/original pixel drift')
        n=spec['series_counts'][g['study']+'/'+g['series']]
        distances=np.asarray(g['positions_mm']);names=g['ordered_files']
        if (g['n_slices']!=n or len(names)!=n or len(set(names))!=n or distances.shape!=(n,)
            or not np.isfinite(distances).all() or (n>1 and np.min(np.diff(distances))<=1e-4)):
            raise ValueError('Invalid physical stack')
        center=round(.5*(n-1))
        for arm in spec['arms'][1:]:
            expected=([round(q*(n-1)) for q in [.25,.5,.75]] if arm=='physical_quartiles'
                      else [max(0,min(n-1,center+d)) for d in [-1,0,1]])
            selected=g['arms'][arm]
            if (selected['indices']!=expected or selected['files']!=[names[j] for j in expected]
                or not np.array_equal(selected['positions_mm'],distances[expected])
                or not np.allclose(selected['gaps_mm'],np.diff(distances[expected]),atol=1e-8,rtol=0)):
                raise ValueError('Sampling contract changed')
            changed=selected['pixel_sha256']!=s['image_sha256']
            summary[arm]['changed_pixel_series']+=int(changed)
            summary[arm]['gaps_mm'].extend(selected['gaps_mm'])
            if not changed and not np.allclose(features[arm][i//3,i%3],features['control'][i//3,i%3],atol=1e-4,rtol=1e-5):
                raise ValueError('Same pixels produce different embeddings')
    labels=np.asarray([row['labels'] for row in rows[299:]],dtype=np.float32);results=[]
    reference=json.loads((baseline/'v02_baseline_receipt.json').read_text())
    if [(s['arm'],s['seed']) for s in r['seeds']]!=[(a,s) for a in spec['arms'] for s in [2026,42]]:
        raise ValueError('Run inventory drift')
    for s in r['seeds']:
        arm,seed=s['arm'],s['seed'];history=s['history']
        scores=[h['development']['mean_soft_bce'] for h in history]
        if (s['epochs']!=20 or [h['epoch'] for h in history]!=list(range(1,21)) or not np.isfinite(scores).all()
            or any(not np.isfinite(h['train_soft_bce']) or h['train_soft_bce']<0 for h in history)):
            raise ValueError('History drift')
        best=int(np.argmin(scores))+1
        if s['best_epoch']!=best:raise ValueError('Selection changed')
        fingerprint=hashlib.sha256(f"{contract}:{r['feature_sha256']}:{arm}:{seed}".encode()).hexdigest()
        folder=directory/arm;predpath=folder/f'v02_seed{seed}_development.npz'
        if digest(predpath)!=s['prediction_sha256'] or s['fingerprint']!=fingerprint:raise ValueError('Prediction contract drift')
        with np.load(predpath,allow_pickle=False) as a:
            if a['ids'].tolist()!=ids[299:]:raise ValueError('Dev order drift')
            predictions={kind:a[kind+'_logits'].copy() for kind in ['best','last']}
        deltas={}
        for kind,epoch in [('best',best),('last',20)]:
            cp=folder/f'v02_seed{seed}_{kind}.pt'
            if digest(cp)!=s[kind+'_checkpoint_sha256']:raise ValueError('Checkpoint SHA drift')
            state=torch.load(cp,map_location='cpu',weights_only=True)
            if (state['seed'],state['epoch'],state['fingerprint'])!=(seed,epoch,fingerprint):raise ValueError('Checkpoint contract drift')
            if kind=='last' and (state['history']!=history or not state['optimizer']['state']):raise ValueError('Resume state missing')
            expected=replay(features[arm][299:],state['model']);observed=predictions[kind]
            if observed.shape!=labels.shape or not np.allclose(expected,observed,atol=1e-4,rtol=1e-5):raise ValueError('Head replay failed')
            deltas[kind]=float(np.max(np.abs(expected-observed)))
            if abs(float(loss(observed,labels).mean())-scores[epoch-1])>2e-6:raise ValueError('BCE replay failed')
        if arm=='control':
            previous=next(v for v in reference['seeds'] if v['seed']==seed)
            path=baseline/f'v02_seed{seed}_development.npz'
            if digest(path)!=previous['prediction_sha256'] or best!=previous['best_epoch']:raise ValueError('Control reference drift')
            with np.load(path,allow_pickle=False) as a:
                if any(not np.allclose(predictions[k],a[k+'_logits'],atol=1e-4,rtol=1e-5) for k in predictions):raise ValueError('Control did not reproduce')
        per=loss(predictions['best'],labels).mean(axis=0)
        if not np.allclose(per,s['best_metrics']['per_target_soft_bce'],atol=2e-6,rtol=0):raise ValueError('Target metric drift')
        results.append({'arm':arm,'seed':seed,'selected_epoch':best,'mean_soft_bce':float(per.mean()),
                        'last_soft_bce':float(loss(predictions['last'],labels).mean()),
                        'per_target_soft_bce':dict(zip(payload['targets'],per.tolist())),
                        'replay_max_delta':deltas})
    for a,v in summary.items():
        v['gap_mm_min_median_max']=np.quantile(v.pop('gaps_mm'),[0,.5,1]).tolist()
    out={'status':'PASSED_G01_AUDIT','build_sha256':digest(build),'results':results,'geometry':summary,
         'selected_reference':decide(results),'confirmation_evaluated':False,'submission_eligible':False,
         'seconds':r['seconds'],'preprocessing_seconds':r['preprocessing_seconds'],
         'cuda_peak_allocated_bytes':r['cuda_peak_allocated_bytes'],
         'limitations':['Weak-label development selects epochs/recipe, not clinical test or Kaggle score.',
                        'Physical adjacency also narrows coverage; encoder remains frozen.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','build','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assess(a.directory,a.build,a.baseline,a.output)
