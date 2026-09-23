"""#RSNA #Kaggle #Pesquisa — one-shot grouped confirmation, never a tuning loop."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scripts.prepare_v04_confirmation import assemble,digest,literal,R02_ROOT
from scripts.assess_v02_baseline import replay,loss


def grouped_bootstrap(delta,groups,replicates=5000,seed=20260921):
    delta=np.asarray(delta,dtype=np.float64);groups=np.asarray(groups,dtype=str)
    if delta.ndim!=1 or not len(delta) or groups.shape!=delta.shape or not np.isfinite(delta).all() or any(not g for g in groups):
        raise ValueError('Invalid bootstrap inputs')
    unique,inverse=np.unique(groups,return_inverse=True)
    if len(unique)<2 or replicates<100:raise ValueError('Insufficient bootstrap groups/replicates')
    sums=np.bincount(inverse,weights=delta);counts=np.bincount(inverse)
    draws=np.random.default_rng(seed).integers(0,len(unique),size=(replicates,len(unique)))
    samples=sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta_mean':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'groups':len(unique),'studies':len(delta),'replicates':replicates,'seed':seed}


def weak_auc(logits,labels,targets):
    result={}
    for j,t in enumerate(targets):
        keep=labels[:,j]!=.5;y=labels[keep,j]>.5;s=logits[keep,j]
        pos=int(y.sum());neg=len(y)-pos
        result[t]={'positive':pos,'negative':neg,'uncertain':int((~keep).sum()),
                   'auc':float(roc_auc_score(y,s)) if pos and neg else None}
    available=[v['auc'] for v in result.values() if v['auc'] is not None]
    return {'per_target':result,'macro_auc':float(np.mean(available)) if available else None,
            'valid_targets':len(available),'reference':'Frozen weak teacher; 0.5 excluded, others >0.5; not clinical AUC.'}


def compare(logits,labels,groups,targets):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    if (x.shape!=(4,4,len(y),12) or y.shape!=(len(y),12) or not len(y)
        or len(targets)!=12 or len(set(targets))!=12 or len(groups)!=len(y)
        or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y<0)|(y>1))):
        raise ValueError('Invalid confirmation predictions/labels')
    losses=np.logaddexp(0.,x)-y[None,None]*x
    results=[]
    for i,(arm,seed) in enumerate([('control',2026),('control',42),('paired_consistency',2026),('paired_consistency',42)]):
        per=losses[i,0].mean(0)
        results.append({'arm':arm,'seed':seed,'mean_soft_bce':float(per.mean()),
                        'mean_missing_soft_bce':float(losses[i,1:].mean()),
                        'missing_soft_bce':dict(zip(['Sagittal','Coronal','Axial'],losses[i,1:].mean((1,2)).tolist())),
                        'per_target_soft_bce':dict(zip(targets,per.tolist())),
                        'weak_auc_diagnostic':weak_auc(x[i,0],y,targets)})
    intact_deltas=losses[2:,0].mean((1,2))-losses[:2,0].mean((1,2))
    missing_deltas=losses[2:,1:].mean((1,2,3))-losses[:2,1:].mean((1,2,3))
    study_delta=(losses[2:,0]-losses[:2,0]).mean((0,2))
    per_target=(losses[2:,0]-losses[:2,0]).mean((0,1))
    boot=grouped_bootstrap(study_delta,groups)
    gates={'intact_both_seeds':bool(np.all(intact_deltas < -2e-6)),
           'missing_both_seeds':bool(np.all(missing_deltas < -2e-6)),
           'bootstrap_upper_below_zero':boot['ci95'][1]<0,
           'no_target_regression_over_0_01':bool(np.all(per_target<=.01))}
    return {'results':results,'intact_delta_by_seed':dict(zip(['2026','42'],intact_deltas.tolist())),
            'missing_delta_by_seed':dict(zip(['2026','42'],missing_deltas.tolist())),
            'target_delta_mean_seeds':dict(zip(targets,per_target.tolist())),'bootstrap':boot,
            'gates':gates,'confirmation_decision':'CONFIRMED' if all(gates.values()) else 'NOT_CONFIRMED'}


def check_geometry(records,rows,forbidden):
    if len(records)!=len(rows)*3:raise ValueError('Geometry coverage drift')
    for i,g in enumerate(records):
        row=rows[i//3];s=row['series'][i%3];headers=g['headers'];plan=g['plan'];chosen=g['selected']
        if (g['study'],g['series'],g['plane'])!=(row['StudyInstanceUID'],s['series_uid'],s['plane']):raise ValueError('Geometry identity drift')
        if g['pixel_sha256'] in forbidden:raise ValueError('Reserved pixel collision')
        names=[h['name'] for h in headers];n=len(names)
        iop=np.asarray([h['iop'] for h in headers],dtype=float);ipp=np.asarray([h['ipp'] for h in headers],dtype=float)
        if (not n or len(set(names))!=n or iop.shape!=(n,6) or ipp.shape!=(n,3)
            or not np.isfinite(iop).all() or not np.isfinite(ipp).all()
            or np.max(abs(iop-iop[0]))>1e-3):raise ValueError('Invalid geometry headers')
        row_dirs,col_dirs=iop[:,:3],iop[:,3:]
        if (np.max(abs(np.linalg.norm(row_dirs,axis=1)-1))>1e-3 or np.max(abs(np.linalg.norm(col_dirs,axis=1)-1))>1e-3
            or np.max(abs((row_dirs*col_dirs).sum(1)))>1e-3):raise ValueError('Invalid direction cosines')
        normal=np.cross(row_dirs[0],col_dirs[0]);normal/=np.linalg.norm(normal)
        distances=ipp@normal;order=np.argsort(distances,kind='stable');distances=distances[order]
        ordered=[names[j] for j in order];center=round(.5*(n-1));indices=[min(n-1,max(0,center+d)) for d in [-1,0,1]]
        if (n>1 and np.diff(distances).min()<=1e-4):raise ValueError('Duplicate physical locations')
        if (plan['n_slices']!=n or plan['center_index']!=center or plan['ordered_files']!=ordered
            or not np.allclose(plan['normal'],normal,atol=1e-10,rtol=0)
            or not np.allclose(plan['positions_mm'],distances,atol=1e-8,rtol=0)
            or chosen!=plan['arms']['physical_adjacent'] or chosen['indices']!=indices
            or chosen['files']!=[ordered[j] for j in indices]
            or not np.allclose(chosen['positions_mm'],distances[indices],atol=1e-8,rtol=0)
            or not np.allclose(chosen['gaps_mm'],np.diff(distances[indices]),atol=1e-8,rtol=0)):
            raise ValueError('Physical sampling drift')


def assess(directory,build,output):
    if output.exists():raise FileExistsError(output)
    source=build.read_text()
    if source!=assemble():raise ValueError('Build/source drift')
    spec=literal(source,'V04');contract=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    r=json.loads((directory/'v04_receipt.json').read_text())
    if (r['status']!='COMPLETE_V04_RESERVED_INFERENCE_NOT_SUBMISSION' or r['spec']!=spec
        or r['contract_hash']!=contract or r['gpu_names']!=['Tesla T4']*2
        or not r['confirmation_predictions_generated'] or r['metrics_evaluated_in_kernel'] or r['submission_eligible']):
        raise ValueError('Receipt scope drift')
    for name in ['feature','geometry','prediction','exposure']:
        filename={'feature':'v04_features.npz','geometry':'v04_geometry.json','prediction':'v04_predictions.npz','exposure':'v04_exposure.json'}[name]
        if digest(directory/filename)!=r[name+'_sha256']:raise ValueError('Artifact SHA drift')
    exposure=json.loads((directory/'v04_exposure.json').read_text())
    if exposure!={'contract_hash':contract,'stage':'predictions_complete','studies':150,'metrics_evaluated_in_kernel':False}:
        raise ValueError('Exposure marker drift')
    rows=spec['rows'];ids=[row['StudyInstanceUID'] for row in rows]
    with np.load(directory/'v04_features.npz',allow_pickle=False) as a:
        features=a['features'].copy()
        if a['ids'].tolist()!=ids or str(a['contract_hash'])!=contract:raise ValueError('Feature identity drift')
    if features.shape!=(150,3,384) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Invalid features')
    check_geometry(json.loads((directory/'v04_geometry.json').read_text()),rows,set(spec['forbidden_pixel_hashes']))
    with np.load(directory/'v04_predictions.npz',allow_pickle=False) as a:
        logits=a['logits'].copy()
        if (a['ids'].tolist()!=ids or str(a['contract_hash'])!=contract
            or a['heads'].tolist()!=[f"{c['arm']}/{c['seed']}" for c in spec['checkpoints']]
            or a['modes'].tolist()!=['intact','without_Sagittal','without_Coronal','without_Axial']):raise ValueError('Prediction identity drift')
    if logits.shape!=(4,4,150,12) or not np.isfinite(logits).all():raise ValueError('Invalid logits')
    deltas=[];torch.set_num_threads(1)
    for i,c in enumerate(spec['checkpoints']):
        cp=R02_ROOT/c['arm']/f"v02_seed{c['seed']}_best.pt"
        if digest(cp)!=c['sha256']:raise ValueError('Checkpoint drift')
        state=torch.load(cp,map_location='cpu',weights_only=True)
        if (state['seed'],state['epoch'],state['fingerprint'])!=(c['seed'],c['epoch'],c['fingerprint']):raise ValueError('Checkpoint identity drift')
        for j,missing in enumerate([-1,0,1,2]):
            f=features if missing<0 else features[:,[k for k in range(3) if k!=missing]]
            expected=replay(f,state['model'])
            if not np.allclose(expected,logits[i,j],atol=1e-4,rtol=1e-5):raise ValueError('Independent head replay failed')
            deltas.append(float(np.abs(expected-logits[i,j]).max()))
    labels=np.asarray([row['labels'] for row in rows],dtype=np.float32)
    result=compare(logits,labels,[row['report_hash'] for row in rows],spec['targets'])
    result.update(status='PASSED_V04_CONFIRMATION_AUDIT',build_sha256=digest(build),receipt_sha256=digest(directory/'v04_receipt.json'),
                  confirmation_evaluated=True,submission_eligible=False,training_performed=False,
                  max_head_replay_delta=max(deltas),seconds=r['seconds'],feature_seconds=r['feature_seconds'],
                  limitations=['Weak teacher reference, not independently adjudicated clinical labels or Kaggle score.',
                               'Reserved set now exposed; do not retune and reconsult it.',
                               'Exact pixel hashes do not exclude approximate/patient duplicates.',
                               'Encoder features verified by provenance/hash/shape; no independent second DICOM extraction.',
                               'Missing planes are synthetic, not real incomplete-protocol validation.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','build','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assess(a.directory,a.build,a.output)
