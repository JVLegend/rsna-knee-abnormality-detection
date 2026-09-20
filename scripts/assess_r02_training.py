"""#RSNA #Kaggle #Pesquisa — independent intact/masked replay and mask RNG audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_r02_training import assemble, literal, digest, ARMS
from scripts.assess_v02_baseline import loss, replay


def decide(results):
    expected={(a,s) for a in ARMS for s in [2026,42]}
    values={(r['arm'],r['seed']):(r['mean_soft_bce'],r['mean_missing_soft_bce']) for r in results}
    if len(results)!=6 or set(values)!=expected or not np.isfinite(list(values.values())).all():
        raise ValueError('Incomplete/nonfinite result matrix')
    eligible=[a for a in ARMS[1:] if all(values[a,s][i]<values['control',s][i]-2e-6
                                        for s in [2026,42] for i in [0,1])]
    return min(eligible,key=lambda a:(np.mean([values[a,s][0] for s in [2026,42]]),ARMS.index(a))) if eligible else 'control'


def mask_history(seed, arm, epochs=20, size=1000, batch=4):
    if arm not in ARMS[1:]: raise ValueError('No masks for this arm')
    rng=torch.Generator().manual_seed(seed+10000);count=np.zeros(3,dtype=np.int64)
    chain='0'*64;seen=0;history=[];probability=.25 if arm=='dropout25' else 1.
    for epoch in range(epochs):
        for start in range(0,size,batch):
            n=min(batch,size-start)
            drop=torch.rand(n,generator=rng).numpy()<probability
            plane=torch.randint(3,(n,),generator=rng).numpy()
            mask=np.ones((n,3),dtype=bool);mask[np.flatnonzero(drop),plane[drop]]=False
            count+=(~mask).sum(axis=0);seen+=n
            chain=hashlib.sha256(bytes.fromhex(chain)+mask.tobytes()).hexdigest()
        history.append({'seen':seen,'dropped':count.tolist(),'chain':chain})
    return history,rng.get_state()


def assess(directory,build,baseline,output):
    if output.exists(): raise FileExistsError(output)
    source=build.read_text()
    if source!=assemble(): raise ValueError('Source/build drift')
    spec=literal(source,'R02');payload=literal(source,'V02_BASELINE')
    contract=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    r=json.loads((directory/'r02_receipt.json').read_text())
    if (r['status']!='COMPLETE_R02_TRAINING_NOT_SUBMISSION' or r['spec']!=spec
        or r['contract_hash']!=contract or r['gpu_names']!=['Tesla T4']*2
        or r['confirmation_evaluated'] or r['submission_eligible']): raise ValueError('Receipt scope drift')
    if [(s['arm'],s['seed']) for s in r['seeds']]!=[(a,s) for a in ARMS for s in [2026,42]]:
        raise ValueError('Run matrix drift')
    fp=baseline/'v03_features.npz';rp=baseline/'v03_receipt.json'
    if digest(fp)!=spec['feature_sha256'] or digest(rp)!=spec['baseline_receipt_sha256']:
        raise ValueError('Reference artifact drift')
    prior=json.loads(rp.read_text());ids=[r['StudyInstanceUID'] for r in payload['splits']['development']]
    with np.load(fp,allow_pickle=False) as a:
        dev=a['development'].copy()
        if a['development_ids'].tolist()!=ids or str(a['contract_hash'])!=spec['baseline_contract']:
            raise ValueError('Development identity drift')
    labels=np.asarray([r['labels'] for r in payload['splits']['development']],dtype=np.float32);results=[]
    for s in r['seeds']:
        arm,seed=s['arm'],s['seed'];scores=[h['development']['mean_soft_bce'] for h in s['history']]
        if (s['epochs']!=20 or [h['epoch'] for h in s['history']]!=list(range(1,21))
            or not np.isfinite(scores).all() or any(not np.isfinite(h['train_soft_bce']) or h['train_soft_bce']<0 for h in s['history'])):
            raise ValueError('Training history drift')
        best=int(np.argmin(scores))+1
        if best!=s['best_epoch'] or abs(scores[best-1]-s['best_metrics']['mean_soft_bce'])>2e-6:
            raise ValueError('Selection rule drift')
        fingerprint=hashlib.sha256(f'{contract}:{arm}:{seed}'.encode()).hexdigest()
        if s['fingerprint']!=fingerprint: raise ValueError('Fingerprint drift')
        folder=directory/arm;pp=folder/f'v02_seed{seed}_development.npz'
        if digest(pp)!=s['prediction_sha256']: raise ValueError('Prediction SHA drift')
        with np.load(pp,allow_pickle=False) as a:
            if a['ids'].tolist()!=ids: raise ValueError('Prediction order drift')
            predictions={k:a[k+'_logits'].copy() for k in ['best','last']}
        deltas={};states={}
        for kind,epoch in [('best',best),('last',20)]:
            cp=folder/f'v02_seed{seed}_{kind}.pt'
            if digest(cp)!=s[kind+'_checkpoint_sha256']: raise ValueError('Checkpoint SHA drift')
            state=torch.load(cp,map_location='cpu',weights_only=True);states[kind]=state
            if (state['seed'],state['epoch'],state['fingerprint'])!=(seed,epoch,fingerprint):
                raise ValueError('Checkpoint identity drift')
            if kind=='last' and (state['history']!=s['history'] or not state['optimizer']['state']):
                raise ValueError('Resume state incomplete')
            expected=replay(dev,state['model']);observed=predictions[kind]
            if observed.shape!=labels.shape or not np.allclose(expected,observed,atol=1e-4,rtol=1e-5):
                raise ValueError('Intact replay failed')
            deltas[kind]=float(np.abs(expected-observed).max())
            if abs(float(loss(observed,labels).mean())-scores[epoch-1])>2e-6: raise ValueError('BCE drift')
        if arm=='control':
            old=next(s for s in prior['seeds'] if (s['arm'],s['seed'])==('expanded',seed))
            path=baseline/'expanded'/f'v02_seed{seed}_development.npz'
            if digest(path)!=old['prediction_sha256'] or best!=old['best_epoch']: raise ValueError('Control reference drift')
            with np.load(path,allow_pickle=False) as a:
                if any(not np.allclose(predictions[k],a[k+'_logits'],atol=1e-4,rtol=1e-5) for k in predictions):
                    raise ValueError('Control did not reproduce')
            if not np.allclose(scores,[h['development']['mean_soft_bce'] for h in old['history']],atol=2e-6,rtol=0):
                raise ValueError('Control learning curve changed')
        else:
            trace,rng=mask_history(seed,arm)
            if ([h['mask_trace'] for h in s['history']]!=trace or s['mask_trace']!=trace[-1]
                or states['last']['mask_trace']!=trace[-1] or not torch.equal(rng,states['last']['mask_rng'])):
                raise ValueError('Mask RNG/trace replay failed')
        mp=folder/f'r02_seed{seed}_missing.npz'
        if digest(mp)!=s['missing']['prediction_sha256']: raise ValueError('Missing predictions SHA drift')
        with np.load(mp,allow_pickle=False) as a:
            masked=a['logits'].copy()
            if a['ids'].tolist()!=ids or masked.shape!=(3,250,12): raise ValueError('Missing prediction identity drift')
        missing=[]
        for plane in range(3):
            expected=replay(dev[:,[j for j in range(3) if j!=plane]],states['best']['model'])
            if not np.allclose(expected,masked[plane],atol=1e-4,rtol=1e-5): raise ValueError('Masked replay failed')
            deltas['missing_'+str(plane)]=float(np.abs(expected-masked[plane]).max())
            per=loss(masked[plane],labels).mean(0);metric=s['missing']['metrics'][plane]
            if (abs(float(per.mean())-metric['mean_soft_bce'])>2e-6
                or not np.allclose(per,metric['per_target_soft_bce'],atol=2e-6,rtol=0)):
                raise ValueError('Missing metric drift')
            missing.append(float(per.mean()))
        per=loss(predictions['best'],labels).mean(0)
        if not np.allclose(per,s['best_metrics']['per_target_soft_bce'],atol=2e-6,rtol=0): raise ValueError('Per-target drift')
        results.append({'arm':arm,'seed':seed,'selected_epoch':best,'mean_soft_bce':float(per.mean()),
                        'per_target_soft_bce':dict(zip(payload['targets'],per.tolist())),
                        'missing_soft_bce':dict(zip(['Sagittal','Coronal','Axial'],missing)),
                        'mean_missing_soft_bce':float(np.mean(missing)),'replay_max_delta':deltas})
    out={'status':'PASSED_R02_TRAINING_AUDIT','build_sha256':digest(build),'results':results,
         'selected_reference':decide(results),'control_reproduced':True,'seconds':r['seconds'],
         'cuda_peak_allocated_bytes':r['cuda_peak_allocated_bytes'],
         'confirmation_evaluated':False,'submission_eligible':False,
         'limitations':['Weak-label development agreement, not clinical or Kaggle score.',
                        'Synthetic missing planes, not real incomplete-protocol validation.',
                        'Paired arm changes exposure and objective, not an isolated consistency ablation.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','build','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assess(a.directory,a.build,a.baseline,a.output)
