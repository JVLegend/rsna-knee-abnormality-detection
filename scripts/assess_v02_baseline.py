"""#RSNA #Kaggle #Pesquisa — independent loss and NumPy head replay."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.freeze_weak_validation import digest,freeze
from scripts.prepare_v02_baseline import SPEC,assemble


def loss(logits,labels):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid logits/labels')
    return np.logaddexp(0.,x)-y*x


def replay(features,state):
    names={'score.0.weight','score.0.bias','score.2.weight','score.2.bias','classifier.weight','classifier.bias'}
    if set(state)!=names:raise ValueError('Unexpected head parameters')
    weights={k:v.detach().cpu().numpy().astype(np.float64) for k,v in state.items()}
    if any(not np.isfinite(v).all() for v in weights.values()):raise ValueError('Nonfinite head weights')
    x=np.asarray(features,dtype=np.float64)
    scores=(np.tanh(x@weights['score.0.weight'].T+weights['score.0.bias'])@weights['score.2.weight'].T+weights['score.2.bias'])[...,0]
    attention=np.exp(scores-scores.max(axis=1,keepdims=True));attention/=attention.sum(axis=1,keepdims=True)
    return (x*attention[...,None]).sum(axis=1)@weights['classifier.weight'].T+weights['classifier.bias']


def assess(directory,cache,build,output):
    if build.read_text()!=assemble(cache):raise ValueError('Source/cache/recipe drift')
    tree=ast.parse(build.read_text());payload=next(ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='V02_BASELINE')
    contract=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    r=json.loads((directory/'v02_baseline_receipt.json').read_text())
    if r['status']!='COMPLETE_V02_WEAK_BASELINE_NOT_SUBMISSION' or r['contract_hash']!=contract or r['spec']!=SPEC:
        raise ValueError('Receipt contract drift')
    if (r['manifest_sha256']!=payload['manifest_sha256'] or
        r['model_sha256']!='1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050'):
        raise ValueError('Model/split provenance drift')
    if (r['studies'],r['series'],r['dicom_files'])!=(549,1647,4941) or r['gpu_names']!=['Tesla T4']*2:
        raise ValueError('Runtime coverage/hardware drift')
    if not r['development_evaluated'] or any(r[k] for k in ['confirmation_evaluated','auc_measured','submission_eligible']):
        raise ValueError('Evaluation scope violation')
    train=payload['splits']['train'];dev=payload['splits']['development'];rows=train+dev
    labels=np.asarray([row['labels'] for row in dev],dtype=np.float32)
    expected_ids=[row['StudyInstanceUID'] for row in rows]
    features_path=directory/'v02_baseline_features.npz'
    if digest(features_path)!=r['feature_sha256']:raise ValueError('Feature archive drift')
    with np.load(features_path,allow_pickle=False) as a:
        features=a['features']
        if (a['ids'].tolist()!=expected_ids or str(a['contract_hash'])!=contract
            or a['image_hashes'].tolist()!=[s['image_sha256'] for row in rows for s in row['series']]
            or features.shape!=(549,3,384) or features.dtype!=np.float32 or not np.isfinite(features).all()):
            raise ValueError('Feature identity/content drift')
    feature_hash=hashlib.sha256(features.tobytes()).hexdigest()
    if feature_hash!=r['feature_fingerprint']:raise ValueError('Feature fingerprint mismatch')
    prior=np.clip(np.array([row['labels'] for row in train],dtype=np.float32).astype(np.float64).mean(axis=0),1e-6,1-1e-6)
    prior_logits=np.broadcast_to(np.log(prior/(1-prior)).astype(np.float32),labels.shape)
    prior_path=directory/'v02_prior_development.npz'
    if digest(prior_path)!=r['prior_prediction_sha256']:raise ValueError('Prior prediction drift')
    with np.load(prior_path,allow_pickle=False) as a:
        if not np.allclose(a['logits'],prior_logits,rtol=0,atol=1e-6):raise ValueError('Prior uses wrong training labels')
    prior_bce=float(loss(prior_logits,labels).mean());results=[]
    if abs(prior_bce-r['prior_metrics']['mean_soft_bce'])>2e-6:raise ValueError('Prior metric replay failed')
    if [s['seed'] for s in r['seeds']]!=SPEC['seeds']:raise ValueError('Seed inventory drift')
    for s in r['seeds']:
        seed=s['seed'];history=s['history']
        if s['epochs']!=20 or [h['epoch'] for h in history]!=list(range(1,21)):raise ValueError('Incomplete training history')
        scores=[h['development']['mean_soft_bce'] for h in history]
        if not np.isfinite(scores).all() or any(h['train_soft_bce']<0 or not np.isfinite(h['train_soft_bce']) for h in history):
            raise ValueError('Invalid history')
        best=int(np.argmin(scores))+1
        if s['best_epoch']!=best:raise ValueError('Selection rule drift')
        fp=hashlib.sha256(f'{contract}:{feature_hash}:{seed}'.encode()).hexdigest()
        if fp!=s['fingerprint']:raise ValueError('Checkpoint contract drift')
        prediction=directory/f'v02_seed{seed}_development.npz'
        if digest(prediction)!=s['prediction_sha256']:raise ValueError('Predictions changed')
        with np.load(prediction,allow_pickle=False) as a:
            if a['ids'].tolist()!=expected_ids[299:]:raise ValueError('Development order drift')
            predictions={kind:a[kind+'_logits'].copy() for kind in ['best','last']}
        deltas={}
        for kind,epoch in [('best',best),('last',20)]:
            cp=directory/f'v02_seed{seed}_{kind}.pt'
            if digest(cp)!=s[kind+'_checkpoint_sha256']:raise ValueError('Checkpoint bytes changed')
            state=torch.load(cp,map_location='cpu',weights_only=True)
            if state['seed']!=seed or state['epoch']!=epoch or state['fingerprint']!=fp:raise ValueError('Checkpoint metadata changed')
            if kind=='last' and (state['history']!=history or not state['optimizer']['state']):raise ValueError('Resume state incomplete')
            predicted=replay(features[299:],state['model']);observed=predictions[kind]
            if observed.shape!=labels.shape or not np.allclose(predicted,observed,atol=1e-4,rtol=1e-5):
                raise ValueError('Independent head replay failed')
            deltas[kind]=float(np.max(np.abs(predicted-observed)))
            if abs(float(loss(observed,labels).mean())-scores[epoch-1])>2e-6:raise ValueError('Independent loss replay failed')
        per_target=loss(predictions['best'],labels).mean(axis=0)
        if not np.allclose(per_target,s['best_metrics']['per_target_soft_bce'],rtol=0,atol=2e-6):
            raise ValueError('Per-target loss replay failed')
        score=float(per_target.mean())
        results.append({'seed':seed,'selected_epoch':best,'mean_soft_bce':score,
            'delta_vs_training_prior':score-prior_bce,'per_target_soft_bce':dict(zip(payload['targets'],per_target.tolist())),
            'head_replay_max_absolute_deltas':deltas})
    out={'status':'PASSED_V02_BASELINE_AUDIT','build_sha256':digest(build),'manifest_sha256':payload['manifest_sha256'],
        'training_studies':299,'development_studies':250,'confirmation_evaluated':False,
        'prior_soft_bce':prior_bce,'results':results,'timings_seconds':r['timings_seconds'],
        'cuda_peak_allocated_bytes':r['cuda_peak_allocated_bytes'],'auc_measured':False,'submission_eligible':False,
        'limitations':['Weak teacher agreement, not independently annotated clinical accuracy.',
            'Development selects epoch; not an untouched final test.',
            'Seeds vary the head and shuffle only; encoder is frozen.',
            'Numerical CPU/NumPy head replay tolerance differs from exact pixel cache gate.']}
    freeze(output,out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','cache','build','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assess(a.directory,a.cache,a.build,a.output)
