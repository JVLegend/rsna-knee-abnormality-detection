"""#RSNA #Kaggle #Pesquisa — standalone GPU baseline, no competition CSV.

Builder prepends frozen V02_BASELINE, CACHE_SOURCE and audited pilot helpers.
"""
import argparse

CONFIG_SHA='1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'


def metrics(logits,labels):
    losses=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    if not torch.isfinite(losses).all():raise ValueError('Nonfinite loss')
    return {'mean_soft_bce':float(losses.mean().cpu()),
            'per_target_soft_bce':losses.mean(dim=0).cpu().tolist()}


def atomic_save(path,payload):
    temporary=path.with_suffix('.partial');torch.save(payload,temporary);os.replace(temporary,path)


def train_seed(x,y,dev,dev_y,seed,fingerprint,output,guard,resume=None):
    spec=V02_BASELINE['spec'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=StudyAttention().cuda();optimizer=torch.optim.AdamW(head.parameters(),
        lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    train_mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
    dev_mask=torch.ones(dev.shape[:2],dtype=torch.bool,device=dev.device)
    history=[];best_state=None;best_loss=float('inf');best_epoch=0;start_epoch=0
    if resume is not None and resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint or state['seed']!=seed:raise ValueError('Resume contract drift')
        head.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];start_epoch=state['epoch'];best_state=state['best_state']
        best_loss=state['best_loss'];best_epoch=state['best_epoch']
        if not 1<=start_epoch<=spec['epochs'] or len(history)!=start_epoch:raise ValueError('Resume epoch drift')
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start_epoch+1,spec['epochs']+1):
        guard();head.train();order=torch.randperm(len(x));train_sum=0.
        for start in range(0,len(x),spec['batch_size']):
            indices=order[start:start+spec['batch_size']].to(x.device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(head(x[indices],train_mask[indices]),y[indices])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError('Nonfinite/missing head gradients')
            optimizer.step();train_sum+=float(loss.detach().cpu())*len(indices)
        head.eval()
        with torch.no_grad():evaluation=metrics(head(dev,dev_mask),dev_y)
        score=evaluation['mean_soft_bce']
        history.append({'epoch':epoch,'train_soft_bce':train_sum/len(x),'development':evaluation})
        if score<best_loss:
            best_loss=score;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        checkpoint_path=output/f'v02_seed{seed}_last.pt'
        atomic_save(checkpoint_path,{'model':head.state_dict(),'optimizer':optimizer.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'fingerprint':fingerprint,'seed':seed,'epoch':epoch,'history':history,
            'best_state':best_state,'best_loss':best_loss,'best_epoch':best_epoch})
        print('seed',seed,'epoch',epoch,'dev_soft_bce',round(score,7),flush=True)
    # Preserve latest checkpoint even when resuming an already complete seed.
    if start_epoch==spec['epochs']:
        atomic_save(output/f'v02_seed{seed}_last.pt',state)
    with torch.no_grad():last_logits=head(dev,dev_mask).detach().cpu().numpy()
    head.load_state_dict(best_state);head.eval()
    with torch.no_grad():
        best_logits=head(dev,dev_mask).detach().cpu().numpy()
        best_metrics=metrics(head(dev,dev_mask),dev_y)
    if best_metrics['mean_soft_bce']!=best_loss:raise ValueError('Best model reload is not exact')
    best_path=output/f'v02_seed{seed}_best.pt'
    atomic_save(best_path,{'model':best_state,'seed':seed,'epoch':best_epoch,'fingerprint':fingerprint})
    prediction_path=output/f'v02_seed{seed}_development.npz'
    np.savez_compressed(prediction_path,best_logits=best_logits,last_logits=last_logits,
        ids=np.array([r['StudyInstanceUID'] for r in V02_BASELINE['splits']['development']]))
    return {'seed':seed,'epochs':len(history),'best_epoch':best_epoch,'best_metrics':best_metrics,
        'history':history,'prediction_sha256':sha(prediction_path),'best_checkpoint_sha256':sha(best_path),
        'last_checkpoint_sha256':sha(output/f'v02_seed{seed}_last.pt'),'fingerprint':fingerprint}


def main(resume_directory=None):
    from transformers import Dinov2Model,Dinov2Config
    import transformers,pydicom,PIL
    started=time.perf_counter();output=Path('/kaggle/working');spec=V02_BASELINE['spec']
    def guard():
        if time.perf_counter()-started>spec['timeout_seconds']-150:raise TimeoutError('Budget guard; retain completed checkpoints')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('T4x2 required, only cuda:0 used')
    torch.manual_seed(spec['encoder_seed']);torch.cuda.manual_seed_all(spec['encoder_seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    contract=hashlib.sha256(json.dumps(V02_BASELINE,sort_keys=True).encode()).hexdigest()
    rows=V02_BASELINE['splits']['train']+V02_BASELINE['splits']['development']
    ids=[r['StudyInstanceUID'] for r in rows]
    image_hashes=[s['image_sha256'] for r in rows for s in r['series']]
    feature_start=time.perf_counter();feature_path=output/'v02_baseline_features.npz'
    if resume_directory is None:
        model_files,roots=discover_inputs(Path('/kaggle/input'))
        models=[p for p in model_files if p.stat().st_size==88297097 and sha(p)==MODEL_SHA]
        if len(models)!=1 or len(roots)!=1:raise ValueError('Ambiguous model/competition attachment')
        config_path=models[0].parent/'config.json'
        if sha(config_path)!=CONFIG_SHA:raise ValueError('Model config drift')
        encoder=Dinov2Model(Dinov2Config.from_json_file(str(config_path)))
        encoder.load_state_dict(torch.load(models[0],map_location='cpu',weights_only=True),strict=True)
        encoder=encoder.eval().requires_grad_(False).cuda()
        helper={'__name__':'v02_cache_helper','__file__':'/kaggle/working/v02_cache_helper.py'}
        exec(compile(CACHE_SOURCE,'<local-cache-helper>','exec'),helper)
        helper['normalize_slice']=normalize_cache_compatible
        features=[]
        # Six series at a time, same encoder batch size as the audited pilot.
        with torch.inference_mode():
            for begin in range(0,len(rows),2):
                guard();images=[]
                for row in rows[begin:begin+2]:
                    for series in row['series']:
                        image,_=rebuild_series(roots[0],row['StudyInstanceUID'],series,helper,pydicom,output)
                        images.append(image)
                tensor=torch.from_numpy(np.stack(images)).float().div_(255)
                tensor=(tensor-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
                features.append(encoder(pixel_values=tensor.cuda()).last_hidden_state[:,0].cpu().numpy())
                if begin%50==0:print('features',min(begin+2,len(rows)),'/',len(rows),flush=True)
        features=np.concatenate(features).reshape(len(rows),3,384)
        if any(p.grad is not None for p in encoder.parameters()):raise ValueError('Frozen encoder received gradients')
        del encoder
    else:
        with np.load(resume_directory/'v02_baseline_features.npz',allow_pickle=False) as a:
            if str(a['contract_hash'])!=contract or a['ids'].tolist()!=ids or a['image_hashes'].tolist()!=image_hashes:
                raise ValueError('Resume feature contract mismatch')
            features=a['features'].copy()
    if features.shape!=(549,3,384) or features.dtype!=np.float32 or not np.isfinite(features).all():
        raise ValueError('Feature identity/shape/dtype/value drift')
    feature_fingerprint=hashlib.sha256(features.tobytes()).hexdigest()
    np.savez_compressed(feature_path,features=features,ids=np.array(ids),image_hashes=np.array(image_hashes),contract_hash=np.array(contract))
    feature_seconds=time.perf_counter()-feature_start
    x=torch.from_numpy(features.copy()).cuda()
    y=torch.tensor([r['labels'] for r in rows],dtype=torch.float32,device='cuda')
    if not torch.isfinite(y).all() or not ((y>=0)&(y<=1)).all():raise ValueError('Invalid teacher')
    train=x[:299];dev=x[299:];train_y=y[:299];dev_y=y[299:]
    prior=train_y.double().mean(dim=0).clamp(1e-6,1-1e-6)
    prior_logits=torch.logit(prior).float().expand(len(dev),-1)
    prior_metrics=metrics(prior_logits,dev_y)
    np.savez_compressed(output/'v02_prior_development.npz',logits=prior_logits.cpu().numpy())
    results=[]
    for seed in spec['seeds']:
        fingerprint=hashlib.sha256(f'{contract}:{feature_fingerprint}:{seed}'.encode()).hexdigest()
        results.append(train_seed(train,train_y,dev,dev_y,seed,fingerprint,output,guard,
            None if resume_directory is None else resume_directory/f'v02_seed{seed}_last.pt'))
    receipt={'status':'COMPLETE_V02_WEAK_BASELINE_NOT_SUBMISSION','contract_hash':contract,
        'manifest_sha256':V02_BASELINE['manifest_sha256'],'model_sha256':MODEL_SHA,'spec':spec,
        'feature_sha256':sha(feature_path),'feature_fingerprint':feature_fingerprint,
        'prior_prediction_sha256':sha(output/'v02_prior_development.npz'),
        'prior_metrics':prior_metrics,'seeds':results,'studies':549,'series':1647,'dicom_files':4941,
        'development_evaluated':True,'confirmation_evaluated':False,'auc_measured':False,
        'submission_eligible':False,'resumed':resume_directory is not None,
        'timings_seconds':{'features':feature_seconds,'total':time.perf_counter()-started},
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
                    'pydicom':pydicom.__version__,'pillow':PIL.__version__},
        'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)]}
    (output/'v02_baseline_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE',[(s['seed'],s['best_epoch'],s['best_metrics']['mean_soft_bce']) for s in results],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-directory',type=Path)
    main(parser.parse_args().resume_directory)
