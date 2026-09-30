"""#RSNA #Kaggle #Pesquisa — builder supplies frozen payload and original control."""


def sample_presence(size, probability, generator):
    if size < 1 or not 0 <= probability <= 1: raise ValueError('Invalid mask parameters')
    selected = torch.rand(size,generator=generator) < probability
    planes = torch.randint(3,(size,),generator=generator)
    mask = torch.ones((size,3),dtype=torch.bool)
    rows = torch.arange(size)[selected]
    mask[rows,planes[selected]] = False
    return mask


def trace_masks(trace, mask):
    trace['seen'] += len(mask)
    trace['dropped'] = [a+b for a,b in zip(trace['dropped'],(~mask).sum(0).tolist())]
    trace['chain'] = hashlib.sha256(bytes.fromhex(trace['chain'])+mask.numpy().tobytes()).hexdigest()


def paired_loss(clean, masked, labels):
    return (.5*nn.functional.binary_cross_entropy_with_logits(clean,labels)
            +.5*nn.functional.binary_cross_entropy_with_logits(masked,labels)
            +.1*nn.functional.mse_loss(masked.sigmoid(),clean.detach().sigmoid()))


def robust_loss(head, features, labels, arm, generator, trace):
    if arm not in ['dropout25','paired_consistency']: raise ValueError('Unknown robustness arm')
    mask = sample_presence(len(features),.25 if arm=='dropout25' else 1.,generator)
    trace_masks(trace,mask)
    masked = head(features,mask.to(features.device))
    if arm=='dropout25': return nn.functional.binary_cross_entropy_with_logits(masked,labels)
    clean = head(features,torch.ones(features.shape[:2],dtype=torch.bool,device=features.device))
    return paired_loss(clean,masked,labels)


def load_features(path):
    if sha(path)!=R02['feature_sha256']: raise ValueError('Feature SHA changed')
    with np.load(path,allow_pickle=False) as a:
        if str(a['contract_hash'])!=R02['baseline_contract']: raise ValueError('Feature contract changed')
        arrays=[]
        for split,count in [('train',1000),('development',250)]:
            x=a[split].copy()
            if (x.shape!=(count,3,384) or x.dtype!=np.float32 or not np.isfinite(x).all()
                or a[split+'_ids'].tolist()!=[r['StudyInstanceUID'] for r in V02_BASELINE['splits'][split]]):
                raise ValueError('Feature identity/shape drift')
            arrays.append(x)
    return arrays


def evaluate_missing(directory,seed,dev,labels):
    state=torch.load(directory/f'v02_seed{seed}_best.pt',map_location='cpu',weights_only=True)
    head=StudyAttention().cuda();head.load_state_dict(state['model']);head.eval()
    values=[];predictions=[]
    with torch.no_grad():
        for i in range(3):
            mask=torch.ones(dev.shape[:2],dtype=torch.bool,device=dev.device);mask[:,i]=False
            logits=head(dev,mask);predictions.append(logits.cpu().numpy());values.append(metrics(logits,labels))
    path=directory/f'r02_seed{seed}_missing.npz'
    np.savez_compressed(path,logits=np.stack(predictions),ids=[r['StudyInstanceUID'] for r in V02_BASELINE['splits']['development']])
    return {'metrics':values,'prediction_sha256':sha(path)}


def main():
    start=time.perf_counter();output=Path('/kaggle/working')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('Require T4x2, only cuda:0 used')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    def guard():
        if time.perf_counter()-start>450: raise TimeoutError('R02 budget guard')
    archives=sorted(Path('/kaggle/input').rglob('v03_features.npz'))
    if len(archives)!=1: raise ValueError('Ambiguous V03 attachment')
    train,dev=load_features(archives[0])
    if sha(archives[0].parent/'v03_receipt.json')!=R02['baseline_receipt_sha256']:
        raise ValueError('Input receipt changed')
    x=torch.from_numpy(train).cuda();dx=torch.from_numpy(dev).cuda()
    y=torch.tensor([r['labels'] for r in V02_BASELINE['splits']['train']],dtype=torch.float32,device='cuda')
    dy=torch.tensor([r['labels'] for r in V02_BASELINE['splits']['development']],dtype=torch.float32,device='cuda')
    if any(not torch.isfinite(z).all() or not ((z>=0)&(z<=1)).all() for z in [y,dy]):
        raise ValueError('Invalid teacher')
    contract=hashlib.sha256(json.dumps(R02,sort_keys=True).encode()).hexdigest();results=[]
    for arm in R02['arms']:
        folder=output/arm;folder.mkdir(exist_ok=False)
        for seed in [2026,42]:
            guard();fp=hashlib.sha256(f'{contract}:{arm}:{seed}'.encode()).hexdigest()
            args=(x,y,dx,dy,seed,fp,folder,guard)
            result=train_seed(*args) if arm=='control' else train_candidate(*args,arm=arm)
            result['arm']=arm;result['missing']=evaluate_missing(folder,seed,dx,dy);results.append(result)
            (output/'r02_progress.json').write_text(json.dumps(results,indent=2))
    receipt={'status':'COMPLETE_R02_TRAINING_NOT_SUBMISSION','spec':R02,'contract_hash':contract,
             'seeds':results,'seconds':time.perf_counter()-start,'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'confirmation_evaluated':False,'submission_eligible':False}
    (output/'r02_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE_R02_TRAINING_NOT_SUBMISSION',receipt['seconds'],flush=True)


if __name__=='__main__':main()
