"""#RSNA #Kaggle #Pesquisa — GPU-only engineering pilot, never a submission.

Builder prepends V02_PILOT and the locally authored cache helper as CACHE_SOURCE.
Only generic official DINO weights; random study-attention head, no public heads.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

MODEL_SHA='1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class StudyAttention(nn.Module):
    def __init__(self,dim=384):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
        self.classifier=nn.Linear(dim,12)

    def forward(self,features,mask):
        if features.ndim!=3 or mask.shape!=features.shape[:2] or mask.dtype!=torch.bool:
            raise ValueError('Invalid bag/mask shape or dtype')
        if not mask.any(dim=1).all():raise ValueError('Empty study bag')
        weights=self.score(features).squeeze(-1).masked_fill(~mask,float('-inf')).softmax(dim=1)
        return self.classifier((features*weights.unsqueeze(-1)).sum(dim=1))


def checkpoint(path,model,optimizer,step,fingerprint):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
             'fingerprint':fingerprint,'cpu_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all()}
    temp=path.with_suffix('.partial')
    torch.save(payload,temp);os.replace(temp,path)


def restore(path,model,optimizer,fingerprint):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['fingerprint']!=fingerprint:raise ValueError('Checkpoint data/recipe drift')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def main():
    from transformers import Dinov2Model,Dinov2Config
    import transformers,pydicom,PIL
    start=time.perf_counter()
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('Expected T4x2 allocation; pilot uses only cuda:0')
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    roots=[p.parent for p in Path('/kaggle/input').rglob('pytorch_model.bin') if p.stat().st_size==88297097]
    valid=[p for p in roots if sha(p/'pytorch_model.bin')==MODEL_SHA]
    if len(valid)!=1:raise ValueError('Exactly one official pretrained model required')
    model_dir=valid[0]
    if sha(model_dir/'config.json')!=V02_PILOT['config_sha256']:raise ValueError('Official config drift')
    config=Dinov2Config.from_json_file(str(model_dir/'config.json'))
    if config.hidden_size!=384 or config.patch_size!=14 or config.num_hidden_layers!=12:raise ValueError('Wrong encoder')
    encoder=Dinov2Model(config)
    encoder.load_state_dict(torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
    encoder=encoder.eval().requires_grad_(False).cuda()
    helper={'__name__':'v02_cache_helper','__file__':'/kaggle/working/v02_cache_helper.py'}
    exec(compile(CACHE_SOURCE,'<local-cache-helper>','exec'),helper)
    roots=[p.parent for p in Path('/kaggle/input').rglob('train_series.csv') if (p.parent/'train_series').is_dir()]
    if len(roots)!=1:raise ValueError('Expected one official competition root')
    root=roots[0]
    rebuild_start=time.perf_counter();images=[];pixel_receipts=[]
    for study in V02_PILOT['pilot']:
        for series in study['series']:
            channels=[]
            for name in series['selected_files']:
                path=root/'train_series'/study['StudyInstanceUID']/series['series_uid']/name
                ds=pydicom.dcmread(path,specific_tags=helper['PIXEL_TAGS'],force=False)
                normalized,_,_=helper['normalize_slice'](helper['_pixel_array'](ds))
                channels.append(helper['resize_slice'](normalized,224))
            image=np.stack(channels);observed=hashlib.sha256(image.tobytes()).hexdigest()
            if observed!=series['image_sha256']:raise ValueError('Rebuilt pixels differ from HD cache')
            images.append(image)
            pixel_receipts.append({'study':study['StudyInstanceUID'],'series':series['series_uid'],'sha256':observed})
        print('rebuilt',len(images)//3,'/12 studies',flush=True)
    rebuild_seconds=time.perf_counter()-rebuild_start
    tensor=torch.from_numpy(np.stack(images)).float().div_(255)
    tensor=(tensor-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    forward_start=time.perf_counter();features=[]
    with torch.inference_mode():
        for i in range(0,len(tensor),6):
            features.append(encoder(pixel_values=tensor[i:i+6].cuda()).last_hidden_state[:,0].cpu())
    torch.cuda.synchronize();forward_seconds=time.perf_counter()-forward_start
    x=torch.cat(features).reshape(12,3,384).clone().cuda()
    y=torch.tensor([s['labels'] for s in V02_PILOT['pilot']],dtype=torch.float32,device='cuda')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not ((y>=0)&(y<=1)).all():raise ValueError('Invalid features/labels')
    feature_path=Path('/kaggle/working/v02_pilot_features.npz')
    np.savez_compressed(feature_path,features=x.cpu().numpy(),ids=np.array([s['StudyInstanceUID'] for s in V02_PILOT['pilot']]))
    # No encoder training or validation evaluation. Soft teacher labels unchanged.
    head=StudyAttention().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    mask=torch.ones((12,3),dtype=torch.bool,device='cuda')
    invalid=mask.clone();invalid[0]=False
    try:head(x,invalid)
    except ValueError:pass
    else:raise AssertionError('Empty bag gate failed')
    with torch.no_grad():
        masked=mask.clone();masked[:,2]=False;changed=x.clone();changed[:,2]=123
        if not torch.equal(head(x,masked),head(changed,masked)):raise AssertionError('Masked slot influences prediction')
    losses=[];steps=[]
    def step():
        begin=time.perf_counter();selection=torch.randperm(12)[:4].cuda()
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(head(x[selection],mask[selection]),y[selection])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Invalid head gradient')
        optimizer.step();torch.cuda.synchronize();steps.append(time.perf_counter()-begin)
        return float(loss.detach().cpu())
    for _ in range(8):losses.append(step())
    fingerprint=hashlib.sha256(json.dumps(V02_PILOT,sort_keys=True).encode()).hexdigest()
    path=Path('/kaggle/working/v02_pilot_checkpoint.pt');checkpoint(path,head,optimizer,8,fingerprint)
    continued=[step() for _ in range(4)];expected={k:v.detach().clone() for k,v in head.state_dict().items()}
    if restore(path,head,optimizer,fingerprint)!=8:raise ValueError('Wrong resume step')
    resumed=[step() for _ in range(4)]
    if resumed!=continued or any(not torch.equal(expected[k],v) for k,v in head.state_dict().items()):
        raise ValueError('Resume does not exactly reproduce continued training')
    if any(p.grad is not None for p in encoder.parameters()):raise ValueError('Frozen encoder received gradients')
    receipt={'status':'PASSED_V02_ENGINEERING_PILOT_NOT_A_BASELINE','manifest_sha256':V02_PILOT['manifest_sha256'],
        'model_sha256':MODEL_SHA,'config_sha256':sha(model_dir/'config.json'),'model_root':str(model_dir),
        'input_contract':V02_PILOT['input_contract'],'pilot_fingerprint':fingerprint,
        'studies':12,'series':36,'dicom_files':108,'pixel_receipts':pixel_receipts,
        'features_sha256':sha(feature_path),'features_shape':list(x.shape),'checkpoint_sha256':sha(path),
        'checkpoint_step':8,'resume_exact':True,'mask_invariance':True,'encoder_frozen':True,
        'training_losses_first8':losses,'continued_losses':continued,'resumed_losses':resumed,
        'timings_seconds':{'rebuild':rebuild_seconds,'forward':forward_seconds,'head_steps':steps,'total':time.perf_counter()-start},
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
                    'pydicom':pydicom.__version__,'pillow':PIL.__version__},
        'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],'active_gpu_count':1,
        'development_evaluated':False,'confirmation_evaluated':False,'auc_measured':False,
        'submission_eligible':False,'full_training_started':False}
    Path('/kaggle/working/v02_pilot_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='pixel_receipts'},indent=2),flush=True)


if __name__=='__main__':main()
