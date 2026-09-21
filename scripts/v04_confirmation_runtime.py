"""#RSNA #Kaggle #Pesquisa — inference only; no metrics, labels fitting or competition CSV."""


def check_pixels(image, forbidden):
    if image.shape!=(3,224,224) or image.dtype!=np.uint8 or any(float(c.std())==0 for c in image):
        raise ValueError('Invalid/constant confirmation image')
    fingerprint=hashlib.sha256(image.tobytes()).hexdigest()
    if fingerprint in forbidden:raise ValueError('Confirmation pixels duplicate train/development; do not exclude case')
    return fingerprint


def find_receipt(root):
    paths=[];count=0
    for folder,dirs,files in os.walk(root):
        count+=1
        if count>256:raise ValueError('Attachment discovery budget exceeded')
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','kaggle_test_series'}]
        if 'r02_receipt.json' in files:paths.append(Path(folder)/'r02_receipt.json')
    if len(paths)!=1:raise ValueError('Ambiguous R02 attachment')
    return paths[0]


def main():
    import pydicom
    from transformers import Dinov2Model,Dinov2Config
    started=time.perf_counter();output=Path('/kaggle/working');rows=V04['rows']
    contract=hashlib.sha256(json.dumps(V04,sort_keys=True).encode()).hexdigest()
    def guard():
        if time.perf_counter()-started>V04['guard_seconds']:raise TimeoutError('V04 inference budget')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('Require T4x2; only cuda:0 used')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model_files,roots=discover_inputs(Path('/kaggle/input'))
    models=[p for p in model_files if p.stat().st_size==88297097 and sha(p)==V04['model_sha256']]
    if len(models)!=1 or len(roots)!=1:raise ValueError('Ambiguous encoder/competition')
    rp=find_receipt(Path('/kaggle/input'))
    if sha(rp)!=V04['r02_receipt_sha256']:raise ValueError('R02 receipt changed')
    for c in V04['checkpoints']:
        if sha(rp.parent/c['arm']/f"v02_seed{c['seed']}_best.pt")!=c['sha256']:raise ValueError('Selected checkpoint changed')
    config=models[0].parent/'config.json'
    if sha(config)!=V04['config_sha256']:raise ValueError('Encoder config changed')
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    encoder=Dinov2Model(Dinov2Config.from_json_file(str(config)))
    encoder.load_state_dict(torch.load(models[0],map_location='cpu',weights_only=True),strict=True)
    encoder=encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    helper={'__name__':'v04_cache_helper','__file__':'/kaggle/working/v04_cache_helper.py'}
    exec(compile(CACHE_SOURCE,'<audited-cache-helper>','exec'),helper)
    forbidden=set(V04['forbidden_pixel_hashes']);chunks=[];records=[]
    for begin in range(0,len(rows),2):
        guard();images=[]
        for study in rows[begin:begin+2]:
            for s in study['series']:
                folder=roots[0]/'train_series'/study['StudyInstanceUID']/s['series_uid'];files=sorted(folder.glob('*.dcm'))
                if not files:raise ValueError('Confirmation series missing; no replacement permitted')
                headers=[(p.name,pydicom.dcmread(p,stop_before_pixels=True,specific_tags=['ImageOrientationPatient','ImagePositionPatient'])) for p in files]
                plan=physical_plan(headers,[files[0].name]*3);selected=plan['arms']['physical_adjacent'];channels=[]
                for name in selected['files']:
                    ds=pydicom.dcmread(folder/name,specific_tags=helper['PIXEL_TAGS'])
                    norm,_,_=normalize_cache_compatible(helper['_pixel_array'](ds))
                    channels.append(helper['resize_slice'](norm,224))
                image=np.stack(channels);fingerprint=check_pixels(image,forbidden)
                records.append({'study':study['StudyInstanceUID'],'series':s['series_uid'],'plane':s['plane'],
                                'pixel_sha256':fingerprint,'selected':selected,'plan':plan,
                                'headers':[{'name':name,'iop':list(map(float,h.ImageOrientationPatient)),
                                            'ipp':list(map(float,h.ImagePositionPatient))} for name,h in headers]})
                images.append(image)
        tensor=torch.from_numpy(np.stack(images)).cuda().float()/255
        with torch.no_grad():features=encoder(pixel_values=(tensor-mean)/std).last_hidden_state[:,0]
        chunks.append(features.cpu().numpy().reshape(-1,3,384))
        if begin%30==0:print('confirmation_features',min(begin+2,len(rows)),'/',len(rows),flush=True)
    features=np.concatenate(chunks);del encoder
    if features.shape!=(150,3,384) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Invalid reserved features')
    ids=[r['StudyInstanceUID'] for r in rows];feature_path=output/'v04_features.npz';geometry_path=output/'v04_geometry.json'
    np.savez_compressed(feature_path,features=features,ids=ids,contract_hash=contract)
    geometry_path.write_text(json.dumps(records));feature_seconds=time.perf_counter()-started
    # All cases pass exclusion checks before any head prediction is generated.
    exposure={'contract_hash':contract,'stage':'prediction_started','studies':150,'metrics_evaluated_in_kernel':False}
    exposure_path=output/'v04_exposure.json';exposure_path.write_text(json.dumps(exposure))
    x=torch.from_numpy(features).cuda();all_logits=[]
    for c in V04['checkpoints']:
        guard();cp=rp.parent/c['arm']/f"v02_seed{c['seed']}_best.pt"
        state=torch.load(cp,map_location='cpu',weights_only=True)
        if (state['seed'],state['epoch'],state['fingerprint'])!=(c['seed'],c['epoch'],c['fingerprint']):raise ValueError('Checkpoint identity drift')
        head=StudyAttention().cuda();head.load_state_dict(state['model'],strict=True);head.eval();values=[]
        with torch.no_grad():
            for missing in [-1,0,1,2]:
                mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
                if missing>=0:mask[:,missing]=False
                logits=head(x,mask)
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite confirmation logits')
                values.append(logits.cpu().numpy())
        all_logits.append(np.stack(values))
    pred_path=output/'v04_predictions.npz'
    np.savez_compressed(pred_path,logits=np.stack(all_logits),ids=ids,contract_hash=contract,
                        heads=[f"{c['arm']}/{c['seed']}" for c in V04['checkpoints']],
                        modes=['intact','without_Sagittal','without_Coronal','without_Axial'])
    exposure['stage']='predictions_complete';exposure_path.write_text(json.dumps(exposure))
    receipt={'status':'COMPLETE_V04_RESERVED_INFERENCE_NOT_SUBMISSION','spec':V04,'contract_hash':contract,
             'feature_sha256':sha(feature_path),'geometry_sha256':sha(geometry_path),'prediction_sha256':sha(pred_path),
             'exposure_sha256':sha(exposure_path),'confirmation_predictions_generated':True,'metrics_evaluated_in_kernel':False,
             'submission_eligible':False,'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
             'seconds':time.perf_counter()-started,'feature_seconds':feature_seconds,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (output/'v04_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE_V04_RESERVED_INFERENCE_NOT_SUBMISSION',receipt['seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        (Path('/kaggle/working')/'v04_failure.json').write_text(json.dumps({'error':type(error).__name__,'message':str(error),
            'prediction_marker_exists':(Path('/kaggle/working')/'v04_exposure.json').exists()}))
        raise
