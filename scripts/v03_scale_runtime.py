"""#RSNA #Kaggle #Pesquisa — frozen features plus additional train-only studies."""


def main():
    import pydicom
    from transformers import Dinov2Model,Dinov2Config
    started=time.perf_counter();output=Path('/kaggle/working');m=V03['manifest'];spec=V02_BASELINE['spec']
    def guard():
        if time.perf_counter()-started>1050:raise TimeoutError('V03 budget guard')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('Require T4x2, only cuda:0 used')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    contract=hashlib.sha256(json.dumps(V03,sort_keys=True).encode()).hexdigest()
    model_files,roots=discover_inputs(Path('/kaggle/input'))
    models=[p for p in model_files if p.stat().st_size==88297097 and sha(p)=='1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050']
    archives=[]
    for directory,dirs,files in os.walk('/kaggle/input'):
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','kaggle_test_series'}]
        if 'g01_features.npz' in files:archives.append(Path(directory)/'g01_features.npz')
    if len(roots)!=1 or len(models)!=1 or len(archives)!=1:raise ValueError('Ambiguous inputs')
    archive=archives[0];gp=archive.parent/'g01_geometry.json'
    if sha(archive)!=V03['feature_sha256'] or sha(gp)!=V03['geometry_sha256']:raise ValueError('Reference artifact changed')
    with np.load(archive,allow_pickle=False) as a:
        prior=a['physical_adjacent'].copy()
        if a['ids'].tolist()!=[r['StudyInstanceUID'] for r in m['train'][:299]+m['development']]:raise ValueError('Frozen feature order changed')
    if prior.shape!=(549,3,384) or not np.isfinite(prior).all():raise ValueError('Invalid reference features')
    reserved_ids={r['StudyInstanceUID'] for r in m['development']+m['reserved']}
    reserved_groups={r['report_hash'] for r in m['development']+m['reserved']}
    if any(r['StudyInstanceUID'] in reserved_ids or r['report_hash'] in reserved_groups for r in m['train']):
        raise ValueError('Reserved data in train')
    geometry=json.loads(gp.read_text())['series']
    dev_hashes={g['arms']['physical_adjacent']['pixel_sha256'] for g in geometry[299*3:]}
    helper={'__name__':'v03_cache_helper','__file__':'/kaggle/working/v03_cache_helper.py'}
    exec(compile(CACHE_SOURCE,'<audited-helper>','exec'),helper)
    config=models[0].parent/'config.json'
    if sha(config)!='1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d':raise ValueError('Encoder config drift')
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    encoder=Dinov2Model(Dinov2Config.from_json_file(str(config)))
    encoder.load_state_dict(torch.load(models[0],map_location='cpu',weights_only=True),strict=True)
    encoder=encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    extra=m['train'][299:];chunks=[];records=[]
    for begin in range(0,len(extra),2):
        guard();images=[]
        for study in extra[begin:begin+2]:
            for s in study['series']:
                folder=roots[0]/'train_series'/study['StudyInstanceUID']/s['series_uid']
                files=sorted(folder.glob('*.dcm'))
                if not files:raise ValueError('Selected series missing; do not exclude silently')
                headers=[(p.name,pydicom.dcmread(p,stop_before_pixels=True,specific_tags=['ImageOrientationPatient','ImagePositionPatient'])) for p in files]
                plan=physical_plan(headers,[files[0].name]*3)
                names=plan['arms']['physical_adjacent']['files'];channels=[]
                for name in names:
                    ds=pydicom.dcmread(folder/name,specific_tags=helper['PIXEL_TAGS'])
                    pixels=helper['_pixel_array'](ds);norm,_,_=normalize_cache_compatible(pixels)
                    channels.append(helper['resize_slice'](norm,224))
                image=np.stack(channels);pixel_sha=hashlib.sha256(image.tobytes()).hexdigest()
                if pixel_sha in dev_hashes:raise ValueError('New training pixels duplicate development')
                if any(float(c.std())==0 for c in image):raise ValueError('Constant image channel')
                records.append({'study':study['StudyInstanceUID'],'series':s['series_uid'],'plane':s['plane'],
                                'pixel_sha256':pixel_sha,'selected':plan['arms']['physical_adjacent'],
                                'n_slices':plan['n_slices'],'positions_mm':plan['positions_mm']})
                images.append(image)
        tensor=torch.from_numpy(np.stack(images)).cuda().float()/255
        with torch.no_grad():features=encoder(pixel_values=(tensor-mean)/std).last_hidden_state[:,0]
        chunks.append(features.cpu().numpy().reshape(-1,3,384))
        if begin%50==0:print('extra_features',min(begin+2,len(extra)),'/',len(extra),flush=True)
    extra_features=np.concatenate(chunks)
    if extra_features.shape!=(len(extra),3,384) or not np.isfinite(extra_features).all():raise ValueError('Invalid extra features')
    del encoder
    train=np.concatenate([prior[:299],extra_features]);dev=prior[299:]
    feature_path=output/'v03_features.npz';record_path=output/'v03_geometry.json'
    np.savez_compressed(feature_path,train=train,development=dev,train_ids=[r['StudyInstanceUID'] for r in m['train']],
                        development_ids=[r['StudyInstanceUID'] for r in m['development']],contract_hash=contract)
    record_path.write_text(json.dumps(records));feature_sha=sha(feature_path)
    feature_seconds=time.perf_counter()-started
    y=torch.tensor([r['labels'] for r in m['train']],dtype=torch.float32,device='cuda')
    dy=torch.tensor([r['labels'] for r in m['development']],dtype=torch.float32,device='cuda')
    dx=torch.from_numpy(dev).cuda();results=[]
    for arm,size in [('control',299),('expanded',len(train))]:
        folder=output/arm;folder.mkdir(exist_ok=False);x=torch.from_numpy(train[:size]).cuda()
        for seed in spec['seeds']:
            fp=hashlib.sha256(f'{contract}:{feature_sha}:{arm}:{seed}'.encode()).hexdigest()
            result=train_seed(x,y[:size],dx,dy,seed,fp,folder,guard);result['arm']=arm;result['training_studies']=size;results.append(result)
        (output/'v03_progress.json').write_text(json.dumps(results))
    receipt={'status':'COMPLETE_V03_NOT_SUBMISSION','contract_hash':contract,'spec':V03,
             'feature_sha256':feature_sha,'geometry_sha256':sha(record_path),'seeds':results,
             'training_studies':len(train),'development_studies':250,'extra_studies':len(extra),
             'confirmation_evaluated':False,'submission_eligible':False,
             'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
             'seconds':time.perf_counter()-started,'feature_seconds':feature_seconds,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (output/'v03_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE_V03_NOT_SUBMISSION',receipt['seconds'],flush=True)


if __name__=='__main__':main()
