"""#RSNA #Kaggle #Pesquisa — geometry first, frozen encoder, paired heads."""
MODEL_SHA = '1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050'
CONFIG_SHA = '1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'


def main():
    import pydicom
    from transformers import Dinov2Model, Dinov2Config
    started = time.perf_counter();output = Path('/kaggle/working')
    spec = V02_BASELINE['spec']
    def guard():
        if time.perf_counter()-started > spec['timeout_seconds']-150:
            raise TimeoutError('G01 budget guard')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('T4x2 required, only cuda:0 used')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    contract = hashlib.sha256(json.dumps(G01,sort_keys=True).encode()).hexdigest()
    rows = V02_BASELINE['splits']['train']+V02_BASELINE['splits']['development']
    ids = [r['StudyInstanceUID'] for r in rows]
    # Discover without walking tens of thousands of DICOM paths.
    models,roots = discover_inputs(Path('/kaggle/input'))
    candidates = [p for p in models if p.stat().st_size==88297097 and sha(p)==MODEL_SHA]
    if len(roots)!=1 or len(candidates)!=1:raise ValueError('Ambiguous input attachments')
    root = roots[0];model_path = candidates[0]
    archives = []
    for directory,dirs,files in os.walk('/kaggle/input'):
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','kaggle_test_series'}]
        if 'v02_baseline_features.npz' in files:archives.append(Path(directory)/'v02_baseline_features.npz')
    if len(archives)!=1 or sha(archives[0])!=G01['feature_sha256']:raise ValueError('Baseline features changed')
    with np.load(archives[0],allow_pickle=False) as a:
        control=a['features']
        if (a['ids'].tolist()!=ids or str(a['contract_hash'])!=G01['baseline_contract']
            or control.shape!=(549,3,384) or control.dtype!=np.float32 or not np.isfinite(control).all()
            or hashlib.sha256(control.tobytes()).hexdigest()!=G01['feature_fingerprint']):
            raise ValueError('Baseline feature identity drift')
    helper={'__name__':'g01_cache_helper','__file__':'/kaggle/working/g01_cache_helper.py'}
    exec(compile(CACHE_SOURCE,'<audited-cache-helper>','exec'),helper)
    images={arm:[] for arm in G01['arms'][1:]};geometry=[]
    # Hold <0.5 GiB uint8 images. No model evaluation until every geometry/pixel gate passes.
    for study_index,study in enumerate(rows):
        guard();study_images={arm:[] for arm in images}
        for series in study['series']:
            folder=root/'train_series'/study['StudyInstanceUID']/series['series_uid']
            files=sorted(folder.glob('*.dcm'))
            if len(files)!=G01['series_counts'][study['StudyInstanceUID']+'/'+series['series_uid']]:
                raise ValueError('Series file count drift')
            headers=[(p.name,pydicom.dcmread(p,stop_before_pixels=True,
                       specific_tags=['ImageOrientationPatient','ImagePositionPatient'])) for p in files]
            try:
                plan=physical_plan(headers,series['selected_files'])
            except ValueError as e:
                (output/'g01_failure.json').write_text(json.dumps({'status':'FAILED_GEOMETRY_GATE',
                    'study':study['StudyInstanceUID'],'series':series['series_uid'],'reason':str(e),
                    'completed_series':len(geometry),'confirmation_evaluated':False}))
                raise
            selected=set(series['selected_files'])
            for arm in images:selected.update(plan['arms'][arm]['files'])
            decoded={}
            for name in sorted(selected):
                ds=pydicom.dcmread(folder/name,specific_tags=helper['PIXEL_TAGS'])
                raw=helper['_pixel_array'](ds)
                norm,_,_=normalize_cache_compatible(raw)
                decoded[name]=helper['resize_slice'](norm,224)
            original=np.stack([decoded[name] for name in series['selected_files']])
            if hashlib.sha256(original.tobytes()).hexdigest()!=series['image_sha256']:
                raise ValueError('Original V02 pixel parity failed')
            plan.update(study=study['StudyInstanceUID'],series=series['series_uid'],plane=series['plane'],
                        original_files=series['selected_files'],original_pixel_sha256=series['image_sha256'])
            for arm in images:
                arr=np.stack([decoded[name] for name in plan['arms'][arm]['files']])
                plan['arms'][arm]['pixel_sha256']=hashlib.sha256(arr.tobytes()).hexdigest()
                study_images[arm].append(arr)
            geometry.append(plan)
        for arm in images:images[arm].append(np.stack(study_images[arm]))
        if (study_index+1)%25==0:print('geometry studies',study_index+1,flush=True)
    geometry_path=output/'g01_geometry.json'
    geometry_path.write_text(json.dumps({'contract_hash':contract,'series':geometry}))
    preprocessing_seconds=time.perf_counter()-started
    config_path=model_path.parent/'config.json'
    if sha(config_path)!=CONFIG_SHA:raise ValueError('Encoder config drift')
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    encoder=Dinov2Model(Dinov2Config.from_json_file(str(config_path)))
    encoder.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True),strict=True)
    encoder=encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    features={'control':control}
    for arm in images:
        chunks=[]
        for start in range(0,549,2):
            guard();batch=np.stack(images[arm][start:start+2]).reshape(-1,3,224,224)
            tensor=torch.from_numpy(batch).cuda().float()/255
            with torch.no_grad():f=encoder(pixel_values=(tensor-mean)/std).last_hidden_state[:,0]
            chunks.append(f.cpu().numpy().reshape(-1,3,384))
        features[arm]=np.concatenate(chunks)
        if not np.isfinite(features[arm]).all():raise ValueError('Nonfinite encoder output')
        print('features complete',arm,flush=True)
    del encoder,images
    feature_path=output/'g01_features.npz'
    np.savez_compressed(feature_path,**features,ids=np.asarray(ids),contract_hash=contract)
    feature_hash=sha(feature_path);results=[]
    y=torch.tensor([r['labels'] for r in rows[:299]],dtype=torch.float32,device='cuda')
    dy=torch.tensor([r['labels'] for r in rows[299:]],dtype=torch.float32,device='cuda')
    for arm in G01['arms']:
        folder=output/arm;folder.mkdir(exist_ok=False)
        x=torch.from_numpy(features[arm][:299]).cuda();dx=torch.from_numpy(features[arm][299:]).cuda()
        for seed in spec['seeds']:
            fp=hashlib.sha256(f'{contract}:{feature_hash}:{arm}:{seed}'.encode()).hexdigest()
            result=train_seed(x,y,dx,dy,seed,fp,folder,guard);result['arm']=arm;results.append(result)
        (output/'g01_progress.json').write_text(json.dumps(results))
    receipt={'status':'COMPLETE_G01_NOT_SUBMISSION','contract_hash':contract,'spec':G01,
             'feature_sha256':feature_hash,'geometry_sha256':sha(geometry_path),'seeds':results,
             'seconds':time.perf_counter()-started,'preprocessing_seconds':preprocessing_seconds,
             'confirmation_evaluated':False,'submission_eligible':False,
             'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'versions':{'torch':str(torch.__version__),'numpy':np.__version__,'pydicom':pydicom.__version__}}
    (output/'g01_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE_G01_NOT_SUBMISSION',receipt['seconds'],flush=True)


if __name__=='__main__':main()
