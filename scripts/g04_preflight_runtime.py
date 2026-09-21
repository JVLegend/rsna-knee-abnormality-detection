"""#RSNA #Kaggle #Pesquisa — no labels, head fitting or held-out studies."""


def find_archive(root):
    found=[]; count=0
    for directory, dirs, files in os.walk(root):
        count+=1
        if count>256: raise ValueError('Attachment discovery budget')
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','kaggle_test_series'}]
        if 'v03_features.npz' in files: found.append(Path(directory)/'v03_features.npz')
    if len(found)!=1: raise ValueError('Ambiguous V03 feature attachment')
    return found[0]


def check_features(value, rows):
    if value.shape!=(rows,3,384) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid encoder features')


def main():
    import pydicom
    from transformers import Dinov2Model,Dinov2Config
    started=time.perf_counter(); output=Path('/kaggle/working'); rows=G04['rows']
    contract=hashlib.sha256(json.dumps(G04,sort_keys=True).encode()).hexdigest()
    def guard():
        if time.perf_counter()-started>G04['guard_seconds']: raise TimeoutError('G04 pilot budget')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):
        raise ValueError('Require T4x2; only cuda:0 used')
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model_files, roots=discover_inputs(Path('/kaggle/input'))
    models=[p for p in model_files if p.stat().st_size==88297097 and sha(p)==G04['model_sha256']]
    if len(models)!=1 or len(roots)!=1: raise ValueError('Ambiguous model/competition')
    archive=find_archive(Path('/kaggle/input'))
    if sha(archive)!=G04['feature_sha256']: raise ValueError('Frozen features changed')
    ids=[r['StudyInstanceUID'] for r in rows]
    with np.load(archive,allow_pickle=False) as a:
        all_ids=a['train_ids'].tolist()
        if len(all_ids)!=len(set(all_ids)) or not set(ids)<=set(all_ids): raise ValueError('Pilot not training-only')
        reference=a['train'][[all_ids.index(uid) for uid in ids]]
    check_features(reference,len(rows))
    config=models[0].parent/'config.json'
    if sha(config)!=G04['config_sha256']: raise ValueError('Encoder config changed')
    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    encoder=Dinov2Model(Dinov2Config.from_json_file(str(config)))
    encoder.load_state_dict(torch.load(models[0],map_location='cpu',weights_only=True),strict=True)
    encoder=encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    helper={'__name__':'g04_cache_helper','__file__':'/kaggle/working/g04_cache_helper.py'}
    exec(compile(CACHE_SOURCE,'<frozen-cache-helper>','exec'),helper)
    warmup_started=time.perf_counter()
    with torch.no_grad():
        for size in G04['resolutions']:
            encoder(pixel_values=torch.zeros((6,3,size,size),device='cuda'))
    torch.cuda.synchronize(); warmup_seconds=time.perf_counter()-warmup_started
    chunks={size:[] for size in G04['resolutions']}; timings=[]; records=[]
    for begin in range(0,len(rows),2):
        guard(); decode_started=time.perf_counter(); images={size:[] for size in G04['resolutions']}
        for study in rows[begin:begin+2]:
            for s in study['series']:
                folder=roots[0]/'train_series'/study['StudyInstanceUID']/s['series_uid']; files=sorted(folder.glob('*.dcm'))
                if not files: raise ValueError('Missing pilot series')
                headers=[(p.name,pydicom.dcmread(p,stop_before_pixels=True,specific_tags=['ImageOrientationPatient','ImagePositionPatient'])) for p in files]
                plan=physical_plan(headers,[files[0].name]*3); selected=plan['arms']['physical_adjacent']
                expected=G04['expected_geometry'][len(records)]
                if selected!=expected['selected']: raise ValueError('Pilot physical sampling drift')
                channels={size:[] for size in G04['resolutions']}
                for name in selected['files']:
                    ds=pydicom.dcmread(folder/name,specific_tags=helper['PIXEL_TAGS'])
                    norm,_,_=normalize_cache_compatible(helper['_pixel_array'](ds))
                    for size in G04['resolutions']: channels[size].append(helper['resize_slice'](norm,size))
                hashes={}
                for size in G04['resolutions']:
                    value=np.stack(channels[size])
                    if value.shape!=(3,size,size) or value.dtype!=np.uint8 or any(float(c.std())==0 for c in value): raise ValueError('Invalid resized pixels')
                    hashes[str(size)]=hashlib.sha256(value.tobytes()).hexdigest(); images[size].append(value)
                if hashes['224']!=expected['pixel_sha256']: raise ValueError('224 pixel baseline mismatch')
                records.append({'study':study['StudyInstanceUID'],'series':s['series_uid'],'plane':s['plane'],
                                'selected':selected,'pixel_sha256':hashes})
        timing={'begin':begin,'studies':len(rows[begin:begin+2]),'decode_and_both_resizes_seconds':time.perf_counter()-decode_started,'inference_seconds':{},'peak_allocated_bytes':{}}
        sizes=G04['resolutions'] if begin//2%2==0 else list(reversed(G04['resolutions']))
        timing['order']=sizes
        for size in sizes:
            guard(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick=time.perf_counter()
            tensor=torch.from_numpy(np.stack(images[size])).cuda().float()/255
            with torch.no_grad(): features=encoder(pixel_values=(tensor-mean)/std).last_hidden_state[:,0]
            result=features.cpu().numpy().reshape(-1,3,384); torch.cuda.synchronize()
            check_features(result,timing['studies']); chunks[size].append(result)
            timing['inference_seconds'][str(size)]=time.perf_counter()-tick
            timing['peak_allocated_bytes'][str(size)]=torch.cuda.max_memory_allocated()
            del tensor,features
        timings.append(timing); print('pilot_studies',begin+timing['studies'],'/',len(rows),flush=True)
    values={size:np.concatenate(chunks[size]) for size in G04['resolutions']}
    for value in values.values(): check_features(value,len(rows))
    if not np.allclose(values[224],reference,atol=G04['atol'],rtol=G04['rtol']): raise ValueError('224 frozen encoder parity failed')
    feature_path=output/'g04_features.npz'; geometry_path=output/'g04_geometry.json'
    np.savez_compressed(feature_path,features224=values[224],features336=values[336],ids=ids,contract_hash=contract)
    geometry_path.write_text(json.dumps(records))
    receipt={'status':'COMPLETE_G04_PREFLIGHT_NOT_MODEL_VALIDATION','spec':G04,'contract_hash':contract,
             'feature_sha256':sha(feature_path),'geometry_sha256':sha(geometry_path),'timings':timings,
             'warmup_seconds':warmup_seconds,'seconds':time.perf_counter()-started,
             'max_224_feature_delta':float(np.abs(values[224]-reference).max()),
             'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
             'training_performed':False,'head_predictions_generated':False,'confirmation_evaluated':False,
             'submission_eligible':False}
    (output/'g04_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE_G04_PREFLIGHT_NOT_MODEL_VALIDATION',receipt['seconds'],flush=True)


if __name__=='__main__':
    try: main()
    except Exception as e:
        (Path('/kaggle/working')/'g04_failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e)}))
        raise
