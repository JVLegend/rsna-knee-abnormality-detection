"""#RSNA #Kaggle #Pesquisa — bounded train-only pixel audit for V02."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.freeze_weak_validation import digest, freeze, TARGETS

MANIFEST_SHA='365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df'
PLANES=('Sagittal','Coronal','Axial')


def check_image(path, series, record):
    if path.stat().st_size != series['bytes']:raise ValueError('Frozen array size changed')
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'image','sample_indices'}:raise ValueError('Unexpected cache keys')
        image,indices=a['image'],a['sample_indices']
    if image.shape!=(3,224,224) or image.dtype!=np.uint8:raise ValueError('Invalid image shape/dtype')
    expected=[int(round(q*(series['slices_recorded']-1))) for q in [.25,.5,.75]]
    if indices.tolist()!=expected or record['sample_indices']!=expected or len(set(expected))!=3:
        raise ValueError('Sampling indices drift or repeated slices')
    if any(float(channel.std())<=0 for channel in image):raise ValueError('Constant image channel')
    return {'npz_sha256':digest(path),'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'sample_indices':expected,'channel_std':[float(x.std()) for x in image],
            'sample_gaps':np.diff(indices).tolist()}


def audit(manifest_path):
    if digest(manifest_path)!=MANIFEST_SHA:raise ValueError('V01 drift')
    m=json.loads(manifest_path.read_text());index_path=Path(m['sources']['cache_index']['path'])
    if digest(index_path)!=m['sources']['cache_index']['sha256']:raise ValueError('Cache index drift')
    index=json.loads(index_path.read_text());root=index_path.parent.resolve()
    if (index['size'],index['channels'],index['quantiles'])!=(224,3,[.25,.5,.75]):raise ValueError('Cache recipe drift')
    records={(r['study_uid'],r['series_uid']):r for r in index['records']}
    if len(records)!=len(index['records']):raise ValueError('Duplicate index series')
    seen_ids,seen_groups=set(),set()
    for rows in m['splits'].values():
        ids=[r['StudyInstanceUID'] for r in rows];groups={r['report_hash'] for r in rows}
        if len(ids)!=len(set(ids)) or seen_ids.intersection(ids) or seen_groups&groups:raise ValueError('Split overlap')
        seen_ids.update(ids);seen_groups|=groups
    studies=[];hashes=Counter();gaps=[]
    for study in m['splits']['train']:
        uid=study['StudyInstanceUID'];out=[]
        if sorted(s['plane'] for s in study['series'])!=sorted(PLANES):raise ValueError('Require exactly three planes')
        for s in sorted(study['series'],key=lambda x:PLANES.index(x['plane'])):
            p=(root/s['array_path']).resolve()
            if not p.is_relative_to(root):raise ValueError('Array outside cache root')
            r=records[(uid,s['series_uid'])]
            if r['n_slices']!=s['slices_recorded'] or r['anatomical_plane']!=s['plane']:raise ValueError('Series metadata drift')
            c=check_image(p,s,r);hashes[c['image_sha256']]+=1;gaps.extend(c['sample_gaps'])
            files=[x['file'] for x in r['selected']]
            if len(files)!=3 or any(Path(f).name!=f for f in files):raise ValueError('Unsafe selected filename')
            out.append(dict(s,**c,selected_files=files,selected_shapes=[x['input_shape'] for x in r['selected']]))
        studies.append({'StudyInstanceUID':uid,'report_hash':study['report_hash'],
                        'labels':[study['labels'][t] for t in TARGETS],'series':out})
    # Coverage/cost strata use pixel dimensions only, never labels.
    ordered=sorted(studies,key=lambda s:(sum(h*w for r in s['series'] for h,w in r['selected_shapes']),s['StudyInstanceUID']))
    positions=np.linspace(0,len(ordered)-1,12).round().astype(int).tolist()
    pilot=sorted([ordered[i] for i in positions],key=lambda s:s['StudyInstanceUID'])
    if len({s['StudyInstanceUID'] for s in pilot})!=12:raise ValueError('Insufficient pilot coverage')
    return {'status':'PASSED_TRAIN_CACHE_PIXELS','manifest_sha256':MANIFEST_SHA,
        'cache_index_sha256':digest(index_path),'train_studies':len(studies),'series':sum(len(s['series']) for s in studies),
        'channels':len(gaps)//2*3,'bytes':sum(r['bytes'] for s in studies for r in s['series']),
        'pixel_hash_duplicates':sum(v-1 for v in hashes.values()),
        'sample_gap_min':min(gaps),'sample_gap_max':max(gaps),'all_triplets_adjacent':all(x==1 for x in gaps),
        'pilot':pilot,'studies':studies,'development_pixels_read':0,'confirmation_pixels_read':0,
        'limitations':['Pixel integrity is not correctness of anatomical orientation.',
            'Selected-file rebuild still required; complete DICOM series geometry not reaudited.',
            'Within-train exact array hashes do not exclude patient/visual leakage across reserved splits.',
            'Cache uses spaced quantile channels, not adjacent slices.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/processed/validation_weak_v1/manifest.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.manifest);freeze(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ['studies','pilot']},indent=2))


if __name__=='__main__':main()
