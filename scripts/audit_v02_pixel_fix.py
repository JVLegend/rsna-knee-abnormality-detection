"""#RSNA #Kaggle #Pesquisa — reproduce the V02 pixel failure, train-only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pydicom
import PIL
from scripts import build_dicom_25d_features as legacy
from scripts.freeze_weak_validation import digest,freeze
from scripts.prepare_v02_pilot import validate
from scripts.preflight_v02_cache import MANIFEST_SHA


def audit(cache,remote,output):
    manifest=Path('data/processed/validation_weak_v1/manifest.json')
    if digest(manifest)!=MANIFEST_SHA:raise ValueError('Frozen split drift')
    m=json.loads(manifest.read_text());index=Path(m['sources']['cache_index']['path'])
    if digest(index)!=m['sources']['cache_index']['sha256']:raise ValueError('Frozen index drift')
    payload=json.loads(cache.read_text());pilot=validate(payload)
    source=Path('scripts/v02_pilot_runtime.py').read_text();tree=ast.parse(source)
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_cache_compatible']
    if len(node)!=1:raise ValueError('Missing normalization function')
    namespace={'np':np};exec(compile(ast.Module(body=node,type_ignores=[]),'<normalization>','exec'),namespace)
    normalize=namespace['normalize_cache_compatible']
    evidence=json.loads((remote/'v02_pixel_mismatch.json').read_text())
    lookup={(s['StudyInstanceUID'],r['series_uid']):r for s in pilot for r in s['series']}
    key=(evidence['study'],evidence['series']);series=lookup[key]
    if evidence['expected_sha256']!=series['image_sha256']:raise ValueError('Remote expected image drift')
    with np.load(remote/'v02_pixel_mismatch.npz',allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files}
    if hashlib.sha256(arrays['image'].tobytes()).hexdigest()!=evidence['observed_sha256']:
        raise ValueError('Remote image evidence drift')
    if [r['file'] for r in evidence['stages']]!=series['selected_files']:raise ValueError('Remote file order drift')
    comparisons=[];rebuilt=[]
    for study in pilot:
        for series in study['series']:
            cache_path=index.parent/series['array_path']
            if digest(cache_path)!=series['npz_sha256']:raise ValueError('Cache archive drift')
            with np.load(cache_path,allow_pickle=False) as a:expected=a['image']
            if hashlib.sha256(expected.tobytes()).hexdigest()!=series['image_sha256']:raise ValueError('Cache pixel drift')
            fixed=[];old=[]
            for i,name in enumerate(series['selected_files']):
                path=Path('data/raw/train_series')/study['StudyInstanceUID']/series['series_uid']/name
                pixels=legacy._pixel_array(pydicom.dcmread(path,specific_tags=legacy.PIXEL_TAGS))
                normalized,low,high=normalize(pixels)
                previous,previous_low,previous_high=legacy.normalize_slice(pixels)
                fixed.append(legacy.resize_slice(normalized,224));old.append(legacy.resize_slice(previous,224))
                if (study['StudyInstanceUID'],series['series_uid'])==key:
                    recorded=evidence['stages'][i];raw=arrays[f'pixels_{i}'];norm=arrays[f'normalized_{i}']
                    if (digest(path)!=recorded['dicom_sha256'] or not np.array_equal(raw,pixels)
                        or hashlib.sha256(raw.tobytes()).hexdigest()!=recorded['pixel_sha256']
                        or hashlib.sha256(norm.tobytes()).hexdigest()!=recorded['normalized_sha256']):
                        raise ValueError('Raw DICOM or stage evidence differs')
                    comparisons.append({'channel':i,'file_and_raw_pixels_equal':True,
                        'remote_bounds':[recorded['low'],recorded['high']],
                        'legacy_local_bounds':[previous_low,previous_high],'fixed_bounds':[low,high],
                        'legacy_normalization_matches_remote':bool(np.array_equal(previous,norm)),
                        'remote_resize_reproduced_locally':bool(np.array_equal(legacy.resize_slice(norm,224),arrays['image'][i])),
                        'remote_vs_cache_changed_pixels':int(np.count_nonzero(arrays['image'][i]!=expected[i])),
                        'fixed_vs_cache_changed_pixels':int(np.count_nonzero(fixed[-1]!=expected[i]))})
            rebuilt.append({'study':study['StudyInstanceUID'],'series':series['series_uid'],
                'fixed_exact':bool(np.array_equal(np.stack(fixed),expected)),
                'legacy_exact':bool(np.array_equal(np.stack(old),expected)),
                'fixed_image_sha256':hashlib.sha256(np.stack(fixed).tobytes()).hexdigest()})
        print('rebuilt',len(rebuilt),'series',flush=True)
    passed=len(rebuilt)==36 and all(r['fixed_exact'] for r in rebuilt)
    result={'status':'PASSED_FIXED_PIXEL_REBUILD' if passed else 'FAILED_FIXED_PIXEL_REBUILD',
        'versions':{'numpy':np.__version__,'pillow':PIL.__version__,'pydicom':pydicom.__version__},
        'runtime_sha256':hashlib.sha256(source.encode()).hexdigest(),'cache_audit_sha256':digest(cache),
        'remote_evidence_sha256':digest(remote/'v02_pixel_mismatch.npz'),
        'manifest_sha256':MANIFEST_SHA,'studies':12,'series':len(rebuilt),'dicom_files':len(rebuilt)*3,
        'fixed_exact_series':sum(r['fixed_exact'] for r in rebuilt),
        'legacy_exact_series':sum(r['legacy_exact'] for r in rebuilt),
        'stages':comparisons,'rebuilt':rebuilt,'development_pixels_read':0,'confirmation_pixels_read':0,
        'training_started':False,'auc_measured':False}
    freeze(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rebuilt'},indent=2))
    if not passed:raise ValueError('Corrected reconstruction not exact')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--remote',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit(a.cache,a.remote,a.output)
