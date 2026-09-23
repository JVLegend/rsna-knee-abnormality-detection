"""#RSNA #Kaggle #Pesquisa — independent provenance/cost audit, no AUC."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.freeze_weak_validation import digest, freeze
from scripts.prepare_v02_pilot import CONFIG_SHA, validate, build as build_pilot

MODEL_SHA='1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050'


def assess(directory,cache,build):
    audit=json.loads(cache.read_text());pilot=validate(audit)
    tree=ast.parse(build.read_text())
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                 and n.targets[0].id in ['V02_PILOT','CACHE_SOURCE']}
    p=assignments['V02_PILOT']
    if p['pilot']!=pilot or p['config_sha256']!=CONFIG_SHA:raise ValueError('Build pilot/config drift')
    expected_source=build_pilot(audit,Path('scripts/build_dicom_25d_features.py').read_text(),
                                Path('scripts/v02_pilot_runtime.py').read_text())
    if build.read_text()!=expected_source:raise ValueError('Runtime/helper source drift')
    r=json.loads((directory/'v02_pilot_receipt.json').read_text())
    if (r['status']!='PASSED_V02_ENGINEERING_PILOT_NOT_A_BASELINE'
        or r['manifest_sha256']!=audit['manifest_sha256'] or r['model_sha256']!=MODEL_SHA
        or r['config_sha256']!=CONFIG_SHA or r['input_contract']!=p['input_contract']
        or (r['studies'],r['series'],r['dicom_files'])!=(12,36,108)
        or r['gpu_names']!=['Tesla T4']*2 or r['active_gpu_count']!=1):raise ValueError('Pilot recipe/provenance failed')
    if r['pilot_fingerprint']!=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest():raise ValueError('Pilot fingerprint drift')
    if any(r[k] is not True for k in ['resume_exact','mask_invariance','encoder_frozen']):raise ValueError('Training gates failed')
    if any(r[k] is not False for k in ['development_evaluated','confirmation_evaluated','auc_measured','submission_eligible','full_training_started']):
        raise ValueError('Pilot used reserved data or claims unmeasured results')
    expected=[{'study':s['StudyInstanceUID'],'series':x['series_uid'],'sha256':x['image_sha256']} for s in pilot for x in s['series']]
    if r['pixel_receipts']!=expected:raise ValueError('Pixel reconstruction mismatch')
    fp=directory/'v02_pilot_features.npz';cp=directory/'v02_pilot_checkpoint.pt'
    if digest(fp)!=r['features_sha256'] or digest(cp)!=r['checkpoint_sha256'] or r['checkpoint_step']!=8:
        raise ValueError('Feature/checkpoint hash drift')
    with np.load(fp,allow_pickle=False) as a:
        if a['ids'].tolist()!=[s['StudyInstanceUID'] for s in pilot] or a['features'].shape!=(12,3,384):raise ValueError('Feature identity/shape drift')
        if not np.isfinite(a['features']).all() or float(a['features'].std())==0:raise ValueError('Invalid feature values')
    losses=r['training_losses_first8']+r['continued_losses']+r['resumed_losses']
    if (len(r['training_losses_first8'])!=8 or len(r['continued_losses'])!=4 or len(losses)!=16
        or r['continued_losses']!=r['resumed_losses'] or any(not math.isfinite(v) or v<0 for v in losses)):
        raise ValueError('Checkpoint replay/loss evidence failed')
    t=r['timings_seconds'];steps=t['head_steps']
    if len(steps)!=16 or any(not math.isfinite(v) or v<=0 for v in steps+[t['rebuild'],t['forward'],t['total']]):
        raise ValueError('Invalid timings')
    median=float(np.median(steps[8:]));epochs=20;steps_per_epoch=math.ceil(299/4)
    projected=(t['rebuild']+t['forward'])*(299+250)/12 + median*steps_per_epoch*epochs
    return {'status':'PASSED_V02_PILOT_AUDIT','build_sha256':digest(build),'cache_audit_sha256':digest(cache),
        'model_sha256':MODEL_SHA,'studies':12,'series':36,'pixels_match_hd':True,
        'resume_exact_runtime_verified':True,'features_shape':[12,3,384],
        'timings_seconds':t,'cuda_peak_allocated_bytes':r['cuda_peak_allocated_bytes'],'versions':r['versions'],
        'planning_only':{'train':299,'development':250,'epochs':epochs,'batch_size':4,
            'median_warm_head_step_seconds':median,'projected_seconds_before_overhead':projected,
            'projected_seconds_with_2x_margin':projected*2,
            'not_a_runtime_guarantee':True,'confirmation_excluded':True},
        'eligible_for_baseline_planning':True,'auc_measured':False,'full_baseline_trained':False,
        'limitations':['Only 12 training studies; no validation/model quality measured.',
            'Resume exactness checked inside the same GPU process, not across Kaggle sessions.',
            'Pilot uses one GPU of a two-T4 allocation.',
            'Full train/dev cache parity and prospective budget remain required.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=assess(a.directory,a.cache,a.build);freeze(a.output,r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
