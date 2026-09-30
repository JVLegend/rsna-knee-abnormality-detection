"""#RSNA #Kaggle #Pesquisa — preregistered robustness recipes on frozen V03."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_m01_ablation import digest, literal

BASE = Path('reports/avance_av024_v03/v03_v1.py')
BASE_SHA = '3ae4654579db73f31779962b8ea25a620be67ec257522973ce0c36947e57088b'
FEATURE_SHA = 'bbc34ca678aed570c4325b283b7a33901134cd29b6b4314ad11c207c07012971'
CONTRACT = '31930692c9d22401be690a25ed960d55f2167510e53dd75b75454028a5812bf5'
ARMS = ['control', 'dropout25', 'paired_consistency']


def modify_training(source):
    replacements = [
        ('def train_seed(x,y,dev,dev_y,seed,fingerprint,output,guard,resume=None):',
         'def train_candidate(x,y,dev,dev_y,seed,fingerprint,output,guard,arm,resume=None):'),
        ('    history=[];best_state=None;',
         "    mask_rng=torch.Generator().manual_seed(seed+10000)\n"
         "    mask_trace={'seen':0,'dropped':[0,0,0],'chain':'0'*64}\n"
         '    history=[];best_state=None;'),
        ('        torch.set_rng_state(state[\'cpu_rng\']);torch.cuda.set_rng_state_all(state[\'cuda_rng\'])',
         "        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])\n"
         "        mask_rng.set_state(state['mask_rng']);mask_trace=state['mask_trace']"),
        ('loss=nn.functional.binary_cross_entropy_with_logits(head(x[indices],train_mask[indices]),y[indices])',
         'loss=robust_loss(head,x[indices],y[indices],arm,mask_rng,mask_trace)'),
        ('        if score<best_loss:',
         "        history[-1]['mask_trace']=json.loads(json.dumps(mask_trace))\n        if score<best_loss:"),
        ("            'fingerprint':fingerprint,'seed':seed,'epoch':epoch,'history':history,",
         "            'mask_rng':mask_rng.get_state(),'mask_trace':mask_trace,\n"
         "            'fingerprint':fingerprint,'seed':seed,'epoch':epoch,'history':history,"),
        ("    return {'seed':seed,'epochs':len(history),", "    return {'mask_trace':mask_trace,'seed':seed,'epochs':len(history),"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Training template drift: '+old)
        source = source.replace(old, new)
    ast.parse(source)
    return source


def assemble():
    if digest(BASE) != BASE_SHA: raise ValueError('Frozen V03 source changed')
    source = BASE.read_text(); v03 = literal(source, 'V03')
    if hashlib.sha256(json.dumps(v03,sort_keys=True).encode()).hexdigest() != CONTRACT:
        raise ValueError('V03 contract drift')
    audit_path = Path('reports/avance_av024_v03/v03_audit_v1.json')
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'PASSED_V03_AUDIT' or audit['selected_reference'] != 'expanded'
            or audit['build_sha256'] != BASE_SHA): raise ValueError('V03 not approved')
    prior = Path('reports/avance_av024_v03_v1/v03_receipt.json')
    receipt = json.loads(prior.read_text())
    if receipt['spec'] != v03 or receipt['feature_sha256'] != FEATURE_SHA: raise ValueError('Receipt drift')
    if digest(Path('reports/avance_av024_v03_v1/v03_features.npz')) != FEATURE_SHA:
        raise ValueError('Feature bytes changed')
    original = literal(source, 'V02_BASELINE')
    m = v03['manifest']
    payload = {'spec':dict(original['spec'],timeout_seconds=600), 'targets':m['targets'],
               'splits':{split:[{'StudyInstanceUID':r['StudyInstanceUID'],'labels':r['labels']}
                                for r in m[split]] for split in ['train','development']}}
    runtime = Path('scripts/r02_training_runtime.py').read_text()
    spec = {'name':'R02_robustness_v1','arms':ARMS,'baseline_contract':CONTRACT,
            'baseline_build_sha256':BASE_SHA,'feature_sha256':FEATURE_SHA,
            'baseline_receipt_sha256':digest(prior),'baseline_audit_sha256':digest(audit_path),
            'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest(),
            'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            'mask_seed_offset':10000,'dropout_probability':.25,'paired_mask_probability':1.,
            'paired_clean_weight':.5,'paired_masked_weight':.5,'consistency_mse_weight':.1,
            'decision':'both_seeds_intact_and_mean_masked_better_2e-6_then_lowest_mean_intact_tie_dropout25',
            'confirmation_evaluated':False,'submission_eligible':False}
    wanted = {'sha','StudyAttention','metrics','atomic_save','train_seed'}
    definitions = {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
                   if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted}
    if set(definitions) != wanted: raise ValueError('Missing audited definitions')
    imports = "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nimport hashlib,json,time\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\n"
    result = (imports+'\nV02_BASELINE = '+repr(payload)+'\nR02 = '+repr(spec)+'\n'
              +'\n\n'.join(definitions.values())+'\n'+modify_training(definitions['train_seed'])+'\n'+runtime)
    ast.parse(result)
    if len(result.encode()) >= 1_000_000: raise ValueError('Kaggle source size limit')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=assemble();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(source)
    print(json.dumps({'path':str(a.output),'sha256':digest(a.output),'bytes':len(source.encode())}))
