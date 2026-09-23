"""#RSNA #Kaggle #Pesquisa — fixed 0.943 candidate with audited assets/release gate.

Build only. No Kaggle upload, inference or automatic competition submission.
Arithmetic is retained; strengthened gates can abort previously degraded runs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h46_preflight import SOURCE, SOURCE_SHA, assemble
from scripts.prepare_h43_parent import replace_once


def build(raw, receipt):
    preflight, spec = assemble(raw)
    if receipt.get('source_sha256') != SOURCE_SHA or receipt.get('status') != 'PASSED_H46_ASSET_AUDIT_NOT_INFERENCE':
        raise ValueError('Missing successful CPU asset receipt')
    if len(receipt.get('families', {})) != 2 or receipt.get('shared', {}).get('dino_members') != 20:
        raise ValueError('Incomplete CPU asset receipt')
    groups = [receipt['shared']['files']] + [r['files'] for r in receipt['families'].values()]
    lock = sorted({(Path(r['path']).name, r['sha256']) for group in groups for r in group.values()})
    if len(lock) < 100:
        raise ValueError('Incomplete artifact lock')
    preflight = replace_once(preflight, '\nh46_preflight()\n', '\nH43_ARTIFACT_LOCK = '+repr(lock)+'\nh46_preflight()\n')
    nb = json.loads(raw)
    cells = nb['cells']
    for c in cells:
        if c['cell_type'] == 'code':
            c['source'] = ''.join(c['source']); c['outputs'] = []; c['execution_count'] = None
    cells[2]['source'] = replace_once(cells[2]['source'], 'PRESET = os.environ.get("RSNA_PRESET", "speedy")', 'PRESET = "speedy"')
    cells[40]['source'] += '\nH46_A5_COUNT = len(models)\nif H46_A5_COUNT != 5: raise RuntimeError("H46 A5 fold count")\n'
    # Convert a logged incomplete preparation identity into an immediate abort.
    cells[49]['source'] = replace_once(cells[49]['source'],
        'if _ke_input_ids != _expected_preparations:\n',
        'if _ke_input_ids != _expected_preparations:\n    raise RuntimeError("H46 Raptor preparation identity mismatch")\n')
    gate = Path('scripts/h46_runtime_gate.py').read_text()
    before_publish = '''
H46_RELEASE = validate_h46_release(
    json.loads(Path('/kaggle/working/h46_preflight.json').read_text()),
    _RSNA_AUDIT['events'],
    json.loads(Path('/kaggle/working/_coat_raptor_blend_receipt.json').read_text()),
    {'dino': _DINOV2_MATCHED_MEMBERS, 'a5': H46_A5_COUNT, 'raptor_views': len(_KE_NS['ARMS'])},
    RUN, H46_SPEC['recipe'])
_RSNA_AUDIT['h46_release'] = H46_RELEASE
_RSNA_AUDIT['coats']['global96_epochs'] = [16, 23, 18]
_RSNA_AUDIT['family_mix'] = H46_RELEASE['family_weights']
rsna_json('/kaggle/working/h46_release_gate.json', H46_RELEASE)
'''
    cells[49]['source'] = replace_once(cells[49]['source'],
        "_final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')",
        before_publish+"\n_final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')")
    setup = {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':preflight+'\n'+gate}
    cells.insert(2, setup)
    for c in cells:
        if c['cell_type']=='code': ast.parse(c['source'])
    # Keep author attribution/credits, discard widget state and stale output metadata.
    nb['metadata'] = {k:v for k,v in nb['metadata'].items() if k in {'kernelspec','language_info'}}
    nb['metadata']['h46_build'] = {'source_sha256':SOURCE_SHA,'recipe':'speedy',
        'asset_lock_entries':len(lock),'gpu_smoke_executed':False,
        'gate_sha256':hashlib.sha256(gate.encode()).hexdigest(),
        'modified_original_cells':[2,40,49], 'submission_eligible':False,
        'remaining_gates':['T4x2 numerical/coverage/runtime smoke','hidden-set runtime budget','submission quota check']}
    return nb


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    nb=build(SOURCE.read_bytes(),json.loads(a.receipt.read_text()))
    body=json.dumps(nb,ensure_ascii=False,indent=1)+'\n';a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(body)
    print(json.dumps({'sha256':hashlib.sha256(body.encode()).hexdigest(),'bytes':len(body.encode()),**nb['metadata']['h46_build']},indent=2))
