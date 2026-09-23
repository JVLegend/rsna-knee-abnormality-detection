"""#RSNA #Kaggle #Pesquisa — stable recipe plus fixed probe22, no automatic dispatch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_probe22 import PROBE22, literal_assignment

SMOKE_SHA = '9ca08d0ab5465e7da77dd3459201901e7f2ea9ceb8ab64b70def386a428470e7'
SMOKE_CSV_SHA = '7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770'
PURPOSE = 'ordered_probe22_competition_candidate_v1'


def build(raw, audit):
    if (audit.get('status') != 'PASSED_ORDERED_OFFICIAL_SMOKE'
            or (audit.get('studies'), audit.get('series')) != (3, 15)
            or audit.get('prediction_sha256') != SMOKE_CSV_SHA
            or any(audit.get(k) is not True for k in ['all_five_stages_present',
                   'dino_independent_replay', 'coat_independent_replay', 'official_ids_exact'])):
        raise ValueError('Approved official smoke required')
    if hashlib.sha256(raw).hexdigest() != SMOKE_SHA:
        raise ValueError('Approved smoke source drift')
    n = json.loads(raw)
    run = literal_assignment(n['cells'][5]['source'], 'RUN')
    if run['coatnet_w'] != PROBE22: raise ValueError('Published weights drift')
    s = replace_once(n['cells'][3]['source'],
        "h43_require(run['coatnet_w'] == {'__default__': .60}, 'not the fixed parent')",
        f"h43_require(run == {run!r}, 'not the complete fixed ordered probe22 recipe')")
    s = replace_once(s, "receipt.update(status='PASSED_PARENT_INTEGRITY',",
        "receipt.update(status='PASSED_ORDERED_PROBE22_INTEGRITY', preset='probe22', "
        "run=run, historical_probe22_ref=56263721,")
    s = replace_once(s, "with_name('h43_parent_integrity.json')",
                     "with_name('h43_ordered_probe22_integrity.json')")
    n['cells'][3]['source'] = s
    n['cells'][4]['source'] = f"H43_RECEIPT['purpose'] = {PURPOSE!r}\n"
    n['cells'][5]['source'] = replace_once(n['cells'][5]['source'], 'PRESET = "parent"', 'PRESET = "probe22"')
    n['cells'][61]['source'] = (
        "# Validated recipe; visible execution is not a hidden score.\n"
        "__import__('numpy').savez('/kaggle/working/ordered_probe22_fusion_inputs.npz',\n"
        "    ids=_blend_output['StudyInstanceUID'].astype(str).to_numpy(dtype=str),\n"
        "    transformer=_blend_tr.to_numpy(), hybrid=_blend_cr.to_numpy())\n"
        "h43_publish('/kaggle/working/h43_candidate.partial.csv', '/kaggle/working/submission.csv',\n"
        "    pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist(), H43_RECEIPT, RUN)\n"
        "Path('/kaggle/working/ordered_probe22_receipt.json').write_text(json.dumps({\n"
        f"    'status': 'ORDERED_PROBE22_READY_FOR_REVIEW', 'purpose': {PURPOSE!r},\n"
        f"    'source_smoke_sha256': {SMOKE_SHA!r},\n"
        "    'prediction_sha256': h43_sha(Path('/kaggle/working/submission.csv')),\n"
        "    'fusion_inputs_sha256': h43_sha(Path('/kaggle/working/ordered_probe22_fusion_inputs.npz')),\n"
        "    'studies': H43_RECEIPT['studies'], 'official_root': str(ROOT),\n"
        "    'auc_measured': False, 'automatically_submitted': False}, indent=2))\n"
        "print('H43_ORDERED_PROBE22_READY_FOR_REVIEW', flush=True)\n")
    for c in n['cells']:
        if c['cell_type'] != 'code': continue
        ast.parse(c['source'])
        if c['outputs'] or c['execution_count'] is not None: raise ValueError('Saved output not allowed')
        if 'H43_BENCHMARK_ROOT' in c['source'] or '/kaggle/working/h43_benchmark_input' in c['source']:
            raise ValueError('Training input leaked into candidate')
    n['metadata'].pop('h43_ordered_smoke')
    n['metadata']['h43_ordered_probe22'] = {'purpose': PURPOSE, 'source_smoke_sha256': SMOKE_SHA,
        'modified_code_cells': [3,4,5,61], 'published_outer_weights': PROBE22,
        'historical_probe22_ref': 56263721, 'no_independent_oof': True,
        'session_timeout_seconds_required': 32400}
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from scripts.assess_h43_ordered_smoke import assess
    audit = assess(Path('reports/avance_av015_smoke_v1'), Path('data/raw/sample_submission.csv'),
                   Path('data/raw/test_series.csv'))
    n = build(Path('reports/avance_av015_build/h43_ordered_official_smoke_v1.ipynb').read_bytes(), audit)
    body = json.dumps(n, ensure_ascii=False, indent=1)+'\n'
    if a.output.exists() and a.output.read_text() != body: raise ValueError('Refuse different overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not a.output.exists():
        with a.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(a.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_ordered_probe22']}, indent=2))


if __name__ == '__main__': main()
