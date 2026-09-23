"""#RSNA #Kaggle #Pesquisa — exemplos oficiais usando a receita ordered validada."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_ordered_fullstack import recipe as coat_recipe

SOURCE_SHA = '11ddd8e5ad7123ae7f6d322f77b42f37c6ccaf0dd0ca139d59454161bcb58927'
PURPOSE = 'official_example_ordered_smoke_only_no_submission'
REQUIRED = ['eligible_for_real_test_smoke', 'exact_csv', 'dino_raw_predictions_exact',
            'raptor_inputs_exact', 'raptor_raw_exact', 'all_components_byte_identical',
            'coat_inputs_batches_environment_exact']


def build(raw, audit):
    if audit.get('status') != 'PASSED_ORDERED_FULLSTACK_PARITY' or any(audit.get(k) is not True for k in REQUIRED):
        raise ValueError('Complete successful ordered fullstack audit required')
    if audit.get('ordered_coat_recipe') != coat_recipe(): raise ValueError('CoAt recipe differs')
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA: raise ValueError('Validated prefetch build drift')
    n = json.loads(raw)
    n['cells'][4]['source'] = f"H43_RECEIPT['purpose'] = {PURPOSE!r}\n"
    n['cells'][24]['source'] = replace_once(n['cells'][24]['source'],
        "def find_root():\n    return Path(os.environ['H43_BENCHMARK_ROOT'])\n", 'def find_root():\n')
    n['cells'][43]['source'] = replace_once(n['cells'][43]['source'],
        "COMP = Path(os.environ['H43_BENCHMARK_ROOT'])", "COMP = _find_dir('rsna-knee-abnormality-detection')")
    tree = ast.parse(n['cells'][55]['source'])
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign): raise ValueError('Raptor cell drift')
    raptor = ast.literal_eval(tree.body[0].value)
    raptor = replace_once(raptor,"def find_test_root():\n    return os.environ['H43_BENCHMARK_ROOT']\n",'def find_test_root():\n')
    n['cells'][55]['source'] = '_KE_SRC = ' + repr(raptor) + '\n'
    n['cells'][57]['source'] = replace_once(n['cells'][57]['source'],
        "competition_root=Path('/kaggle/working/h43_benchmark_input'),",
        'competition_root=rt.base.find_competition_root(),')
    s = replace_once(n['cells'][-1]['source'],"'/kaggle/working/benchmark_predictions.csv'",
                     "'/kaggle/working/smoke_predictions.csv'")
    s = replace_once(s,'BENCHMARK ONLY, never submit this kernel','OFFICIAL EXAMPLE SMOKE ONLY, never submit this kernel')
    s += "Path('/kaggle/working/ordered_smoke_receipt.json').write_text(json.dumps({\n"
    s += f"    'status': 'ORDERED_SMOKE_COMPLETE_NOT_SUBMITTED', 'purpose': {PURPOSE!r},\n"
    s += f"    'source_build_sha256': {SOURCE_SHA!r},\n"
    s += "    'prediction_sha256': h43_sha(Path('/kaggle/working/smoke_predictions.csv')),\n"
    s += "    'studies': H43_RECEIPT['studies'], 'official_root': str(ROOT),\n"
    s += "    'recipe_receipts_retain_validated_protocol_names': True,\n"
    s += "    'auc_measured': False, 'submission_authorized': False}, indent=2))\n"
    n['cells'][-1]['source'] = s
    for c in n['cells']:
        if c['cell_type'] != 'code': continue
        ast.parse(c['source'])
        if 'H43_BENCHMARK_ROOT' in c['source'] or '/kaggle/working/h43_benchmark_input' in c['source']:
            raise ValueError('Benchmark root leaked into smoke')
    # Legacy recipe names remain provenance identifiers, not claims of 36 executed cases.
    n['metadata'] = {k:v for k,v in n['metadata'].items() if k in ['kernelspec','language_info','kaggle']}
    n['metadata']['h43_ordered_smoke'] = {'source_build_sha256': SOURCE_SHA,
        'purpose': PURPOSE, 'modified_code_cells': [4,24,43,55,57,61],
        'benchmark_prediction_sha256': audit['candidate_sha256'], 'submission_authorized': False}
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    from scripts.assess_h43_ordered_fullstack import assess
    audit = assess(args.reference,args.candidate,Path('reports/avance_av013_coat_order_v1'))
    n = build(Path('reports/avance_av014_build/h43_ordered36_prefetch_v1.ipynb').read_bytes(),audit)
    body = json.dumps(n,indent=1,ensure_ascii=False)+'\n'
    if args.output.exists() and args.output.read_text()!=body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f:f.write(body)
    print(json.dumps({'path':str(args.output),'sha256':hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_ordered_smoke']},indent=2))


if __name__ == '__main__': main()
