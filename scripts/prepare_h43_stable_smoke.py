"""#RSNA #Kaggle #Pesquisa — smoke real só após paridade do stable36.

Constrói candidato separado. Nunca produz submission.csv nem submete.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_probe22 import PARENT_SHA
from scripts.prepare_h43_e03 import adapt_raptor


def build(raw, rank_helper, prefetch_helper, audit):
    required = ['eligible_for_real_test_smoke', 'exact_csv', 'dino_raw_predictions_exact',
                'raptor_inputs_exact', 'raptor_raw_exact', 'all_components_byte_identical']
    if audit.get('status') != 'PASSED_STABLE_FULLSTACK_PARITY' or any(audit.get(k) is not True for k in required):
        raise ValueError('Complete successful stable36 audit required before smoke')
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('Confirmed parent build drift')
    rank_sha = hashlib.sha256(rank_helper.encode()).hexdigest()
    receipts = audit.get('stable_receipts', [])
    if len(receipts) != 2 or any(r.get('helper_sha256') != rank_sha for r in receipts):
        raise ValueError('Rank helper differs from audited candidate')
    n = json.loads(raw)
    changed = []
    for i, cell in enumerate(n['cells']):
        if cell['cell_type'] != 'code': continue
        source = cell['source']
        if 'def infer_from_package(' in source:
            source = replace_once(source,
                '        frontier_ids, frontier_acc = _combine(public_frontier_members)',
                "        frontier_ids, frontier_acc = h43_combine_public_exact(\n"
                "            public_frontier_members, H43_RECEIPT['preflight']['dino_ids'], TARGETS)")
            cell['source'] = rank_helper + '\n' + source
            changed.append(i)
        nodes = [x for x in ast.parse(source).body if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in x.targets)]
        if nodes:
            if len(nodes) != 1 or len(ast.parse(source).body) != 1:
                raise ValueError('Raptor cell boundary drift')
            raptor = adapt_raptor(ast.literal_eval(nodes[0].value))
            # Keep the original official test discovery; never point smoke at train.
            raptor = replace_once(raptor,
                "def find_test_root():\n    return os.environ['H43_BENCHMARK_ROOT']\n",
                'def find_test_root():\n')
            cell['source'] = '_KE_SRC = ' + repr(raptor) + '\n'
            changed.append(i)
        if "_KE_NS['main']()" in source:
            cell['source'] = replace_once(source, "_KE_NS['main']()",
                'exec(compile(' + repr(prefetch_helper) + ", '<e03-prefetch>', 'exec'), _KE_NS)\n"
                "_KE_NS.update(E03_MODE='prefetch', E03_INPUTS=[], E03_RUN_INDEX=0)\n"
                "_KE_NS['main']()")
            changed.append(i)
    if len(changed) != 3:
        raise ValueError('Expected DINO, Raptor and namespace patches only')
    final = replace_once(n['cells'][-1]['source'], "'/kaggle/working/submission.csv'",
                         "'/kaggle/working/smoke_predictions.csv'")
    final = "H43_RECEIPT['purpose'] = 'stable_prefetch_smoke_only_no_submission'\n" + final
    final += ("Path('/kaggle/working/stable_smoke_receipt.json').write_text(json.dumps({\n"
              "    'status': 'PASSED_STABLE_PREFETCH_SMOKE_NOT_SUBMITTED',\n"
              "    'prediction_sha256': h43_sha(Path('/kaggle/working/smoke_predictions.csv')),\n"
              "    'studies': H43_RECEIPT['studies'], 'recipe_changed_from_parent': True,\n"
              "    'rank_helper_sha256': " + repr(rank_sha) + ",\n"
              "    'next_step': 'review runtime and competition submission separately'}, indent=2))\n")
    n['cells'][-1]['source'] = final
    for cell in n['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            if 'H43_BENCHMARK_ROOT' in cell['source']:
                raise ValueError('Train benchmark root leaked into real smoke')
    n['metadata']['h43_stable_smoke'] = {
        'parent_build_sha256': PARENT_SHA, 'rank_helper_sha256': rank_sha,
        'prefetch_helper_sha256': hashlib.sha256(prefetch_helper.encode()).hexdigest(),
        'modified_cells': changed + [len(n['cells'])-1],
        'purpose': 'real_test_smoke_only_not_a_submission', 'legacy_recipe_changed': True,
        'source_benchmark_csv_sha256': audit['candidate_sha256']}
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from scripts.assess_h43_stable_fullstack import assess
    # Recompute the audit from actual outputs; do not trust a hand-edited approval JSON.
    audit = assess(args.reference, args.candidate)
    helper = Path('scripts/ordered_prefetch.py').read_text() + '\n' + Path('scripts/h43_e03_runtime.py').read_text()
    n = build(Path('reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb').read_bytes(),
              Path('scripts/h43_stable_rank.py').read_text(), helper, audit)
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse different build overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_stable_smoke']}, indent=2))


if __name__ == '__main__': main()
