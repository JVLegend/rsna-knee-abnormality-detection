"""#RSNA #Kaggle #Pesquisa — par completo mais âncora CoAt ordenada auditada."""
import argparse
import json
from pathlib import Path
from scripts.assess_h43_deterministic_fullstack import assess as assess_fullstack
from scripts.assess_h43_coat_order import assess_abba, load_preflight, load_run, compare_arrays, batch_contract
from scripts.prepare_h43_ordered_fullstack import recipe, ORDERED_SHA
from scripts.h43_integrity import h43_sha


def validate_ordered_run(directory):
    r = json.loads((directory/'h43_ordered_coat_recipe.json').read_text())
    if r != recipe(): raise ValueError('Ordered CoAt recipe mismatch')
    if h43_sha(directory/'h43_coat_ordered_runtime.py') != ORDERED_SHA:
        raise ValueError('Ordered CoAt runtime mismatch')
    if 'PACKED FALLBACK' in (directory/'h43_coat_worker.log').read_text():
        raise ValueError('CoAt used fallback microbatch')
    pre,ids = load_preflight(directory)
    receipt,arr = load_run(directory,ids,'_coat_arm.csv',pre)
    return arr, [batch_contract(s,'ordered',ids) for s in receipt['shards']]


def assess(reference, candidate, isolated):
    isolated_audit = assess_abba(isolated)
    if not isolated_audit['eligible_for_fullstack_test']: raise ValueError('CoAt isolated gate failed')
    pre,ids = load_preflight(isolated)
    r,a = load_run(isolated/'coat_order_run1',ids,'coat_diagnostic.csv',pre)
    anchor_contract = [batch_contract(s,'ordered',ids) for s in r['shards']]
    result = assess_fullstack(reference,candidate,Path('reports/avance_av011_raptor_repeat_v1'))
    comparisons, contracts = [], []
    for d in [reference,candidate]:
        arr,contract = validate_ordered_run(d)
        comparisons.append(compare_arrays(a,arr))
        contracts.append(contract)
    contract_exact = all(c == anchor_contract for c in contracts)
    raw_exact = all(c[k]['exact'] for c in comparisons
                    for k in ['raw_probabilities','checkpoint_percentile_ranks','rank_ensemble'])
    passed = result['status'] == 'PASSED_DETERMINISTIC_FULLSTACK_PARITY' and contract_exact and raw_exact
    result.update(status='PASSED_ORDERED_FULLSTACK_PARITY' if passed else 'FAILED_ORDERED_FULLSTACK_PARITY',
        eligible_for_real_test_smoke=bool(passed and result['eligible_for_real_test_smoke']),
        coat_anchor_comparisons=comparisons, coat_inputs_batches_environment_exact=contract_exact,
        ordered_coat_recipe=recipe(), automatic_submission_authorized=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--isolated',type=Path,default=Path('reports/avance_av013_coat_order_v1'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    r = assess(args.reference,args.candidate,args.isolated)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(r,f,indent=2); f.write('\n')
    print(json.dumps(r,indent=2))


if __name__ == '__main__': main()
