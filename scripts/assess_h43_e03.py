#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — auditoria local independente dos outputs E03."""
import argparse
import hashlib
import json
import math
from pathlib import Path

MODES = ['serial', 'prefetch', 'prefetch', 'serial']


def require(condition, message):
    if not condition:
        raise ValueError('E03 audit: ' + message)


def evaluate_records(records, parity, studies=12):
    require(studies in (12, 36), 'supported study count')
    require(len(records) == 4, 'four runs required')
    for i, r in enumerate(records):
        require(r['index'] == i and r['mode'] == MODES[i], 'ABBA order')
        require(math.isfinite(r['seconds']) and r['seconds'] > 0, 'invalid time')
        require(r['studies'] == studies and r['prepared_recipe_studies'] == 3 * studies, 'coverage')
    a = (records[0]['seconds'] + records[3]['seconds']) / 2
    b = (records[1]['seconds'] + records[2]['seconds']) / 2
    return {'serial_mean_seconds': a, 'prefetch_mean_seconds': b,
            'abba_speedup': a / b,
            'warm_serial_vs_prefetch_speedup': records[3]['seconds'] / b,
            'eligible_for_full_stack_test': bool(parity and min(a, records[3]['seconds']) / b >= 1.05)}


def assess(directory, studies=12, series=70):
    import numpy as np
    load = lambda name: json.loads((directory / name).read_text())
    pre = load('h43_preflight.json')
    require(pre['status'] == 'PASSED_ARTIFACT_PREFLIGHT' and not pre['errors'], 'preflight')
    require(pre['artifact_lock_entries'] == 56 and pre['gpus'] == ['Tesla T4'] * 2, 'hardware/lock')
    selection = load('h43_benchmark_selection.json')
    ids = selection['ids']
    require(studies in (12, 36) and len(ids) == len(set(ids)) == studies and selection['series'] == series, 'sample identity/count')
    require(selection['uses_labels_for_selection'] is False and selection['partition'] == 'V01 train only',
            'training-only contract')
    reports, arrays, contracts, comparisons = [], [], [], []
    for i in range(4):
        r = load(f'e03_run{i}_receipt.json')
        raw = directory / f'e03_run{i}_raw.npz'
        require(hashlib.sha256(raw.read_bytes()).hexdigest() == r['raw_sha256'], 'raw hash')
        with np.load(raw, allow_pickle=False) as f:
            require(set(f.files) == {'ids', 'arm_probs', 'ranks'}, 'array keys')
            a = {k: f[k].copy() for k in f.files}
        require(a['ids'].tolist() == ids, 'study ID order')
        require(a['arm_probs'].shape == (4, studies, 12) and a['ranks'].shape == (studies, 12), 'array shapes')
        for name in ['arm_probs', 'ranks']:
            require(np.isfinite(a[name]).all() and ((a[name] >= 0) & (a[name] <= 1)).all(), 'prediction range')
        rows = r['inputs']
        require([(x['recipe'], x['uid']) for x in rows] == [(recipe, uid) for recipe in range(3) for uid in ids],
                'recipe/study coverage')
        require(all(math.isfinite(x['prepare_seconds']) and x['prepare_seconds'] >= 0 for x in rows), 'prep time')
        require(math.isclose(sum(x['prepare_seconds'] for x in rows), r['prepare_seconds_sum_overlaps_gpu'],
                             rel_tol=1e-10), 'prep receipt mismatch')
        contracts.append([{k: v for k, v in x.items() if k != 'prepare_seconds'} for x in rows])
        reports.append(r); arrays.append(a)
        base = arrays[0]
        comparisons.append({'run': i, 'ids_exact': np.array_equal(base['ids'], a['ids']),
            'inputs_exact': contracts[0] == contracts[i],
            'probabilities_exact': np.array_equal(base['arm_probs'], a['arm_probs']),
            'ranks_exact': np.array_equal(base['ranks'], a['ranks']),
            'max_abs_probability_delta': float(np.max(np.abs(base['arm_probs'] - a['arm_probs'])))})
    parity = all(all(c[k] for k in ['ids_exact', 'inputs_exact', 'probabilities_exact', 'ranks_exact'])
                 for c in comparisons)
    metrics = evaluate_records(reports, parity, studies=studies)
    remote = load('e03_comparison.json')
    require(remote['purpose'] == 'raptor_efficiency_only_no_auc_no_submission', 'purpose')
    require(remote['sequence'] == MODES and remote['comparisons'] == comparisons, 'remote parity summary')
    require(remote['deployment_authorized'] is False, 'never auto-deploy')
    require(remote['status'] == ('PASSED_EXACT_PARITY' if parity else 'FAILED_PARITY'), 'remote status')
    for key, value in metrics.items():
        require(remote[key] == value, 'remote metric mismatch: ' + key)
    return {'status': 'VERIFIED_LOCAL_AUDIT', 'parity': parity, **metrics,
            'comparisons': comparisons,
            'runs': [{k: v for k, v in r.items() if k != 'inputs'} for r in reports],
            'deployment_authorized': False, 'auc_measured': False,
            'next_action': 'full-stack benchmark only' if metrics['eligible_for_full_stack_test'] else
                           'do not promote this variant',
            'limitations': remote['limitations']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = assess(args.directory)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
