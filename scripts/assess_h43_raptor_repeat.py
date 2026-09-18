"""#RSNA #Kaggle #Pesquisa — comparação de probabilidades Raptor entre duas sessões."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.assess_h43_raptor_determinism import assess as assess_reference, check_environment


def compare_arrays(a, b):
    if set(a) != set(b) or set(a) != {'ids', 'arm_probs', 'ranks'}:
        raise ValueError('Different/missing raw arrays')
    if a['ids'].shape != (36,) or b['ids'].shape != (36,) or not np.array_equal(a['ids'], b['ids']):
        raise ValueError('Different ID coverage/order')
    result = {}
    for key, shape in [('arm_probs', (4, 36, 12)), ('ranks', (36, 12))]:
        for values in [a[key], b[key]]:
            if values.shape != shape or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError('Raw shape/finitude/range')
        result[key] = {'exact': bool(np.array_equal(a[key], b[key])),
                       'changed_values': int(np.count_nonzero(a[key] != b[key])),
                       'max_abs_delta': float(np.max(np.abs(a[key] - b[key])))}
    return result


def assess(reference, candidate):
    base = assess_reference(reference)
    if not base['parity']: raise ValueError('Reference ABBA failed')
    load = lambda name: json.loads((candidate / name).read_text())
    env = load('raptor_determinism_environment.json')
    if env['purpose'] != 'second_worker_36_study_prefetch_no_submission': raise ValueError('Wrong repeat protocol')
    normalized = dict(env, purpose=base['environment']['purpose'])
    check_environment(normalized)
    if normalized != base['environment']: raise ValueError('Different backend/runtime environment')
    pre = load('h43_preflight.json')
    old_pre = json.loads((reference / 'h43_preflight.json').read_text())
    if pre['status'] != 'PASSED_ARTIFACT_PREFLIGHT' or pre['errors'] or pre['artifact_lock_entries'] != 56 or pre['gpus'] != ['Tesla T4'] * 2:
        raise ValueError('Repeat preflight failed')
    pins = lambda p: sorted((Path(f['path']).name, f['sha256']) for f in p['files'].values())
    if pins(pre) != pins(old_pre): raise ValueError('Different mounted model artifacts')
    selection = load('h43_benchmark_selection.json')
    if selection != json.loads((reference / 'h43_benchmark_selection.json').read_text()): raise ValueError('Different sample')
    r = load('raptor_repeat_receipt.json')
    if (r['status'] != 'RAPTOR_REPEAT_COMPLETE_NOT_SUBMITTED' or r['mode'] != 'prefetch'
            or r['studies'] != 36 or r['recipe_studies'] != 108 or r['ids'] != selection['ids']
            or not math.isfinite(r['seconds']) or r['seconds'] <= 0):
        raise ValueError('Invalid repeat receipt')
    path = candidate / 'e03_run0_raw.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != r['raw_sha256']: raise ValueError('Raw hash mismatch')
    with np.load(reference / 'e03_run2_raw.npz', allow_pickle=False) as x, np.load(path, allow_pickle=False) as y:
        arrays = compare_arrays(dict(x), dict(y))
        if y['ids'].tolist() != selection['ids']: raise ValueError('Raw IDs mismatch')
    old_r = json.loads((reference / 'e03_run2_receipt.json').read_text())
    contract = lambda rows: [{k: v for k,v in row.items() if k != 'prepare_seconds'} for row in rows]
    inputs_exact = contract(r['inputs']) == contract(old_r['inputs'])
    parity = inputs_exact and all(row['exact'] for row in arrays.values())
    return {'status': 'PASSED_CROSS_SESSION_PARITY' if parity else 'FAILED_CROSS_SESSION_PARITY',
            'inputs_exact': inputs_exact, 'arrays': arrays, 'reference_seconds': old_r['seconds'],
            'repeat_seconds': r['seconds'], 'environment': env,
            'eligible_for_fullstack_pair': bool(parity and base['eligible_for_full_stack_test']),
            'autotuner_cause_confirmed': False, 'submission_authorized': False,
            'limitations': ['Two separate Kaggle sessions; physical GPU host identity unavailable.',
                           'Only Raptor36, not fullstack or hidden test. No AUC or automatic promotion.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = assess(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
