"""#RSNA #Kaggle #Pesquisa — auditoria independente do par stable36."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.assess_h43_fullstack import assess as assess_fullstack, compare_components
from scripts.dino_rank_replay import aggregate
from scripts.h43_integrity import H43_TARGETS, h43_sha, h43_validate_rows

RANK_SHA = '0b660440c08ac21979b8e805f86d1ab3453ecfc5033f327e289a7edb13565a71'


def validate_capture(directory, mode):
    receipt = json.loads((directory / 'dino_stable_receipt.json').read_text())
    if (receipt['status'] != 'DINO_STABLE_AGGREGATED_NOT_SUBMISSION'
            or receipt['rule'] != 'public_uniform_doubled_integer_ranks_v1'
            or receipt['mode'] != mode or receipt['helper_sha256'] != RANK_SHA
            or (receipt['members'], receipt['studies']) != (20, 36)
            or receipt['targets'] != H43_TARGETS):
        raise ValueError('Wrong stable aggregation receipt')
    path = directory / 'dino_stable_inputs.npz'
    if h43_sha(path) != receipt['input_sha256']:
        raise ValueError('DINO capture hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        members, ids, pred = data['member_ids'].tolist(), data['study_ids'].tolist(), data['predictions'].copy()
    if len(members) != 20 or len(set(members)) != 20 or len(ids) != 36 or len(set(ids)) != 36:
        raise ValueError('Duplicate/missing captured member or study')
    if pred.shape != (20, 36, 12) or not np.isfinite(pred).all():
        raise ValueError('DINO capture shape/finitude')
    pre = json.loads((directory / 'h43_preflight.json').read_text())
    selection = json.loads((directory / 'h43_benchmark_selection.json').read_text())
    if set(pre['dino_ids']) != set(members) or set(selection['ids']) != set(ids):
        raise ValueError('DINO capture does not match pinned member/study sets')
    expected = np.asarray(aggregate(pred.tolist(), list(range(20)), exact=True))
    with (directory / 'submission_public_0899.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    h43_validate_rows(rows, selection['ids'], ['StudyInstanceUID'] + H43_TARGETS)
    by_id = {row['StudyInstanceUID']: row for row in rows}
    actual = np.asarray([[float(by_id[uid][t]) for t in H43_TARGETS] for uid in ids])
    if not np.array_equal(actual, expected):
        raise ValueError('Stable CSV does not match independent integer-rank replay')
    # Canonical axis order allows comparing workers with different completion order.
    canonical = pred[[members.index(m) for m in sorted(members)]][:, [ids.index(s) for s in sorted(ids)]]
    return {'mode': mode, 'helper_sha256': RANK_SHA, 'capture_sha256': receipt['input_sha256'],
            'independent_replay_exact': True,
            'canonical_predictions_sha256': hashlib.sha256(canonical.tobytes()).hexdigest()}, canonical


def assess(reference, candidate):
    result = assess_fullstack(candidate, reference)
    serial, a = validate_capture(reference, 'serial')
    prefetch, b = validate_capture(candidate, 'prefetch')
    components = compare_components(reference, candidate)
    required = {'_raptor.csv', '_coat_arm.csv', 'submission_public_0899.csv',
                'submission_legacy_fold_blend.csv', 'submission_native_v38.csv',
                'submission_e10_v2.csv', 'submission_parent_exact.csv'}
    if not required <= components.keys():
        raise ValueError('Missing fullstack component CSVs')
    def input_contract(directory):
        return [{k: v for k, v in row.items() if k != 'prepare_seconds'}
                for row in json.loads((directory / 'e03_fullstack_inputs.json').read_text())]
    inputs_exact = input_contract(reference) == input_contract(candidate)
    with np.load(reference / 'e03_run0_raw.npz', allow_pickle=False) as x, \
            np.load(candidate / 'e03_run0_raw.npz', allow_pickle=False) as y:
        raw_exact = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        if not {'ids', 'arm_probs', 'ranks'} <= set(x.files) & set(y.files):
            raise ValueError('Missing Raptor raw arrays')
    components_exact = all(v['byte_identical'] for v in components.values())
    passed = result['exact_csv'] and np.array_equal(a, b) and inputs_exact and raw_exact and components_exact
    result.update(status='PASSED_STABLE_FULLSTACK_PARITY' if passed else 'FAILED_STABLE_FULLSTACK_PARITY',
                  stable_receipts=[serial, prefetch], dino_raw_predictions_exact=bool(np.array_equal(a, b)),
                  raptor_inputs_exact=inputs_exact, raptor_raw_exact=raw_exact,
                  components=components, all_components_byte_identical=components_exact,
                  eligible_for_real_test_smoke=bool(passed and result['times']['total']['time_reduction_pct'] > 0),
                  automatic_submission_authorized=False, legacy_recipe_changed=True,
                  score_improvement_demonstrated=False)
    return result


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
