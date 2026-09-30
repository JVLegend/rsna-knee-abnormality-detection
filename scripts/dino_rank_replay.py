"""#RSNA #Kaggle #Pesquisa — replay sem GPU do rank mean público uniforme.

Diagnóstico apenas: não altera os notebooks aprovados nem publica submissão.
"""
import argparse
import json
import math
from pathlib import Path
import random
import csv
import hashlib
import re


def average_ranks(values, pct=False):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Empty/nonfinite values')
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for i in order[start:stop]:
            result[i] = rank / len(values) if pct else rank
        start = stop
    return result


def rank_columns(matrix, pct):
    return list(map(list, zip(*(average_ranks(list(c), pct) for c in zip(*matrix)))))


def aggregate(predictions, order, exact=False):
    m = len(predictions)
    if sorted(order) != list(range(m)) or not m:
        raise ValueError('Missing/duplicate member')
    n = len(predictions[0])
    if not n:
        raise ValueError('Empty studies')
    targets = len(predictions[0][0])
    if not n or not targets or any(len(p) != n or any(len(r) != targets for r in p) for p in predictions):
        raise ValueError('Incomplete shape')
    acc = [[0 if exact else 0. for _ in range(targets)] for _ in range(n)]
    for index in order:
        ranks = rank_columns(predictions[index], pct=not exact)
        for i in range(n):
            for j in range(targets):
                acc[i][j] += int(2 * ranks[i][j]) if exact else ranks[i][j]
    # Na rota exata, o denominador 2*N*M é comum: não precisa dividir antes do rank.
    if not exact:
        acc = [[v / m for v in row] for row in acc]
    return rank_columns(acc, pct=True)


def changes(a, b):
    return [sum(x[j] != y[j] for x, y in zip(a, b)) for j in range(len(a[0]))]


def synthetic_probe():
    rng = random.Random(2026)
    a = list(range(20)); b = a.copy(); b[14], b[15] = b[15], b[14]
    for trial in range(200):
        p = [[[rng.randrange(12) for _ in range(2)] for _ in range(36)] for _ in range(20)]
        delta = changes(aggregate(p, a), aggregate(p, b))
        if any(delta):
            return {'trial': trial, 'changed_values_by_column': delta,
                'exact_integer_changes': changes(aggregate(p, a, True), aggregate(p, b, True)),
                'conclusion': 'Same predictions + swapped completion order can change legacy final ranks.',
                'actual_competition_cause_confirmed': False}
    raise RuntimeError('No synthetic counterexample found')


def log_order(path):
    order = [member for row in json.loads(path.read_text())
             for member in re.findall(r'banked ([a-z0-9]+) fold', row.get('data', ''))]
    if len(order) != 20 or len(set(order)) != 20:
        raise ValueError('Expected 20 unique completion IDs')
    return order


def replay_capture(directory):
    import numpy as np
    receipt = json.loads((directory / 'dino_capture_receipt.json').read_text())
    path = directory / 'dino_replay_inputs.npz'
    if receipt['status'] != 'DINO_CAPTURED_NOT_SUBMISSION' or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Capture receipt/hash mismatch')
    with np.load(path, allow_pickle=False) as f:
        members, ids, preds = f['member_ids'].tolist(), f['study_ids'].tolist(), f['predictions']
        if preds.shape != (20, 36, 12) or not np.isfinite(preds).all():
            raise ValueError('Capture shape/finitude')
        predictions = preds.tolist()
    if len(set(members)) != 20 or len(set(ids)) != 36:
        raise ValueError('Duplicate member/study')
    reference_dir = Path('reports/avance_av008_reference_components')
    candidate_dir = Path('reports/avance_av008_candidate_components')
    orders = {
        'capture': members,
        'historical_reference': log_order(reference_dir / 'rsna-knee-h43-runtime-benchmark.log'),
        'historical_candidate': log_order(candidate_dir / 'rsna-knee-e03-fullstack36-prefetch.log')}
    if any(set(order) != set(members) for order in orders.values()):
        raise ValueError('Different member set')
    from scripts.h43_integrity import H43_TARGETS
    def csv_matrix(p):
        with p.open(newline='') as f: rows = list(csv.DictReader(f))
        by_id = {r['StudyInstanceUID']: r for r in rows}
        if len(rows) != 36 or set(by_id) != set(ids): raise ValueError('Historical IDs')
        return [[float(by_id[uid][t]) for t in H43_TARGETS] for uid in ids]
    outputs = {name: aggregate(predictions, [members.index(m) for m in order]) for name, order in orders.items()}
    exact = {name: aggregate(predictions, [members.index(m) for m in order], True) for name, order in orders.items()}
    parity = changes(outputs['capture'], csv_matrix(directory / 'submission_public_0899.csv'))
    if any(parity): raise ValueError('Local replay does not match capture serialization')
    reference_match = changes(outputs['historical_reference'], csv_matrix(reference_dir / 'submission_public_0899.csv'))
    candidate_match = changes(outputs['historical_candidate'], csv_matrix(candidate_dir / 'submission_public_0899.csv'))
    order_delta = changes(outputs['historical_reference'], outputs['historical_candidate'])
    exact_delta = changes(exact['historical_reference'], exact['historical_candidate'])
    rng = random.Random(2026)
    for _ in range(20):
        shuffled = list(range(20)); rng.shuffle(shuffled)
        if any(changes(exact['capture'], aggregate(predictions, shuffled, True))):
            raise AssertionError('Exact aggregation depends on member order')
    return {'status': 'REPLAY_VERIFIED_AGAINST_CAPTURE', 'orders': orders,
        'capture_sha256': receipt['sha256'],
        'exact_integer_random_permutations_passed': 20,
        'exact_vs_legacy_capture_changed_by_target': dict(zip(H43_TARGETS, changes(exact['capture'], outputs['capture']))),
        'changed_values_by_target': dict(zip(H43_TARGETS, order_delta)),
        'exact_integer_changed_by_target': dict(zip(H43_TARGETS, exact_delta)),
        'reference_mismatch_by_target': dict(zip(H43_TARGETS, reference_match)),
        'candidate_mismatch_by_target': dict(zip(H43_TARGETS, candidate_match)),
        'both_historical_csvs_reproduced_from_same_predictions': not any(reference_match + candidate_match),
        'actual_order_dependence_reproduced': any(order_delta),
        'deployment_authorized': False,
        'limitation': 'Uniform complete public-member rank mean only; no score/OOF or production change.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture-directory', type=Path)
    args = p.parse_args()
    result = replay_capture(args.capture_directory) if args.capture_directory else synthetic_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
