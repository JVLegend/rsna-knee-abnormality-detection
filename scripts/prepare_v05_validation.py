"""#RSNA #Kaggle #Pesquisa — new grouped validation, never recycle V04 for tuning."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.freeze_weak_validation import KEY, TARGETS, digest, freeze, group_hash, rows_by_id
from scripts.prepare_v03_scale import PLANES, V01, V01_SHA
from scripts.prepare_v04_confirmation import V03_BUILD, V03_SHA, literal

V04_AUDIT = Path('reports/avance_av028_v04/audit_v1.json')
V04_SHA = 'b821b2caafbdfe51f3c1be4961b8707548632a4f5bdb28a9f1912369326f5be1'
SEED = 20260921


def order_key(group, namespace):
    return hashlib.sha256(f'V05:{SEED}:{namespace}:{group}'.encode()).hexdigest()


def eligible_pool(train, teacher, series, exposed):
    """Exclude whole known-exposed/gold groups before any new split selection."""
    ids = [r[KEY] for r in exposed]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate exposure identity')
    hashes = {}
    for uid, r in train.items():
        try:
            hashes[uid] = group_hash(r['Report'])
        except ValueError:
            pass
    if any(hashes.get(r[KEY]) != r['report_hash'] for r in exposed):
        raise ValueError('Exposure/report identity drift')
    blocked_ids = set(ids)
    blocked_groups = {r['report_hash'] for r in exposed}
    gold_ids = {uid for uid, r in train.items() if any(r[t].strip() for t in TARGETS)}
    gold_groups = {hashes[u] for u in gold_ids if u in hashes}
    by_id = defaultdict(list)
    seen = set()
    for s in series:
        key = (s[KEY], s['SeriesInstanceUID'])
        if key in seen:
            raise ValueError('Duplicate series metadata')
        seen.add(key)
        by_id[key[0]].append(s)
    eligible = []
    excluded = Counter()
    for uid in sorted(train):
        if uid in blocked_ids or hashes.get(uid) in blocked_groups:
            reason = 'known_exposed_id_or_group'
        elif uid in gold_ids or hashes.get(uid) in gold_groups:
            reason = 'gold_id_or_group'
        elif uid not in hashes:
            reason = 'blank_report'
        elif uid not in teacher:
            reason = 'missing_teacher'
        else:
            try:
                labels = [float(teacher[uid][t]) for t in TARGETS]
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in labels):
                    raise ValueError('Invalid labels')
                chosen = []
                for plane in PLANES:
                    options = [s for s in by_id[uid] if s['Anatomical_Plane'] == plane]
                    options.sort(key=lambda s: (-int(s['Fluid_Sensitive']), -int(s['Fat_Suppression']), s['SeriesInstanceUID']))
                    if not options:
                        raise ValueError('Missing plane')
                    chosen.append({'plane': plane, 'series_uid': options[0]['SeriesInstanceUID']})
                eligible.append({KEY: uid, 'report_hash': hashes[uid], 'labels': labels, 'series': chosen})
                continue
            except (ValueError, KeyError, TypeError):
                reason = 'invalid_labels_or_series'
        excluded[reason] += 1
    return eligible, dict(excluded)


def allocate(rows, confirmation=300, development=300):
    if confirmation < 1 or development < 1:
        raise ValueError('Invalid requested sizes')
    groups = defaultdict(list)
    for row in rows:
        groups[row['report_hash']].append(row)
    if len({r[KEY] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate identity')
    result = {'confirmation': [], 'development': [], 'unused': []}
    # One seed, no search/rebalancing based on class counts or performance.
    for g in sorted(groups, key=lambda x: order_key(x, 'new-splits')):
        name = ('confirmation' if len(result['confirmation']) < confirmation else
                'development' if len(result['development']) < development else 'unused')
        result[name].extend(sorted(groups[g], key=lambda r: r[KEY]))
    if len(result['confirmation']) < confirmation or len(result['development']) < development or not result['unused']:
        raise ValueError('Insufficient whole groups; never split/reseed to fit')
    return result


def oof_folds(rows, folds=5):
    if folds < 2 or len({r[KEY] for r in rows}) != len(rows):
        raise ValueError('Invalid OOF population')
    groups = defaultdict(list)
    for r in rows:
        groups[r['report_hash']].append(r[KEY])
    if len(groups) < folds:
        raise ValueError('Insufficient OOF groups')
    counts = [0] * folds
    assignments = {}
    for g in sorted(groups, key=lambda x: (-len(groups[x]), order_key(x, 'oof'))):
        fold = min(range(folds), key=lambda f: (counts[f], f))
        counts[fold] += len(groups[g])
        assignments[g] = fold
    return {'folds': folds, 'study_counts': counts,
            'assignments': [{KEY: r[KEY], 'report_hash': r['report_hash'], 'fold': assignments[r['report_hash']]} for r in rows],
            'executed': False,
            'warning': 'Design only. Held-out fold must not select epochs; use fixed epochs or inner training-only validation.'}


def coverage(rows):
    return {t: {'positive': sum(r['labels'][j] > .5 for r in rows),
                'negative': sum(r['labels'][j] < .5 for r in rows),
                'uncertain': sum(r['labels'][j] == .5 for r in rows)} for j, t in enumerate(TARGETS)}


def build():
    for p, sha in [(V01, V01_SHA), (V03_BUILD, V03_SHA), (V04_AUDIT, V04_SHA)]:
        if digest(p) != sha:
            raise ValueError('Frozen provenance drift')
    audit = json.loads(V04_AUDIT.read_text())
    if not audit['confirmation_evaluated'] or audit['confirmation_decision'] != 'NOT_CONFIRMED':
        raise ValueError('V04 not reconciled')
    v03 = literal(V03_BUILD.read_text(), 'V03')['manifest']
    for p, sha in v03['sources'].items():
        if digest(Path(p)) != sha:
            raise ValueError('Metadata source drift')
    old = json.loads(V01.read_text())
    exposed = v03['train'] + v03['development'] + v03['reserved']
    train = rows_by_id(Path(old['sources']['train']['path']))
    teacher = rows_by_id(Path(old['sources']['teacher']['path']))
    with Path('data/raw/train_series.csv').open(newline='') as f:
        series = list(csv.DictReader(f))
    pool, excluded = eligible_pool(train, teacher, series, exposed)
    splits = allocate(pool)
    counts = {name: coverage(rows) for name, rows in splits.items() if name != 'unused'}
    failures = [f'{name}/{t}' for name, targets in counts.items() for t, c in targets.items()
                if not c['positive'] or not c['negative']]
    # Validate frozen train labels and IDs against current pinned metadata.
    for r in v03['train']:
        if r['labels'] != [float(teacher[r[KEY]][t]) for t in TARGETS] or r['report_hash'] != group_hash(train[r[KEY]]['Report']):
            raise ValueError('Frozen training identity drift')
    oof = oof_folds(v03['train'] + splits['development'])
    return {'format': 'V05_grouped_validation_v1', 'seed': SEED, 'targets': TARGETS, 'planes': PLANES,
            'requested_sizes': {'confirmation': 300, 'development': 300},
            'sources': {**v03['sources'], str(V03_BUILD): V03_SHA, str(V04_AUDIT): V04_SHA},
            'known_exposure': [{KEY: r[KEY], 'report_hash': r['report_hash']} for r in exposed],
            'splits': {'train': v03['train'], **splits}, 'oof': oof,
            'eligible_count': len(pool), 'excluded_counts': excluded, 'class_counts': counts,
            'class_gate_passed': not failures, 'class_gate_failures': failures,
            'confirmation_model_evaluations': 0, 'development_model_evaluations': 0,
            'pixel_integrity_checked': False, 'submission_eligible': False,
            'limitations': ['Fresh for own-model line only; historical global exposure unknown.',
                           'Report hash is not a patient ID; approximate/pixel duplicates still require checks.',
                           'Labels remain weak and frozen, with 0.5 uncertainty preserved.',
                           'OOF design includes historically selected training cases; not an unbiased external test.',
                           'Only metadata inventoried; DICOM availability/geometry/pixels not yet checked.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = build()
    freeze(a.output, data)
    print(json.dumps({'sha256': digest(a.output), 'sizes': {k: len(v) for k, v in data['splits'].items()},
                      'eligible': data['eligible_count'], 'excluded': data['excluded_counts'],
                      'oof_counts': data['oof']['study_counts'], 'class_gate_passed': data['class_gate_passed'],
                      'class_counts': data['class_counts']}, indent=2))
