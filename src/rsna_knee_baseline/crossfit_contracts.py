"""#RSNA #Kaggle #Pesquisa — leakage gates for visual teachers, no training IO.

Provenance lists must describe ALL exposure, including checkpoint selection
and task-specific pretraining. A caller's declaration is not an independent
audit of a public checkpoint. No gold/confirmation predictions are generated.
"""
import hashlib
import math

KEY = 'StudyInstanceUID'


def identity_index(rows):
    result = {}
    for row in rows:
        uid, group = row[KEY], row['report_hash']
        if not isinstance(uid, str) or not uid or not isinstance(group, str) or not group:
            raise ValueError('Missing study/group identity')
        if uid in result:
            raise ValueError('Duplicate study identity')
        result[uid] = group
    return result


def nested_partitions(rows, assignments, blocked_rows, inner_fraction=.15):
    """Honor frozen outer folds; pick inner groups deterministically, no labels."""
    if not 0 < inner_fraction < .5:
        raise ValueError('Invalid inner selection fraction')
    index, blocked = identity_index(rows), identity_index(blocked_rows)
    if set(index) & set(blocked) or set(index.values()) & set(blocked.values()):
        raise ValueError('Reserved/gold group entered cross-fitting')
    amap = identity_index(assignments)
    if amap != index:
        raise ValueError('Fold identity does not match population')
    group_fold = {}
    for row in assignments:
        fold, group = row['fold'], row['report_hash']
        if type(fold) is not int or fold < 0:
            raise ValueError('Invalid fold')
        if group in group_fold and group_fold[group] != fold:
            raise ValueError('Report group crosses outer folds')
        group_fold[group] = fold
    folds = sorted(set(group_fold.values()))
    if len(folds) < 2 or folds != list(range(len(folds))):
        raise ValueError('Missing/non-contiguous outer folds')
    result = []
    for fold in folds:
        available = [g for g, f in group_fold.items() if f != fold]
        available.sort(key=lambda g: hashlib.sha256(f'AV030:inner:{fold}:{g}'.encode()).hexdigest())
        n = max(1, math.ceil(len(available)*inner_fraction))
        if n >= len(available):
            raise ValueError('Not enough inner training groups')
        selection = set(available[:n])
        partitions = {'train': [], 'selection': [], 'prediction': []}
        for uid, group in sorted(index.items()):
            role = ('prediction' if group_fold[group] == fold else
                    'selection' if group in selection else 'train')
            partitions[role].append({KEY: uid, 'report_hash': group})
        result.append({'fold': fold, **partitions})
    return result


def validate_teacher(provenance, prediction_rows, forbidden_rows):
    """Fail closed unless every task-specific exposure set is declared."""
    expected = identity_index(prediction_rows)
    forbidden = identity_index(forbidden_rows)
    if set(expected) & set(forbidden) or set(expected.values()) & set(forbidden.values()):
        raise ValueError('Requested teacher predictions include a forbidden group')
    if provenance.get('format') != 'visual_teacher_exposure_v1':
        raise ValueError('Missing teacher provenance format')
    if provenance.get('exposure_complete') is not True:
        raise ValueError('Unknown teacher exposure')
    digest = provenance.get('checkpoint_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid checkpoint digest')
    for role in ['train', 'selection', 'task_pretraining']:
        exposed = identity_index(provenance[role])
        if set(exposed) & set(expected) or set(exposed.values()) & set(expected.values()):
            raise ValueError(f'Teacher {role} leaks prediction study/group')
        if set(exposed) & set(forbidden) or set(exposed.values()) & set(forbidden.values()):
            raise ValueError(f'Teacher {role} leaks reserved/student-outer group')
    if identity_index(provenance['prediction']) != expected:
        raise ValueError('Teacher prediction coverage drift')
    return True


def combine_teachers(teachers, prediction_rows, targets, forbidden_rows):
    """Equal soft-probability mean; no weights tuned on held-out/gold labels."""
    ids = list(identity_index(prediction_rows))
    if not ids or not teachers or len(targets) != 12 or len(set(targets)) != 12:
        raise ValueError('Empty population/teachers or invalid target schema')
    seen = set()
    totals = {uid: [0.] * len(targets) for uid in ids}
    for teacher in teachers:
        validate_teacher(teacher['provenance'], prediction_rows, forbidden_rows)
        sha = teacher['provenance']['checkpoint_sha256']
        if sha in seen:
            raise ValueError('Repeated checkpoint is not a second teacher')
        seen.add(sha)
        if teacher['targets'] != targets or set(teacher['probabilities']) != set(ids):
            raise ValueError('Teacher target/ID mismatch')
        for uid in ids:
            values = teacher['probabilities'][uid]
            if len(values) != len(targets) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError('Invalid teacher probabilities')
            totals[uid] = [a + b/len(teachers) for a, b in zip(totals[uid], values)]
    return totals
