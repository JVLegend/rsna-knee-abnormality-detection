"""#RSNA #Kaggle #Pesquisa — agregação exata só do DINO público uniforme.

Retorna somas de ranks duplicados (não probabilidades); o consumidor deve
fazer rank global antes de misturar com outros modelos. Não substitui _combine.
"""
import numpy as np
import pandas as pd


def h43_combine_public_exact(members, expected_members, targets):
    expected = list(expected_members)
    if len(expected) != 20 or len(set(expected)) != 20:
        raise ValueError('Exactly 20 pinned public members required')
    names = [m['id'] for m in members]
    if len(names) != 20 or set(names) != set(expected):
        raise ValueError('Missing, duplicate or unknown public member')
    if len(targets) != 12 or len(set(targets)) != 12:
        raise ValueError('Expected 12 unique targets')
    ids = sorted(members[0]['ids'])
    n = len(ids)
    if not n or len(set(ids)) != n or not all(isinstance(s, str) for s in ids):
        raise ValueError('Empty/duplicate/nonstring study IDs')
    # Bound both integer accumulation and exact representation at final rank.
    if 2 * n * len(members) > 2**53:
        raise ValueError('Exact rank accumulation bound exceeded')
    position = {uid: i for i, uid in enumerate(ids)}
    total = np.zeros((n, len(targets)), dtype=np.int64)
    for member in members:
        if float(member.get('weight', 1.0)) != 1.0:
            raise ValueError('Only unit public member weights supported')
        weights = member.get('target_weight')
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(targets),) or not np.all(weights == 1):
                raise ValueError('Target-specific weights not supported')
        member_ids = list(member['ids'])
        if len(member_ids) != n or set(member_ids) != set(ids):
            raise ValueError('Incomplete/duplicate study coverage')
        pred = np.asarray(member['pred'])
        if pred.shape != total.shape or not np.isfinite(pred).all():
            raise ValueError('Wrong prediction shape or nonfinite value')
        doubled = 2 * pd.DataFrame(pred).rank(method='average', pct=False).to_numpy()
        if not np.array_equal(doubled, np.floor(doubled)):
            raise AssertionError('Average ranks must be half-integers')
        total[[position[s] for s in member_ids]] += doubled.astype(np.int64)
    return ids, total
