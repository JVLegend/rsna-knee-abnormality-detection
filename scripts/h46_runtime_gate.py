"""#RSNA #Kaggle #Testes — strict release gate, never infer a score from success."""
import math

H46_MEMBERS = {'resgated_top3', 'global96_top3', 'd4_swa3'}


def validate_h46_release(preflight, events, family, counts, recipe, expected_recipe):
    if preflight.get('status') != 'PASSED_H46_ASSET_AUDIT_NOT_INFERENCE':
        raise ValueError('H46 missing asset preflight')
    if counts != {'dino': 20, 'a5': 5, 'raptor_views': 4}:
        raise ValueError('H46 incomplete model inventory')
    if recipe != expected_recipe:
        raise ValueError('H46 recipe drift: no implicit preset/weight changes')
    members = family.get('members', [])
    if len(members) != 3 or set(members) != H46_MEMBERS:
        raise ValueError('H46 missing/duplicate/unexpected CoAt family member')
    if family.get('family_reduction') != 'rank_of_member_probability_mean':
        raise ValueError('H46 probability-family reduction changed')
    weights = family.get('within_coat', {})
    if set(weights) != H46_MEMBERS or any(not math.isclose(v, 1/3, abs_tol=1e-12) for v in weights.values()):
        raise ValueError('H46 family weights changed')
    if not math.isclose(family.get('public_raptor_alpha', -1), .6, abs_tol=1e-12):
        raise ValueError('H46 outer family blend changed')
    for event in events:
        kind = event.get('kind', '')
        # Scratch relocation and a finite FP32 retry do not discard a model.
        if kind in {'scratch_fallback'} or kind.endswith('_retry_fp32'):
            continue
        if kind.endswith('_fallback_studies') and event.get('count') == 0:
            continue
        if any(token in kind for token in ['failed', 'failure', 'dropped', 'partial', 'rejected',
                                           'neutral', 'nonfinite', 'repaired', 'fallback',
                                           'incomplete', 'mismatch']):
            raise ValueError(f'H46 degraded computation: {kind}')
    return {'status': 'PASSED_H46_RELEASE_GATE_NOT_SCORE_VALIDATION',
            'members': sorted(H46_MEMBERS), 'family_weights': weights,
            'family_reduction': family['family_reduction'], 'counts': counts,
            'runtime_parity_independently_verified': False, 'score_reproduced': False}
