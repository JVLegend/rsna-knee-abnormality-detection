"""#RSNA #Kaggle #Testes — source-compatible release gate.

The public H46 runtime deliberately repairs individual unreadable studies and
records those repairs as audit events.  The hidden cohort is much larger than
the three-row visible cohort, so treating every recorded repair as a fatal
error changes the published runtime contract and can abort an otherwise finite,
complete submission.  This gate therefore blocks composition/schema drift and
reports row-level fallbacks without turning them into a late exception.
"""
from collections import Counter
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
    warning_tokens = ('failed', 'failure', 'dropped', 'partial', 'rejected',
                      'neutral', 'nonfinite', 'repaired', 'fallback',
                      'incomplete', 'mismatch', 'unreadable', 'unfilled')
    warnings = Counter()
    for event in events:
        kind = str(event.get('kind', ''))
        if any(token in kind for token in warning_tokens):
            warnings[kind] += 1
    return {'status': 'PASSED_H46_SOURCE_COMPATIBLE_GATE_NOT_SCORE_VALIDATION',
            'members': sorted(H46_MEMBERS), 'family_weights': weights,
            'family_reduction': family['family_reduction'], 'counts': counts,
            'warning_event_counts': dict(sorted(warnings.items())),
            'source_compatible_fallbacks_observed': bool(warnings),
            'runtime_parity_independently_verified': False, 'score_reproduced': False}
