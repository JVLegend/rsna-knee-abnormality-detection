"""#RSNA #Kaggle #Pesquisa — gate completo preservado e proveniência de backend."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.assess_h43_stable_fullstack import assess as assess_stable
from scripts.assess_h43_raptor_determinism import check_environment, EXPECTED
from scripts.assess_h43_raptor_repeat import compare_arrays, assess as assess_repeat
from scripts.prepare_h43_deterministic_fullstack import recipe, PURPOSE


def validate_recipe(directory, mode):
    r = json.loads((directory / 'h43_deterministic_recipe.json').read_text())
    if r != dict(recipe(), mode=mode): raise ValueError('Fullstack recipe mismatch')
    env = json.loads((directory / 'raptor_determinism_environment.json').read_text())
    if env.get('purpose') != PURPOSE: raise ValueError('Wrong fullstack environment purpose')
    check_environment(dict(env, purpose=EXPECTED['purpose']))
    return env


def assess(reference, candidate, isolated):
    # Revalidate actual isolated files, not only a cached success receipt.
    repeat = assess_repeat(Path('reports/avance_av010_raptor_determinism_v1'), isolated)
    if not repeat['eligible_for_fullstack_pair']: raise ValueError('Isolated reference failed')
    environments = [validate_recipe(reference, 'serial'), validate_recipe(candidate, 'prefetch')]
    if environments[0] != environments[1]: raise ValueError('Different fullstack environments')
    if dict(environments[0], purpose=repeat['environment']['purpose']) != repeat['environment']:
        raise ValueError('Different isolated/fullstack runtime')
    result = assess_stable(reference, candidate)  # Includes native CSV: gate is NOT relaxed.
    anchors = []
    with np.load(isolated / 'e03_run0_raw.npz', allow_pickle=False) as x:
        for directory in [reference, candidate]:
            with np.load(directory / 'e03_run0_raw.npz', allow_pickle=False) as y:
                anchors.append(compare_arrays(dict(x), dict(y)))
    anchor_exact = all(v['exact'] for a in anchors for v in a.values())
    passed = result['status'] == 'PASSED_STABLE_FULLSTACK_PARITY' and anchor_exact
    result.update(status='PASSED_DETERMINISTIC_FULLSTACK_PARITY' if passed else 'FAILED_DETERMINISTIC_FULLSTACK_PARITY',
                  eligible_for_real_test_smoke=bool(passed and result['eligible_for_real_test_smoke']),
                  isolated_raptor_comparisons=anchors, environments=environments, recipe=recipe(),
                  smoke_requires_this_recipe=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--isolated', type=Path, default=Path('reports/avance_av011_raptor_repeat_v1'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = assess(args.reference, args.candidate, args.isolated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
