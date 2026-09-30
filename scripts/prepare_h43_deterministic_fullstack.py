"""#RSNA #Kaggle #Pesquisa — par completo com receita determinística rastreável."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_stable_fullstack import build as build_stable
from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_raptor_determinism import SETUP, FLAGS
from scripts.assess_h43_raptor_repeat import assess as assess_repeat

PURPOSE = 'fullstack36_fixed_backend_no_submission'
FIXED_FLAGS = FLAGS.replace('same_worker_36_study_abba_no_submission', PURPOSE)
SCOPE = '''from contextlib import contextmanager
@contextmanager
def h43_raptor_fixed_scope():
    import random
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        exec(H43_FIXED_FLAGS, globals())
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[:2]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[2:4]
        torch.use_deterministic_algorithms(flags[4], warn_only=flags[5])
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
with h43_raptor_fixed_scope():
    main()
'''
NATIVE_RULE = "    per_member = sorted(per_member, key=lambda m: m['id'])\n"


def recipe():
    return {'protocol': 'fullstack36_fixed_backend_v1', 'purpose': PURPOSE,
            'setup_sha256': hashlib.sha256(SETUP.encode()).hexdigest(),
            'flags_sha256': hashlib.sha256(FIXED_FLAGS.encode()).hexdigest(),
            'scope_sha256': hashlib.sha256(SCOPE.encode()).hexdigest(),
            'native_rule': 'weighted_sum_sorted_by_member_id_v1',
            'gate': 'all_components_exact_including_native', 'submission_authorized': False}


def build(raw, helper, rank_helper, mode, audit):
    if audit.get('status') != 'PASSED_CROSS_SESSION_PARITY' or audit.get('eligible_for_fullstack_pair') is not True:
        raise ValueError('Cross-session parity required')
    n = build_stable(raw, helper, rank_helper, mode)
    code = [c for c in n['cells'] if c['cell_type'] == 'code']
    code[0]['source'] = SETUP + code[0]['source']  # Before preflight/CUDA initialization.
    changed = []
    for i, c in enumerate(n['cells']):
        if c['cell_type'] != 'code': continue
        s = c['source']
        if 'def _combine(per_member):\n' in s:
            c['source'] = replace_once(s, 'def _combine(per_member):\n',
                                      'def _combine(per_member):\n' + NATIVE_RULE)
            changed.append(i)
        if "_KE_NS['main']()" in s:
            execution = "H43_FIXED_FLAGS = " + repr(FIXED_FLAGS) + '\n' + SCOPE
            receipt = dict(recipe(), mode=mode)
            c['source'] = replace_once(s, "_KE_NS['main']()",
                "exec(compile(" + repr(execution) + ", '<h43-fixed-scope>', 'exec'), _KE_NS)\n"
                "_KePath('/kaggle/working/h43_deterministic_recipe.json').write_text(" +
                repr(json.dumps(receipt, indent=2)) + ")")
            changed.append(i)
    if changed != [33, 57]: raise ValueError('Unexpected fullstack injection sites')
    for c in code: ast.parse(c['source'])
    n['metadata']['h43_deterministic_fullstack'] = dict(recipe(), mode=mode, modified_cells=[3, *changed])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['serial', 'prefetch'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit = assess_repeat(Path('reports/avance_av010_raptor_determinism_v1'),
                          Path('reports/avance_av011_raptor_repeat_v1'))
    helper = Path('scripts/ordered_prefetch.py').read_text() + '\n' + Path('scripts/h43_e03_runtime.py').read_text()
    n = build(Path('reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb').read_bytes(),
              helper, Path('scripts/h43_stable_rank.py').read_text(), args.mode, audit)
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_deterministic_fullstack']}, indent=2))


if __name__ == '__main__': main()
