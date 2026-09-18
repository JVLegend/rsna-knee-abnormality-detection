"""#RSNA #Kaggle #Pesquisa — integra somente o CoAt ordenado ao par completo."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h43_parent import replace_once

STACK_SHA = {
    'serial': '7e4fce5475b117e7301f2590f9edbcbd4c881fa7b0305dd2e5386e630a6e4f6d',
    'prefetch': 'a6a49f5ef3b84f7ce0118235f6dbdc57aacbbe5418bf7dfd6e05a2c3edebe760'}
HELPER_SHA = '8f77911f466fabe64678a6e05c24be02d3cb47dc0bb45ee0cf6e9aee8273384c'
ORDERED_SHA = '31c4acb53b666374b02ead1dc1d2554cbb80c467ede780b3377ef0ace8952f3a'


def recipe():
    return {'protocol': 'h43_fullstack_ordered_coat_v1', 'mode': 'ordered',
            'helper_sha256': HELPER_SHA, 'runtime_sha256': ORDERED_SHA,
            'purpose': 'training36_fullstack_parity_only', 'submission_authorized': False}


def child_loader(helper):
    return ('exec(' + repr(helper) + ')\n' +
        "import coatnet_resgated_ep10_top3_inference as original_rt\n"
        "import importlib.util, hashlib\nfrom pathlib import Path\n"
        "source = patch_coat(Path(original_rt.__file__).read_text(), 'ordered')\n"
        f"assert hashlib.sha256(source.encode()).hexdigest() == {ORDERED_SHA!r}\n"
        "runtime = Path('/kaggle/working/h43_coat_ordered_runtime.py')\n"
        "runtime.write_text(source)\n"
        "spec = importlib.util.spec_from_file_location('h43_ordered_coat', runtime)\n"
        "rt = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = rt\nspec.loader.exec_module(rt)\n"
        "Path('/kaggle/working/h43_ordered_coat_recipe.json').write_text(" +
        repr(json.dumps(recipe(), indent=2)) + ")\n")


def build(raw, helper, mode, audit):
    if (audit.get('status') != 'PASSED_ORDERED_REPEAT'
            or audit.get('eligible_for_fullstack_test') is not True
            or audit.get('inputs_exact') is not True or audit.get('environments_exact') is not True):
        raise ValueError('Successful ordered CoAt repeat required')
    if mode not in STACK_SHA or hashlib.sha256(raw).hexdigest() != STACK_SHA[mode]:
        raise ValueError('Pinned fullstack build drift')
    if hashlib.sha256(helper.encode()).hexdigest() != HELPER_SHA:
        raise ValueError('Audited CoAt helper drift')
    n = json.loads(raw)
    s = n['cells'][57]['source']
    s = replace_once(s, '        "import coatnet_resgated_ep10_top3_inference as rt\\n"',
                     '        ' + repr(child_loader(helper)))
    s = replace_once(s, '    if proc.returncode != 0:\n',
        "    _P('/kaggle/working/h43_coat_worker.log').write_text(proc.stdout + '\\n' + proc.stderr)\n"
        "    if 'PACKED FALLBACK' in proc.stdout:\n"
        "        raise RuntimeError('CoAt changed inference microbatch through fallback')\n"
        '    if proc.returncode != 0:\n')
    n['cells'][57]['source'] = s
    for c in n['cells']:
        if c['cell_type'] == 'code': ast.parse(c['source'])
    n['metadata']['h43_ordered_coat'] = dict(recipe(), raptor_mode=mode,
        baseline_build_sha256=STACK_SHA[mode], modified_cells=[57])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=list(STACK_SHA), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from scripts.assess_h43_coat_order import assess_abba
    audit = assess_abba(Path('reports/avance_av013_coat_order_v1'))
    n = build(Path(f'reports/avance_av012_build/h43_deterministic36_{args.mode}_v1.ipynb').read_bytes(),
              Path('scripts/h43_coat_order.py').read_text(), args.mode, audit)
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_ordered_coat']},indent=2))


if __name__ == '__main__': main()
