#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — prefetch no benchmark completo, nunca submissão."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_e03 import adapt_raptor
from scripts.prepare_h43_parent import replace_once

BENCH_SHA = '835083352f0c0b7d8785794bb6846b6b2bd74a13846d5c68fbab7da713c4d614'


def build(raw, helper):
    if hashlib.sha256(raw).hexdigest() != BENCH_SHA:
        raise ValueError('Audited 36-study benchmark drift')
    n = json.loads(raw)
    changed = []
    for i, cell in enumerate(n['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        nodes = [x for x in ast.parse(source).body if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in x.targets)]
        if nodes:
            if len(nodes) != 1 or len(ast.parse(source).body) != 1:
                raise ValueError('Raptor cell drift')
            raptor = ast.literal_eval(nodes[0].value)
            # Retira só o redirecionamento já existente, pois adapt_raptor o repõe.
            raptor = replace_once(raptor,
                "def find_test_root():\n    return os.environ['H43_BENCHMARK_ROOT']\n",
                'def find_test_root():\n')
            cell['source'] = '_KE_SRC = ' + repr(adapt_raptor(raptor)) + '\n'
            changed.append(i)
        if "_KE_NS['main']()" in source:
            cell['source'] = replace_once(source, "_KE_NS['main']()",
                "exec(compile(" + repr(helper) + ", '<e03-prefetch>', 'exec'), _KE_NS)\n"
                "_KE_NS.update(E03_MODE='prefetch', E03_INPUTS=[], E03_RUN_INDEX=0)\n"
                "_KE_NS['main']()\n"
                "_KePath('/kaggle/working/e03_fullstack_inputs.json').write_text(\n"
                "    __import__('json').dumps(_KE_NS['E03_INPUTS'], indent=2))")
            changed.append(i)
    if len(changed) != 2:
        raise ValueError('Expected exactly two changed cells')
    for c in n['cells']:
        if c['cell_type'] == 'code':
            ast.parse(c['source'])
    n['metadata']['e03_fullstack'] = {
        'baseline_benchmark_sha256': BENCH_SHA, 'modified_cells': changed,
        'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
        'purpose': 'full_stack_runtime_parity_only_no_submission',
        'only_model_execution_change': 'one-ahead CPU study preparation in Raptor',
        'expected_reference_csv_sha256': 'd774e25c03db117dc708851d8fb9eb588c722f92935377a117825bac674832ae'}
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_path = Path('reports/avance_av007_e03_v1/local_assessment.json')
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'VERIFIED_LOCAL_AUDIT' or not audit.get('eligible_for_full_stack_test'):
        raise ValueError('Missing successful isolated E03 audit')
    helper = Path('scripts/ordered_prefetch.py').read_text() + '\n' + Path('scripts/h43_e03_runtime.py').read_text()
    result = build(Path('reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb').read_bytes(), helper)
    result['metadata']['e03_fullstack']['isolated_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    body = json.dumps(result, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse overwrite of different build')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f:
            f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **result['metadata']['e03_fullstack']}, indent=2))


if __name__ == '__main__':
    main()
