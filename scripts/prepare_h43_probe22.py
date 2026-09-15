#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — A02: somente o preset publicado probe22.

Parte do notebook parent com score confirmado 0.939. Não altera o builder
nem o gate parent originais. Não executa código de terceiros localmente,
não lança kernel e não submete. Recusa drift e sobrescrita de outro build.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_parent import replace_once

PARENT_SHA = 'a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306'
PROBE22 = {'__default__': 0.60, 'ACL': 0.75, 'Medial Meniscus': 0.80,
           'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75}


def literal_assignment(source, name):
    values = [node.value for node in ast.parse(source).body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(values) != 1:
        raise ValueError(f'Expected exactly one literal {name}')
    return ast.literal_eval(values[0])


def build_probe22(raw):
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('Confirmed parent SHA mismatch')
    notebook = copy.deepcopy(json.loads(raw))
    cells = notebook['cells']
    # Índices pertencem ao artefato completo fixado acima, não a outra versão.
    run = literal_assignment(cells[4]['source'], 'RUN')
    if run['coatnet_w'] != PROBE22:
        raise ValueError('Published probe22 map drift')
    cells[4]['source'] = replace_once(cells[4]['source'], 'PRESET = "parent"',
                                    'PRESET = "probe22"')
    runtime = replace_once(cells[3]['source'],
        "h43_require(run['coatnet_w'] == {'__default__': .60}, 'not the fixed parent')",
        f"h43_require(run == {run!r}, 'not the complete fixed probe22 recipe')")
    runtime = replace_once(runtime, "receipt.update(status='PASSED_PARENT_INTEGRITY',",
        "receipt.update(status='PASSED_PROBE22_INTEGRITY', preset='probe22', "
        "run=run, parent_submission_ref=56253529,")
    runtime = replace_once(runtime, "with_name('h43_parent_integrity.json')",
                           "with_name('h43_probe22_integrity.json')")
    cells[3]['source'] = runtime
    cells[-1]['source'] = replace_once(cells[-1]['source'],
        "H43_PARENT_INTEGRITY_PASSED: ready for runtime/coverage review, not auto-submitted",
        "H43_PROBE22_INTEGRITY_PASSED: ready for paired review, not auto-submitted")
    for cell in cells:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Expected clean parent without saved outputs')
    parent_metadata = notebook['metadata'].pop('h43_build')
    notebook['metadata']['h43_parent_build'] = parent_metadata
    notebook['metadata']['h43_build'] = {
        'preset': 'probe22', 'parent_build_sha256': PARENT_SHA,
        'parent_submission_ref': 56253529, 'parent_public_score': '0.939',
        'source_sha256': parent_metadata['source_sha256'],
        'modified_parent_cells': [3, 4, len(cells) - 1],
        'embedded_setup_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
        'artifact_lock_entries': parent_metadata['artifact_lock_entries'],
        'published_outer_weights': PROBE22, 'no_independent_oof': True,
    }
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path,
        default=Path('reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_probe22(args.parent.read_bytes())
    body = json.dumps(result, ensure_ascii=False, indent=1) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Output exists with different content; choose new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as stream:
            stream.write(body)
    print(json.dumps({'path': str(args.output),
        'sha256': hashlib.sha256(body.encode()).hexdigest(),
        **result['metadata']['h43_build']}, indent=2))


if __name__ == '__main__':
    main()
