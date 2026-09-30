"""#RSNA #Kaggle #Pesquisa — teste isolado do encaminhamento native/public DINO.

Executa somente o trecho de promoção auditado por SHA, com inferência simulada,
em diretório temporário. Não carrega pesos, labels, GPUs ou código de treino.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scripts.h43_integrity import H43_TARGETS, h43_require

BENCH_SHA = 'b1e409de5e4192f61bc4d41c91a8d21c8f404bddeda605188affa6d82c5a6f41'


def extract(raw):
    if hashlib.sha256(raw).hexdigest() != BENCH_SHA:
        raise ValueError('Audited stable36 serial build drift')
    notebook = json.loads(raw)
    source = notebook['cells'][37]['source']
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    main = copy.deepcopy(functions['main'])
    # A execução real retorna no ramo de pesos; não compilar o fallback de treino.
    if not isinstance(main.body[2], ast.If) or not isinstance(main.body[2].body[-1], ast.Return):
        raise ValueError('Promotion boundary drift')
    main.body = main.body[:3]
    tree = ast.Module(body=[functions['_v37_validate_submission'], main], type_ignores=[])
    occurrences = []
    for index, cell in enumerate(notebook['cells']):
        for line, text in enumerate(cell['source'].splitlines(), 1):
            if 'submission_native_v38.csv' in text:
                occurrences.append({'cell': index, 'line': line, 'source': text.strip()})
    if len(occurrences) != 1 or occurrences[0]['source'] != "native.to_csv('submission_native_v38.csv', index=False)":
        raise ValueError('Native artifact routing changed; inspect consumers')
    return compile(ast.fix_missing_locations(tree), '<audited-dino-promotion>', 'exec'), occurrences


def exercise(code, native_value, *, omit_public=False, wrong_ids=False):
    ids = ['study_a', 'study_b', 'study_c']
    with tempfile.TemporaryDirectory(prefix='rsna-native-route-') as temporary:
        root = Path(temporary)
        pd.DataFrame({'StudyInstanceUID': ids}).to_csv(root / 'test.csv', index=False)
        native = pd.DataFrame(native_value, index=range(3), columns=H43_TARGETS)
        native.insert(0, 'StudyInstanceUID', ids)
        public = pd.DataFrame(np.arange(36).reshape(3, 12) / 36, columns=H43_TARGETS)
        public.insert(0, 'StudyInstanceUID', ids if not wrong_ids else ['other', 'study_b', 'study_c'])
        def fake_inference(*args):
            native.to_csv(root / 'h43_candidate.partial.csv', index=False)
            if not omit_public: public.to_csv(root / 'submission_public_0899.csv', index=False)
        env = {'pd': pd, 'np': np, 'Path': Path, 'TARGETS': H43_TARGETS, 'ROOT': root,
               'h43_require': h43_require, 'find_weights': lambda: root,
               'infer_from_package': fake_inference, 'DEVS': ['CPU_STUB'], 'log': lambda *a: None}
        exec(code, env)
        # Single-threaded diagnostic only; cwd restored even on failure.
        with contextlib.chdir(root): env['main']()
        candidate = (root / 'h43_candidate.partial.csv').read_bytes()
        diagnostic = pd.read_csv(root / 'submission_native_v38.csv')
        expected = pd.read_csv(root / 'submission_public_0899.csv')
        actual = pd.read_csv(root / 'h43_candidate.partial.csv')
        if not actual.equals(expected): raise AssertionError('Public candidate was not promoted exactly')
        if not (diagnostic[H43_TARGETS].to_numpy() == native_value).all():
            raise AssertionError('Native diagnostic not preserved')
        return hashlib.sha256(candidate).hexdigest()


def assess(raw):
    code, occurrences = extract(raw)
    hashes = [exercise(code, value) for value in [0., .25, .75, 1.]]
    if len(set(hashes)) != 1: raise AssertionError('Native predictions changed public candidate')
    for kwargs in [{'omit_public': True}, {'wrong_ids': True}]:
        try: exercise(code, .5, **kwargs)
        except RuntimeError: pass
        else: raise AssertionError('Invalid public output did not fail closed')
    return {'status': 'VERIFIED_NATIVE_DIAGNOSTIC_ROUTING', 'build_sha256': BENCH_SHA,
            'native_artifact_occurrences': occurrences, 'native_constants_tested': [0., .25, .75, 1.],
            'candidate_sha256_all_variants': hashes[0], 'missing_or_invalid_public_rejected': True,
            'classification': 'Native CSV is a diagnostic; public CSV replaces primary at this boundary.',
            'submission_authorized': False, 'fullstack_parity_gate_changed': False,
            'limitations': ['Controlled promotion-boundary test, not GPU/OOF or score evaluation.',
                           'Literal path audit is not a general dynamic dependency proof.',
                           'Does not resolve active Raptor raw prediction differences.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = assess(Path('reports/avance_av009_build/h43_stable36_serial_v1.ipynb').read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
