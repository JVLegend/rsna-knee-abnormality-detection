"""#RSNA #Kaggle #Testes — integração, ordem de soma e gate de receita."""
import ast
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.prepare_h43_stable_fullstack import build as build_stable
from scripts.prepare_h43_deterministic_fullstack import build, recipe, PURPOSE, SCOPE
from scripts.assess_h43_deterministic_fullstack import validate_recipe
from scripts.assess_h43_raptor_determinism import EXPECTED

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb'


class DeterministicFullstackTests(unittest.TestCase):
    def test_closed_prerequisite(self):
        for audit in [{}, {'status': 'PASSED_CROSS_SESSION_PARITY', 'eligible_for_fullstack_pair': False}]:
            with self.assertRaises(ValueError): build(b'{}', '', '', 'serial', audit)

    def test_recipe_and_environment_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            r = dict(recipe(), mode='serial')
            (d / 'h43_deterministic_recipe.json').write_text(json.dumps(r))
            env = dict(EXPECTED, purpose=PURPOSE, torch='test', cuda='test', cudnn=9000)
            (d / 'raptor_determinism_environment.json').write_text(json.dumps(env))
            self.assertEqual(validate_recipe(d, 'serial'), env)
            with self.assertRaises(ValueError): validate_recipe(d, 'prefetch')
            r['flags_sha256'] = 'wrong'
            (d / 'h43_deterministic_recipe.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError): validate_recipe(d, 'serial')

    def test_scope_restores_flags_and_rng_even_on_failure(self):
        # Lightweight torch stand-in: verifies actual scope control flow on CPU.
        from types import SimpleNamespace as NS
        for fails in [False, True]:
            state = {'rng': 11, 'cuda': [12], 'enabled': False, 'warn': True}
            def set_alg(value, warn_only=False): state.update(enabled=value, warn=warn_only)
            cuda = NS(get_rng_state_all=lambda: state['cuda'], set_rng_state_all=lambda v: state.update(cuda=v))
            t = NS(backends=NS(cudnn=NS(benchmark=True, deterministic=False, allow_tf32=True),
                              cuda=NS(matmul=NS(allow_tf32=True))), cuda=cuda,
                   get_rng_state=lambda: state['rng'], set_rng_state=lambda v: state.update(rng=v),
                   are_deterministic_algorithms_enabled=lambda: state['enabled'],
                   is_deterministic_algorithms_warn_only_enabled=lambda: state['warn'],
                   use_deterministic_algorithms=set_alg)
            before = random.getstate(); before_np = np.random.get_state()
            def main():
                self.assertFalse(t.backends.cudnn.benchmark)
                state.update(rng=99, cuda=[98]); random.seed(1); np.random.seed(1)
                if fails: raise RuntimeError('fixture')
            ns = {'torch': t, 'np': np, 'main': main,
                  'H43_FIXED_FLAGS': 'torch.backends.cudnn.benchmark=False\ntorch.use_deterministic_algorithms(True)'}
            if fails:
                with self.assertRaises(RuntimeError): exec(SCOPE, ns)
            else: exec(SCOPE, ns)
            self.assertTrue(t.backends.cudnn.benchmark)
            self.assertEqual(state, {'rng': 11, 'cuda': [12], 'enabled': False, 'warn': True})
            self.assertEqual(random.getstate(), before)
            self.assertTrue(np.array_equal(np.random.get_state()[1], before_np[1]))

    @unittest.skipUnless(BENCH.exists(), 'External benchmark not committed')
    def test_only_expected_cells_change_and_native_order_is_fixed(self):
        helper = (ROOT / 'scripts/ordered_prefetch.py').read_text() + '\n' + (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        rank = (ROOT / 'scripts/h43_stable_rank.py').read_text()
        raw = BENCH.read_bytes()
        audit = {'status': 'PASSED_CROSS_SESSION_PARITY', 'eligible_for_fullstack_pair': True}
        for mode in ['serial', 'prefetch']:
            base = build_stable(raw, helper, rank, mode)
            n = build(raw, helper, rank, mode, audit)
            self.assertEqual([i for i,(a,b) in enumerate(zip(base['cells'], n['cells'])) if a != b], [3,33,57])
            self.assertTrue(n['cells'][3]['source'].startswith('import os\n'))
            self.assertEqual(n['cells'][55], base['cells'][55])  # Raptor model/decode unchanged.
            s = n['cells'][57]['source']
            self.assertLess(s.index("exec(compile(_KE_SRC"), s.index("'<h43-fixed-scope>'"))
            fn = next(x for x in ast.parse(n['cells'][33]['source']).body
                      if isinstance(x, ast.FunctionDef) and x.name == '_combine')
            ns = {'np': np, 'pd': pd, 'TARGETS': ['a','b']}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<combine>', 'exec'), ns)
            members = [{'id': str(i), 'ids': ['0','1','2'], 'weight': 0.1+i,
                        'pred': np.asarray([[i%3, 1], [2, i%2], [0,3]])} for i in range(20)]
            expected = ns['_combine'](members)[1]
            rng = random.Random(2026)
            for _ in range(20):
                rng.shuffle(members)
                self.assertTrue(np.array_equal(expected, ns['_combine'](members)[1]))


if __name__ == '__main__': unittest.main()
