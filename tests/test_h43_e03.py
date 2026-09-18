"""#RSNA #Kaggle #Testes — paralelismo limitado e receitas preservadas."""
import ast
from contextlib import closing
import json
from pathlib import Path
import threading
import unittest

from scripts.ordered_prefetch import ordered_one_ahead
from scripts.prepare_h43_e03 import build, adapt_raptor

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'


class PrefetchTests(unittest.TestCase):
    def test_empty_and_order(self):
        self.assertEqual(list(ordered_one_ahead([], lambda x: x)), [])
        self.assertEqual(list(ordered_one_ahead([3, 1, 2], lambda x: x * 2)),
                         [(0, 3, 6), (1, 1, 2), (2, 2, 4)])

    def test_only_one_ahead_and_close(self):
        seen, threads = [], []
        second_ready = threading.Event()
        def prepare(x):
            seen.append(x); threads.append(threading.get_ident())
            if x == 1:
                second_ready.set()
            return x
        with closing(ordered_one_ahead(range(5), prepare)) as g:
            self.assertEqual(next(g), (0, 0, 0))
            self.assertTrue(second_ready.wait(2))
            self.assertEqual(seen, [0, 1])
        self.assertEqual(seen, [0, 1])
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())

    def test_exception_propagates(self):
        def prepare(x):
            if x == 1:
                raise ValueError('decode failed')
            return x
        with closing(ordered_one_ahead(range(3), prepare)) as g:
            self.assertEqual(next(g), (0, 0, 0))
            with self.assertRaisesRegex(ValueError, 'decode failed'):
                next(g)

    def test_parent_drift_rejected(self):
        with self.assertRaises(ValueError):
            build(b'{}', list(map(str, range(12))), '', '')

    @unittest.skipUnless(PARENT.exists(), 'External parent not committed')
    def test_build_and_model_functions_preserved(self):
        p = json.loads(PARENT.read_bytes())
        node = next(n for n in ast.parse(p['cells'][48]['source']).body if isinstance(n, ast.Assign))
        raw = ast.literal_eval(node.value)
        adapted = adapt_raptor(raw)
        funcs = lambda text: {n.name: ast.dump(n) for n in ast.parse(text).body
                              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        before, after = funcs(raw), funcs(adapted)
        self.assertEqual(set(before), set(after))
        for name in before:
            if name not in ['main', 'find_test_root']:
                self.assertEqual(before[name], after[name], name)
        prefetch = (ROOT / 'scripts/ordered_prefetch.py').read_text()
        runtime = (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        n = build(PARENT.read_bytes(), list(map(str, range(12))), prefetch, runtime)
        self.assertEqual(n, build(PARENT.read_bytes(), list(map(str, range(12))), prefetch, runtime))
        for c in n['cells']:
            if c['cell_type'] == 'code':
                ast.parse(c['source'])
                self.assertNotIn('h43_publish(', c['source'].split('def h43_publish')[0])
        self.assertEqual(n['cells'][-1]['source'], 'e03_benchmark()\n')
        self.assertIn('with _e03_studies(', adapted)


if __name__ == '__main__':
    unittest.main()
