"""#RSNA #Kaggle #Testes — benchmark completo sem retunar os modelos."""
import ast
import json
from pathlib import Path
import unittest
from scripts.prepare_h43_e03_fullstack import build

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb'


class FullStackTests(unittest.TestCase):
    def test_source_drift_rejected(self):
        with self.assertRaises(ValueError):
            build(b'{}', '')

    @unittest.skipUnless(BASE.exists(), 'External benchmark not committed')
    def test_only_raptor_and_helper_wiring_changed(self):
        raw = BASE.read_bytes(); before = json.loads(raw)
        helper = (ROOT / 'scripts/ordered_prefetch.py').read_text() + '\n' + (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        n = build(raw, helper)
        self.assertEqual(n, build(raw, helper))
        changed = [i for i, (a, b) in enumerate(zip(before['cells'], n['cells'])) if a != b]
        self.assertEqual(len(changed), 2)
        self.assertEqual(changed, n['metadata']['e03_fullstack']['modified_cells'])
        self.assertEqual(n['cells'][-1], before['cells'][-1])
        self.assertIn('benchmark_predictions.csv', n['cells'][-1]['source'])
        self.assertEqual(n['metadata']['h43_benchmark'], before['metadata']['h43_benchmark'])
        for c in n['cells']:
            if c['cell_type'] == 'code':
                ast.parse(c['source'])


if __name__ == '__main__':
    unittest.main()
