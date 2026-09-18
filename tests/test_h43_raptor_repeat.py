"""#RSNA #Kaggle #Testes — repetição somente após gate e comparação sem tolerância."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scripts.prepare_h43_raptor_repeat import build
from scripts.assess_h43_raptor_repeat import compare_arrays

BUILD = Path(__file__).resolve().parents[1] / 'reports/avance_av010_build/raptor36_deterministic_abba_v1.ipynb'


class RaptorRepeatTests(unittest.TestCase):
    def test_reject_failed_audit(self):
        with self.assertRaises(ValueError): build(b'{}', {})

    @unittest.skipUnless(BUILD.exists(), 'External build not committed')
    def test_model_and_flags_preserved(self):
        raw = BUILD.read_bytes(); old = json.loads(raw)
        n = build(raw, {'status': 'VERIFIED_LOCAL_AUDIT', 'parity': True, 'eligible_for_full_stack_test': True})
        self.assertEqual(n['cells'][1:-1], old['cells'][1:-1])
        self.assertIn('torch.use_deterministic_algorithms(True)', n['cells'][-1]['source'])
        self.assertIn('raptor_repeat_receipt.json', n['cells'][-1]['source'])
        self.assertEqual(n['metadata']['e03']['sequence'], ['prefetch'])

    def test_exact_drift_and_invalid_arrays(self):
        a = {'ids': np.asarray([str(i) for i in range(36)]), 'arm_probs': np.zeros((4,36,12)), 'ranks': np.zeros((36,12))}
        self.assertTrue(all(v['exact'] for v in compare_arrays(a,a).values()))
        b = copy.deepcopy(a); b['arm_probs'][0,0,0] = 1e-8
        self.assertEqual(compare_arrays(a,b)['arm_probs']['changed_values'], 1)
        b['arm_probs'][0,0,0] = np.nan
        with self.assertRaises(ValueError): compare_arrays(a,b)
        with self.assertRaises(ValueError): compare_arrays(a,dict(a, ids=a['ids'][::-1]))


if __name__ == '__main__': unittest.main()
