"""#RSNA #Kaggle #Testes — gate de promoção, teste real e saída não submetível."""
import hashlib
import json
from pathlib import Path
import unittest
from scripts.prepare_h43_stable_smoke import build

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'


class StableSmokeTests(unittest.TestCase):
    def setUp(self):
        self.rank = (ROOT / 'scripts/h43_stable_rank.py').read_text()
        self.helper = (ROOT / 'scripts/ordered_prefetch.py').read_text() + '\n' + (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        self.audit = {'status': 'PASSED_STABLE_FULLSTACK_PARITY', 'candidate_sha256': 'fixture',
            'stable_receipts': [{'helper_sha256': hashlib.sha256(self.rank.encode()).hexdigest()}] * 2,
            **dict.fromkeys(['eligible_for_real_test_smoke', 'exact_csv', 'dino_raw_predictions_exact',
                'raptor_inputs_exact', 'raptor_raw_exact', 'all_components_byte_identical'], True)}

    def test_reject_missing_or_failed_audit(self):
        for audit in [{}, dict(self.audit, exact_csv=False), dict(self.audit, eligible_for_real_test_smoke=False)]:
            with self.assertRaises(ValueError): build(b'{}', self.rank, self.helper, audit)

    def test_reject_parent_drift(self):
        with self.assertRaisesRegex(ValueError, 'parent build drift'):
            build(b'{}', self.rank, self.helper, self.audit)

    @unittest.skipUnless(PARENT.exists(), 'External audited parent not committed')
    def test_real_roots_and_other_models_preserved(self):
        raw = PARENT.read_bytes(); old = json.loads(raw)
        n = build(raw, self.rank, self.helper, self.audit)
        changed = [i for i,(a,b) in enumerate(zip(old['cells'], n['cells'])) if a != b]
        self.assertEqual(changed, [32, 48, 49, len(n['cells'])-1])
        self.assertEqual(changed, n['metadata']['h43_stable_smoke']['modified_cells'])
        self.assertNotIn('H43_BENCHMARK_ROOT', json.dumps(n))
        self.assertIn('smoke_predictions.csv', n['cells'][-1]['source'])
        self.assertNotIn("'/kaggle/working/submission.csv'", n['cells'][-1]['source'])
        with self.assertRaisesRegex(ValueError, 'helper differs'):
            build(raw, self.rank + '\n', self.helper, self.audit)


if __name__ == '__main__': unittest.main()
