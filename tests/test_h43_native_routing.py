"""#RSNA #Kaggle #Testes — diagnóstico não vira fallback silencioso."""
from pathlib import Path
import unittest
from scripts.audit_h43_native_routing import extract, assess

BUILD = Path(__file__).resolve().parents[1] / 'reports/avance_av009_build/h43_stable36_serial_v1.ipynb'


class NativeRoutingTests(unittest.TestCase):
    def test_build_drift_rejected(self):
        with self.assertRaises(ValueError): extract(b'{}')

    @unittest.skipUnless(BUILD.exists(), 'External audited build not committed')
    def test_candidate_invariance_and_fail_closed(self):
        before = Path.cwd()
        result = assess(BUILD.read_bytes())
        self.assertEqual(Path.cwd(), before)
        self.assertTrue(result['missing_or_invalid_public_rejected'])
        self.assertEqual(result['native_constants_tested'], [0., .25, .75, 1.])
        self.assertFalse(result['submission_authorized'])
        self.assertFalse(result['fullstack_parity_gate_changed'])


if __name__ == '__main__': unittest.main()
