"""#RSNA #Kaggle #Testes — flags fixadas depois da fonte e amostra íntegra."""
import json
from pathlib import Path
import unittest
from scripts.prepare_h43_raptor_determinism import build
from scripts.prepare_h43_e03 import build as build_abba
from scripts.assess_h43_e03 import evaluate_records
from scripts.assess_h43_raptor_determinism import check_environment, EXPECTED

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'


class RaptorDeterminismTests(unittest.TestCase):
    def test_environment_accepts_fixed_settings(self):
        check_environment(dict(EXPECTED, torch='fixture', cuda='fixture', cudnn=9000))

    def test_environment_rejects_missing_or_changed_flags(self):
        for env in [{}, dict(EXPECTED, cudnn_benchmark=True), dict(EXPECTED, deterministic_algorithms='True')]:
            with self.assertRaises(ValueError): check_environment(env)

    def test_metrics_require_explicit_36(self):
        records = [dict(index=i, mode=m, seconds=100., studies=36, prepared_recipe_studies=108)
                   for i,m in enumerate(['serial','prefetch','prefetch','serial'])]
        with self.assertRaises(ValueError): evaluate_records(records, True)
        self.assertEqual(evaluate_records(records, True, studies=36)['abba_speedup'], 1)

    @unittest.skipUnless(PARENT.exists(), 'External parent not committed')
    def test_only_environment_and_description_change(self):
        raw = PARENT.read_bytes(); ids = [str(i) for i in range(36)]
        prefetch = (ROOT / 'scripts/ordered_prefetch.py').read_text()
        runtime = (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        base = build_abba(raw, ids, prefetch, runtime, study_count=36)
        n = build(raw, ids, prefetch, runtime)
        self.assertEqual(n['cells'][5], base['cells'][5])  # Every Raptor model/decode function unchanged.
        self.assertEqual(n['cells'][2], base['cells'][2])
        self.assertTrue(n['cells'][1]['source'].startswith("import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']"))
        self.assertIn('torch.backends.cudnn.benchmark = False', n['cells'][-1]['source'])
        self.assertTrue(n['cells'][-1]['source'].endswith('e03_benchmark()\n'))
        self.assertIn('36 training studies', n['cells'][-2]['source'])
        with self.assertRaises(ValueError): build(raw, ids[:-1], prefetch, runtime)


if __name__ == '__main__': unittest.main()
