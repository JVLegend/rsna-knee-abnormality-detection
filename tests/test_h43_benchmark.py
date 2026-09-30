"""#RSNA #Kaggle #Testes — amostra de runtime não vira submissão."""
import ast
import unittest
from pathlib import Path
from scripts.prepare_h43_benchmark import select_studies, adapt_notebook, benchmark_setup

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'


class BenchmarkTests(unittest.TestCase):
    def test_selection_deterministic_unique_spans_counts(self):
        ids = [str(i) for i in range(100)]
        counts = {uid: 4 + int(uid) // 10 for uid in ids}
        a = select_studies(ids, counts)
        self.assertEqual(a, select_studies(ids[::-1], counts))
        self.assertEqual(len(set(a)), 36)
        self.assertEqual([counts[a[0]], counts[a[-1]]], [4, 13])

    def test_missing_data_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            select_studies(['a', 'a'], {'a': 3}, 2)
        with self.assertRaises(ValueError):
            select_studies(['a', 'b'], {'a': 3}, 2)

    @unittest.skipUnless(PILOT.exists(), 'external pilot build absent')
    def test_all_runners_routed_and_no_submission_publication(self):
        output = adapt_notebook(PILOT.read_bytes(), ['a', 'b'])
        sources = [c['source'] for c in output['cells'] if c['cell_type'] == 'code']
        for source in sources:
            ast.parse(source)
        self.assertIn("benchmark_predictions.csv", sources[-1])
        self.assertNotIn("'/kaggle/working/submission.csv'", sources[-1])
        self.assertTrue(any("return Path(os.environ['H43_BENCHMARK_ROOT'])" in s for s in sources))
        self.assertTrue(any("COMP = Path(os.environ['H43_BENCHMARK_ROOT'])" in s for s in sources))
        self.assertTrue(any("competition_root=Path('/kaggle/working/h43_benchmark_input')" in s for s in sources))
        self.assertEqual(sum("H43_RECEIPT.setdefault('timings_seconds'" in s for s in sources), 4)
        self.assertIn('runtime_benchmark_only_no_auc_no_submission', benchmark_setup(['a', 'b']))


if __name__ == '__main__':
    unittest.main()
