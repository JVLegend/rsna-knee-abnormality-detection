"""#RSNA #Kaggle #Testes — patches ancorados; código público nunca executado."""
import ast
import json
from pathlib import Path
import unittest

from scripts.prepare_h43_parent import build_notebook, partial_names, replace_once

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/avance_av001_sources/maverick/rsna-knee-0941-restructured.ipynb'


class BuilderTests(unittest.TestCase):
    def test_patch_rejects_missing_and_ambiguous(self):
        self.assertEqual(replace_once('abc', 'b', 'B'), 'aBc')
        for source in ['', 'bb']:
            with self.assertRaises(ValueError):
                replace_once(source, 'b', 'B')

    def test_partial_does_not_rename_sample(self):
        self.assertEqual(partial_names("'sample_submission.csv'"), "'sample_submission.csv'")
        self.assertEqual(partial_names("'/kaggle/working/submission.csv'"),
                         "'/kaggle/working/h43_candidate.partial.csv'")

    def test_hash_drift_blocks_build(self):
        with self.assertRaises(ValueError):
            build_notebook(b'{}', '')

    @unittest.skipUnless(SOURCE.exists(), 'external audited source not distributed in Git')
    def test_rejects_invalid_remote_receipt(self):
        with self.assertRaisesRegex(ValueError, 'Invalid artifact receipt'):
            build_notebook(SOURCE.read_bytes(), '', {'status': 'FAILED_ARTIFACT_PREFLIGHT'})

    @unittest.skipUnless(SOURCE.exists(), 'external audited source not distributed in Git')
    def test_real_source_build_is_deterministic_and_compiles(self):
        raw = SOURCE.read_bytes()
        runtime = (ROOT / 'scripts/h43_integrity.py').read_text()
        a = build_notebook(raw, runtime)
        self.assertEqual(a, build_notebook(raw, runtime))
        for cell in a['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        self.assertEqual(a['metadata']['h43_build']['preset'], 'parent')
        self.assertIn('h43_publish(', a['cells'][-1]['source'])
        self.assertIn("ROOT / 'test.csv'", a['cells'][-1]['source'])
        original = json.loads(raw)
        # Seções não modificadas mantêm aritmética; somente o caminho temporário muda.
        for i in [15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 38, 39, 42]:
            self.assertEqual(a['cells'][i + 1]['source'], partial_names(''.join(original['cells'][i]['source'])))
        self.assertEqual(a['cells'][37]['source'], "main()\nlog('done')\n")
        self.assertIn("h43_check_members([m['id'] for m in members]", a['cells'][32]['source'])
        raptor_assignment = next(node for node in ast.parse(a['cells'][48]['source']).body
                                 if isinstance(node, ast.Assign))
        raptor = ast.literal_eval(raptor_assignment.value)
        ast.parse(raptor)
        self.assertIn("raise RuntimeError('H43 Raptor arm failed')", raptor)


if __name__ == '__main__':
    unittest.main()
