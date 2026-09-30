"""#RSNA #Kaggle #Testes — preset fixo sem relaxar o gate parent."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_h43_probe22 import build_probe22, literal_assignment, PROBE22
from scripts.h43_integrity import H43_TARGETS
from scripts.assess_h43_probe22 import paired_counts

PARENT = Path(__file__).resolve().parents[1] / 'reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb'


class Probe22Tests(unittest.TestCase):
    def test_paired_target_contract(self):
        parent = [{'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)}]
        candidate = copy.deepcopy(parent); candidate[0]['ACL'] = .75
        self.assertEqual(paired_counts(parent, candidate)['ACL'], 1)
        candidate[0]['MCL'] = .75
        with self.assertRaisesRegex(RuntimeError, 'Unchanged target drift'):
            paired_counts(parent, candidate)

    def test_paired_ids_and_nan_rejected(self):
        parent = [{'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)}]
        for key, value in [('StudyInstanceUID', 'b'), ('ACL', float('nan'))]:
            candidate = copy.deepcopy(parent); candidate[0][key] = value
            with self.assertRaises(RuntimeError):
                paired_counts(parent, candidate)

    def test_parent_drift_rejected(self):
        with self.assertRaises(ValueError):
            build_probe22(b'{}')

    def test_literal_only_and_unique(self):
        self.assertEqual(literal_assignment('RUN = {"a": 1}', 'RUN'), {'a': 1})
        for source in ['RUN = f()', 'RUN = 1\nRUN = 2', 'x = 1']:
            with self.assertRaises(ValueError):
                literal_assignment(source, 'RUN')

    @unittest.skipUnless(PARENT.exists(), 'External parent not committed')
    def test_only_preset_and_gates_changed(self):
        raw = PARENT.read_bytes()
        a = build_probe22(raw)
        self.assertEqual(a, build_probe22(raw))
        parent = json.loads(raw)
        changed = [i for i, (x, y) in enumerate(zip(parent['cells'], a['cells'])) if x != y]
        self.assertEqual(changed, [3, 4, len(a['cells']) - 1])
        self.assertEqual(a['metadata']['h43_build']['published_outer_weights'], PROBE22)
        self.assertEqual(a['metadata']['h43_build']['artifact_lock_entries'], 56)
        self.assertEqual(a['cells'][49], parent['cells'][49])
        for c in a['cells']:
            if c['cell_type'] == 'code':
                ast.parse(c['source'])

    @unittest.skipUnless(PARENT.exists(), 'External parent not committed')
    def test_probe_gate_checks_full_recipe_and_receipt(self):
        n = build_probe22(PARENT.read_bytes())
        # Executa somente o gate stdlib de autoria local, sem preflight/GPU.
        setup = n['cells'][3]['source'].split('\nH43_ARTIFACT_LOCK = ', 1)[0]
        ns = {}
        exec(compile(setup, '<local-stdlib-gate>', 'exec'), ns)
        run = literal_assignment(n['cells'][4]['source'], 'RUN')
        receipt = {'preflight': {'status': 'PASSED_ARTIFACT_PREFLIGHT'},
                   'stages': ['dino', 'a5', 'rad', 'raptor', 'coat']}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'partial.csv'
            output = Path(tmp) / 'submission.csv'
            with p.open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['StudyInstanceUID'] + H43_TARGETS)
                w.writerow(['uid'] + [.5] * 12)
            for key, value in [('coatnet_w', {'__default__': .6}), ('a5_w', .44),
                               ('rad_flip_tta', 'all'), ('raptor_extra_arm_w', .1)]:
                bad = copy.deepcopy(run); bad[key] = value
                with self.assertRaisesRegex(RuntimeError, 'fixed probe22 recipe'):
                    ns['h43_publish'](p, output, ['uid'], copy.deepcopy(receipt), bad)
                self.assertFalse(output.exists())
            bad_receipt = copy.deepcopy(receipt); bad_receipt['stages'].remove('coat')
            with self.assertRaisesRegex(RuntimeError, 'incomplete stages'):
                ns['h43_publish'](p, output, ['uid'], bad_receipt, run)
            ns['h43_publish'](p, output, ['uid'], receipt, run)
            result = json.loads((Path(tmp) / 'h43_probe22_integrity.json').read_text())
            self.assertEqual(result['status'], 'PASSED_PROBE22_INTEGRITY')
            self.assertEqual(result['run'], run)
            self.assertFalse(result['oof_independent'])
            self.assertEqual(result['parent_submission_ref'], 56253529)
            self.assertFalse((Path(tmp) / 'h43_parent_integrity.json').exists())
            self.assertTrue(output.exists())


if __name__ == '__main__':
    unittest.main()
