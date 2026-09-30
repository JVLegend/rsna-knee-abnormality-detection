"""#RSNA #Kaggle #Testes — auditoria do CSV independente do helper de produção."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scripts.assess_h43_stable_fullstack import validate_capture, RANK_SHA
from scripts.h43_integrity import H43_TARGETS
from scripts.h43_stable_rank import h43_combine_public_exact


class StableAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        names = [f'm{i}' for i in range(20)]
        ids = [f's{i:02}' for i in range(36)]
        pred = np.random.default_rng(2026).integers(0, 12, (20, 36, 12)) / 12.
        path = self.directory / 'dino_stable_inputs.npz'
        np.savez_compressed(path, member_ids=names, study_ids=ids, predictions=pred)
        self.receipt = dict(status='DINO_STABLE_AGGREGATED_NOT_SUBMISSION',
            rule='public_uniform_doubled_integer_ranks_v1', mode='serial',
            helper_sha256=RANK_SHA, members=20, studies=36, targets=H43_TARGETS,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.write_json('dino_stable_receipt.json', self.receipt)
        self.write_json('h43_preflight.json', {'dino_ids': names})
        self.write_json('h43_benchmark_selection.json', {'ids': ids})
        rows = [dict(id=m, ids=ids, pred=p) for m,p in zip(names, pred)]
        ordered, total = h43_combine_public_exact(rows, names, H43_TARGETS)
        frame = pd.DataFrame(total).rank(method='average', pct=True)
        frame.columns = H43_TARGETS; frame.insert(0, 'StudyInstanceUID', ordered)
        frame.to_csv(self.directory / 'submission_public_0899.csv', index=False)

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def test_valid_runtime_matches_independent_audit(self):
        receipt, pred = validate_capture(self.directory, 'serial')
        self.assertTrue(receipt['independent_replay_exact'])
        self.assertEqual(pred.shape, (20, 36, 12))
        self.assertEqual(hashlib.sha256(Path('scripts/h43_stable_rank.py').read_bytes()).hexdigest(), RANK_SHA)

    def test_reject_wrong_mode_or_helper(self):
        with self.assertRaises(ValueError): validate_capture(self.directory, 'prefetch')
        self.receipt['helper_sha256'] = 'wrong'
        self.write_json('dino_stable_receipt.json', self.receipt)
        with self.assertRaises(ValueError): validate_capture(self.directory, 'serial')

    def test_reject_bad_capture_hash(self):
        self.receipt['input_sha256'] = 'wrong'
        self.write_json('dino_stable_receipt.json', self.receipt)
        with self.assertRaises(ValueError): validate_capture(self.directory, 'serial')

    def test_reject_csv_drift(self):
        path = self.directory / 'submission_public_0899.csv'
        with path.open(newline='') as f: rows = list(csv.DictReader(f))
        rows[0]['ACL'] = '0.123456789'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with self.assertRaises(ValueError): validate_capture(self.directory, 'serial')

    def test_reject_wrong_member_pins(self):
        self.write_json('h43_preflight.json', {'dino_ids': ['unknown']})
        with self.assertRaises(ValueError): validate_capture(self.directory, 'serial')


if __name__ == '__main__': unittest.main()
