"""#RSNA #Kaggle #Testes — falhas fechadas antes de publicar CSV."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.h43_integrity import (H43_TARGETS, h43_check_members, h43_publish,
                                   h43_preflight, h43_select, h43_sha, h43_validate_rows)


class IntegrityTests(unittest.TestCase):
    def test_preflight_missing_artifact_persists_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / 'failure.json'
            with self.assertRaises(RuntimeError):
                h43_preflight(root, receipt)
            result = json.loads(receipt.read_text())
            self.assertEqual(result['status'], 'FAILED_ARTIFACT_PREFLIGHT')
            self.assertTrue(result['errors'])
            self.assertFalse((root / 'submission.csv').exists())

    def test_members_exact_not_just_count(self):
        h43_check_members(['a', 'b'], ['b', 'a'])
        for actual in [['a'], ['a', 'a'], ['a', 'c']]:
            with self.assertRaises(RuntimeError):
                h43_check_members(['a', 'b'], actual)

    def test_csv_ids_nan_and_range(self):
        rows = [{'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)}]
        fields = ['StudyInstanceUID'] + H43_TARGETS
        h43_validate_rows(rows, ['a'], fields)
        with self.assertRaises(RuntimeError):
            h43_validate_rows(rows, ['b'], fields)
        for value in [float('nan'), float('inf'), -1, 1.1]:
            rows[0]['ACL'] = value
            with self.assertRaises(RuntimeError):
                h43_validate_rows(rows, ['a'], fields)

    def test_duplicate_names_resolved_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(2):
                (root / str(i)).mkdir()
                path = root / str(i) / 'heads.pt'
                path.write_bytes(str(i).encode())
                paths.append(path)
            self.assertEqual(h43_select(paths, 'heads.pt', h43_sha(paths[1])), paths[1])
            with self.assertRaises(RuntimeError):
                h43_select(paths, 'heads.pt')
            with self.assertRaises(RuntimeError):
                h43_select(paths, 'heads.pt', '0' * 64)

    def test_failed_gate_leaves_no_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial, output = root / 'partial.csv', root / 'submission.csv'
            with partial.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['StudyInstanceUID'] + H43_TARGETS)
                writer.writeheader()
                writer.writerow({'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)})
            receipt = {'preflight': {'status': 'PASSED_ARTIFACT_PREFLIGHT'},
                       'stages': ['dino', 'a5', 'rad', 'raptor']}
            run = {'coatnet_w': {'__default__': .6}}
            with self.assertRaises(RuntimeError):
                h43_publish(partial, output, ['a'], receipt, run)
            self.assertFalse(output.exists())
            self.assertTrue(partial.exists())
            receipt['stages'].append('coat')
            h43_publish(partial, output, ['a'], receipt, run)
            self.assertTrue(output.exists())
            self.assertFalse(partial.exists())
            self.assertEqual(receipt['status'], 'PASSED_PARENT_INTEGRITY')


if __name__ == '__main__':
    unittest.main()
