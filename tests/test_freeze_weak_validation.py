"""#RSNA #Kaggle #Testes — contrato do manifesto V01."""
import tempfile
import unittest
import csv
import json
from pathlib import Path

from scripts.freeze_weak_validation import TARGETS, assign_groups, build, class_counts, freeze, group_hash


class FreezeTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(group_hash("  LＥSÃO\nNORMAL "), group_hash("lesão normal"))
        with self.assertRaises(ValueError):
            group_hash(" \n")

    def test_deterministic_disjoint_whole_groups(self):
        rows = [{"StudyInstanceUID": str(i), "report_hash": str(i // 2)} for i in range(40)]
        a = assign_groups(rows, 9, 7, 2026)
        self.assertEqual(a, assign_groups(list(reversed(rows)), 9, 7, 2026))
        sets = [{r["report_hash"] for r in v} for v in a.values()]
        for i, group in enumerate(sets):
            for other in sets[i + 1:]:
                self.assertFalse(group & other)
        self.assertEqual(sum(map(len, a.values())), 40)
        self.assertGreaterEqual(len(a["development"]), 9)
        self.assertGreaterEqual(len(a["confirmation"]), 7)

    def test_insufficient_groups_fail_closed(self):
        rows = [{"StudyInstanceUID": str(i), "report_hash": "same"} for i in range(40)]
        with self.assertRaises(ValueError):
            assign_groups(rows, 9, 7, 2026)

    def test_ambiguous_labels_not_positive(self):
        rows = [{"labels": {t: v for t in TARGETS}} for v in [.1, .5, .9]]
        self.assertEqual(class_counts(rows)["ACL"], {
            "positive_gt_0.5": 1, "negative_lt_0.5": 1, "uncertain_eq_0.5": 1})

    def test_frozen_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assertTrue(freeze(path, {"a": 1}))
            self.assertFalse(freeze(path, {"a": 1}))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                freeze(path, {"a": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_inventory_excludes_gold_alias_missing_plane_invalid_and_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, teacher, records = [], [], []
            for i in range(20):
                uid = str(i)
                report = "gold report" if i in [0, 1] else f"report {i}"
                train.append({"StudyInstanceUID": uid, "Report": " " if i == 4 else report,
                              **{t: "1" if i == 0 else "" for t in TARGETS}})
                teacher.append({"StudyInstanceUID": uid,
                                **{t: "nan" if i == 3 else str(i % 2) for t in TARGETS}})
                for plane in ["Sagittal", "Coronal", "Axial"]:
                    array = f"{uid}_{plane}.npz"
                    # Existência/tamanho, não um teste de decodificação de pixels.
                    if not (i == 2 and plane == "Axial"):
                        (root / array).write_bytes(b"fixture")
                    records.append({"study_uid": uid, "series_uid": plane,
                                    "anatomical_plane": plane, "array_path": array, "n_slices": 3})
            for name, rows in [("train.csv", train), ("teacher.csv", teacher)]:
                with (root / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (root / "index.json").write_text(json.dumps({"records": records}))
            payload = build(root / "train.csv", root / "teacher.csv", root / "index.json", 4, 3)
            self.assertEqual(payload["coverage"]["eligible_studies"], 15)
            self.assertEqual(payload["excluded"], {
                "0": "gold_or_shared_gold_report", "1": "gold_or_shared_gold_report",
                "2": "missing_plane_cache", "3": "invalid_teacher_labels", "4": "blank_report"})
            ids = [row["StudyInstanceUID"] for rows in payload["splits"].values() for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(set(ids) & {"0", "1", "2", "3", "4"})
            records.append(records[0])
            (root / "index.json").write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "duplicada"):
                build(root / "train.csv", root / "teacher.csv", root / "index.json", 4, 3)


if __name__ == "__main__":
    unittest.main()
