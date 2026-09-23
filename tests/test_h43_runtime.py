"""#RSNA #Kaggle #Testes — orçamento não vira score nem permissão de envio."""
import unittest
from scripts.assess_h43_runtime import scenarios


class RuntimeTests(unittest.TestCase):
    def test_conservative_projection(self):
        t = {'dino': 36, 'a5': 36, 'rad': 36, 'raptor_coat_fusion': 36}
        r = scenarios(t, 36, 60, [100, 1000])
        self.assertEqual(r['seconds_per_study_including_stage_loads'], 4)
        self.assertAlmostEqual(r['cases'][0]['estimated_hours_with_margin'], (400 + 60 + 120) * 1.25 / 3600)
        self.assertFalse(r['submission_authorized_by_this_report'])
        self.assertLess(r['cases'][0]['raw_dino_cache_gib_only'], r['cases'][1]['raw_dino_cache_gib_only'])

    def test_smoke_or_missing_stage_rejected(self):
        t = {'dino': 1, 'a5': 1, 'rad': 1, 'raptor_coat_fusion': 1}
        with self.assertRaises(ValueError):
            scenarios(t, 3, 1)
        del t['rad']
        with self.assertRaises(ValueError):
            scenarios(t, 36, 1)

    def test_nonfinite_and_negative_rejected(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                scenarios({'dino': value, 'a5': 1, 'rad': 1, 'raptor_coat_fusion': 1}, 36, 1)


if __name__ == '__main__':
    unittest.main()
