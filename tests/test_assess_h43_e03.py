"""#RSNA #Kaggle #Testes — rejeitar ganhos falsos de cache ou paridade."""
import copy
import unittest
from scripts.assess_h43_e03 import evaluate_records, MODES


def records(times):
    return [{'index': i, 'mode': mode, 'seconds': t, 'studies': 12,
             'prepared_recipe_studies': 36} for i, (mode, t) in enumerate(zip(MODES, times))]


class E03AssessmentTests(unittest.TestCase):
    def test_real_gain_requires_both_comparators(self):
        r = evaluate_records(records([110, 80, 80, 100]), True)
        self.assertEqual(r['abba_speedup'], 105 / 80)
        self.assertEqual(r['warm_serial_vs_prefetch_speedup'], 100 / 80)
        self.assertTrue(r['eligible_for_full_stack_test'])

    def test_cold_serial_alone_cannot_approve(self):
        r = evaluate_records(records([200, 100, 100, 101]), True)
        self.assertGreater(r['abba_speedup'], 1.05)
        self.assertFalse(r['eligible_for_full_stack_test'])

    def test_parity_failure_cannot_approve(self):
        self.assertFalse(evaluate_records(records([200, 50, 50, 200]), False)['eligible_for_full_stack_test'])

    def test_invalid_inputs_rejected(self):
        base = records([100] * 4)
        for field, value in [('seconds', float('nan')), ('seconds', float('inf')), ('seconds', -1),
                              ('mode', 'prefetch'), ('index', 2), ('studies', 11),
                              ('prepared_recipe_studies', 35)]:
            bad = copy.deepcopy(base); bad[0][field] = value
            with self.assertRaises(ValueError):
                evaluate_records(bad, True)
        with self.assertRaises(ValueError):
            evaluate_records(base[:3], True)


if __name__ == '__main__':
    unittest.main()
