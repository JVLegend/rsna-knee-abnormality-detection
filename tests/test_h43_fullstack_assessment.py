"""#RSNA #Kaggle #Testes — comparação completa rejeita deriva e tempo inválido."""
import copy
import unittest
from scripts.assess_h43_fullstack import compare_rows, compare_times
from scripts.h43_integrity import H43_TARGETS


class FullStackAssessmentTests(unittest.TestCase):
    def test_identical_and_changed_predictions(self):
        a = [{'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)}]
        self.assertTrue(all(v['changed_values'] == 0 for v in compare_rows(a, a, ['a']).values()))
        b = copy.deepcopy(a); b[0]['ACL'] = .75
        self.assertEqual(compare_rows(a, b, ['a'])['ACL'], {'changed_values': 1, 'max_abs_delta': .25})

    def test_ids_and_nan_rejected(self):
        a = [{'StudyInstanceUID': 'a', **dict.fromkeys(H43_TARGETS, .5)}]
        for key, value in [('StudyInstanceUID', 'b'), ('ACL', float('nan'))]:
            b = copy.deepcopy(a); b[0][key] = value
            with self.assertRaises(RuntimeError):
                compare_rows(a, b, ['a'])

    def test_time_gain_and_regression(self):
        r = compare_times({'a': 100, 'b': 100}, {'a': 50, 'b': 150})
        self.assertEqual(r['a']['time_reduction_pct'], 50)
        self.assertEqual(r['b']['time_reduction_pct'], -50)
        self.assertEqual(r['total']['time_reduction_pct'], 0)

    def test_invalid_time_or_stage_rejected(self):
        for bad in [{'a': 0}, {'a': float('nan')}, {'b': 1}]:
            with self.assertRaises(ValueError):
                compare_times({'a': 100}, bad)


if __name__ == '__main__':
    unittest.main()
