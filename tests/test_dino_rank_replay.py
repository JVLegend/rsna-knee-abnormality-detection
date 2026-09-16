"""#RSNA #Kaggle #Testes — empates e ordem de conclusão não são hiperparâmetros."""
import json
from pathlib import Path
import unittest
from scripts.dino_rank_replay import average_ranks, aggregate, synthetic_probe
from scripts.prepare_h43_dino_capture import build

BASE = Path(__file__).resolve().parents[1] / 'reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb'


class DinoReplayTests(unittest.TestCase):
    def test_average_ties(self):
        self.assertEqual(average_ranks([3, 1, 1, 4]), [3, 1.5, 1.5, 4])
        self.assertEqual(average_ranks([3, 1, 1, 4], True), [.75, .375, .375, 1])

    def test_exact_order_invariant(self):
        a = [[[.1], [.2], [.2]], [[.3], [.2], [.1]], [[.5], [.5], [.1]]]
        self.assertEqual(aggregate(a, [0, 1, 2], True), aggregate(a, [2, 0, 1], True))
        with self.assertRaises(ValueError): aggregate(a, [0, 0, 1])

    def test_synthetic_failure_reproduced(self):
        r = synthetic_probe()
        self.assertTrue(any(r['changed_values_by_column']))
        self.assertEqual(r['exact_integer_changes'], [0, 0])
        self.assertFalse(r['actual_competition_cause_confirmed'])

    @unittest.skipUnless(BASE.exists(), 'External benchmark not committed')
    def test_capture_stops_before_a5(self):
        raw = BASE.read_bytes(); n = build(raw); old = json.loads(raw)
        changed = [i for i, (a, b) in enumerate(zip(old['cells'], n['cells'][:-1])) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn('dino_replay_inputs.npz', n['cells'][changed[0]]['source'])
        self.assertEqual(n['cells'][-2]['source'], "main()\nlog('done')\n")
        self.assertIn('DINO_CAPTURED_NOT_SUBMISSION', n['cells'][-1]['source'])
        with self.assertRaises(ValueError): build(b'{}')


if __name__ == '__main__': unittest.main()
