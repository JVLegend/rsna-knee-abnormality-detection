"""#RSNA #Kaggle #Testes — regra exata, pesos, cobertura e builds pareados."""
import copy
import json
from pathlib import Path
import random
import unittest

import numpy as np
import pandas as pd
from scripts.dino_rank_replay import aggregate
from scripts.h43_stable_rank import h43_combine_public_exact
from scripts.prepare_h43_stable_fullstack import build

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb'
CAPTURE = ROOT / 'reports/avance_av008_dino_capture_v1/dino_replay_inputs.npz'


class StableRankTests(unittest.TestCase):
    def setUp(self):
        self.names = [str(i) for i in range(20)]
        self.targets = [str(i) for i in range(12)]
        self.members = [{'id': m, 'ids': ['b', 'a', 'c'],
                         'pred': np.asarray([[3] * 12, [1] * 12, [1] * 12])} for m in self.names]

    def test_ties_and_id_alignment(self):
        ids, total = h43_combine_public_exact(self.members, self.names, self.targets)
        self.assertEqual(ids, ['a', 'b', 'c'])
        self.assertEqual(total[:, 0].tolist(), [60, 120, 60])
        self.members[1]['ids'].reverse()
        self.members[1]['pred'] = self.members[1]['pred'][::-1]
        _, reordered = h43_combine_public_exact(self.members[::-1], self.names, self.targets)
        np.testing.assert_array_equal(total, reordered)

    def test_reject_nonuniform_weights(self):
        for key, value in [('weight', .9), ('weight', float('nan')), ('target_weight', [1] * 11),
                           ('target_weight', [.5] * 12)]:
            bad = copy.deepcopy(self.members); bad[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                h43_combine_public_exact(bad, self.names, self.targets)

    def test_reject_missing_duplicate_unknown_members(self):
        for bad in [self.members[:-1], [self.members[0]] * 20,
                    self.members[:-1] + [dict(self.members[-1], id='unknown')]]:
            with self.assertRaises(ValueError): h43_combine_public_exact(bad, self.names, self.targets)

    def test_reject_coverage_shape_finitude(self):
        for key, value in [('ids', ['a', 'a', 'c']), ('ids', ['a', 'b']),
                           ('pred', np.zeros((3, 11))), ('pred', np.full((3, 12), np.nan))]:
            bad = copy.deepcopy(self.members); bad[0][key] = value
            with self.assertRaises(ValueError): h43_combine_public_exact(bad, self.names, self.targets)

    @unittest.skipUnless(CAPTURE.exists(), 'External capture not committed')
    def test_real_capture_matches_independent_replay_20_orders(self):
        with np.load(CAPTURE, allow_pickle=False) as data:
            names, ids, pred = data['member_ids'].tolist(), data['study_ids'].tolist(), data['predictions']
        members = [dict(id=m, ids=ids, pred=p) for m, p in zip(names, pred)]
        reference = np.asarray(aggregate(pred.tolist(), list(range(20)), exact=True))
        rng = random.Random(2026)
        for _ in range(20):
            rng.shuffle(members)
            ordered_ids, result = h43_combine_public_exact(members, names, self.targets)
            got = pd.DataFrame(result).rank(method='average', pct=True).to_numpy()
            np.testing.assert_array_equal(got, reference[[ids.index(uid) for uid in ordered_ids]])

    def test_invalid_build(self):
        with self.assertRaises(ValueError): build(b'{}', '', '', 'serial')
        with self.assertRaises(ValueError): build(b'{}', '', '', 'bad')

    @unittest.skipUnless(BASE.exists(), 'External benchmark not committed')
    def test_paired_builds_preserve_other_models_and_legacy_combine(self):
        raw = BASE.read_bytes(); old = json.loads(raw)
        helper = (ROOT / 'scripts/ordered_prefetch.py').read_text() + '\n' + (ROOT / 'scripts/h43_e03_runtime.py').read_text()
        rank = (ROOT / 'scripts/h43_stable_rank.py').read_text()
        serial, prefetch = [build(raw, helper, rank, mode) for mode in ['serial', 'prefetch']]
        self.assertEqual(serial, build(raw, helper, rank, 'serial'))
        changed = [i for i,(a,b) in enumerate(zip(old['cells'], serial['cells'])) if a != b]
        self.assertEqual(changed, [33, 55, 57])
        for a,b in zip(serial['cells'], prefetch['cells']):
            normalized = a['source'].replace("H43_STABLE_MODE = 'serial'", "H43_STABLE_MODE = 'prefetch'")
            normalized = normalized.replace("E03_MODE='serial'", "E03_MODE='prefetch'")
            self.assertEqual(normalized, b['source'])
        original_combine = old['cells'][33]['source'].split('def _combine(')[1].split('def combine_public_members_by_fold')[0]
        self.assertIn('def _combine(' + original_combine, serial['cells'][33]['source'])
        self.assertEqual(serial['cells'][-1], old['cells'][-1])
        self.assertNotIn('expected_reference_csv_sha256', serial['metadata']['e03_fullstack'])


if __name__ == '__main__': unittest.main()
