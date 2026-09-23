"""#RSNA #Kaggle #Testes — lotes fixos e preservação do forward CoAt."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
from scripts.h43_coat_order import patch_coat, WAIT_ORIGINAL, WAIT_ORDERED
from scripts.prepare_h43_coat_order import build
from scripts.assess_h43_coat_order import compare_arrays, batch_contract, load_preflight, load_run

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/avance_av013_sources/coat/coatnet_resgated_ep10_top3_inference.py'
STACK = ROOT / 'reports/avance_av012_build/h43_deterministic36_serial_v1.ipynb'


class CoAtOrderTests(unittest.TestCase):
    def test_source_drift_rejected(self):
        with self.assertRaises(ValueError): patch_coat('wrong', 'ordered')
        with self.assertRaises(ValueError): build(b'{}', '', '')

    def test_raw_drift_is_not_hidden_by_equal_ranks(self):
        import numpy as np
        a = {'study_uids': np.asarray(['a','b']), 'raw_probabilities': np.full((3,2,12), .5),
             'checkpoint_percentile_ranks': np.full((3,2,12), .5), 'rank_ensemble': np.full((2,12), .5)}
        b = {k:v.copy() for k,v in a.items()}
        self.assertTrue(compare_arrays(a,b)['raw_probabilities']['exact'])
        b['raw_probabilities'][0,1,0] = .50001
        r = compare_arrays(a,b)
        self.assertEqual(r['raw_changed_study_indices'],[1])
        self.assertFalse(r['raw_probabilities']['exact'])
        self.assertTrue(r['rank_ensemble']['exact'])
        b['study_uids'] = b['study_uids'][::-1]
        with self.assertRaises(ValueError): compare_arrays(a,b)

    def test_fixed_batch_contract_rejects_reordering(self):
        import copy
        ids = ['a','b','c']
        probe = {'mode': 'ordered', 'inputs': [{'index': i,'uid': u} for i,u in enumerate(ids)],
                 'batches': [[0,1],[2]], 'torch': 'fixture'}
        shard = {'start':0, 'stop':3, 'order_probe':probe}
        self.assertEqual(batch_contract(shard,'ordered',ids)[1],[[0,1],[2]])
        for batches in [[[0,2],[1]], [[0,1],[1]]]:
            bad = copy.deepcopy(shard); bad['order_probe']['batches'] = batches
            with self.assertRaises(ValueError): batch_contract(bad,'ordered',ids)

    @unittest.skipUnless((ROOT/'reports/avance_av012_prefetch_v1/coat_top3_part1.npz').exists(), 'External outputs unavailable')
    def test_real_coat_rank_replay_and_raw_differences(self):
        arrays = []
        for mode in ['serial','prefetch']:
            d = ROOT / f'reports/avance_av012_{mode}_v1'
            pre,ids = load_preflight(d)
            arrays.append(load_run(d,ids,'_coat_arm.csv',pre)[1])
        r = compare_arrays(*arrays)
        self.assertEqual(r['raw_probabilities']['changed_values'],71)
        self.assertEqual(r['raw_changed_study_indices'],[11,13])
        self.assertEqual(r['rank_ensemble']['changed_values'],2)

    @unittest.skipUnless(SOURCE.exists(), 'Pinned external source not committed')
    def test_only_shard_changes(self):
        raw = SOURCE.read_text()
        functions = lambda s: {n.name: ast.dump(n) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
        baseline = functions(raw)
        for mode in ['completion','ordered']:
            s = patch_coat(raw, mode); new = functions(s)
            self.assertEqual(set(baseline), set(new))
            self.assertEqual([k for k in baseline if baseline[k] != new[k]], ['_infer_shard'])
            self.assertIn(WAIT_ORDERED if mode == 'ordered' else WAIT_ORIGINAL, s)
        with self.assertRaises(ValueError): patch_coat(raw, 'unknown')

    @unittest.skipUnless(SOURCE.exists(), 'Pinned external source not committed')
    def test_actual_scheduling_loop_is_independent_of_completion(self):
        raw = SOURCE.read_text()
        def run(mode, completion_order):
            fn = next(n for n in ast.parse(patch_coat(raw, mode)).body
                      if isinstance(n, ast.FunctionDef) and n.name == '_infer_shard')
            loop = next(n for n in fn.body if isinstance(n, ast.Try)
                        and any(isinstance(x, ast.While) for x in n.body))
            class Future:
                def __init__(self, i): self.i = i
                def result(self):
                    import numpy as np
                    return self.i, SimpleNamespace(images=np.zeros(1, dtype='uint8'),
                        slots=np.zeros(1, dtype='int64'), metadata=np.zeros(1, dtype='float32'))
            pending = {Future(i): i for i in range(6)}
            batches = []
            def fail(*args): raise AssertionError('Unexpected fallback')
            ns = dict(pending=pending, ready=[], completed=0, prepared_studies=0,
                      next_submit=6, study_uids=[str(i) for i in range(6)], input_contracts=[],
                      gpu_batch_studies=2, FIRST_COMPLETED='unused',
                      wait=lambda futures, **kw: ({min(futures, key=lambda f: completion_order.index(f.i))}, set()),
                      executor=SimpleNamespace(shutdown=lambda **kw: None), record_failure=fail,
                      infer_batch=lambda batch: batches.append([i for i,b in batch]),
                      progress_prefix='', time=SimpleNamespace(monotonic=lambda: 0), started=0)
            exec(compile(ast.Module(body=[loop], type_ignores=[]), '<actual-loop>', 'exec'), ns)
            return batches
        normal = [0,1,2,3,4,5]; shuffled = [0,2,1,3,5,4]
        expected = [[0,1],[2,3],[4,5]]
        self.assertEqual(run('ordered', normal), expected)
        self.assertEqual(run('ordered', shuffled), expected)
        self.assertNotEqual(run('completion', shuffled), expected)

    @unittest.skipUnless(SOURCE.exists() and STACK.exists(), 'External build unavailable')
    def test_only_coat_executes_and_sample_preflight_preserved(self):
        import json
        raw = STACK.read_bytes(); old = json.loads(raw)
        n = build(raw, (ROOT/'scripts/h43_coat_order.py').read_text(), SOURCE.read_text())
        self.assertEqual(n['cells'][1], old['cells'][3])
        self.assertEqual(n['cells'][2], old['cells'][4])
        self.assertIn("'ordered', 'ordered'", n['cells'][-1]['source'])
        self.assertNotIn("_KE_NS['main']()", json.dumps(n))
        self.assertIn('coat_diagnostic.csv', n['cells'][-1]['source'])


if __name__ == '__main__': unittest.main()
