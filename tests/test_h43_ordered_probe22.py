"""#RSNA #Kaggle #Testes — fixed recipe, exact inheritance and independent replay."""
import ast
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.prepare_h43_ordered_probe22 import build, PURPOSE, SMOKE_SHA, SMOKE_CSV_SHA
from scripts.prepare_h43_probe22 import PROBE22, literal_assignment
from scripts.assess_h43_ordered_probe22 import check_receipts, replay_fusion, compare_raw
from scripts.h43_integrity import H43_TARGETS

SOURCE = Path('reports/avance_av015_build/h43_ordered_official_smoke_v1.ipynb')


def audit():
    return dict(status='PASSED_ORDERED_OFFICIAL_SMOKE',studies=3,series=15,
        prediction_sha256=SMOKE_CSV_SHA,all_five_stages_present=True,
        dino_independent_replay=True,coat_independent_replay=True,official_ids_exact=True)


class OrderedProbeTests(unittest.TestCase):
    def test_reject_source_and_incomplete_gates(self):
        with self.assertRaisesRegex(ValueError,'source drift'):build(b'{}',audit())
        for key in audit():
            a=audit();a.pop(key)
            with self.assertRaisesRegex(ValueError,'smoke required'):build(b'{}',a)

    @unittest.skipUnless(SOURCE.exists(),'External evidence not committed')
    def test_only_four_cells_change_and_repeatable(self):
        raw=SOURCE.read_bytes(); n=build(raw,audit()); original=json.loads(raw)
        self.assertEqual(n,build(raw,audit()))
        self.assertEqual([i for i,(a,b) in enumerate(zip(n['cells'],original['cells'])) if a!=b],[3,4,5,61])
        self.assertEqual(literal_assignment(n['cells'][5]['source'],'RUN')['coatnet_w'],PROBE22)
        for c in n['cells']:
            if c['cell_type']=='code':
                ast.parse(c['source'])
                self.assertNotIn('H43_BENCHMARK_ROOT',c['source'])
        self.assertIn("'/kaggle/working/submission.csv'",n['cells'][61]['source'])
        self.assertNotIn('ordered_smoke_receipt.json',n['cells'][61]['source'])

    @unittest.skipUnless(SOURCE.exists(),'External evidence not committed')
    def test_publication_gate(self):
        n=build(SOURCE.read_bytes(),audit()); ns={}
        exec(compile(n['cells'][3]['source'].split('\nH43_ARTIFACT_LOCK = ',1)[0],'<local-gate>','exec'),ns)
        run=literal_assignment(n['cells'][5]['source'],'RUN')
        r={'preflight':{'status':'PASSED_ARTIFACT_PREFLIGHT'},'stages':['dino','a5','rad','raptor','coat']}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'partial.csv';o=Path(tmp)/'submission.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['StudyInstanceUID']+H43_TARGETS);w.writerow(['a']+[.5]*12)
            for key,value in [('coatnet_w',{'__default__':.6}),('a5_w',.44),('rad_flip_tta','all')]:
                bad=copy.deepcopy(run);bad[key]=value
                with self.assertRaisesRegex(RuntimeError,'fixed ordered probe22'):ns['h43_publish'](p,o,['a'],copy.deepcopy(r),bad)
                self.assertFalse(o.exists())
            bad=copy.deepcopy(r);bad['stages'].remove('coat')
            with self.assertRaisesRegex(RuntimeError,'incomplete stages'):ns['h43_publish'](p,o,['a'],bad,run)
            ns['h43_publish'](p,o,['a'],r,run)
            self.assertTrue(o.exists())
            saved=json.loads((Path(tmp)/'h43_ordered_probe22_integrity.json').read_text())
            self.assertEqual(saved['run'],run)
            self.assertEqual(saved['status'],'PASSED_ORDERED_PROBE22_INTEGRITY')

    def test_receipts_fail_closed(self):
        run={'coatnet_w':PROBE22}
        r=dict(status='PASSED_ORDERED_PROBE22_INTEGRITY',purpose=PURPOSE,preset='probe22',run=run,
            historical_probe22_ref=56263721,stages=['dino','a5','rad','raptor','coat'],studies=3,
            preflight=dict(status='PASSED_ARTIFACT_PREFLIGHT',errors=[],artifact_lock_entries=56,gpus=['Tesla T4']*2))
        p=dict(status='ORDERED_PROBE22_READY_FOR_REVIEW',purpose=PURPOSE,source_smoke_sha256=SMOKE_SHA,
            studies=3,automatically_submitted=False,official_root='/kaggle/input/competition')
        check_receipts(r,p,run,['a','b','c'])
        for key in r:
            bad=copy.deepcopy(r);bad.pop(key)
            with self.assertRaises((ValueError,KeyError)):check_receipts(bad,p,run,['a','b','c'])
        for key in p:
            bad=copy.deepcopy(p);bad.pop(key)
            with self.assertRaises(ValueError):check_receipts(r,bad,run,['a','b','c'])

    def test_replay_matches_pandas_and_rejects_nan(self):
        rng=np.random.default_rng(2026)
        left=pd.DataFrame(rng.integers(0,8,(36,12))).rank(pct=True).to_numpy()
        right=pd.DataFrame(rng.integers(0,8,(36,12))).rank(pct=True).to_numpy()
        w=np.array([PROBE22.get(t,.6) for t in H43_TARGETS])
        expected=pd.DataFrame((1-w)*left+w*right).rank(pct=True).to_numpy()
        np.testing.assert_array_equal(replay_fusion(left,right,PROBE22),expected)
        np.testing.assert_array_equal(expected[:,3],pd.DataFrame(right).rank(pct=True).to_numpy()[:,3])
        bad=left.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):replay_fusion(bad,right,PROBE22)
        with self.assertRaises(ValueError):replay_fusion(left[:2],right,PROBE22)

    def test_raw_comparison_canonicalizes_dino_but_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'reference',Path(tmp)/'candidate';a.mkdir();b.mkdir()
            pred=np.arange(48).reshape(2,2,12)
            np.savez(a/'dino_stable_inputs.npz',member_ids=['a','b'],study_ids=['x','y'],predictions=pred)
            np.savez(b/'dino_stable_inputs.npz',member_ids=['b','a'],study_ids=['y','x'],predictions=pred[::-1,::-1])
            for name in ['e03_run0_raw.npz','coat_resgated_ep10_top3_predictions.npz']:
                for d in [a,b]:np.savez(d/name,predictions=pred)
            compare_raw(a,b)
            np.savez(b/'e03_run0_raw.npz',predictions=pred+1)
            with self.assertRaisesRegex(ValueError,'Raw predictions differ'):compare_raw(a,b)


if __name__=='__main__':unittest.main()
