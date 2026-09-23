"""#RSNA #Kaggle #Testes — smoke bloqueado sem paridade e sem caminho de treino."""
import ast
import json
from pathlib import Path
import unittest
import tempfile
import csv
import numpy as np
from scripts.prepare_h43_ordered_smoke import build, REQUIRED
from scripts.prepare_h43_ordered_fullstack import recipe
from scripts.prepare_h43_ordered_smoke import PURPOSE, SOURCE_SHA
from scripts.assess_h43_ordered_smoke import check_receipts
from scripts.assess_h43_coat_order import load_run
from scripts.h43_integrity import H43_TARGETS, h43_sha
from scripts.dino_rank_replay import rank_columns

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/avance_av014_build/h43_ordered36_prefetch_v1.ipynb'


class OrderedSmokeTests(unittest.TestCase):
    def setUp(self):
        self.audit = dict(status='PASSED_ORDERED_FULLSTACK_PARITY',ordered_coat_recipe=recipe(),
                          candidate_sha256='fixture',**dict.fromkeys(REQUIRED,True))

    def test_every_gate_required(self):
        for k in REQUIRED:
            with self.assertRaises(ValueError): build(b'{}',dict(self.audit,**{k:False}))
        with self.assertRaises(ValueError): build(b'{}',{})
        with self.assertRaisesRegex(ValueError,'build drift'): build(b'{}',self.audit)
        with self.assertRaisesRegex(ValueError,'recipe differs'): build(b'{}',dict(self.audit,ordered_coat_recipe={}))

    def test_smoke_receipt_rejects_wrong_count_root_and_stages(self):
        ids=['a','b','c']
        pre={'status':'PASSED_ARTIFACT_PREFLIGHT','errors':[],'artifact_lock_entries':56,'gpus':['Tesla T4']*2}
        r={'status':'PASSED_PARENT_INTEGRITY','purpose':PURPOSE,'studies':3,
           'stages':['dino','a5','rad','raptor','coat'],'preflight':pre}
        s={'status':'ORDERED_SMOKE_COMPLETE_NOT_SUBMITTED','purpose':PURPOSE,'studies':3,
           'source_build_sha256':SOURCE_SHA,'official_root':'/kaggle/input/competitions/rsna-knee-abnormality-detection',
           'submission_authorized':False}
        check_receipts(r,s,ids)
        for bad in [dict(s,studies=36),dict(s,official_root='/kaggle/working/h43_benchmark_input'),dict(s,submission_authorized=True)]:
            with self.assertRaises(ValueError):check_receipts(r,bad,ids)
        with self.assertRaises(ValueError):check_receipts(dict(r,stages=['dino']),s,ids)

    def test_coat_loader_supports_explicit_official_sample_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);ids=['a','b','c']
            raw=np.random.default_rng(2026).random((3,3,12)).astype(np.float32)
            ranks=np.asarray([rank_columns(p.tolist(),pct=True) for p in raw]);mean=ranks.mean(0)
            np.savez(d/'coat_resgated_ep10_top3_predictions.npz',study_uids=ids,
                     raw_probabilities=raw,checkpoint_percentile_ranks=ranks,rank_ensemble=mean)
            with (d/'_coat_arm.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['StudyInstanceUID']+H43_TARGETS)
                w.writerows([[u,*row] for u,row in zip(ids,mean)])
            bounds=[[0,2],[2,3]]
            for i,(start,stop) in enumerate(bounds):
                np.savez(d/f'coat_top3_part{i}.npz',study_uids=ids[start:stop],
                         failures_json='[]',prediction=raw[:,start:stop])
            checkpoints=[{'epoch':e,'name':f'e{e}.pt','sha256':str(e)} for e in [4,6,8]]
            pre={'files':{c['name']:{'path':c['name'],'sha256':c['sha256']} for c in checkpoints}}
            receipt=dict(status='VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION',models=3,studies=3,
                series=18,gpu_names=['Tesla T4']*2,failures=[],fallback_studies=0,gpu_batch_studies=2,
                backbone_micro_images=8,serving_precision='float16',cudnn_benchmark=False,
                checkpoint_rank_weights=[1/3]*3,checkpoints=checkpoints,parallel_bounds=bounds,
                processes=[dict(start=a,stop=b,returncode=0) for a,b in bounds],
                shards=[dict(start=a,stop=b) for a,b in bounds],output_sha256=h43_sha(d/'_coat_arm.csv'),
                predictions_sha256=h43_sha(d/'coat_resgated_ep10_top3_predictions.npz'))
            (d/'coat_resgated_ep10_top3_submission_receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(load_run(d,ids,'_coat_arm.csv',pre,expected_series=18)[1]['raw_probabilities'].shape,(3,3,12))
            with self.assertRaises(ValueError):load_run(d,ids,'_coat_arm.csv',pre)

    @unittest.skipUnless(SOURCE.exists(),'External audited build unavailable')
    def test_only_paths_and_final_gate_change(self):
        raw = SOURCE.read_bytes(); old = json.loads(raw); n = build(raw,self.audit)
        self.assertEqual([i for i,(a,b) in enumerate(zip(old['cells'],n['cells'])) if a!=b],[4,24,43,55,57,61])
        text = json.dumps(n)
        self.assertNotIn('H43_BENCHMARK_ROOT',text)
        self.assertNotIn('/kaggle/working/h43_benchmark_input',text)
        self.assertIn('smoke_predictions.csv',n['cells'][-1]['source'])
        self.assertNotIn("'/kaggle/working/submission.csv'",n['cells'][-1]['source'])
        for i in [3,33,37,45,51]:self.assertEqual(n['cells'][i],old['cells'][i])
        # Reverse the sole CoAt discovery change: flags, loader, source, fusion all exact.
        self.assertEqual(n['cells'][57]['source'].replace('competition_root=rt.base.find_competition_root(),',
            "competition_root=Path('/kaggle/working/h43_benchmark_input'),"),old['cells'][57]['source'])
        for i in [24,43,55,57]:ast.parse(n['cells'][i]['source'])


if __name__ == '__main__':unittest.main()
