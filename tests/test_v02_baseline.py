"""#RSNA #Kaggle #Testes — split gates, soft metrics, checkpoint resume."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from scripts.prepare_v02_baseline import SPEC,MANIFEST_SHA,PLANES,validate
from scripts.assess_v02_baseline import replay,loss


def fixture():
    splits={}
    for name,count in [('train',299),('development',250)]:
        splits[name]=[{'StudyInstanceUID':f'{name}{i}','report_hash':f'{name}{i}','labels':[.5]*12,
                       'series':[{'plane':p,'selected_files':['a','b','c'],'image_sha256':f'{name}{i}{p}'} for p in PLANES]}
                      for i in range(count)]
    return {'status':'PASSED_BASELINE_CACHE','manifest_sha256':MANIFEST_SHA,'splits':splits,'confirmation_pixels_read':0}


def test_frozen_baseline_gates():
    good=fixture();validate(good)
    for kind in ['label','id','group','pixel','count','reserved','plane','file']:
        bad=copy.deepcopy(good);r=bad['splits']['development'][0]
        if kind=='label':r['labels'][0]=float('nan')
        if kind=='id':r['StudyInstanceUID']='train0'
        if kind=='group':r['report_hash']='train0'
        if kind=='pixel':r['series'][0]['image_sha256']='train0Sagittal'
        if kind=='count':bad['splits']['train'].pop()
        if kind=='reserved':bad['confirmation_pixels_read']=1
        if kind=='plane':r['series'].reverse()
        if kind=='file':r['series'][0]['selected_files'][0]='../bad'
        with pytest.raises(ValueError):validate(bad)
    assert SPEC['seeds']==[2026,42] and SPEC['epochs']==20
    assert not SPEC['auc_measured'] and not SPEC['submission_eligible']


def runtime():
    namespace={'torch':torch,'nn':nn,'np':np,'Path':Path,'os':os,
               'sha':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    pilot=ast.parse(Path('scripts/v02_pilot_runtime.py').read_text())
    main=ast.parse(Path('scripts/v02_baseline_runtime.py').read_text())
    nodes=[n for n in pilot.body if isinstance(n,ast.ClassDef) and n.name=='StudyAttention']
    nodes += [n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in {'metrics','atomic_save','train_seed'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<baseline-unit>','exec'),namespace)
    return namespace


def test_soft_bce_and_invalid_predictions():
    score=runtime()['metrics'];x=torch.tensor([[0.]*12,[0.]*12]);y=torch.tensor([[0.]*12,[1.]*12])
    assert abs(score(x,y)['mean_soft_bce']-np.log(2))<1e-6
    with pytest.raises(ValueError):score(x*float('nan'),y)


def test_cpu_resume_reproduces_full_training():
    ns=runtime();spec=dict(SPEC,epochs=3);ns['V02_BASELINE']={'spec':spec,'splits':{'development':[{'StudyInstanceUID':str(i)} for i in range(5)]}}
    generator=torch.Generator().manual_seed(23)
    x=torch.randn((9,3,384),generator=generator);y=torch.rand((9,12),generator=generator)
    dev=torch.randn((5,3,384),generator=generator);dev_y=torch.rand((5,12),generator=generator)
    with TemporaryDirectory() as tmp,patch.object(nn.Module,'cuda',lambda self:self):
        full=Path(tmp)/'full';partial=Path(tmp)/'partial';resumed=Path(tmp)/'resumed'
        for p in [full,partial,resumed]:p.mkdir()
        reference=ns['train_seed'](x,y,dev,dev_y,2026,'fixed',full,lambda:None)
        calls=[]
        def interrupt():
            calls.append(1)
            if len(calls)>1:raise TimeoutError('synthetic interruption')
        with pytest.raises(TimeoutError):ns['train_seed'](x,y,dev,dev_y,2026,'fixed',partial,interrupt)
        checkpoint=partial/'v02_seed2026_last.pt'
        result=ns['train_seed'](x,y,dev,dev_y,2026,'fixed',resumed,lambda:None,checkpoint)
        assert result['history']==reference['history'] and result['best_epoch']==reference['best_epoch']
        with np.load(full/'v02_seed2026_development.npz') as a,np.load(resumed/'v02_seed2026_development.npz') as b:
            np.testing.assert_array_equal(a['best_logits'],b['best_logits'])
            np.testing.assert_array_equal(a['last_logits'],b['last_logits'])
        with pytest.raises(ValueError,match='contract'):
            ns['train_seed'](x,y,dev,dev_y,2026,'wrong',resumed,lambda:None,checkpoint)


def test_runtime_has_no_submission_and_confirms_frozen_encoder():
    source=Path('scripts/v02_baseline_runtime.py').read_text();ast.parse(source)
    assert 'submission.csv' not in source and 'requires_grad_(False)' in source
    assert 'rebuild_series' in source and 'weights_only=True' in source


def test_independent_numpy_head_and_loss_replay():
    head=runtime()['StudyAttention']();x=torch.randn(7,3,384)
    with torch.no_grad():expected=head(x,torch.ones((7,3),dtype=torch.bool)).numpy()
    np.testing.assert_allclose(replay(x.numpy(),head.state_dict()),expected,atol=1e-5,rtol=1e-5)
    y=np.full_like(expected,.5)
    np.testing.assert_allclose(loss(expected,y),np.logaddexp(0.,expected.astype(float))-.5*expected,atol=1e-7)
    with pytest.raises(ValueError):replay(x.numpy(),{})
    with pytest.raises(ValueError):loss(expected*np.nan,y)
