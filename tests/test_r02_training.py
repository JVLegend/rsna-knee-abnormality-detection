"""#RSNA #Kaggle #Pesquisa — synthetic loss, RNG, resume and promotion gates."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from scripts.prepare_r02_training import assemble,modify_training,BASE,ARMS
from scripts.assess_r02_training import decide,mask_history


def runtime():
    tree=ast.parse(Path('scripts/r02_training_runtime.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    scope={'torch':torch,'nn':nn,'np':np,'hashlib':hashlib,'json':json,'os':os,'Path':Path}
    exec(compile(tree,'<local-r02-functions>','exec'),scope)
    return scope


@pytest.mark.parametrize('probability',[0.,.25,1.])
def test_masks_keep_two_planes_and_do_not_advance_global_rng(probability):
    scope=runtime();torch.manual_seed(9);before=torch.get_rng_state().clone()
    a=scope['sample_presence'](10000,probability,torch.Generator().manual_seed(6))
    b=scope['sample_presence'](10000,probability,torch.Generator().manual_seed(6))
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(a,b)
    assert a.dtype==torch.bool and a.sum(1).min()>=2
    if probability in [0.,1.]: assert torch.all(a.sum(1)==3-int(probability))
    else: assert .22 < float((a.sum(1)==2).float().mean()) < .28
    if probability: assert np.max((~a).sum(0).numpy())/np.min((~a).sum(0).numpy())<1.2


@pytest.mark.parametrize('arm',ARMS[1:])
def test_mask_trace_matches_independent_replay(arm):
    scope=runtime();rng=torch.Generator().manual_seed(10042)
    trace={'seen':0,'dropped':[0,0,0],'chain':'0'*64};observed=[]
    for epoch in range(3):
        for size in [4,4,3]:
            mask=scope['sample_presence'](size,.25 if arm=='dropout25' else 1.,rng)
            scope['trace_masks'](trace,mask)
        observed.append(copy.deepcopy(trace))
    expected,end=mask_history(42,arm,epochs=3,size=11,batch=4)
    assert observed==expected and torch.equal(rng.get_state(),end)


def test_paired_loss_formula_and_detached_clean_target():
    scope=runtime();clean=torch.tensor([[.2,-.8]],requires_grad=True)
    masked=torch.tensor([[.9,-.1]],requires_grad=True);labels=torch.tensor([[.5,1.]])
    result=scope['paired_loss'](clean,masked,labels);result.backward()
    expected=.5*nn.functional.binary_cross_entropy_with_logits(clean,labels)+.5*nn.functional.binary_cross_entropy_with_logits(masked,labels)+.1*((masked.sigmoid()-clean.detach().sigmoid())**2).mean()
    assert result.item()==pytest.approx(expected.item())
    torch.testing.assert_close(clean.grad,.5*(clean.detach().sigmoid()-labels)/labels.numel())
    assert torch.isfinite(masked.grad).all()


def test_decision_requires_every_gate_and_fixed_tie_rule():
    rows=[{'arm':a,'seed':s,'mean_soft_bce':.6,'mean_missing_soft_bce':.65} for a in ARMS for s in [2026,42]]
    assert decide(rows)=='control'
    for r in rows[2:4]:r['mean_missing_soft_bce']=.60
    assert decide(rows)=='control'  # Robustness alone is insufficient.
    rows[2]['mean_soft_bce']=.59
    assert decide(rows)=='control'  # Both seeds required.
    rows[3]['mean_soft_bce']=.6-1e-6
    assert decide(rows)=='control'
    rows[3]['mean_soft_bce']=.59
    assert decide(rows)=='dropout25'
    for r in rows[4:]:r.update(mean_soft_bce=.59,mean_missing_soft_bce=.60)
    assert decide(rows)=='dropout25'
    rows[5]['mean_soft_bce']=.58
    assert decide(rows)=='paired_consistency'
    with pytest.raises(ValueError):decide(rows[:-1])
    rows[0]['mean_soft_bce']=float('nan')
    with pytest.raises(ValueError):decide(rows)


def test_build_preserves_control_and_frozen_scope():
    source=assemble();tree=ast.parse(source)
    def fn(text,name):return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    original=fn(BASE.read_text(),'train_seed')
    assert fn(source,'train_seed')==original
    assert fn(source,'train_candidate')==modify_training(original)
    assert len(source.encode())<1_000_000 and 'submission.csv' not in source


@pytest.mark.parametrize('arm',ARMS[1:])
def test_candidate_resume_exact_on_small_synthetic_cpu_batch(tmp_path,arm):
    scope=runtime();original=BASE.read_text();tree=ast.parse(original)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'sha','StudyAttention','metrics','atomic_save','train_seed'}]
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<local-baseline>','exec'),scope)
    source=next(ast.get_source_segment(original,n) for n in selected if n.name=='train_seed')
    exec(compile(modify_training(source),'<candidate>','exec'),scope)
    scope['V02_BASELINE']={'spec':{'epochs':2,'batch_size':4,'learning_rate':.001,'weight_decay':.0001},
                         'splits':{'development':[{'StudyInstanceUID':str(i)} for i in range(3)]}}
    torch.set_num_threads(1);g=torch.Generator().manual_seed(91)
    x=torch.randn((6,3,384),generator=g);y=torch.rand((6,12),generator=g)
    dev=x[:3].clone();dy=y[:3].clone();full=tmp_path/'full';partial=tmp_path/'partial'
    full.mkdir();partial.mkdir();calls=[]
    def interrupt():
        calls.append(1)
        if len(calls)==2:raise TimeoutError('synthetic interruption')
    with patch.object(nn.Module,'cuda',lambda self:self):
        one=scope['train_candidate'](x,y,dev,dy,42,'synthetic',full,lambda:None,arm=arm)
        with pytest.raises(TimeoutError):scope['train_candidate'](x,y,dev,dy,42,'synthetic',partial,interrupt,arm=arm)
        resumed=scope['train_candidate'](x,y,dev,dy,42,'synthetic',partial,lambda:None,arm=arm,resume=partial/'v02_seed42_last.pt')
    assert one['history']==resumed['history'] and one['mask_trace']==resumed['mask_trace']
    a=torch.load(full/'v02_seed42_last.pt',weights_only=True)
    b=torch.load(partial/'v02_seed42_last.pt',weights_only=True)
    assert torch.equal(a['mask_rng'],b['mask_rng'])
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
