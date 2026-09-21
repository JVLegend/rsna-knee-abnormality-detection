"""#RSNA #Kaggle #Pesquisa — train-only pilot, parity, scope and timing checks."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from scripts.prepare_g04_preflight import assemble, literal, select_training, V05
from scripts.assess_g04_preflight import check_arrays, summarize_timings


def test_selection_training_only_no_labels():
    rows=[{'StudyInstanceUID':str(i),'series':[],'labels':[.5]*12} for i in range(25)]
    result=select_training(rows)
    assert len(result)==20 and result==select_training(list(reversed(rows)))
    assert all(set(r)=={'StudyInstanceUID','series'} for r in result)
    with pytest.raises(ValueError): select_training(rows[:5])


def test_arrays_enforce_parity_resolution_and_finitude():
    a=np.zeros((20,3,384),np.float32); b=np.ones_like(a)
    assert check_arrays(a,b,a)==0
    with pytest.raises(ValueError): check_arrays(a,a,a)
    with pytest.raises(ValueError): check_arrays(b,a,a)
    b[0,0,0]=np.nan
    with pytest.raises(ValueError): check_arrays(a,b,a)


def timings():
    return [{'begin':i*2,'studies':2,'order':[224,336] if i%2==0 else [336,224],
             'decode_and_both_resizes_seconds':2.,'inference_seconds':{'224':.2,'336':.6},
             'peak_allocated_bytes':{'224':1000000,'336':2000000}} for i in range(10)]


def test_timing_balanced_order_and_non_guaranteed_budget():
    t=timings(); result=summarize_timings(t)
    assert result['forward336_over224']==pytest.approx(3.)
    assert result['planning_1300_both_resolutions_seconds']==pytest.approx(3150.)
    assert 'Not a runtime guarantee' in result['planning_warning']
    t[1]['order']=[224,336]
    with pytest.raises(ValueError): summarize_timings(t)


@pytest.mark.parametrize('fault',['nan','memory','count'])
def test_invalid_timing_rejected(fault):
    t=timings()
    if fault=='nan': t[0]['inference_seconds']['336']=float('nan')
    if fault=='memory': t[0]['peak_allocated_bytes']['336']=15*1024**3
    if fault=='count': t.pop()
    with pytest.raises(ValueError): summarize_timings(t)


def test_bounded_archive_discovery(tmp_path):
    import os
    tree=ast.parse(Path('scripts/g04_preflight_runtime.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    scope={'os':os,'Path':Path,'np':np}; exec(compile(tree,'<g04-functions>','exec'),scope)
    (tmp_path/'v03_features.npz').touch()
    (tmp_path/'train_series').mkdir(); (tmp_path/'train_series'/'v03_features.npz').touch()
    assert scope['find_archive'](tmp_path)==tmp_path/'v03_features.npz'
    (tmp_path/'other').mkdir(); (tmp_path/'other'/'v03_features.npz').touch()
    with pytest.raises(ValueError): scope['find_archive'](tmp_path)


def test_source_has_no_held_out_ids_labels_or_training():
    import json
    source=assemble(); spec=literal(source,'G04'); data=json.loads(V05.read_text())
    train={r['StudyInstanceUID'] for r in data['splits']['train']}
    assert {r['StudyInstanceUID'] for r in spec['rows']}<=train
    assert all('labels' not in r for r in spec['rows'])
    assert spec['resolutions']==[224,336] and not spec['confirmation_evaluated']
    assert 'AdamW' not in source and 'submission.csv' not in source and 'StudyAttention' not in source
    assert all(r['StudyInstanceUID'] not in source for r in data['splits']['confirmation']+data['splits']['development'])
