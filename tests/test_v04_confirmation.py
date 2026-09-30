"""#RSNA #Kaggle #Pesquisa — reserved identity, one-shot gates and grouped uncertainty."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.prepare_v04_confirmation import assemble,reserved_rows,literal,PLANES,TARGETS,V03_BUILD
from scripts.assess_v04_confirmation import grouped_bootstrap,compare,weak_auc,check_geometry
from rsna_knee_baseline.physical_triplets import physical_plan


def metadata():
    row={'StudyInstanceUID':'reserved','report_hash':'reserved-group','labels':{t:.5 for t in TARGETS},
         'series':[{'plane':p,'series_uid':str(i)} for i,p in enumerate(PLANES)]}
    old={'splits':{'confirmation':[row]}}
    expanded={'train':[{'StudyInstanceUID':'train','report_hash':'train-group'}],'development':[],
              'reserved':[{'StudyInstanceUID':'reserved','report_hash':'reserved-group'}]}
    series=[{'StudyInstanceUID':'reserved','SeriesInstanceUID':str(i),'Anatomical_Plane':p,
             'Fluid_Sensitive':'1','Fat_Suppression':'1'} for i,p in enumerate(PLANES)]
    return old,expanded,series


def test_reserved_metadata_preserves_uncertainty_and_series():
    rows=reserved_rows(*metadata())
    assert rows[0]['labels']==[.5]*12 and [s['plane'] for s in rows[0]['series']]==PLANES


@pytest.mark.parametrize('fault',['overlap','group','identity','labels','missing','duplicate','policy'])
def test_reserved_drift_blocks(fault):
    old,expanded,series=metadata()
    if fault=='overlap':expanded['train'][0]['StudyInstanceUID']='reserved'
    if fault=='group':expanded['train'][0]['report_hash']='reserved-group'
    if fault=='identity':expanded['reserved'][0]['report_hash']='different'
    if fault=='labels':old['splits']['confirmation'][0]['labels'][TARGETS[0]]=float('nan')
    if fault=='missing':series.pop()
    if fault=='duplicate':series.append(copy.deepcopy(series[0]))
    if fault=='policy':old['splits']['confirmation'][0]['series'][0]['series_uid']='different'
    with pytest.raises(ValueError):reserved_rows(old,expanded,series)


def test_group_bootstrap_weights_all_studies_not_group_means():
    delta=np.array([-2.,-2.,1.]);groups=['a','a','b']
    got=grouped_bootstrap(delta,groups,replicates=100,seed=4)
    draw=np.random.default_rng(4).integers(0,2,size=(100,2))
    expected=np.array([-4.,1.])[draw].sum(1)/np.array([2,1])[draw].sum(1)
    np.testing.assert_allclose(got['ci95'],np.quantile(expected,[.025,.975]))
    assert got['delta_mean']==-1 and got['groups']==2
    same=grouped_bootstrap(np.repeat(-.002,8),['a','a','b','b','c','c','d','d'])
    np.testing.assert_allclose(same['ci95'],[-.002,-.002],atol=1e-15)
    with pytest.raises(ValueError):grouped_bootstrap([1.,np.nan],['a','b'])
    with pytest.raises(ValueError):grouped_bootstrap([1.,2.],['a','a'])


def predictions():
    y=np.tile(np.arange(12)%2,(12,1)).astype(float)
    # Alternate rows too, ensuring both weak-reference classes in each target.
    y[1::2]=1-y[1::2]
    x=np.zeros((4,4,12,12));x[2:]=(2*y-1)[None,None]
    return x,y,[str(i//2) for i in range(12)]


def test_confirmation_requires_every_fixed_gate():
    x,y,g=predictions();r=compare(x,y,g,TARGETS)
    assert r['confirmation_decision']=='CONFIRMED' and all(r['gates'].values())
    same=x.copy();same[3,0]=0
    assert not compare(same,y,g,TARGETS)['gates']['intact_both_seeds']
    same=x.copy();same[3,1:]=0
    assert not compare(same,y,g,TARGETS)['gates']['missing_both_seeds']
    same=x.copy();same[2:,:,:,0]=-3*(2*y[:,0]-1)[None,None]
    r=compare(same,y,g,TARGETS)
    assert r['gates']['intact_both_seeds'] and not r['gates']['no_target_regression_over_0_01']
    assert r['confirmation_decision']=='NOT_CONFIRMED'
    x[0,0,0,0]=np.nan
    with pytest.raises(ValueError):compare(x,y,g,TARGETS)


def test_weak_auc_excludes_half_and_reports_undefined():
    y=np.array([[.5,.5],[.2,.5],[.8,.5],[.9,.5]])
    r=weak_auc(np.array([[999,0],[-1,0],[1,0],[2,0]]),y,['a','b'])
    assert r['per_target']['a']=={'positive':2,'negative':1,'uncertain':1,'auc':1.}
    assert r['per_target']['b']['auc'] is None and r['valid_targets']==1


def geometry_fixture():
    rows=reserved_rows(*metadata());headers=[(str(i),SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0],ImagePositionPatient=[0,0,i*3])) for i in range(5)]
    plan=physical_plan(headers,['0']*3);records=[]
    for s in rows[0]['series']:
        records.append({'study':'reserved','series':s['series_uid'],'plane':s['plane'],'pixel_sha256':'a'*64,
                        'selected':copy.deepcopy(plan['arms']['physical_adjacent']),'plan':copy.deepcopy(plan),
                        'headers':[{'name':name,'iop':h.ImageOrientationPatient,'ipp':h.ImagePositionPatient} for name,h in headers]})
    return records,rows


def test_geometry_reconstruction_and_collision_gate():
    records,rows=geometry_fixture();check_geometry(records,rows,set())
    with pytest.raises(ValueError,match='collision'):check_geometry(records,rows,{'a'*64})
    records[0]['selected']['indices']=[0,1,2]
    with pytest.raises(ValueError):check_geometry(records,rows,set())
    records,rows=geometry_fixture();records[0]['headers'][1]['ipp']=[0,0,0]
    with pytest.raises(ValueError):check_geometry(records,rows,set())


def test_pixel_gate_and_bounded_attachment_discovery(tmp_path):
    tree=ast.parse(Path('scripts/v04_confirmation_runtime.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    import os
    scope={'np':np,'hashlib':hashlib,'os':os,'Path':Path}
    exec(compile(tree,'<local-v04-functions>','exec'),scope)
    image=np.arange(3*224*224,dtype=np.int64).astype(np.uint8).reshape(3,224,224)
    sha=scope['check_pixels'](image,set())
    with pytest.raises(ValueError,match='duplicate'):scope['check_pixels'](image,{sha})
    with pytest.raises(ValueError):scope['check_pixels'](np.zeros_like(image),set())
    input_path=tmp_path/'input';input_path.mkdir();(input_path/'r02_receipt.json').touch()
    pruned=input_path/'train_series';pruned.mkdir();(pruned/'r02_receipt.json').touch()
    assert scope['find_receipt'](tmp_path)==input_path/'r02_receipt.json'
    other=tmp_path/'other';other.mkdir();(other/'r02_receipt.json').touch()
    with pytest.raises(ValueError):scope['find_receipt'](tmp_path)


def test_frozen_build_has_no_training_or_csv_and_reuses_image_contract():
    source=assemble();payload=literal(source,'V04')
    assert len(payload['rows'])==150 and len(payload['checkpoints'])==4
    assert payload['forbidden_series_count']==3750
    assert payload['bootstrap_replicates']==5000 and payload['bootstrap_seed']==20260921
    assert payload['margin']==2e-6 and payload['max_target_regression']==.01
    assert len(source.encode())<1_000_000 and 'submission.csv' not in source and 'AdamW' not in source
    def definition(text,name):return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for name in ['normalize_cache_compatible','physical_plan','StudyAttention','discover_inputs']:
        assert definition(source,name)==definition(V03_BUILD.read_text(),name)
