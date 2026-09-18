"""#RSNA #Kaggle #Pesquisa — geometric contracts on synthetic DICOM headers."""
from types import SimpleNamespace
import numpy as np
import pytest
from rsna_knee_baseline.physical_triplets import physical_plan


def records(n=9):
    return [(str(i)+'.dcm',SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0],
             ImagePositionPatient=[0,0,i*3.])) for i in reversed(range(n))]


def test_order_center_and_spacing():
    p=physical_plan(records(),['2.dcm','4.dcm','6.dcm'])
    assert p['arms']['physical_quartiles']['indices']==[2,4,6]
    assert p['arms']['physical_adjacent']['indices']==[3,4,5]
    assert p['arms']['physical_quartiles']['gaps_mm']==[6,6]
    assert p['arms']['physical_adjacent']['gaps_mm']==[3,3]
    assert p['arms']['physical_adjacent']['files'][1]==p['arms']['physical_quartiles']['files'][1]


@pytest.mark.parametrize('n',[1,2,3,4,32,97])
def test_short_even_odd_series(n):
    p=physical_plan(records(n),['0.dcm']*3)
    for arm in p['arms'].values():
        assert len(arm['files'])==3 and all(0<=i<n for i in arm['indices'])
    assert p['arms']['physical_adjacent']['indices'][1]==p['center_index']


@pytest.mark.parametrize('problem',['missing','nan','orientation','duplicate','direction','files'])
def test_invalid_geometry_stops(problem):
    r=records()
    if problem=='missing':del r[0][1].ImagePositionPatient
    if problem=='nan':r[0][1].ImagePositionPatient[1]=float('nan')
    if problem=='orientation':r[0][1].ImageOrientationPatient=[0,1,0,1,0,0]
    if problem=='duplicate':r[0][1].ImagePositionPatient=r[1][1].ImagePositionPatient
    if problem=='direction':r[0][1].ImageOrientationPatient=[1,0,0,1,0,0]
    if problem=='files':r[0]=r[1]
    with pytest.raises(ValueError):physical_plan(r,['0.dcm']*3)


def test_oblique_projection_not_patient_z():
    r=records(3)
    for _,h in r:
        z=h.ImagePositionPatient[2]
        h.ImageOrientationPatient=[0,1,0,0,0,1]
        h.ImagePositionPatient=[z,0,0]
    p=physical_plan(r,['0.dcm']*3)
    np.testing.assert_allclose(p['normal'],[1,0,0])
    assert p['ordered_files']==['0.dcm','1.dcm','2.dcm']


def test_build_limit_and_frozen_counts():
    from scripts.prepare_g01_ablation import assemble, literal
    source=assemble();spec=literal(source,'G01')
    assert len(source.encode())<1_000_000
    assert len(spec['series_counts'])==1647
    assert all(isinstance(n,int) and n>0 for n in spec['series_counts'])
    assert spec['confirmation_evaluated'] is False
    assert spec['submission_eligible'] is False
    assert 'submission.csv' not in source


def test_decision_requires_both_seeds():
    from scripts.assess_g01_ablation import decide
    rows=[{'arm':a,'seed':s,'mean_soft_bce':.6} for a in
          ['control','physical_quartiles','physical_adjacent'] for s in [2026,42]]
    assert decide(rows)=='control'
    rows[4]['mean_soft_bce']=.5
    assert decide(rows)=='control'
    rows[5]['mean_soft_bce']=.55
    assert decide(rows)=='physical_adjacent'
    with pytest.raises(ValueError):decide(rows[:-1])
