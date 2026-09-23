"""#RSNA #Kaggle #Testes — source pins and OOF diagnostics."""
import ast
import copy
import json
from pathlib import Path
import pytest
from scripts.prepare_h46_preflight import SOURCE, inspect_source, assemble
from scripts.audit_av030_oof import fold_summary
from scripts.h46_runtime_gate import validate_h46_release, H46_MEMBERS
from scripts.prepare_h46_candidate import build


def test_oof_report_groups_cannot_hide_behind_unique_ids():
    r=fold_summary(['a','b','c'],[0,1,2],{'a':'same','b':'same','c':'other'})
    assert r['cross_fold_report_groups']==1 and r['studies_in_crossing_groups']==2
    r=fold_summary(['a','b','c'],[0,1,2],{'a':'same','b':'same','c':'other'},['a'])
    assert r['cross_fold_report_groups']==0 and r['studies']==2
    with pytest.raises(ValueError):fold_summary(['a','a'],[0,1],{'a':'same'})
    with pytest.raises(ValueError):fold_summary(['a'],[-1],{'a':'same'})


def test_pinned_public_source_only_parsed_not_executed():
    if not SOURCE.is_file(): pytest.skip('Pinned public notebook is a private local audit fixture')
    raw=SOURCE.read_bytes();r=inspect_source(raw)
    assert r['recipe']['coatnet_w']['__default__']==.65 and not r['submission_eligible']
    with pytest.raises(ValueError):inspect_source(raw+b' ')
    source,spec=assemble(raw);ast.parse(source)
    assert 'h46_preflight()' in source and 'h43_preflight(gpu=False)' in source
    assert 'torch.load(' not in source and 'exec(' not in source and 'pip install' not in source
    assert 'rank_of_member_probability_mean' in spec['required_reduction']


def release_args():
    return [{'status':'PASSED_H46_ASSET_AUDIT_NOT_INFERENCE'},[],
            {'members':sorted(H46_MEMBERS),'within_coat':dict.fromkeys(H46_MEMBERS,1/3),
             'family_reduction':'rank_of_member_probability_mean','public_raptor_alpha':.6},
            {'dino':20,'a5':5,'raptor_views':4},{'fixed':True},{'fixed':True}]


@pytest.mark.parametrize('fault',['member','duplicate','rankfallback','nan','neutral','dropped','count','recipe','weights'])
def test_runtime_gate_rejects_degraded_composition(fault):
    args=release_args();assert not validate_h46_release(*args)['score_reproduced']
    if fault=='member':args[2]['members'].pop()
    if fault=='duplicate':args[2]['members'][0]=args[2]['members'][1]
    if fault=='rankfallback':args[2]['family_reduction']='rank_sum_fallback_probabilities_unavailable'
    if fault in ['nan','neutral','dropped']:
        args[1]=[{'kind':{'nan':'rad_nonfinite_zeroed','neutral':'a5_nonfinite_neutral','dropped':'dino_dropped'}[fault]}]
    if fault=='count':args[3]['dino']=19
    if fault=='recipe':args[4]['fixed']=False
    if fault=='weights':args[2]['within_coat']['d4_swa3']=.5
    with pytest.raises(ValueError):validate_h46_release(*args)


def test_runtime_gate_allows_finite_retry_and_zero_fallback():
    args=release_args();args[1]=[{'kind':'scratch_fallback'},{'kind':'rad_fp16_nonfinite_retry_fp32'},
                                {'kind':'coat_global96_fallback_studies','count':0}]
    validate_h46_release(*args)
    args[1][-1]['count']=1
    with pytest.raises(ValueError):validate_h46_release(*args)


def test_candidate_is_frozen_and_gate_precedes_publication():
    fixture=Path('reports/avance_av030_h46_v1/h46_preflight.json')
    if not SOURCE.is_file() or not fixture.is_file(): pytest.skip('CPU audit fixtures not present in this checkout')
    receipt=json.loads(fixture.read_text())
    nb=build(SOURCE.read_bytes(),receipt)
    source=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'H46_RELEASE = validate_h46_release(' in c['source'])
    assert source.index('H46_RELEASE = validate_h46_release(') < source.index('os.replace(_tmp,_final)')
    assert 'global96_epochs' in source and "PRESET = \"speedy\"" in nb['cells'][3]['source']
    assert nb['metadata']['h46_build']['asset_lock_entries']>=100
    assert not nb['metadata']['h46_build']['gpu_smoke_executed']
    with pytest.raises(ValueError):build(SOURCE.read_bytes(),{})
