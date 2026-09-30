"""#RSNA #Kaggle #Pesquisa — group exclusions, immutable teacher, controlled scale."""
import copy
import pytest
from scripts.prepare_v03_scale import select_pool,group_hash,TARGETS,PLANES


def fixture():
    train={str(i):{'Report':'report '+str(i),**{t:'' for t in TARGETS}} for i in range(12)}
    # Reserved duplicates and a gold duplicate must never enter training.
    train['6']['Report']=train['2']['Report'];train['7']['Report']=train['3']['Report']
    train['8'][TARGETS[0]]='1';train['9']['Report']=train['8']['Report']
    # Complete the existing training group when growing the sample.
    train['5']['Report']=train['0']['Report']
    teacher={uid:{t:'.5' for t in TARGETS} for uid in train}
    series=[{'StudyInstanceUID':uid,'SeriesInstanceUID':uid+'.'+str(j),
             'Anatomical_Plane':plane,'Fluid_Sensitive':'1','Fat_Suppression':'1'}
            for uid in train for j,plane in enumerate(PLANES)]
    def row(uid):return {'StudyInstanceUID':uid,'report_hash':group_hash(train[uid]['Report']),
                         'labels':{t:.5 for t in TARGETS}}
    splits={'train':[row('0'),row('1')],'development':[row('2')],'confirmation':[row('3')]}
    return train,teacher,series,splits


def test_scale_groups_reserved_and_teacher():
    args=fixture();result=select_pool(*args,target=5)
    ids=[r['StudyInstanceUID'] for r in result['train']]
    assert ids[:2]==['0','1'] and '5' in ids
    assert not set(ids)&{'2','3','6','7','8','9'}
    assert result==select_pool(*args,target=5)
    assert all(r['labels']==[.5]*12 for r in result['train'])
    assert all([s['plane'] for s in r['series']]==PLANES for r in result['train'])


def test_block_invalid_original_and_insufficient_pool():
    args=fixture();args[1]['0'][TARGETS[0]]='.9'
    with pytest.raises(ValueError,match='Teacher'):select_pool(*args,target=5)
    args=fixture();args[1]['0'][TARGETS[0]]='nan'
    with pytest.raises(ValueError,match='eligible'):select_pool(*args,target=5)
    with pytest.raises(ValueError,match='Insufficient'):select_pool(*fixture(),target=99)


def test_reserved_group_overlap_and_duplicate_series():
    args=fixture();args[3]['train'][0]['report_hash']=args[3]['development'][0]['report_hash']
    with pytest.raises(ValueError,match='overlap'):select_pool(*args,target=5)
    args=fixture();args[2].append(copy.deepcopy(args[2][0]))
    with pytest.raises(ValueError,match='Duplicate'):select_pool(*args,target=5)


def test_series_priority_not_row_order():
    args=fixture();new=copy.deepcopy(args[2][0]);new['SeriesInstanceUID']='0.000';new['Fluid_Sensitive']='0'
    args[2].insert(0,new)
    result=select_pool(*args,target=5)
    assert result['train'][0]['series'][0]['series_uid']=='0.0'


def test_complete_old_groups_before_new_groups():
    result=select_pool(*fixture(),target=3)
    assert {r['StudyInstanceUID'] for r in result['train']}=={'0','1','5'}


def test_frozen_build_uses_unchanged_training_loop():
    import ast
    from pathlib import Path
    from scripts.prepare_v03_scale import assemble,literal
    source=assemble(Path('data/processed/validation_weak_v3_scale1000/manifest.json'))
    assert source==Path('reports/avance_av024_v03/v03_v1.py').read_text()
    data=literal(source,'V03')['manifest']
    assert len(data['train'])==1000 and data['original_count']==299
    assert len(source.encode())<1_000_000 and 'submission.csv' not in source
    def train(text):return next(ast.get_source_segment(text,n) for n in ast.parse(text).body
                               if isinstance(n,ast.FunctionDef) and n.name=='train_seed')
    assert train(source)==train(Path('scripts/v02_baseline_runtime.py').read_text())
