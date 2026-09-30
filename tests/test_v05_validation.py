"""#RSNA #Kaggle #Pesquisa — new split/exposure and OOF invariants."""
import copy
import pytest
from scripts.prepare_v05_validation import allocate, build, coverage, eligible_pool, oof_folds, TARGETS
from tests.test_v03_scale import fixture


def rows(n=30):
    return [{'StudyInstanceUID': str(i), 'report_hash': str(i // 2), 'labels': [.5] * 12} for i in range(n)]


def test_exclude_whole_exposed_and_gold_groups():
    train, teacher, series, splits = fixture()
    exposed = sum(splits.values(), [])
    pool, excluded = eligible_pool(train, teacher, series, exposed)
    assert {r['StudyInstanceUID'] for r in pool} == {'4', '10', '11'}
    assert excluded['known_exposed_id_or_group'] == 7
    assert excluded['gold_id_or_group'] == 2
    assert all(r['labels'] == [.5] * 12 for r in pool)
    series.reverse()
    assert eligible_pool(train, teacher, series, exposed)[0] == pool


@pytest.mark.parametrize('fault', ['duplicate_series', 'exposure_drift', 'duplicate_exposure'])
def test_metadata_drift_rejected(fault):
    train, teacher, series, splits = fixture()
    exposed = sum(splits.values(), [])
    if fault == 'duplicate_series': series.append(copy.deepcopy(series[0]))
    if fault == 'exposure_drift': exposed[0]['report_hash'] = 'changed'
    if fault == 'duplicate_exposure': exposed.append(copy.deepcopy(exposed[0]))
    with pytest.raises(ValueError): eligible_pool(train, teacher, series, exposed)


def test_invalid_teacher_never_enters_pool():
    train, teacher, series, splits = fixture()
    teacher['4'][TARGETS[0]] = 'nan'
    pool, excluded = eligible_pool(train, teacher, series, sum(splits.values(), []))
    assert '4' not in {r['StudyInstanceUID'] for r in pool}
    assert excluded['invalid_labels_or_series'] == 1


def test_allocation_whole_group_deterministic_and_no_seed_search():
    r = rows()
    a = allocate(r, confirmation=5, development=5)
    assert a == allocate(list(reversed(r)), confirmation=5, development=5)
    assert [len(a[k]) for k in ['confirmation', 'development', 'unused']] == [6, 6, 18]
    groups = [{x['report_hash'] for x in s} for s in a.values()]
    assert all(not x & y for i, x in enumerate(groups) for y in groups[i+1:])
    with pytest.raises(ValueError): allocate(r, confirmation=30, development=1)
    with pytest.raises(ValueError): allocate(r + [r[0]], 5, 5)


def test_oof_has_no_group_overlap_and_complete_coverage():
    r = rows(50)
    result = oof_folds(r)
    assert result['study_counts'] == [10] * 5
    assignments = result['assignments']
    assert len({x['StudyInstanceUID'] for x in assignments}) == 50
    assert all(len({x['fold'] for x in assignments if x['report_hash'] == g}) == 1 for g in {x['report_hash'] for x in r})
    reverse = oof_folds(list(reversed(r)))
    assert {x['StudyInstanceUID']: x['fold'] for x in assignments} == {x['StudyInstanceUID']: x['fold'] for x in reverse['assignments']}
    assert result['executed'] is False
    with pytest.raises(ValueError): oof_folds(rows(4))


def test_counts_preserve_uncertain():
    r = rows(3)
    r[0]['labels'] = [.2] * 12
    r[1]['labels'] = [.8] * 12
    assert coverage(r)[TARGETS[0]] == {'positive': 1, 'negative': 1, 'uncertain': 1}


def test_real_frozen_population_and_source_hashes():
    data = build()
    splits = data['splits']
    assert len(splits['train']) == 1000
    assert len(splits['development']) >= 300 and len(splits['confirmation']) >= 300
    exposed = {r['report_hash'] for r in data['known_exposure']}
    for k in ['development', 'confirmation', 'unused']:
        assert not {r['report_hash'] for r in splits[k]} & exposed
    fold_ids = {r['StudyInstanceUID'] for r in data['oof']['assignments']}
    assert fold_ids == {r['StudyInstanceUID'] for r in splits['train'] + splits['development']}
    assert not fold_ids & {r['StudyInstanceUID'] for r in splits['confirmation']}
    assert data['class_gate_passed'] and not data['pixel_integrity_checked']
    assert data['confirmation_model_evaluations'] == data['development_model_evaluations'] == 0


@pytest.mark.parametrize('fault', ['none', 'group', 'label', 'fold', 'claim'])
def test_independent_metadata_audit_rejects_drift(fault):
    from pathlib import Path
    from scripts.assess_v05_validation import check_partitions
    from scripts.freeze_weak_validation import rows_by_id
    data=build(); training=copy.deepcopy(data['splits']['train'])
    raw=rows_by_id(Path('data/raw/train.csv')); teacher=rows_by_id(Path('data/external_labels/targetwise_teacher.csv'))
    if fault=='group': data['splits']['development'][0]['report_hash']=training[0]['report_hash']
    if fault=='label': data['splits']['confirmation'][0]['labels'][0]=float('nan')
    if fault=='fold': data['oof']['assignments'][0]['fold']=5
    if fault=='claim': data['oof']['executed']=True
    if fault=='none': assert check_partitions(data,training,raw,teacher)['oof_counts']==[260]*5
    else:
        with pytest.raises(ValueError): check_partitions(data,training,raw,teacher)
