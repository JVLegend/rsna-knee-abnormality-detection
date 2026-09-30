"""#RSNA #Kaggle #Pesquisa — synthetic tests, never open reserved predictions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.resolution_comparison import (RECIPE, V05_SHA, contract_hash, freeze,
                                          identity_gate, paired_metrics)
from scripts.run_resolution_heads import resource_gate, read_features, read_protocol


def population():
    rows = {s: [{'StudyInstanceUID': s+str(i), 'report_hash': contract_hash({'r': s+str(i)}),
                 'series': [{'series_uid': s+str(i)+str(j)} for j in range(3)]}
                for i in range(2)] for s in ['train', 'development', 'confirmation']}
    evidence = {'v05_sha256': V05_SHA, 'stable_patient_key_verified': True,
                'stable_patient_key_source': 'synthetic fixture not real evidence',
                'studies': [{'StudyInstanceUID': r['StudyInstanceUID'],
                             'patient_hash': contract_hash({'p': r['StudyInstanceUID']}),
                             'series_uids': [s['series_uid'] for s in r['series']],
                             'headers_verified': 3, 'all_headers_consistent': True}
                            for split in rows.values() for r in split]}
    return rows, evidence


def test_patient_gate_requires_actual_provenance_and_complete_coverage():
    rows, evidence = population()
    assert len(identity_gate(rows, evidence)) == 6
    for field, value in [('stable_patient_key_verified', False), ('stable_patient_key_source', '')]:
        bad = copy.deepcopy(evidence); bad[field] = value
        with pytest.raises(ValueError): identity_gate(rows, bad)
    bad = copy.deepcopy(evidence); bad['studies'].pop()
    with pytest.raises(ValueError): identity_gate(rows, bad)
    bad = copy.deepcopy(evidence); bad['studies'][0]['all_headers_consistent'] = False
    with pytest.raises(ValueError): identity_gate(rows, bad)


@pytest.mark.parametrize('overlap', ['patient', 'report', 'study', 'chain'])
def test_no_cross_split_leakage_or_transitive_aliases(overlap):
    rows, evidence = population()
    if overlap in ('patient', 'chain'):
        evidence['studies'][0]['patient_hash'] = evidence['studies'][2]['patient_hash']
    if overlap == 'report':
        rows['development'][0]['report_hash'] = rows['train'][0]['report_hash']
    if overlap == 'study':
        rows['development'][0]['StudyInstanceUID'] = rows['train'][0]['StudyInstanceUID']
    if overlap == 'chain':
        rows['train'][1]['report_hash'] = rows['train'][0]['report_hash']
    with pytest.raises(ValueError): identity_gate(rows, evidence)


def test_within_split_linked_studies_share_bootstrap_cluster():
    rows, evidence = population()
    evidence['studies'][0]['patient_hash'] = evidence['studies'][1]['patient_hash']
    groups = identity_gate(rows, evidence)
    assert groups['train0'] == groups['train1']


def test_paired_metrics_report_condition_ci_regression_and_uncertain_coverage():
    n = 24
    y = np.tile(np.arange(n)[:, None] % 2, (1, 12)).astype(float)
    y[0, 0] = .5
    direction = 2*y-1
    x = np.stack([np.stack([direction*.5]*2), np.stack([direction*1.5]*2)])
    result = paired_metrics(x, y, list(map(str, range(n))), replicates=100)
    assert result['primary_delta'] < -.001
    assert result['quality_passed']
    assert len(result['per_condition']) == 12
    assert result['primary_interval']['confidence'] == .95
    assert result['per_condition']['ACL']['soft_or_uncertain_excluded_from_auc'] == 1
    assert result['per_condition']['ACL']['bce_delta_simultaneous_interval']['confidence'] > .99
    assert not result['submission_eligible']
    x[1, :, :, 0] = -direction[:, 0]*4
    regressed = paired_metrics(x, y, list(map(str, range(n))), replicates=100)
    assert 'ACL' in regressed['regressions'] and not regressed['quality_passed']


def test_invalid_and_sparse_results_cannot_promote():
    x = np.zeros((2, 2, 4, 12))
    y = np.zeros((4, 12))
    result = paired_metrics(x, y, ['a', 'b', 'c', 'd'], replicates=100)
    assert not result['quality_passed'] and result['per_condition']['ACL']['auc224'] is None
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError): paired_metrics(x, y, ['a', 'b', 'c', 'd'], replicates=100)


def test_resource_receipt_is_required_and_measured():
    with pytest.raises(ValueError): resource_gate({})
    v = {'total_session_seconds': 6000, 'maximum_peak_allocated_bytes': 1000000000,
         'forward336_over224': 2.22}
    assert all(resource_gate(v).values())
    v['total_session_seconds'] = 7201
    assert not resource_gate(v)['bounded_job']
    v['maximum_peak_allocated_bytes'] = np.nan
    with pytest.raises(ValueError): resource_gate(v)


def test_protocol_freeze_refuses_overwrite_and_drift(tmp_path):
    p = tmp_path/'protocol.json'
    v = {'recipe': RECIPE, 'pins': {}}
    v['contract_hash'] = contract_hash(v)
    freeze(p, v)
    assert read_protocol(p) == v
    with pytest.raises(FileExistsError): freeze(p, v)
    v['recipe'] = dict(RECIPE, epochs=1)
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError): read_protocol(p)


def test_unverified_cache_fails_before_array_access(tmp_path):
    receipt = tmp_path/'cache.json'; receipt.write_text('{}')
    with pytest.raises(ValueError): read_features(tmp_path/'absent.npz', receipt, {'contract_hash': 'x'}, [])


def test_source_has_no_dispatch_or_auto_submission_or_dev_epoch_selection():
    source = Path('scripts/run_resolution_heads.py').read_text()
    assert 'save_kernel' not in source and 'competition_submit' not in source
    assert 'confirmation_exposure.json' in source
    assert RECIPE['epoch_selection'] == 'fixed_epoch20_no_dev_early_stopping'
    assert RECIPE['seeds'] == [2026, 42] and RECIPE['resolutions'] == [224, 336]
