"""#RSNA #Kaggle #Pesquisa — synthetic exhaustive evidence never proves stability."""
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.assess_g05_headers import assess
from scripts.prepare_g05_headers import assemble
from scripts.resolution_comparison import digest, load_manifest, V05_SHA


@pytest.fixture
def fixture(tmp_path):
    build = tmp_path/'build.py'; build.write_text(assemble())
    spec = ast.literal_eval(ast.parse(build.read_text()).body[0].value)
    official = defaultdict(set)
    with Path('data/raw/train_series.csv').open(newline='') as h:
        for r in csv.DictReader(h): official[r['StudyInstanceUID']].add(r['SeriesInstanceUID'])
    records = []
    for row in spec['rows']:
        records.append({'StudyInstanceUID': row['StudyInstanceUID'], 'split': row['split'],
            'series_uids': [s['series_uid'] for s in row['series']], 'all_series_uids': sorted(official[row['StudyInstanceUID']]),
            'series_records': [{'series_uid': uid, 'headers': 2, 'patient_keys': 1, 'header_transcript_sha256': 'b'*64} for uid in sorted(official[row['StudyInstanceUID']])],
            'patient_hash': hashlib.sha256(row['StudyInstanceUID'].encode()).hexdigest(), 'patient_keys_seen': 1,
            'all_headers_consistent': True, 'all_official_series_checked': True, 'issues': [],
            'headers_verified': len(official[row['StudyInstanceUID']])*2, 'header_transcript_sha256': 'a'*64})
    evidence = {'status': 'COMPLETE_FULL_HEADERS_NOT_PATIENT_SEMANTICS_CERTIFICATION',
        'studies': records, 'source_contract_hash': spec['contract_hash'], 'v05_sha256': V05_SHA,
        'headers_verified': sum(r['headers_verified'] for r in records),
        'series_verified': sum(len(r['series_records']) for r in records),
        'studies_with_complete_header_audit': 1600, 'cross_study_sop_collisions': [],
        'deidentification_methods': [], 'deidentification_codes': [], 'patient_identity_removed_values': [],
        'gpu_used': False, 'confirmation_pixels_read': 0, 'confirmation_labels_scored': False,
        'stable_patient_key_verified': False, 'stable_patient_key_source': None, 'seconds': 1.}
    receipt = {'source_contract_hash': spec['contract_hash'], 'gpu_used': False, 'confirmation_evaluated': False,
        'headers_verified': evidence['headers_verified'], 'studies_complete': 1600}
    return tmp_path, build, evidence, receipt


def save(f):
    directory, build, evidence, receipt = f
    (directory/'g05_headers.json').write_text(json.dumps(evidence))
    receipt['headers_sha256'] = digest(directory/'g05_headers.json')
    (directory/'g05_headers_receipt.json').write_text(json.dumps(receipt))
    return directory, build


def test_no_collisions_is_not_patient_independence(fixture):
    result = assess(*save(fixture))
    assert result['full_header_technical_audit_passed'] and result['headers'] > 0
    assert result['cross_split_patient_keys'] == 0
    assert not result['patient_gate_passed'] and not result['stable_patient_key_verified']


@pytest.mark.parametrize('fault', ['missing', 'order', 'inventory', 'patient', 'scope', 'count'])
def test_independent_header_assessment_rejects_inconsistent_receipts(fixture, fault):
    _, _, e, _ = fixture
    if fault == 'missing': e['studies'].pop()
    if fault == 'order': e['studies'][0], e['studies'][1] = e['studies'][1], e['studies'][0]
    if fault == 'inventory': e['studies'][0]['all_series_uids'].pop()
    if fault == 'patient': e['studies'][0]['patient_hash'] = 'x'*64
    if fault == 'scope': e['stable_patient_key_verified'] = True
    if fault == 'count': e['headers_verified'] += 1
    with pytest.raises(ValueError): assess(*save(fixture))


def test_cross_split_patient_keys_block_without_resplitting(fixture):
    _, _, e, _ = fixture
    e['studies'][1000]['patient_hash'] = e['studies'][0]['patient_hash']
    result = assess(*save(fixture))
    assert result['full_header_technical_audit_passed']
    assert result['cross_split_patient_keys'] == 1 and not result['patient_gate_passed']
    assert result['split_counts'] == {'train': 1000, 'development': 300, 'confirmation': 300}
