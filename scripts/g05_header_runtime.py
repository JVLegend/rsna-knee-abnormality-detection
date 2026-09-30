"""#RSNA #Kaggle #Dados — exhaustive header-only audit; never certifies ID semantics.

Builder supplies HEADER_SPEC. All official series of each frozen study are read,
not only the three selected planes. No pixel, report, label or PatientName read.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import pydicom

HEADER_TAGS = ['PatientID', 'IssuerOfPatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
               'SOPInstanceUID', 'StudyDate', 'PatientIdentityRemoved',
               'DeidentificationMethod', 'DeidentificationMethodCodeSequence']


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def hashed(value, domain):
    return hashlib.sha256((domain+':'+str(value)).encode()).hexdigest()


def find_competition(root):
    found = []
    for count, (folder, dirs, files) in enumerate(os.walk(root), 1):
        if count > 256:
            raise ValueError('Input discovery budget')
        if 'train_series.csv' in files and 'train_series' in dirs:
            found.append(Path(folder))
        dirs[:] = sorted(d for d in dirs if d not in ('train_series', 'test_series', 'kaggle_test_series'))
    if len(found) != 1:
        raise ValueError('Ambiguous official competition attachment')
    return found[0]


def scan_study(root, row, metadata, guard):
    uid = row['StudyInstanceUID']
    selected = {s['series_uid'] for s in row['series']}
    expected = {s['SeriesInstanceUID'] for s in metadata if s['StudyInstanceUID'] == uid}
    actual = {e.name for e in os.scandir(root/uid) if e.is_dir()} if (root/uid).is_dir() else set()
    issues = []
    if not expected or actual != expected or not selected <= expected:
        issues.append('series_inventory_mismatch')
    patients, issuers, dates, methods, codes, removed = set(), set(), set(), set(), set(), set()
    series_records = []
    headers = 0
    sop_keys = set()
    transcript = hashlib.sha256()
    for series_uid in sorted(expected):
        guard()
        folder = root/uid/series_uid
        files = sorted(Path(e.path) for e in os.scandir(folder) if e.name.endswith('.dcm')) if folder.is_dir() else []
        if not files:
            issues.append('empty_or_missing_series')
        series_digest = hashlib.sha256()
        series_patients = set()
        for path in files:
            guard()
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS, force=False)
            patient = str(getattr(ds, 'PatientID', '')).strip()
            sop = str(getattr(ds, 'SOPInstanceUID', '')).strip()
            issuer = str(getattr(ds, 'IssuerOfPatientID', '')).strip()
            if (str(getattr(ds, 'StudyInstanceUID', '')) != uid or
                    str(getattr(ds, 'SeriesInstanceUID', '')) != series_uid or not patient or not sop):
                issues.append('header_identity_missing_or_drift')
            if patient.upper() in ('', 'ANON', 'ANONYMOUS', 'UNKNOWN', 'NONE', 'NULL'):
                issues.append('placeholder_patient_key')
            pkey = hashed(patient, 'RSNA-G05-patient') if patient else ''
            patients.add(pkey); series_patients.add(pkey)
            issuers.add(hashed(issuer, 'issuer') if issuer else '')
            date = str(getattr(ds, 'StudyDate', ''))
            if date:
                dates.add(hashed(date, 'study-date'))
            method = str(getattr(ds, 'DeidentificationMethod', ''))
            if method:
                methods.add(method)  # deidentification protocol text, never patient names
            for item in getattr(ds, 'DeidentificationMethodCodeSequence', []):
                codes.add(str(getattr(item, 'CodingSchemeDesignator', ''))+':'+str(getattr(item, 'CodeValue', '')))
            removed.add(str(getattr(ds, 'PatientIdentityRemoved', '')))
            skey = hashed(sop, 'sop') if sop else ''
            if skey in sop_keys:
                issues.append('repeated_sop_within_study')
            sop_keys.add(skey)
            material = {'file': path.name, 'study': uid, 'series': series_uid, 'patient_hash': pkey,
                        'issuer_hash': hashed(issuer, 'issuer') if issuer else '', 'sop_hash': skey,
                        'date_hash': hashed(date, 'study-date') if date else '',
                        'method_hash': hashed(method, 'method') if method else ''}
            encoded = json.dumps(material, sort_keys=True).encode()+b'\n'
            transcript.update(encoded); series_digest.update(encoded)
            headers += 1
        series_records.append({'series_uid': series_uid, 'selected': series_uid in selected,
                               'headers': len(files), 'patient_keys': len(series_patients),
                               'header_transcript_sha256': series_digest.hexdigest()})
    if len(patients) != 1 or '' in patients:
        issues.append('patient_inconsistent_within_study')
    complete = not issues and headers > 0
    record = {'StudyInstanceUID': uid, 'split': row['split'],
              'patient_hash': next(iter(patients)) if len(patients) == 1 else '',
              'patient_keys_seen': len(patients), 'series_uids': sorted(selected & actual),
              'all_series_uids': sorted(expected), 'series_records': series_records,
              'headers_verified': headers, 'all_headers_consistent': complete,
              'all_official_series_checked': complete, 'issues': sorted(set(issues)),
              'issuer_hashes': sorted(issuers), 'study_date_hashes': sorted(dates),
              'header_transcript_sha256': transcript.hexdigest()}
    return record, sop_keys, methods, codes, removed


def main():
    started = time.monotonic()
    output = Path('/kaggle/working')
    root = find_competition(Path('/kaggle/input'))
    if digest(root/'train_series.csv') != HEADER_SPEC['series_metadata_sha256']:
        raise ValueError('Official series metadata changed')
    with (root/'train_series.csv').open(newline='') as handle:
        metadata = list(csv.DictReader(handle))
    def guard():
        if time.monotonic()-started > HEADER_SPEC['guard_seconds']:
            raise TimeoutError('Header-only guard; reconcile partial checkpoint')
    records, seen_sops, cross_sop, methods, codes, removed = [], {}, [], set(), set(), set()
    try:
        for row in HEADER_SPEC['rows']:
            record, sops, mm, cc, rr = scan_study(root/'train_series', row, metadata, guard)
            records.append(record); methods.update(mm); codes.update(cc); removed.update(rr)
            for sop in sops:
                if sop in seen_sops and seen_sops[sop] != row['StudyInstanceUID']:
                    cross_sop.append({'first': seen_sops[sop], 'second': row['StudyInstanceUID'], 'sop_hash': sop})
                seen_sops[sop] = row['StudyInstanceUID']
            if len(records) % 25 == 0:
                (output/'g05_headers_progress.json').write_text(json.dumps({'studies': records,
                    'completed': len(records), 'source_contract_hash': HEADER_SPEC['contract_hash']}, sort_keys=True))
                print('FULL_HEADERS', len(records), '/', len(HEADER_SPEC['rows']), flush=True)
        by_patient = {}
        for record in records:
            if record['patient_hash']:
                by_patient.setdefault(record['patient_hash'], []).append(record)
        repeated = [{'patient_hash': p, 'studies': [r['StudyInstanceUID'] for r in group],
                     'splits': sorted({r['split'] for r in group}),
                     'distinct_date_hashes': len({d for r in group for d in r['study_date_hashes']})}
                    for p, group in by_patient.items() if len(group) > 1]
        receipt = {'status': 'COMPLETE_FULL_HEADERS_NOT_PATIENT_SEMANTICS_CERTIFICATION',
                   'v05_sha256': HEADER_SPEC['v05_sha256'], 'source_contract_hash': HEADER_SPEC['contract_hash'],
                   'studies': records, 'studies_with_complete_header_audit': sum(r['all_headers_consistent'] for r in records),
                   'headers_verified': sum(r['headers_verified'] for r in records),
                   'series_verified': sum(len(r['series_records']) for r in records),
                   'repeated_patient_keys': repeated, 'unique_patient_keys': len(by_patient),
                   'cross_split_patient_hashes': sum(len(r['splits']) > 1 for r in repeated),
                   'cross_study_sop_collisions': cross_sop, 'deidentification_methods': sorted(methods),
                   'deidentification_codes': sorted(codes), 'patient_identity_removed_values': sorted(removed),
                   'stable_patient_key_verified': False, 'stable_patient_key_source': None,
                   'gpu_used': False, 'confirmation_pixels_read': 0, 'confirmation_labels_scored': False,
                   'seconds': time.monotonic()-started, 'submission_eligible': False}
        path = output/'g05_headers.json'
        path.write_text(json.dumps(receipt, sort_keys=True))
        (output/'g05_headers_receipt.json').write_text(json.dumps({'status': receipt['status'],
            'headers_sha256': digest(path), 'source_contract_hash': HEADER_SPEC['contract_hash'],
            'headers_verified': receipt['headers_verified'], 'studies_complete': receipt['studies_with_complete_header_audit'],
            'seconds': receipt['seconds'], 'gpu_used': False, 'confirmation_evaluated': False}, indent=2))
        print('COMPLETE_HEADERS_NOT_IDENTITY_CERTIFICATION', receipt['headers_verified'], receipt['seconds'], flush=True)
    except Exception as exc:
        (output/'g05_headers_failure.json').write_text(json.dumps({'error_type': type(exc).__name__,
            'message': str(exc), 'completed_studies': len(records), 'source_contract_hash': HEADER_SPEC['contract_hash']}))
        raise


if __name__ == '__main__':
    main()
