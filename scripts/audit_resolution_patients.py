"""#RSNA #Kaggle #Dados — local bounded header inventory; never score labels/read pixels.

Partial header coverage does not certify patient separation. No raw PatientID
is persisted. A hash is pseudonymization, not proof of stable anonymization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pydicom

from scripts.resolution_comparison import freeze, identity_gate, load_manifest, V05_SHA


def audit(root, seconds=120, all_headers=False):
    manifest = load_manifest()
    started = time.monotonic()
    records = []
    issues = []
    tagged_same_as_study = 0
    for split in ('train', 'development', 'confirmation'):
        for row in manifest['splits'][split]:
            uid = row['StudyInstanceUID']
            patients = set()
            checked = 0
            series_uids = []
            complete = all_headers
            for series in row['series']:
                if time.monotonic()-started > seconds:
                    complete = False
                    issues.append({'study': uid, 'reason': 'header_budget_exhausted'})
                    break
                folder = root/uid/series['series_uid']
                if not folder.is_dir():
                    complete = False
                    issues.append({'study': uid, 'reason': 'missing_series'})
                    continue
                files = sorted(Path(e.path) for e in os.scandir(folder) if e.name.endswith('.dcm'))
                if not files:
                    complete = False
                    issues.append({'study': uid, 'reason': 'no_dicom'})
                    continue
                selected = files if all_headers else files[:1]
                valid = True
                for path in selected:
                    if time.monotonic()-started > seconds:
                        complete = valid = False
                        break
                    try:
                        ds = pydicom.dcmread(path, stop_before_pixels=True,
                                             specific_tags=['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID'])
                        patient = str(getattr(ds, 'PatientID', '')).strip()
                        if (not patient or str(getattr(ds, 'StudyInstanceUID', '')) != uid or
                                str(getattr(ds, 'SeriesInstanceUID', '')) != series['series_uid']):
                            raise ValueError('header identity missing/drift')
                        # No site namespace that could falsely split one patient.
                        patients.add(hashlib.sha256(('RSNA-G05-patient:'+patient).encode()).hexdigest())
                        tagged_same_as_study += int(patient == uid)
                        checked += 1
                    except (OSError, ValueError, pydicom.errors.InvalidDicomError):
                        valid = complete = False
                        issues.append({'study': uid, 'reason': 'header_invalid_or_missing'})
                if valid:
                    series_uids.append(series['series_uid'])
            if len(patients) != 1:
                complete = False
            records.append({'StudyInstanceUID': uid, 'split': split,
                            'patient_hash': next(iter(patients)) if len(patients) == 1 else '',
                            'patient_keys_seen': len(patients), 'series_uids': series_uids,
                            'headers_verified': checked, 'all_headers_consistent': complete})
            if len(records) % 100 == 0:
                print('headers inventoried', len(records), flush=True)
    evidence = {'status': 'INVENTORY_NOT_PATIENT_INDEPENDENCE_CERTIFICATION',
                'v05_sha256': V05_SHA, 'studies': records,
                'stable_patient_key_verified': False, 'stable_patient_key_source': None,
                'all_headers_requested': all_headers, 'issues': issues,
                'headers_with_PatientID_equal_to_study_uid': tagged_same_as_study,
                'confirmation_pixels_read': 0, 'confirmation_labels_scored': False,
                'manifest_contains_previously_frozen_weak_labels': True,
                'seconds': time.monotonic()-started, 'submission_eligible': False}
    patient_splits = {}
    for r in records:
        if r['patient_hash']:
            patient_splits.setdefault(r['patient_hash'], set()).add(r['split'])
    evidence['cross_split_patient_hashes'] = sum(len(s) > 1 for s in patient_splits.values())
    evidence['studies_with_three_series_headers'] = sum(len(r['series_uids']) == 3 for r in records)
    evidence['studies_with_complete_header_audit'] = sum(r['all_headers_consistent'] for r in records)
    try:
        identity_gate(manifest['splits'], evidence)
        evidence['patient_gate_passed'] = True
    except ValueError as exc:
        evidence['patient_gate_passed'] = False
        evidence['blocking_reason'] = str(exc)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/train_series'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--all-headers', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        raise ValueError('Header-only budget must be <=600 seconds')
    evidence = audit(args.root, args.seconds, args.all_headers)
    freeze(args.output, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k not in ('studies', 'issues')}, indent=2))
