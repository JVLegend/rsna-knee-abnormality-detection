"""#RSNA #Kaggle #Pesquisa — verify exhaustive coverage, not PatientID stability."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from scripts.prepare_g05_headers import assemble
from scripts.resolution_comparison import digest, freeze, load_manifest, V05_SHA


def assess(directory, build, output=None):
    if build.read_text() != assemble():
        raise ValueError('Header builder/source drift')
    spec = ast.literal_eval(ast.parse(build.read_text()).body[0].value)
    receipt = json.loads((directory/'g05_headers_receipt.json').read_text())
    evidence_path = directory/'g05_headers.json'
    if (receipt['source_contract_hash'] != spec['contract_hash'] or receipt['gpu_used'] or
            receipt['confirmation_evaluated'] or digest(evidence_path) != receipt['headers_sha256']):
        raise ValueError('Header execution receipt/source/SHA drift')
    evidence = json.loads(evidence_path.read_text())
    if (evidence.get('status') != 'COMPLETE_FULL_HEADERS_NOT_PATIENT_SEMANTICS_CERTIFICATION' or
            evidence.get('gpu_used') or evidence.get('confirmation_pixels_read') != 0 or
            evidence.get('confirmation_labels_scored') or evidence.get('stable_patient_key_verified') or
            evidence.get('stable_patient_key_source')):
        raise ValueError('Header scope/unsupported patient certification claim')
    manifest = load_manifest()
    expected = [(r['StudyInstanceUID'], split, {s['series_uid'] for s in r['series']})
                for split in ['train', 'development', 'confirmation'] for r in manifest['splits'][split]]
    with Path('data/raw/train_series.csv').open(newline='') as handle:
        metadata = list(csv.DictReader(handle))
    official = defaultdict(set)
    for r in metadata:
        official[r['StudyInstanceUID']].add(r['SeriesInstanceUID'])
    records = evidence['studies']
    if len(records) != 1600 or len({r['StudyInstanceUID'] for r in records}) != 1600:
        raise ValueError('Frozen study coverage incomplete/duplicated')
    total = 0
    issues = []
    per_patient = defaultdict(list)
    for record, (uid, split, selected) in zip(records, expected):
        if (record['StudyInstanceUID'], record['split'], set(record['series_uids'])) != (uid, split, selected):
            raise ValueError('Study/split/selected-series identity changed')
        series = record['series_records']
        if (set(record['all_series_uids']) != official[uid] or
                {r['series_uid'] for r in series} != official[uid] or
                len(series) != len(official[uid])):
            raise ValueError('All official series coverage drift')
        if (not record['all_headers_consistent'] or not record['all_official_series_checked'] or
                record['patient_keys_seen'] != 1 or record['issues'] or
                any(r['headers'] < 1 or r['patient_keys'] != 1 for r in series) or
                record['headers_verified'] != sum(r['headers'] for r in series)):
            issues.append(uid)
        if any(len(r['header_transcript_sha256']) != 64 for r in series) or len(record['header_transcript_sha256']) != 64:
            raise ValueError('Header transcript evidence missing')
        total += record['headers_verified']
        patient = record['patient_hash']
        if record['all_headers_consistent'] and (len(patient) != 64 or any(c not in '0123456789abcdef' for c in patient)):
            raise ValueError('Patient hash missing from consistent study')
        per_patient[record['patient_hash']].append(record)
    if (total != evidence['headers_verified'] or total != receipt['headers_verified'] or
            evidence['v05_sha256'] != V05_SHA or evidence['source_contract_hash'] != spec['contract_hash']):
        raise ValueError('Header counts/contract drift')
    if (evidence['series_verified'] != sum(len(r['series_records']) for r in records) or
            evidence['studies_with_complete_header_audit'] != 1600-len(issues) or
            receipt['studies_complete'] != 1600-len(issues)):
        raise ValueError('Header summary coverage drift')
    overlap = [p for p, rows in per_patient.items() if len({r['split'] for r in rows}) > 1]
    complete = not issues and not evidence['cross_study_sop_collisions']
    result = {'status': 'PASSED_FULL_HEADERS_IDENTITY_SEMANTICS_UNVERIFIED' if complete else 'BLOCKED_INCOMPLETE_OR_INCONSISTENT_HEADERS',
              'full_header_technical_audit_passed': complete,
              'patient_independence_certified': False, 'patient_gate_passed': False,
              'stable_patient_key_verified': False,
              'blocking_reasons': ['Dataset-specific stable patient anonymization/mapping not evidenced'] +
                                  (['Observed cross-split PatientID keys'] if overlap else []) +
                                  (['Header coverage/consistency or SOP duplicate failure'] if not complete else []),
              'studies': 1600, 'split_counts': dict(Counter(r['split'] for r in records)),
              'complete_studies': 1600-len(issues), 'headers': total, 'series': evidence['series_verified'],
              'unique_patient_keys': len(per_patient),
              'repeated_patient_keys': sum(len(r) > 1 for r in per_patient.values()),
              'cross_split_patient_keys': len(overlap), 'deidentification_methods': evidence['deidentification_methods'],
              'deidentification_codes': evidence['deidentification_codes'],
              'patient_identity_removed_values': evidence['patient_identity_removed_values'],
              'receipt_sha256': digest(directory/'g05_headers_receipt.json'),
              'headers_sha256': digest(evidence_path), 'build_sha256': digest(build),
              'seconds': evidence['seconds'], 'gpu_used': False,
              'confirmation_pixels_read': 0, 'confirmation_labels_scored': False,
              'submission_eligible': False,
              'limitation': 'Consistent headers, report disjointness and no key collision cannot establish patient independence.'}
    if output is not None:
        freeze(output, result)
        print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['directory', 'build', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); assess(a.directory, a.build, a.output)
