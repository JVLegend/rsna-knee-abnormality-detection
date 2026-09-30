"""#RSNA #Kaggle #Pesquisa — independently replay cache/duplicate gates, no model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.g05_duplicate_audit import descriptor, screen, DUPLICATE_RECIPE
from scripts.resolution_comparison import contract_hash, digest, freeze
import scripts.g05_pair_runtime as runtime


def verify_record(images, record, row, series, reference):
    if ((record['study'], record['split'], record['series'], record['plane']) !=
            (row['StudyInstanceUID'], row['split'], series['series_uid'], series['plane'])):
        raise ValueError('Pixel geometry order/identity changed')
    for size in (224, 336):
        value = images[size]
        if (value.shape != (3, size, size) or value.dtype != np.uint8 or
                any(float(c.std()) == 0 for c in value) or
                hashlib.sha256(value.tobytes()).hexdigest() != record['pixel_sha256'][str(size)]):
            raise ValueError('Per-series pixel hash/array mismatch')
    selected = record['selected']
    if (set(selected) != {'indices', 'files', 'positions_mm', 'gaps_mm'} or
            len(selected['files']) != 3 or len(selected['indices']) != 3 or
            len(selected['positions_mm']) != 3 or len(selected['gaps_mm']) != 2):
        raise ValueError('Physical sampling transcript incomplete')
    if reference is not None:
        if (selected != reference['selected'] or
                record['pixel_sha256']['224'] != reference['pixel_sha256'] or
                record['baseline224_reference_available'] is not True):
            raise ValueError('Baseline224 exact physical/pixel parity failed')
    elif record['baseline224_reference_available']:
        raise ValueError('Unpinned baseline parity claim')
    for field in ('raw_slice_sha256', 'normalized_slice_sha256'):
        if len(record[field]) != 3 or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in record[field]):
            raise ValueError('Slice receipt hash missing')
    vector, phash = descriptor(images[224])
    if phash != record['phash']:
        raise ValueError('Duplicate pHash replay mismatch')
    return vector


def candidate_keys(value):
    exact = [dict(r) for r in value['exact_cross_split_candidates']]
    near = [{k: v for k, v in r.items() if k != 'cosine'} for r in value['near_cross_split_candidates']]
    return exact, near


def verify(directory, headers):
    from scripts.prepare_g05_extraction import assemble
    _, expected_spec = assemble('pixels', headers)
    path = directory/'g05_pixels_receipt.json'
    receipt = json.loads(path.read_text())
    if (receipt['spec'] != expected_spec or receipt['contract_hash'] != expected_spec['contract_hash'] or
            receipt['status'] != 'COMPLETE_G05_PIXELS_NOT_FEATURES_OR_PATIENT_CERTIFICATION' or
            receipt['headers_sha256'] != digest(headers) or receipt['studies'] != 1600 or
            receipt['series'] != 4800 or receipt['gpu_used'] or receipt['confirmation_encoded'] or
            receipt['confirmation_labels_scored'] or receipt['patient_independence_certified'] or
            receipt['confirmation_pixels_for_duplicate_audit_only'] != 900 or
            receipt['baseline224_reference_series'] != 3000 or receipt['seconds'] > expected_spec['pixel_guard_seconds']):
        raise ValueError('Pixel receipt/source/scope/resource drift')
    geometry = directory/'g05_geometry.json'; duplicate_path = directory/'g05_duplicates.json'
    if digest(geometry) != receipt['geometry_sha256'] or digest(duplicate_path) != receipt['duplicates_sha256']:
        raise ValueError('Geometry/duplicate SHA drift')
    records = json.loads(geometry.read_text())
    if len(records) != 4800:
        raise ValueError('Incomplete paired geometry')
    expected = {(r['study'], r['series']): r for r in expected_spec['expected_geometry']}
    rows = expected_spec['rows']
    if len(receipt['shards']) != 80 or len({s['file'] for s in receipt['shards']}) != 80:
        raise ValueError('Incomplete or repeated pixel shards')
    vectors = []
    total_bytes = 0
    runtime.sha = digest
    for shard_index, shard in enumerate(receipt['shards']):
        begin = shard_index*20
        batch = rows[begin:begin+20]
        if shard['file'] != f'g05_pixels_{begin:04d}.npz' or shard['ids'] != [r['StudyInstanceUID'] for r in batch]:
            raise ValueError('Frozen shard order/identity drift')
        arrays = runtime.load_pixel_shard(directory/shard['file'], shard, receipt['contract_hash'])
        total_bytes += (directory/shard['file']).stat().st_size
        for i, row in enumerate(batch):
            for j, series in enumerate(row['series']):
                record = records[(begin+i)*3+j]
                vectors.append(verify_record({s: a[i, j] for s, a in arrays.items()}, record, row, series,
                               expected.get((row['StudyInstanceUID'], series['series_uid']))))
        del arrays
    replay = screen(records, vectors)
    reported = json.loads(duplicate_path.read_text())
    if (reported['recipe'] != DUPLICATE_RECIPE or not reported['screen_completed'] or
            reported['records'] != 4800 or reported['patient_independence_certified'] or
            candidate_keys(replay) != candidate_keys(reported) or
            replay['screen_passed'] != reported['screen_passed'] or
            receipt['duplicate_screen_passed'] != replay['screen_passed'] or
            not np.allclose([r['cosine'] for r in replay['near_cross_split_candidates']],
                            [r['cosine'] for r in reported['near_cross_split_candidates']], atol=1e-6, rtol=0)):
        raise ValueError('Independent duplicate replay differs from execution receipt')
    return {'status': 'PASSED_PAIRED_PIXELS_NOT_PATIENT_CERTIFICATION' if replay['screen_passed'] else 'BLOCKED_DUPLICATE_CANDIDATES_REVIEW_REQUIRED',
            'paired_pixel_gate_passed': replay['screen_passed'], 'studies': 1600, 'series': 4800,
            'baseline224_pixel_parity_series': 3000, 'shards': 80, 'compressed_bytes': total_bytes,
            'exact_cross_split_candidates': len(replay['exact_cross_split_candidates']),
            'near_cross_split_candidates': len(replay['near_cross_split_candidates']),
            'pixel_receipt_sha256': digest(path), 'headers_sha256': digest(headers),
            'contract_hash': receipt['contract_hash'], 'seconds_cpu_job': receipt['seconds'],
            'confirmation_encoded': False, 'confirmation_labels_scored': False,
            'patient_independence_certified': False, 'submission_eligible': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('directory', 'headers', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    result = verify(args.directory, args.headers)
    freeze(args.output, result)
    print(json.dumps(result, indent=2))
