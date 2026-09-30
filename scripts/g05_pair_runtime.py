"""#RSNA #Kaggle #Pesquisa — CPU paired pixels, then identity-gated frozen features.

Builder prepends G05_SPEC, exact pinned decoder/geometry helpers and CACHE_SOURCE.
Pixels of confirmation are used solely for duplicate screening, never encoding,
head fitting or label/prediction scoring in the development stage.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def unique_artifact(root, name, expected_sha):
    found = []
    for count, (folder, dirs, files) in enumerate(os.walk(root), 1):
        if count > 512:
            raise ValueError('Attachment discovery budget')
        if name in files:
            found.append(Path(folder)/name)
        dirs[:] = sorted(d for d in dirs if d not in ('train_series', 'test_series', 'kaggle_test_series'))
    if len(found) != 1 or sha(found[0]) != expected_sha:
        raise ValueError('Ambiguous or incompatible hashed attachment: '+name)
    return found[0]


def checked_header_evidence(path, expected_sha, spec):
    if sha(path) != expected_sha:
        raise ValueError('Full-header attachment SHA drift')
    evidence = json.loads(path.read_text())
    rows = spec['rows']
    ids = {r['StudyInstanceUID'] for r in rows}
    observed = evidence.get('studies', [])
    if (evidence.get('v05_sha256') != spec['v05_sha256'] or
            evidence.get('status') != 'COMPLETE_FULL_HEADERS_NOT_PATIENT_SEMANTICS_CERTIFICATION' or
            len(observed) != len(ids) or {r['StudyInstanceUID'] for r in observed} != ids or
            any(not r['all_headers_consistent'] or not r['all_official_series_checked'] for r in observed) or
            evidence.get('cross_study_sop_collisions')):
        raise ValueError('Exhaustive technical header audit not passed')
    return evidence


def paired_series(folder, study, series, helper, expected, guard):
    import pydicom
    files = sorted(folder.glob('*.dcm'))
    if not files:
        raise ValueError('Missing paired series; do not drop studies')
    guard()
    headers = [(p.name, pydicom.dcmread(p, stop_before_pixels=True,
               specific_tags=['ImageOrientationPatient', 'ImagePositionPatient'])) for p in files]
    selected = physical_plan(headers, [files[0].name]*3)['arms']['physical_adjacent']
    if expected is not None and selected != expected['selected']:
        raise ValueError('Pinned physical sampling changed')
    channels = {size: [] for size in (224, 336)}
    raw_hashes = []
    normalized_hashes = []
    for name in selected['files']:
        guard()
        ds = pydicom.dcmread(folder/name, specific_tags=helper['PIXEL_TAGS'])
        pixels = helper['_pixel_array'](ds)
        normalized, _, _ = normalize_cache_compatible(pixels)
        raw_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        normalized_hashes.append(hashlib.sha256(normalized.tobytes()).hexdigest())
        for size in channels:
            channels[size].append(helper['resize_slice'](normalized, size))
    images = {size: np.stack(channels[size]) for size in channels}
    hashes = {}
    for size, image in images.items():
        if image.shape != (3, size, size) or image.dtype != np.uint8 or any(float(c.std()) == 0 for c in image):
            raise ValueError('Invalid paired pixels')
        hashes[str(size)] = hashlib.sha256(image.tobytes()).hexdigest()
    if expected is not None and hashes['224'] != expected['pixel_sha256']:
        raise ValueError('Exact baseline224 pixel parity failed')
    vector, phash = descriptor(images[224])
    record = {'study': study['StudyInstanceUID'], 'split': study['split'], 'series': series['series_uid'],
              'plane': series['plane'], 'selected': selected, 'pixel_sha256': hashes,
              'raw_slice_sha256': raw_hashes, 'normalized_slice_sha256': normalized_hashes,
              'phash': phash, 'baseline224_reference_available': expected is not None}
    return images, record, vector


def extract_pixels(input_root, output, spec):
    started = globals().get('G05_CODE_STARTED', time.perf_counter())
    header = unique_artifact(input_root, 'g05_headers.json', spec['headers_sha256'])
    checked_header_evidence(header, spec['headers_sha256'], spec)
    _, roots = discover_inputs(input_root)
    if len(roots) != 1:
        raise ValueError('Ambiguous competition attachment')
    helper = {'__name__': 'g05_pinned_pixel_helper', '__file__': '/kaggle/working/g05_pinned_pixel_helper.py'}
    exec(compile(CACHE_SOURCE, '<pinned-pixel-helper>', 'exec'), helper)
    def guard():
        if time.perf_counter()-started > spec['pixel_guard_seconds']:
            raise TimeoutError('Paired pixel budget; no partial cache accepted')
    expected = {(r['study'], r['series']): r for r in spec['expected_geometry']}
    records, vectors, shards, timing = [], [], [], []
    rows = spec['rows']
    for begin in range(0, len(rows), 20):
        tick = time.perf_counter()
        batch = rows[begin:begin+20]
        values = {size: [] for size in (224, 336)}
        for study in batch:
            for series in study['series']:
                images, record, vector = paired_series(roots[0]/'train_series'/study['StudyInstanceUID']/series['series_uid'],
                    study, series, helper, expected.get((study['StudyInstanceUID'], series['series_uid'])), guard)
                for size in values:
                    values[size].append(images[size])
                records.append(record); vectors.append(vector)
        shard = output/f'g05_pixels_{begin:04d}.npz'
        np.savez_compressed(shard, ids=[r['StudyInstanceUID'] for r in batch],
                            images224=np.asarray(values[224]).reshape(len(batch), 3, 3, 224, 224),
                            images336=np.asarray(values[336]).reshape(len(batch), 3, 3, 336, 336),
                            contract_hash=spec['contract_hash'])
        shards.append({'file': shard.name, 'sha256': sha(shard), 'ids': [r['StudyInstanceUID'] for r in batch]})
        timing.append({'begin': begin, 'studies': len(batch), 'decode_resize_write_seconds': time.perf_counter()-tick})
        print('PAIRED_PIXELS', begin+len(batch), '/', len(rows), flush=True)
    guard()
    duplicate = screen(records, vectors)
    guard()
    duplicate_path = output/'g05_duplicates.json'
    duplicate_path.write_text(json.dumps(duplicate, sort_keys=True))
    geometry = output/'g05_geometry.json'
    geometry.write_text(json.dumps(records, sort_keys=True))
    guard()
    receipt = {'status': 'COMPLETE_G05_PIXELS_NOT_FEATURES_OR_PATIENT_CERTIFICATION',
               'contract_hash': spec['contract_hash'], 'protocol_hash': spec['protocol_hash'],
               'spec': spec, 'shards': shards, 'geometry_sha256': sha(geometry),
               'duplicates_sha256': sha(duplicate_path), 'duplicate_screen_passed': duplicate['screen_passed'],
               'same_physical_slices_verified': True,
               'baseline224_pixel_hashes_verified': True, 'baseline224_reference_series': len(expected),
               'studies': len(rows), 'series': len(records), 'headers_sha256': spec['headers_sha256'],
               'seconds': time.perf_counter()-started, 'timings': timing,
               'gpu_used': False, 'confirmation_pixels_for_duplicate_audit_only': sum(len(r['series']) for r in rows if r['split'] == 'confirmation'),
               'confirmation_encoded': False, 'confirmation_labels_scored': False,
               'patient_independence_certified': False, 'submission_eligible': False}
    (output/'g05_pixels_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def load_pixel_shard(path, pinned, contract):
    if sha(path) != pinned['sha256']:
        raise ValueError('Pixel shard hash drift')
    with np.load(path, allow_pickle=False) as archive:
        if archive['ids'].tolist() != pinned['ids'] or str(archive['contract_hash']) != contract:
            raise ValueError('Pixel cache IDs/contract drift')
        values = {size: archive[f'images{size}'].copy() for size in (224, 336)}
    for size, value in values.items():
        if value.dtype != np.uint8 or value.shape != (len(pinned['ids']), 3, 3, size, size):
            raise ValueError('Pixel cache shape/dtype drift')
    return values


def extract_features(input_root, output, spec):
    started = globals().get('G05_CODE_STARTED', time.perf_counter())
    import torch
    from transformers import Dinov2Model, Dinov2Config
    identity_path = unique_artifact(input_root, 'g05_identity.json', spec['identity_sha256'])
    identity = json.loads(identity_path.read_text())
    identity_gate(spec['split_identity_rows'], identity)  # before CUDA/model access
    receipt_path = unique_artifact(input_root, 'g05_pixels_receipt.json', spec['pixel_receipt_sha256'])
    pixels = json.loads(receipt_path.read_text())
    if pixels['contract_hash'] != spec['pixel_contract_hash'] or not pixels['duplicate_screen_passed']:
        raise ValueError('Pixel contract/duplicate gate not passed')
    directory = receipt_path.parent
    if sha(directory/'g05_geometry.json') != pixels['geometry_sha256'] or sha(directory/'g05_duplicates.json') != pixels['duplicates_sha256']:
        raise ValueError('Geometry/duplicate evidence hash drift')
    duplicates = json.loads((directory/'g05_duplicates.json').read_text())
    if not duplicates['screen_completed'] or not duplicates['screen_passed'] or duplicates['recipe'] != DUPLICATE_RECIPE:
        raise ValueError('Fixed duplicate audit not passed')
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_name(i) != 'Tesla T4' for i in range(2)):
        raise ValueError('Existing T4x2 required; only cuda0 used')
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    model_files, _ = discover_inputs(input_root)
    models = [p for p in model_files if p.stat().st_size == 88297097 and sha(p) == spec['model_sha256']]
    if len(models) != 1 or sha(models[0].parent/'config.json') != spec['config_sha256']:
        raise ValueError('Official encoder weights/config hash drift')
    encoder = Dinov2Model(Dinov2Config.from_json_file(str(models[0].parent/'config.json')))
    encoder.load_state_dict(torch.load(models[0], map_location='cpu', weights_only=True), strict=True)
    encoder = encoder.eval().requires_grad_(False).cuda()
    baseline = unique_artifact(input_root, 'v03_features.npz', spec['baseline_archive_sha256'])
    pilot = unique_artifact(input_root, 'g04_features.npz', spec['pilot_archive_sha256'])
    pilot_geometry = unique_artifact(input_root, 'g04_geometry.json', spec['pilot_geometry_sha256'])
    with np.load(baseline, allow_pickle=False) as a:
        baseline_ids = a['train_ids'].tolist(); baseline_features = a['train'].copy()
    with np.load(pilot, allow_pickle=False) as a:
        pilot_ids = a['ids'].tolist(); pilot224 = a['features224'].copy(); pilot336 = a['features336'].copy()
    if baseline_ids != [r['StudyInstanceUID'] for r in spec['rows'] if r['split'] == 'train']:
        raise ValueError('Pinned train feature identities changed')
    if (baseline_features.shape != (1000, 3, 384) or baseline_features.dtype != np.float32 or
            pilot224.shape != (20, 3, 384) or pilot336.shape != pilot224.shape or
            not all(np.isfinite(f).all() for f in (baseline_features, pilot224, pilot336))):
        raise ValueError('Invalid existing feature cache')
    pilot_geometry = {(r['study'], r['series']): r for r in json.loads(pilot_geometry.read_text())}
    geometry = {(r['study'], r['series']): r for r in json.loads((directory/'g05_geometry.json').read_text())}
    for key, record in pilot_geometry.items():
        current = geometry[key]
        if record['selected'] != current['selected'] or record['pixel_sha256'] != current['pixel_sha256']:
            raise ValueError('Pilot336 cache pixel/geometry incompatibility')
    mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
    feature_map = {224: dict(zip(baseline_ids, baseline_features)), 336: dict(zip(pilot_ids, pilot336))}
    timing, parity = [], []
    selected_ids = [r['StudyInstanceUID'] for r in spec['rows'] if r['split'] in ('train', 'development')]
    def guard():
        if time.perf_counter()-started > spec['feature_guard_seconds']:
            raise TimeoutError('Feature GPU budget')
    # Matched, balanced-order timing; missing224 and missing336 populations differ
    # due to cache reuse and must NOT be used to estimate the resolution ratio.
    first = pixels['shards'][0]
    benchmark_images = load_pixel_shard(directory/first['file'], first, pixels['contract_hash'])
    benchmark = []
    with torch.no_grad():
        for size in (224, 336):
            tensor = torch.from_numpy(benchmark_images[size][:2].reshape(-1, 3, size, size)).cuda().float()/255
            encoder(pixel_values=(tensor-mean)/std)
        torch.cuda.synchronize()
        for repetition in range(10):
            for size in ((224, 336) if repetition % 2 == 0 else (336, 224)):
                guard(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
                tensor = torch.from_numpy(benchmark_images[size][:2].reshape(-1, 3, size, size)).cuda().float()/255
                encoder(pixel_values=(tensor-mean)/std)
                torch.cuda.synchronize()
                benchmark.append({'repetition': repetition, 'resolution': size,
                                  'seconds': time.perf_counter()-tick,
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    del benchmark_images, tensor
    for shard in pixels['shards']:
        # Never load confirmation shards for feature encoding.
        if not set(shard['ids']) & set(selected_ids):
            continue
        values = load_pixel_shard(directory/shard['file'], shard, pixels['contract_hash'])
        for size in (224, 336):
            missing = [i for i, uid in enumerate(shard['ids']) if uid in selected_ids and
                       (uid not in feature_map[size] or (size == 224 and uid in pilot_ids))]
            for begin in range(0, len(missing), 2):
                guard(); ix = missing[begin:begin+2]; ids = [shard['ids'][i] for i in ix]
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
                tensor = torch.from_numpy(values[size][ix].reshape(-1, 3, size, size)).cuda().float()/255
                with torch.no_grad():
                    encoded = encoder(pixel_values=(tensor-mean)/std).last_hidden_state[:, 0].cpu().numpy().reshape(len(ix), 3, 384)
                torch.cuda.synchronize()
                if encoded.dtype != np.float32 or not np.isfinite(encoded).all():
                    raise ValueError('Invalid frozen encoder output')
                timing.append({'resolution': size, 'studies': len(ix), 'seconds': time.perf_counter()-tick,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                for uid, value in zip(ids, encoded):
                    if uid in feature_map[size]:
                        if not np.allclose(value, feature_map[size][uid], atol=1e-4, rtol=1e-5):
                            raise ValueError('Replayed224 feature parity failed')
                        parity.append(float(np.abs(value-feature_map[size][uid]).max()))
                    else:
                        feature_map[size][uid] = value
                del tensor
        print('FEATURE_SHARD_COMPLETE', shard['file'], flush=True)
    if len(parity) != 20 or set(feature_map[224]) != set(selected_ids) or set(feature_map[336]) != set(selected_ids):
        raise ValueError('Feature/parity coverage incomplete or confirmation accessed')
    path = output/'g05_features.npz'
    np.savez_compressed(path, ids=selected_ids, features224=np.array([feature_map[224][u] for u in selected_ids]),
                        features336=np.array([feature_map[336][u] for u in selected_ids]), contract_hash=spec['protocol_hash'])
    by_size = {s: sum(t['seconds'] for t in benchmark if t['resolution'] == s)/10 for s in (224, 336)}
    receipt = {'status': 'VERIFIED_G05_PAIRED_FEATURES', 'contract_hash': spec['protocol_hash'],
               'extraction_contract_hash': spec['contract_hash'], 'archive_sha256': sha(path),
               'model_sha256': spec['model_sha256'], 'config_sha256': spec['config_sha256'],
               'same_physical_slices_verified': True, 'baseline224_pixel_hashes_verified': True,
               'baseline224_feature_parity_verified': True, 'parity224_replayed_studies': len(parity),
               'max_parity224_delta': max(parity), 'cached224_reused_studies': 1000, 'cached336_reused_studies': 20,
               'cross_split_exact_and_near_duplicate_audit_passed': True,
               'duplicate_screen_is_not_patient_identity_proof': True,
               'identity_sha256': spec['identity_sha256'], 'pixel_receipt_sha256': spec['pixel_receipt_sha256'],
               'resources': {'total_session_seconds': time.perf_counter()-started,
                             'cpu_pixel_seconds': pixels['seconds'],
                             'maximum_peak_allocated_bytes': max(t['peak_allocated_bytes'] for t in timing+benchmark),
                             'forward336_over224': by_size[336]/by_size[224]},
               'timings': timing, 'matched_resolution_benchmark': benchmark,
               'confirmation_encoded': False, 'confirmation_labels_scored': False,
               'submission_eligible': False}
    (output/'g05_feature_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    output = Path('/kaggle/working')
    try:
        function = extract_pixels if G05_SPEC['stage'] == 'pixels' else extract_features
        result = function(Path('/kaggle/input'), output, G05_SPEC)
        print(result['status'], flush=True)
    except Exception as exc:
        (output/'g05_extraction_failure.json').write_text(json.dumps({'error_type': type(exc).__name__,
            'message': str(exc), 'contract_hash': G05_SPEC['contract_hash'], 'stage': G05_SPEC['stage']}))
        raise


if __name__ == '__main__':
    main()
