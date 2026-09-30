"""#RSNA #Kaggle #Pesquisa — twenty train-only real pixel parity checks, no model.

Uses the same studies as the pinned G04 pilot. This does not satisfy the full
1,600-study extraction/duplicate/identity gates and measures no quality.
"""
import argparse
import ast
import json
from pathlib import Path
import resource
import time

import numpy as np

from scripts.g05_duplicate_audit import descriptor
import scripts.g05_pair_runtime as runtime
from scripts.prepare_g05_extraction import geometry_contract
from scripts.prepare_g04_preflight import select_training
from scripts.prepare_v04_confirmation import V03_BUILD, V03_SHA, literal
from scripts.resolution_comparison import digest, freeze, load_manifest


def check(root, output):
    if output.exists(): raise FileExistsError('Preserve existing pixel parity receipt')
    started = time.perf_counter()
    pilot = Path('reports/avance_av035_g04/v2_output/g04_features.npz')
    geometry_path = pilot.with_name('g04_geometry.json')
    if (digest(V03_BUILD) != V03_SHA or
            digest(pilot) != '3175b1a08d08a80fd371e8c9c10ed1dde2fd4d93b0ed27d1215764dfe45d9948' or
            digest(geometry_path) != '524893ce96b2fb74ef57924a7d1b01dc791c60df1c6ee7500a2ed29a83f0ac84'):
        raise ValueError('Pinned baseline/pilot drift')
    with np.load(pilot, allow_pickle=False) as a: ids = a['ids'].tolist()
    train = load_manifest()['splits']['train']
    if ids != [r['StudyInstanceUID'] for r in select_training(train)]:
        raise ValueError('G04 training-only population changed')
    by_id = {r['StudyInstanceUID']: r for r in train}
    selected_rows = [by_id[uid] for uid in ids]
    source = V03_BUILD.read_text()
    scope = {'np': np}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('physical_plan', 'normalize_cache_compatible'):
            exec(ast.get_source_segment(source, node), scope)
            setattr(runtime, node.name, scope[node.name])
    runtime.descriptor = descriptor
    helper = {'__name__': 'local_pinned_pixel_helper', '__file__': str(output.parent/'pinned_helper.py')}
    exec(compile(literal(source, 'CACHE_SOURCE'), '<pinned-pixel-helper>', 'exec'), helper)
    baseline = {(r['study'], r['series']): r for r in geometry_contract()}
    prior = {(r['study'], r['series']): r for r in json.loads(geometry_path.read_text())}
    records = []
    def guard():
        if time.perf_counter()-started > 240: raise TimeoutError('Bounded local CPU parity sample')
    try:
        for row in selected_rows:
            study = dict(row, split='train')
            for series in row['series']:
                key = row['StudyInstanceUID'], series['series_uid']
                images, record, _ = runtime.paired_series(root/key[0]/key[1], study, series, helper, baseline[key], guard)
                if (record['selected'] != prior[key]['selected'] or
                        record['pixel_sha256'] != prior[key]['pixel_sha256']):
                    raise ValueError('Exact G04 pixel parity at224/336 failed')
                records.append(record)
                del images
            print('LOCAL_PAIRED_PARITY', len(records)//3, '/20', flush=True)
        result = {'status': 'PASSED_20_TRAIN_PIXEL_PARITY_NOT_FULL_EXTRACTION_OR_MODEL_VALIDATION',
                  'studies': 20, 'series': 60, 'pixel224_exact_parity': True, 'pixel336_exact_parity': True,
                  'seconds': time.perf_counter()-started, 'peak_process_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'records': records, 'runtime_sha256': digest(Path('scripts/g05_pair_runtime.py')),
                  'g04_geometry_sha256': digest(geometry_path), 'gpu_used': False,
                  'confirmation_pixels_read': 0, 'confirmation_encoded': False, 'confirmation_labels_scored': False,
                  'full_pixel_gate_passed': False, 'patient_independence_certified': False,
                  'quality_measured': False, 'submission_eligible': False}
    except Exception as exc:
        result = {'status': 'BLOCKED_LOCAL_PIXEL_PARITY_SAMPLE', 'error_type': type(exc).__name__,
                  'message': str(exc), 'series_completed': len(records), 'seconds': time.perf_counter()-started,
                  'gpu_used': False, 'confirmation_pixels_read': 0, 'quality_measured': False,
                  'full_pixel_gate_passed': False, 'submission_eligible': False}
        freeze(output, result)
        raise
    freeze(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/raw/train_series'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); check(args.root, args.output)
