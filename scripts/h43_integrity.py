"""#RSNA #Kaggle #Testes — preflight e gates H43, sem executar pesos/fontes."""
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

H43_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
               'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
H43_RECEIPT = {}
H43_SOURCE_SHA = '5b133e41d951aae8dc9efa9361d1975d5259a8f98f956cc1e84d78ea1aed80e7'
H43_PINS = [
    ('manifest.json', '496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844'),
    ('coat_resgated_ep10_top3_manifest.json', '98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'),
    ('ResNet50.pt', '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'),
    ('v52_radimagenet_heads.pt', '54f657826b3458a7ba3d462e198ba380732f2b136246182312704929874a9a2c'),
    ('v52_radimagenet_heads.pt', '0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa'),
    ('v52_e11_heads.pt', 'ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a'),
    ('opencv_python_headless-4.12.0.88-*.whl', '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'),
]


def h43_require(condition, message):
    if not condition:
        raise RuntimeError('H43 integrity: ' + message)


def h43_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def h43_check_members(expected, actual):
    h43_require(len(expected) == len(set(expected)), 'duplicate expected members')
    h43_require(Counter(expected) == Counter(actual), 'missing/duplicate/unexpected members')


def h43_validate_rows(rows, ids, columns):
    h43_require(columns == ['StudyInstanceUID'] + H43_TARGETS, 'CSV schema')
    actual = [r['StudyInstanceUID'] for r in rows]
    h43_require(len(ids) > 0 and actual == ids and len(ids) == len(set(ids)), 'CSV IDs/order/coverage')
    for row in rows:
        for target in H43_TARGETS:
            value = float(row[target])
            h43_require(math.isfinite(value) and 0 <= value <= 1, 'nonfinite/out-of-range prediction')


def h43_inventory(root):
    result = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {
            'train_series', 'test_series', 'train_images', 'test_images'})
        result.extend(Path(directory) / name for name in sorted(files))
    return result


def h43_select(files, pattern, expected_sha=None, cache=None):
    cache = {} if cache is None else cache
    candidates = [p for p in files if p.match(pattern)]
    valid = []
    for path in candidates:
        key = str(path)
        if key not in cache:
            cache[key] = {'path': key, 'bytes': path.stat().st_size, 'sha256': h43_sha(path)}
        if expected_sha is None or cache[key]['sha256'] == expected_sha:
            valid.append(path)
    h43_require(bool(valid), f'missing/hash mismatch: {pattern} {expected_sha}')
    # Cópias idênticas são aceitas; conteúdo diferente sem pin é ambíguo.
    h43_require(len({cache[str(p)]['sha256'] for p in valid}) == 1,
                f'ambiguous unpinned content: {pattern}')
    return sorted(valid)[0]


def h43_preflight(root='/kaggle/input', output='/kaggle/working/h43_preflight.json', gpu=False):
    start = time.monotonic()
    files = h43_inventory(root)
    observed, resolved = {}, {}
    report = {'status': 'CHECKING', 'source_sha256': H43_SOURCE_SHA,
              'files': observed, 'errors': [], 'gpu_required': gpu}
    try:
        if gpu:
            import torch
            h43_require(torch.cuda.device_count() == 2, 'requires two GPUs')
            names = [torch.cuda.get_device_name(i) for i in range(2)]
            h43_require(all('T4' in n for n in names), f'expected T4, got {names}')
            report['gpus'] = names
        for name, pin in H43_PINS:
            resolved[pin] = h43_select(files, name, pin, observed)
        lock = globals().get('H43_ARTIFACT_LOCK', [])
        for name, pin in lock:
            h43_select(files, name, pin, observed)
        report['artifact_lock_entries'] = len(lock)
        dino_path = resolved[H43_PINS[0][1]]
        dino = json.loads(dino_path.read_text())
        members = dino['members']
        h43_require(len(members) == 20, 'DINO manifest member count')
        h43_require(Counter(m['fold'] for m in members) == dict.fromkeys(range(5), 4), 'DINO folds')
        h43_check_members([m['id'] for m in members], [m['id'] for m in members])
        for member in members:
            path = (dino_path.parent / member['file']).resolve()
            h43_require(path.is_relative_to(dino_path.parent.resolve()), 'manifest path escapes package')
            h43_select([path] if path.is_file() else [], path.name, cache=observed)
        coat_path = resolved[H43_PINS[1][1]]
        coat = json.loads(coat_path.read_text())
        h43_require([m['epoch'] for m in coat['checkpoints']] == [4, 6, 8], 'CoAt epochs')
        for name, pin in coat['files'].items():
            path = (coat_path.parent / name).resolve()
            h43_require(path.is_relative_to(coat_path.parent.resolve()), 'CoAt path escapes package')
            h43_select([path] if path.is_file() else [], path.name, pin, observed)
        for name in [*(f'm_f{i}.pt' for i in range(5)), 'raptor_ft_coatnet_v5_full_swa.pt',
                     'raptor_ft_coatnet_v10_full.pt', 'raptor_ft_coatnet_v8_full_swa.pt']:
            h43_select(files, name, cache=observed)
        report.update(status='PASSED_ARTIFACT_PREFLIGHT', dino_ids=[m['id'] for m in members],
                      dino_members=20, a5_folds=5, raptor_checkpoints=3, coat_checkpoints=3,
                      limitation='Unpinned weights hashed for receipt only; no state_dict, pixel or numerical parity test.')
    except Exception as error:
        report['status'] = 'FAILED_ARTIFACT_PREFLIGHT'
        report['errors'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
    H43_RECEIPT['preflight'] = report
    return report


def h43_publish(partial, output, expected_ids, receipt, run):
    h43_require(run['coatnet_w'] == {'__default__': .60}, 'not the fixed parent')
    h43_require(receipt.get('preflight', {}).get('status') == 'PASSED_ARTIFACT_PREFLIGHT', 'preflight missing')
    h43_require(set(receipt.get('stages', [])) == {'dino', 'a5', 'rad', 'raptor', 'coat'}, 'incomplete stages')
    with Path(partial).open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        h43_validate_rows(rows, list(expected_ids), reader.fieldnames)
    h43_require(not Path(output).exists(), 'refuse overwrite of existing submission')
    # Apenas o gate final publica o nome elegível; diagnósticos usam outro nome.
    receipt.update(status='PASSED_PARENT_INTEGRITY', submission_sha256=h43_sha(partial),
                   studies=len(rows), oof_independent=False)
    Path(output).with_name('h43_parent_integrity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    Path(partial).rename(output)
