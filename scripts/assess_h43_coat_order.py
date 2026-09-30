"""#RSNA #Kaggle #Pesquisa — raw, ranks, lotes e inputs do diagnóstico CoAt."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scripts.dino_rank_replay import rank_columns
from scripts.h43_integrity import h43_sha, h43_validate_rows, H43_TARGETS
from scripts.h43_coat_order import COAT_SOURCE_SHA, patch_coat


def load_run(directory, ids, name, preflight, expected_series=205):
    n = len(ids)
    r = json.loads((directory / 'coat_resgated_ep10_top3_submission_receipt.json').read_text())
    if (r['status'] != 'VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION'
            or (r['models'], r['studies'], r['series']) != (3,n,expected_series)
            or r['gpu_names'] != ['Tesla T4']*2 or r['failures'] or r['fallback_studies'] != 0
            or r['gpu_batch_studies'] != 2 or r['backbone_micro_images'] != 8
            or r['serving_precision'] != 'float16' or r['cudnn_benchmark'] is not False
            or r['checkpoint_rank_weights'] != [1/3]*3
            or [c['epoch'] for c in r['checkpoints']] != [4,6,8]):
        raise ValueError('CoAt composition/coverage failed')
    pins = {(Path(f['path']).name, f['sha256']) for f in preflight['files'].values()}
    if any((c['name'], c['sha256']) not in pins for c in r['checkpoints']):
        raise ValueError('Checkpoint pins differ')
    csv_path = directory / name
    npz_path = directory / 'coat_resgated_ep10_top3_predictions.npz'
    if h43_sha(csv_path) != r['output_sha256'] or h43_sha(npz_path) != r['predictions_sha256']:
        raise ValueError('CoAt artifact hash mismatch')
    with np.load(npz_path, allow_pickle=False) as data:
        arr = {k: data[k].copy() for k in data.files}
    if set(arr) != {'study_uids','raw_probabilities','checkpoint_percentile_ranks','rank_ensemble'}:
        raise ValueError('Unexpected CoAt arrays')
    if arr['study_uids'].tolist() != ids: raise ValueError('Study order drift')
    for k,shape in [('raw_probabilities',(3,n,12)),('checkpoint_percentile_ranks',(3,n,12)),('rank_ensemble',(n,12))]:
        a = arr[k]
        if a.shape != shape or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():
            raise ValueError('CoAt raw shape/finitude/range')
    ranks = np.asarray([rank_columns(a.tolist(), pct=True) for a in arr['raw_probabilities']])
    if not np.array_equal(ranks, arr['checkpoint_percentile_ranks']) or not np.array_equal(ranks.mean(0), arr['rank_ensemble']):
        raise ValueError('Independent CoAt rank replay failed')
    with csv_path.open(newline='') as f: rows = list(csv.DictReader(f))
    h43_validate_rows(rows, ids, ['StudyInstanceUID']+H43_TARGETS)
    values = np.asarray([[float(row[t]) for t in H43_TARGETS] for row in rows])
    if not np.array_equal(values, arr['rank_ensemble']): raise ValueError('CoAt CSV does not match raw replay')
    if len(r['processes']) != 2 or len(r['shards']) != 2 or any(p['returncode'] != 0 for p in r['processes']):
        raise ValueError('CoAt worker failed')
    expected_start = 0
    for i,(p,s) in enumerate(zip(r['processes'], r['shards'])):
        start,stop = s['start'],s['stop']
        if start != expected_start or not start < stop <= n or [start,stop] != r['parallel_bounds'][i]:
            raise ValueError('CoAt shard coverage gap')
        if (p['start'],p['stop']) != (start,stop): raise ValueError('CoAt process bounds drift')
        with np.load(directory / f'coat_top3_part{i}.npz', allow_pickle=False) as shard:
            if shard['study_uids'].tolist() != ids[start:stop] or str(shard['failures_json']) != '[]':
                raise ValueError('CoAt shard identity/failures')
            if not np.array_equal(shard['prediction'],arr['raw_probabilities'][:,start:stop]):
                raise ValueError('Shard/merged raw mismatch')
        expected_start = stop
    if expected_start != n: raise ValueError('Incomplete CoAt shards')
    return r, arr


def compare_arrays(a,b):
    if not np.array_equal(a['study_uids'],b['study_uids']): raise ValueError('Different study order')
    differences = {}
    for key in ['raw_probabilities','checkpoint_percentile_ranks','rank_ensemble']:
        x,y = a[key],b[key]
        if x.shape != y.shape: raise ValueError('Different prediction shape')
        differences[key] = {'exact': bool(np.array_equal(x,y)),
            'changed_values': int(np.count_nonzero(x!=y)), 'max_abs_delta': float(np.max(np.abs(x-y)))}
    differences['raw_changed_study_indices'] = np.flatnonzero(np.any(a['raw_probabilities'] != b['raw_probabilities'],axis=(0,2))).tolist()
    return differences


def load_preflight(directory):
    pre = json.loads((directory/'h43_preflight.json').read_text())
    if pre['status'] != 'PASSED_ARTIFACT_PREFLIGHT' or pre['errors'] or pre['artifact_lock_entries'] != 56 or pre['gpus'] != ['Tesla T4']*2:
        raise ValueError('Preflight failed')
    selection = json.loads((directory/'h43_benchmark_selection.json').read_text())
    baseline = json.loads(Path('reports/avance_av012_serial_v1/h43_benchmark_selection.json').read_text())
    if selection != baseline: raise ValueError('Frozen sample differs')
    return pre, selection['ids']


def batch_contract(shard, mode, ids):
    probe = shard['order_probe']; n = shard['stop']-shard['start']
    if probe['mode'] != mode or len(probe['inputs']) != n: raise ValueError('Order/input receipt missing')
    rows = probe['inputs']
    if [r['index'] for r in rows] != list(range(n)) or [r['uid'] for r in rows] != ids[shard['start']:shard['stop']]:
        raise ValueError('Input receipt coverage/order')
    flat = [i for batch in probe['batches'] for i in batch]
    if sorted(flat) != list(range(n)) or any(not 1<=len(b)<=2 for b in probe['batches']):
        raise ValueError('Invalid GPU batch membership')
    if mode == 'ordered' and probe['batches'] != [list(range(i,min(i+2,n))) for i in range(0,n,2)]:
        raise ValueError('Ordered GPU batches not canonical')
    env = {k:v for k,v in probe.items() if k not in ['mode','inputs','batches']}
    return rows, probe['batches'], env


def assess_abba(directory):
    pre,ids = load_preflight(directory)
    r = json.loads((directory/'coat_order_abba_receipt.json').read_text())
    if r['status'] != 'COAT_ORDER_ABBA_COMPLETE_NOT_SUBMITTED' or r['original_source_sha256'] != COAT_SOURCE_SHA:
        raise ValueError('Wrong CoAt ABBA protocol')
    modes = ['completion','ordered','ordered','completion']
    if len(r['records']) != 4: raise ValueError('Incomplete ABBA')
    original = Path('reports/avance_av013_sources/coat/coatnet_resgated_ep10_top3_inference.py').read_text()
    arrays, contracts = [], []
    for i,(record,mode) in enumerate(zip(r['records'],modes)):
        d = directory / f'coat_order_run{i}'
        if record['index'] != i or record['mode'] != mode or not math.isfinite(record['seconds']) or record['seconds'] <= 0:
            raise ValueError('Invalid run record')
        import hashlib
        if record['source_sha256'] != hashlib.sha256(patch_coat(original,mode).encode()).hexdigest():
            raise ValueError('Patched source provenance mismatch')
        if h43_sha(d / 'coat_order_runtime.py') != record['source_sha256']:
            raise ValueError('Worker runtime file differs from pinned patch')
        if 'PACKED FALLBACK' in (d / 'worker.log').read_text():
            raise ValueError('Packed inference used a different fallback microbatch')
        receipt,arr = load_run(d,ids,'coat_diagnostic.csv',pre)
        if record['csv_sha256'] != receipt['output_sha256'] or record['prediction_sha256'] != receipt['predictions_sha256']:
            raise ValueError('ABBA hashes mismatch')
        arrays.append(arr)
        contracts.append([batch_contract(s,mode,ids) for s in receipt['shards']])
    inputs_exact = all([c[0] for c in x] == [c[0] for c in contracts[0]] for x in contracts)
    env_exact = all([c[2] for c in x] == [c[2] for c in contracts[0]] for x in contracts)
    ordered = compare_arrays(arrays[1],arrays[2]); legacy = compare_arrays(arrays[0],arrays[3])
    passed = inputs_exact and env_exact and all(ordered[k]['exact'] for k in ['raw_probabilities','checkpoint_percentile_ranks','rank_ensemble'])
    return {'status': 'PASSED_ORDERED_REPEAT' if passed else 'FAILED_ORDERED_REPEAT',
        'inputs_exact': inputs_exact, 'environments_exact': env_exact,
        'ordered_comparison': ordered, 'completion_comparison': legacy,
        'completion_vs_ordered': compare_arrays(arrays[0],arrays[1]),
        'batch_sequences': [[c[1] for c in x] for x in contracts],
        'seconds': [row['seconds'] for row in r['records']],
        'eligible_for_fullstack_test': passed, 'automatic_submission_authorized': False,
        'limitations': ['Same Kaggle session; raw parity is not AUC improvement.',
                       'No retroactive explanation of historical batch order without logs.',
                       'Two repeats cannot establish universal determinism.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path)
    p.add_argument('--historical-pair', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.historical_pair:
        if args.directory: p.error('Choose pair or ABBA')
        dirs = [Path('reports/avance_av012_serial_v1'),Path('reports/avance_av012_prefetch_v1')]
        values = [load_run(d,load_preflight(d)[1],'_coat_arm.csv',load_preflight(d)[0]) for d in dirs]
        result = {'status': 'HISTORICAL_COAT_RAW_AUDITED', **compare_arrays(values[0][1],values[1][1]),
                  'causal_batch_order_not_recorded': True, 'no_promotion': True}
    else:
        if not args.directory: p.error('--directory required for ABBA')
        result = assess_abba(args.directory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
