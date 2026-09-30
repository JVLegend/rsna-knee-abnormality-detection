"""#RSNA #Kaggle #Pesquisa — instrumentação E03 injetada no namespace Raptor.

Requer o módulo ordered_prefetch embutido antes deste código. Executado
somente no Kaggle, após definição das funções Raptor e preflight fixado.
"""
from contextlib import contextmanager, closing


@contextmanager
def _e03_studies(ids, series, tsdir, reader, recipe_index):
    def prepare(uid):
        start = time.monotonic()
        volume, mask = build_study(uid, series, tsdir, reader)
        h43_require(bool(mask.any()), 'E03 study has no decoded slot')
        record = {'recipe': recipe_index, 'uid': uid,
                  'volume_sha256': hashlib.sha256(volume.tobytes()).hexdigest(),
                  'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                  'shape': list(volume.shape), 'volume_bytes': int(volume.nbytes),
                  'prepare_seconds': time.monotonic() - start}
        E03_INPUTS.append(record)
        return volume, mask
    if E03_MODE == 'serial':
        iterator = ((index, uid, prepare(uid)) for index, uid in enumerate(ids))
    elif E03_MODE == 'prefetch':
        iterator = ordered_one_ahead(ids, prepare)
    else:
        raise RuntimeError('Unknown E03 mode')
    with closing(iterator):
        yield iterator


def e03_benchmark():
    import json
    import resource
    from pathlib import Path
    global E03_MODE, E03_INPUTS, E03_RUN_INDEX
    records, arrays, input_contracts = [], [], []
    for E03_RUN_INDEX, E03_MODE in enumerate(['serial', 'prefetch', 'prefetch', 'serial']):
        E03_INPUTS = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        main()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        raw_path = Path(f'/kaggle/working/e03_run{E03_RUN_INDEX}_raw.npz')
        with np.load(raw_path) as data:
            arrays.append({name: data[name].copy() for name in data.files})
        input_contracts.append([{k: v for k, v in row.items() if k != 'prepare_seconds'}
                                for row in E03_INPUTS])
        record = {'index': E03_RUN_INDEX, 'mode': E03_MODE, 'seconds': elapsed,
                  'prepare_seconds_sum_overlaps_gpu': sum(r['prepare_seconds'] for r in E03_INPUTS),
                  'peak_reserved_cuda_bytes': torch.cuda.max_memory_reserved(),
                  'process_peak_rss_kib_linux_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'studies': len(arrays[-1]['ids']), 'prepared_recipe_studies': len(E03_INPUTS),
                  'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  'torch_threads': torch.get_num_threads(), 'inputs': E03_INPUTS}
        records.append(record)
        Path(f'/kaggle/working/e03_run{E03_RUN_INDEX}_receipt.json').write_text(json.dumps(record, indent=2))
        print('E03_RUN', {k: v for k, v in record.items() if k != 'inputs'}, flush=True)
    base = arrays[0]
    comparisons = []
    for i, arr in enumerate(arrays):
        comparisons.append({'run': i,
            'ids_exact': np.array_equal(base['ids'], arr['ids']),
            'inputs_exact': input_contracts[0] == input_contracts[i],
            'probabilities_exact': np.array_equal(base['arm_probs'], arr['arm_probs']),
            'max_abs_probability_delta': float(np.max(np.abs(base['arm_probs'] - arr['arm_probs']))),
            'ranks_exact': np.array_equal(base['ranks'], arr['ranks'])})
    parity = all(all(c[k] for k in ['ids_exact', 'inputs_exact', 'probabilities_exact', 'ranks_exact'])
                 for c in comparisons)
    mean_a = (records[0]['seconds'] + records[3]['seconds']) / 2
    mean_b = (records[1]['seconds'] + records[2]['seconds']) / 2
    result = {'status': 'PASSED_EXACT_PARITY' if parity else 'FAILED_PARITY',
        'purpose': 'raptor_efficiency_only_no_auc_no_submission',
        'sequence': [r['mode'] for r in records], 'comparisons': comparisons,
        'serial_mean_seconds': mean_a, 'prefetch_mean_seconds': mean_b,
        'abba_speedup': mean_a / mean_b,
        'warm_serial_vs_prefetch_speedup': records[3]['seconds'] / mean_b,
        'eligible_for_full_stack_test': parity and min(mean_a, records[3]['seconds']) / mean_b >= 1.05,
        'deployment_authorized': False,
        'limitations': ['12 training studies, one worker/session; no confidence interval.',
                       'A1 includes colder caches; A4 is a warm comparison, ABBA does not erase all drift.',
                       'RSS is cumulative process maximum, not per-mode or full-ensemble peak.',
                       'Only Raptor measured, not the full ensemble; no AUC or new score.']}
    Path('/kaggle/working/e03_comparison.json').write_text(json.dumps(result, indent=2))
    print('E03_COMPARISON', result, flush=True)
    if not parity:
        raise RuntimeError('E03 input/prediction parity failed; do not promote')
