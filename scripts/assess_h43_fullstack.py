#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — paridade e custo do benchmark completo E03."""
import argparse
import csv
import json
import math
from pathlib import Path

from scripts.assess_h43_runtime import assess as assess_runtime
from scripts.h43_integrity import H43_TARGETS, h43_sha, h43_validate_rows


def compare_rows(reference, candidate, ids):
    columns = ['StudyInstanceUID'] + H43_TARGETS
    h43_validate_rows(reference, ids, columns)
    h43_validate_rows(candidate, ids, columns)
    return {t: {'changed_values': sum(float(a[t]) != float(b[t]) for a, b in zip(reference, candidate)),
                'max_abs_delta': max(abs(float(a[t]) - float(b[t])) for a, b in zip(reference, candidate))}
            for t in H43_TARGETS}


def compare_times(reference, candidate):
    if set(reference) != set(candidate) or not reference:
        raise ValueError('Stage sets differ')
    if any(not math.isfinite(v) or v <= 0 for d in [reference, candidate] for v in d.values()):
        raise ValueError('Invalid stage time')
    return {k: {'reference_seconds': reference[k], 'candidate_seconds': candidate[k],
                'time_reduction_pct': 100 * (1 - candidate[k] / reference[k])}
            for k in reference} | {'total': {'reference_seconds': sum(reference.values()),
                'candidate_seconds': sum(candidate.values()),
                'time_reduction_pct': 100 * (1 - sum(candidate.values()) / sum(reference.values()))}}


def compare_components(reference, candidate):
    names = {p.name for p in reference.glob('*.csv')} & {p.name for p in candidate.glob('*.csv')}
    if not {'_raptor.csv', '_coat_arm.csv', 'submission_public_0899.csv'} <= names:
        raise ValueError('Missing component outputs')
    result = {}
    for name in sorted(names):
        with (reference / name).open(newline='') as f: a = list(csv.DictReader(f))
        with (candidate / name).open(newline='') as f: b = list(csv.DictReader(f))
        result[name] = {'reference_sha256': h43_sha(reference / name),
            'candidate_sha256': h43_sha(candidate / name),
            'byte_identical': (reference / name).read_bytes() == (candidate / name).read_bytes(),
            'differences': compare_rows(a, b, [r['StudyInstanceUID'] for r in a])}
    return result


def assess(candidate, reference):
    # Mantém os gates de hash/schema/hardware/stages já usados na AV-004.
    new, old = assess_runtime(candidate), assess_runtime(reference)
    load = lambda directory, name: json.loads((directory / name).read_text())
    selection = load(candidate, 'h43_benchmark_selection.json')
    baseline_selection = load(reference, 'h43_benchmark_selection.json')
    if selection != baseline_selection or len(selection['ids']) != 36 or selection['series'] != 205:
        raise ValueError('Not the same 36-study / 205-series benchmark')
    ids = selection['ids']
    def read(directory):
        with (directory / 'benchmark_predictions.csv').open(newline='') as stream:
            return list(csv.DictReader(stream))
    differences = compare_rows(read(reference), read(candidate), ids)
    exact_csv = new['benchmark_prediction_sha256'] == old['benchmark_prediction_sha256']
    inputs = load(candidate, 'e03_fullstack_inputs.json')
    expected_pairs = [(recipe, uid) for recipe in range(3) for uid in ids]
    if [(r['recipe'], r['uid']) for r in inputs] != expected_pairs:
        raise ValueError('Missing/duplicate/out-of-order recipe-study input')
    expected_shapes = [[64, 336, 336], [64, 384, 384], [44, 384, 384]]
    for r in inputs:
        shape = expected_shapes[r['recipe']]
        if r['shape'] != shape or r['volume_bytes'] != math.prod(shape):
            raise ValueError('Raptor tensor recipe drift')
    for directory in [reference, candidate]:
        coat = load(directory, 'coat_resgated_ep10_top3_submission_receipt.json')
        if coat['models'] != 3 or coat['studies'] != 36 or coat['fallback_studies'] != 0 or coat['failures']:
            raise ValueError('CoAt coverage failed')
        if len(coat['processes']) != 2 or any(r['returncode'] != 0 for r in coat['processes']):
            raise ValueError('CoAt workers failed')
    old_pre = load(reference, 'h43_preflight.json')
    new_pre = load(candidate, 'h43_preflight.json')
    pins = lambda p: sorted((Path(r['path']).name, r['sha256']) for r in p['files'].values())
    if pins(old_pre) != pins(new_pre):
        raise ValueError('Mounted artifact hashes differ')
    times = compare_times(old['timings_seconds'], new['timings_seconds'])
    return {'status': 'PASSED_FULLSTACK_PARITY' if exact_csv else 'FAILED_EXACT_CSV_PARITY',
            'exact_csv': exact_csv, 'reference_sha256': old['benchmark_prediction_sha256'],
            'candidate_sha256': new['benchmark_prediction_sha256'],
            'studies': 36, 'series': 205, 'recipe_study_pairs': len(inputs),
            'differences': differences, 'times': times, 'candidate_runtime_scenarios': new['cases'],
            'eligible_for_real_test_smoke': exact_csv and times['total']['time_reduction_pct'] > 0,
            'automatic_submission_authorized': False,
            'limitations': ['Different worker executions: timing differences are not all attributable to prefetch.',
                           '36 training studies, no new AUC, no hidden-test runtime measured.',
                           'Exact final predictions on this sample do not guarantee parity on all hidden cases.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('reports/avance_av004_benchmark_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--component-reference', type=Path)
    p.add_argument('--component-candidate', type=Path)
    args = p.parse_args()
    result = assess(args.candidate, args.reference)
    if bool(args.component_reference) != bool(args.component_candidate):
        raise ValueError('Provide both component directories')
    if args.component_reference:
        result['components'] = compare_components(args.component_reference, args.component_candidate)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
