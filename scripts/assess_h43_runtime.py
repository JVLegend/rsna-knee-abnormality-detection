#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — cenários de runtime, nunca autorização de envio."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.h43_integrity import h43_sha, h43_validate_rows

STAGES = {'dino', 'a5', 'rad', 'raptor_coat_fusion'}


def scenarios(timings, studies, preflight_seconds, sizes=(1322, 2000)):
    if studies < 30 or set(timings) != STAGES:
        raise ValueError('Need representative benchmark >=30 studies and every timed stage')
    if any(not math.isfinite(v) or v <= 0 for v in timings.values()):
        raise ValueError('Invalid stage time')
    if not math.isfinite(preflight_seconds) or preflight_seconds < 0:
        raise ValueError('Invalid preflight time')
    if any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError('Invalid scenario size')
    rate = sum(timings.values()) / studies
    return {
        'seconds_per_study_including_stage_loads': rate,
        'scenarios_are_assumptions_not_actual_hidden_test_size': True,
        'planning_limit_hours_needs_rule_revalidation': 9,
        'time_inflation_factor': 1.25,
        'extra_fixed_seconds': 120,
        'cases': [{
            'studies_assumed': n,
            'estimated_hours_with_margin': (rate * n + preflight_seconds + 120) * 1.25 / 3600,
            'fits_historical_9h_planning_limit': (rate * n + preflight_seconds + 120) * 1.25 <= 9 * 3600,
            'raw_dino_cache_gib_only': n * 6 * 12 * 336 * 336 / 2**30,
            'stage_hours_without_margin': {k: v * n / studies / 3600 for k, v in timings.items()},
        } for n in sizes],
        'submission_authorized_by_this_report': False,
        'limitations': [
            'Systematic sample by series count, not a probabilistic bound for hidden cases.',
            'Stage loads are scaled with N too; conservative but throughput can be nonlinear.',
            'Cache estimate excludes model, decoder, workers and other arrays; not measured peak RAM.',
            'No score, no model selection and no independent OOF evaluation.',
        ],
    }


def assess(directory):
    receipt = json.loads((directory / 'h43_parent_integrity.json').read_text())
    selection = json.loads((directory / 'h43_benchmark_selection.json').read_text())
    timings = json.loads((directory / 'h43_benchmark_timings.json').read_text())
    if receipt.get('status') != 'PASSED_PARENT_INTEGRITY' or receipt.get('purpose') != 'runtime_benchmark_only_no_auc_no_submission':
        raise ValueError('Not a successful runtime-only benchmark')
    if set(receipt.get('stages', [])) != {'dino', 'a5', 'rad', 'raptor', 'coat'}:
        raise ValueError('Missing ensemble stage')
    preflight = receipt['preflight']
    if preflight.get('artifact_lock_entries') != 56 or preflight.get('gpus') != ['Tesla T4', 'Tesla T4']:
        raise ValueError('Hardware or artifact contract differs')
    pred = directory / 'benchmark_predictions.csv'
    if h43_sha(pred) != receipt['submission_sha256']:
        raise ValueError('Prediction hash mismatch')
    with pred.open(newline='') as handle:
        reader = csv.DictReader(handle)
        h43_validate_rows(list(reader), selection['ids'], reader.fieldnames)
    if receipt['studies'] != len(selection['ids']) or receipt.get('timings_seconds') != timings:
        raise ValueError('Count/timing receipt mismatch')
    result = scenarios(timings, receipt['studies'], preflight['elapsed_seconds'])
    result.update(studies=receipt['studies'], series=selection['series'],
                  timings_seconds=timings, benchmark_prediction_sha256=h43_sha(pred))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.directory)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
