#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — comparação pareada de integridade, não AUC."""
import argparse
import csv
import json
from pathlib import Path

from scripts.h43_integrity import H43_TARGETS, h43_sha, h43_validate_rows, h43_require
from scripts.prepare_h43_probe22 import PROBE22


def read_predictions(path, ids):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        h43_validate_rows(rows, ids, reader.fieldnames)
    return rows


def paired_counts(parent, candidate):
    ids = [r['StudyInstanceUID'] for r in parent]
    columns = ['StudyInstanceUID'] + H43_TARGETS
    h43_validate_rows(parent, ids, columns)
    h43_validate_rows(candidate, ids, columns)
    counts = {t: sum(float(p[t]) != float(c[t]) for p, c in zip(parent, candidate))
              for t in H43_TARGETS}
    for target, count in counts.items():
        if target not in PROBE22:
            h43_require(count == 0, f'Unchanged target drift: {target}')
    return counts


def assess(directory, reference, sample):
    with sample.open(newline='') as stream:
        ids = [r['StudyInstanceUID'] for r in csv.DictReader(stream)]
    load = lambda name: json.loads((directory / name).read_text())
    receipt = load('h43_probe22_integrity.json')
    h43_require(receipt['status'] == 'PASSED_PROBE22_INTEGRITY', 'probe22 gate')
    h43_require(receipt['preset'] == 'probe22' and receipt['parent_submission_ref'] == 56253529,
                'wrong reference or preset')
    h43_require(receipt['run']['coatnet_w'] == PROBE22, 'outer weights')
    h43_require(set(receipt['stages']) == {'dino', 'a5', 'rad', 'raptor', 'coat'}, 'stages')
    pre = receipt['preflight']
    h43_require(pre['status'] == 'PASSED_ARTIFACT_PREFLIGHT' and not pre['errors'], 'preflight')
    h43_require(pre['artifact_lock_entries'] == 56 and pre['gpus'] == ['Tesla T4'] * 2,
                'hardware/lock')
    routing = load('probe22_outer_routing_receipt.json')
    expected = {t: PROBE22.get(t, .6) for t in H43_TARGETS}
    h43_require(routing['outer_raptor_weight_by_target'] == expected, 'routing map')
    h43_require(routing['inner_private_coat_alpha'] == .4
                and routing['inner_public_raptor_alpha'] == .6, 'inner blend changed')
    h43_require(routing['final_rank_after_outer_blend'] is True
                and routing['overlay_applied'] is False, 'rank/overlay')
    parent_path = directory / 'submission_0939_parent_exact.csv'
    final_path = directory / 'submission.csv'
    h43_require(h43_sha(parent_path) == routing['parent0939_sha256'], 'parent receipt hash')
    h43_require(h43_sha(parent_path) == h43_sha(reference), 'parent differs from confirmed run')
    h43_require(h43_sha(final_path) == routing['output_sha256'] == receipt['submission_sha256'],
                'candidate receipt hash')
    parent = read_predictions(parent_path, ids)
    candidate = read_predictions(final_path, ids)
    counts = paired_counts(parent, candidate)
    changed = [t for t in H43_TARGETS if t in PROBE22]
    h43_require(routing['changed_targets'] == changed, 'changed target list')
    h43_require(routing['changed_numeric_count_by_target'] == {t: counts[t] for t in changed},
                'paired counts')
    coat = load('coat_resgated_ep10_top3_submission_receipt.json')
    h43_require(coat['models'] == 3 and coat['fallback_studies'] == 0 and not coat['failures'],
                'CoAt coverage')
    h43_require(len(coat['processes']) == 2 and all(p['returncode'] == 0 for p in coat['processes']),
                'CoAt workers')
    h43_require(receipt['studies'] == coat['studies'] == routing['study_count'] == len(ids), 'counts')
    return {'status': 'PASSED_PAIRED_INTEGRITY', 'studies': len(ids),
            'parent_sha256': h43_sha(parent_path), 'candidate_sha256': h43_sha(final_path),
            'changed_numeric_count_by_target': counts, 'seven_unchanged_targets_match': True,
            'oof_independent': False, 'auc_measured': False,
            'limitation': 'Parity on visible examples is not hidden-test parity or accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
        default=Path('reports/avance_av004_submission_v1/submission.csv'))
    parser.add_argument('--sample', type=Path, default=Path('data/raw/sample_submission.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.directory, args.reference, args.sample)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
