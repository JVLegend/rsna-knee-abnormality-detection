"""#RSNA #Kaggle #Pesquisa — validate stable candidate before a single code submission."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.assess_h43_ordered_smoke import validate_components, read_rows
from scripts.assess_h43_probe22 import paired_counts
from scripts.dino_rank_replay import rank_columns
from scripts.h43_integrity import H43_TARGETS, h43_sha
from scripts.prepare_h43_probe22 import PROBE22, literal_assignment
from scripts.prepare_h43_ordered_probe22 import PURPOSE, SMOKE_SHA


def check_receipts(receipt, provenance, run, ids):
    if (receipt.get('status') != 'PASSED_ORDERED_PROBE22_INTEGRITY'
            or receipt.get('purpose') != PURPOSE or receipt.get('preset') != 'probe22'
            or receipt.get('run') != run or run['coatnet_w'] != PROBE22
            or receipt.get('historical_probe22_ref') != 56263721
            or set(receipt.get('stages', [])) != {'dino','a5','rad','raptor','coat'}
            or receipt.get('studies') != len(ids)):
        raise ValueError('Candidate integrity/recipe mismatch')
    if (provenance.get('status') != 'ORDERED_PROBE22_READY_FOR_REVIEW'
            or provenance.get('purpose') != PURPOSE or provenance.get('source_smoke_sha256') != SMOKE_SHA
            or provenance.get('studies') != len(ids) or provenance.get('automatically_submitted') is not False
            or not str(provenance.get('official_root', '')).startswith('/kaggle/input/')):
        raise ValueError('Candidate provenance mismatch')
    pre = receipt['preflight']
    if (pre['status'] != 'PASSED_ARTIFACT_PREFLIGHT' or pre['errors']
            or pre['artifact_lock_entries'] != 56 or pre['gpus'] != ['Tesla T4']*2):
        raise ValueError('Candidate preflight failed')


def replay_fusion(transformer, hybrid, weights):
    if transformer.shape != hybrid.shape or transformer.ndim != 2 or transformer.shape[1] != 12:
        raise ValueError('Invalid fusion shapes')
    if any(not np.isfinite(x).all() or not ((x>=0)&(x<=1)).all() for x in [transformer,hybrid]):
        raise ValueError('Invalid fusion ranks')
    w = np.array([weights.get(t,weights['__default__']) for t in H43_TARGETS])
    return np.asarray(rank_columns(((1-w)*transformer+w*hybrid).tolist(), pct=True))


def compare_raw(reference, candidate):
    for name in ['dino_stable_inputs.npz','e03_run0_raw.npz','coat_resgated_ep10_top3_predictions.npz']:
        with np.load(reference/name,allow_pickle=False) as a, np.load(candidate/name,allow_pickle=False) as b:
            if set(a.files) != set(b.files): raise ValueError('Raw capture schema drift')
            if name == 'dino_stable_inputs.npz':
                def canonical(x):
                    return x['predictions'][np.argsort(x['member_ids'])][:,np.argsort(x['study_ids'])]
                equal = (sorted(a['member_ids'].tolist())==sorted(b['member_ids'].tolist())
                         and sorted(a['study_ids'].tolist())==sorted(b['study_ids'].tolist())
                         and np.array_equal(canonical(a),canonical(b)))
            else:
                equal = all(np.array_equal(a[k],b[k]) for k in a.files)
            if not equal: raise ValueError(f'Raw predictions differ from smoke: {name}')


def assess(directory, sample, series_file, build_file):
    from scripts.assess_h43_ordered_smoke import assess as assess_smoke
    from scripts.prepare_h43_ordered_probe22 import build
    smoke_dir = Path('reports/avance_av015_smoke_v1')
    smoke = assess_smoke(smoke_dir, sample, series_file)  # Recheck evidence, not a cached success flag.
    expected_build = build(Path('reports/avance_av015_build/h43_ordered_official_smoke_v1.ipynb').read_bytes(), smoke)
    n = json.loads(build_file.read_text())
    if n != expected_build: raise ValueError('Candidate build drift')
    run = literal_assignment(n['cells'][5]['source'], 'RUN')
    with sample.open(newline='') as f: ids = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
    with series_file.open(newline='') as f: series = list(csv.DictReader(f))
    load = lambda name: json.loads((directory/name).read_text())
    receipt, provenance = load('h43_ordered_probe22_integrity.json'), load('ordered_probe22_receipt.json')
    check_receipts(receipt, provenance, run, ids)
    for name in ['smoke_predictions.csv','benchmark_predictions.csv','h43_benchmark_selection.json']:
        if (directory/name).exists(): raise ValueError('Smoke/training artifact in candidate')
    final, parent = directory/'submission.csv', directory/'submission_0939_parent_exact.csv'
    rows, parent_rows = read_rows(final, ids), read_rows(parent, ids)
    if h43_sha(final) != receipt['submission_sha256'] or h43_sha(final) != provenance['prediction_sha256']:
        raise ValueError('Final hash mismatch')
    if h43_sha(parent) != smoke['prediction_sha256']: raise ValueError('Parent route changed vs stable smoke')
    result = validate_components(directory, ids, len(series), receipt)
    # Every intermediate CSV remains unchanged; only outer routing may differ.
    names = ['_raptor.csv','_coat_arm.csv','submission_public_0899.csv','submission_e10_v2.csv',
             'submission_native_v38.csv','submission_legacy_fold_blend.csv','legacy_fold_diagnostics.csv']
    if any(h43_sha(directory/f) != h43_sha(smoke_dir/f) for f in names):
        raise ValueError('Component CSV differs from approved smoke')
    compare_raw(smoke_dir,directory)
    counts = paired_counts(parent_rows, rows)
    routing = load('probe22_outer_routing_receipt.json')
    expected = {t:PROBE22.get(t,.6) for t in H43_TARGETS}
    changed = [t for t in H43_TARGETS if t in PROBE22]
    if (routing['outer_raptor_weight_by_target'] != expected
            or routing['inner_private_coat_alpha'] != .4 or routing['inner_public_raptor_alpha'] != .6
            or routing['final_rank_after_outer_blend'] is not True or routing['overlay_applied'] is not False
            or routing['parent0939_sha256'] != h43_sha(parent) or routing['output_sha256'] != h43_sha(final)
            or routing['study_count'] != len(ids) or routing['changed_targets'] != changed
            or routing['changed_numeric_count_by_target'] != {t:counts[t] for t in changed}):
        raise ValueError('Outer routing receipt failed')
    capture = directory/'ordered_probe22_fusion_inputs.npz'
    if h43_sha(capture) != provenance['fusion_inputs_sha256']: raise ValueError('Fusion capture hash mismatch')
    with np.load(capture, allow_pickle=False) as a:
        if a['ids'].tolist() != ids: raise ValueError('Fusion ID mismatch')
        for weights, output in [(PROBE22,rows),({'__default__':.6},parent_rows)]:
            actual = np.asarray([[float(r[t]) for t in H43_TARGETS] for r in output])
            if not np.array_equal(replay_fusion(a['transformer'],a['hybrid'],weights),actual):
                raise ValueError('Independent outer fusion replay failed')
    return dict(result,status='PASSED_ORDERED_PROBE22_CANDIDATE',candidate_sha256=h43_sha(final),
                build_sha256=h43_sha(build_file),components_match_smoke=True,outer_fusion_replay=True,
                raw_predictions_match_smoke=True,
                changed_numeric_count_by_target=counts,hidden_score_measured=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    r=assess(a.directory,Path('data/raw/sample_submission.csv'),Path('data/raw/test_series.csv'),a.build)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))


if __name__ == '__main__':main()
