"""#RSNA #Kaggle #Pesquisa — preregistered paired resolution experiment; no dispatch.

All patient/source/cache gates fail closed. Weak-label results are not a public
score or clinical validation. Confirmation is a separate, one-shot action.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.freeze_weak_validation import digest, TARGETS
from scripts.prepare_g04_preflight import V05, V05_SHA
from scripts.prepare_r02_training import FEATURE_SHA

RECIPE = {
    'name': 'G05_paired_resolution_v1', 'resolutions': [224, 336],
    'seeds': [2026, 42], 'epochs': 20, 'batch_size': 4,
    'learning_rate': .001, 'weight_decay': .0001,
    'encoder': 'official_dinov2_small_frozen_fp32_cls384',
    'model_sha256': '1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050',
    'config_sha256': '1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
    'pooling': 'StudyAttention384_64_tanh_1_classifier384_12',
    'loss': 'unweighted_mean_BCE_with_logits_frozen_soft_teacher_including_0.5',
    'optimizer': 'AdamW', 'epoch_selection': 'fixed_epoch20_no_dev_early_stopping',
    'geometry': 'same_series_physical_adjacent3_native_FOV_no_crop',
    'pixels': 'same_decoder_percentiles1_99_uint8_PIL_bilinear_only_size_changes',
    'encoder_batch_studies': 2, 'augmentation': 'none', 'precision': 'float32_TF32_off',
    'baseline_archive_sha256': FEATURE_SHA, 'v05_sha256': V05_SHA,
    'primary': 'paired_mean_soft_BCE_delta_336_minus224_averaged_over_seeds_and_targets',
    'bootstrap_replicates': 5000, 'bootstrap_seed': 20260930,
    'minimum_BCE_improvement': .001, 'maximum_target_BCE_regression': .01,
    'maximum_target_BCE_simultaneous_upper': .02,
    'maximum_target_AUC_regression': .02,
    'minimum_hard_positive_and_negative': 10,
    'resource_limits': {'gpu_session_seconds': 7200, 'minimum_remaining_quota_seconds': 14400,
                        'maximum_peak_allocated_bytes': 4 * 1024**3,
                        'maximum_forward336_over224': 3.0},
    'patient_gate': 'verified_stable_patient_key_all_studies_all_selected_series_disjoint',
    'promotion': 'same_gates_on_development_then_once_on_untouched_confirmation',
    'submission': 'manual_user_only_no_automatic_upload',
}


def contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def load_manifest():
    if digest(V05) != V05_SHA:
        raise ValueError('Pinned V05 manifest changed; never resplit silently')
    manifest = json.loads(V05.read_text())
    for name, count in [('train', 1000), ('development', 300), ('confirmation', 300)]:
        if len(manifest['splits'][name]) != count:
            raise ValueError('Frozen population changed')
    for path, expected in manifest['sources'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Manifest source hash changed')
    return manifest


def identity_gate(rows, evidence):
    """Use patient AND report connected components for split/CI grouping.

    Evidence requires a stable-key provenance attestation, not merely the
    existence of PatientID. Headers can be study-wise anonymized placeholders.
    """
    expected = {r['StudyInstanceUID']: (split, r['report_hash'], r['series'])
                for split in ('train', 'development', 'confirmation')
                for r in rows[split]}
    if len(expected) != sum(len(rows[s]) for s in ('train', 'development', 'confirmation')):
        raise ValueError('Repeated study identity across splits')
    found = evidence.get('studies', [])
    ids = [r['StudyInstanceUID'] for r in found]
    if set(ids) != set(expected) or len(ids) != len(set(ids)):
        raise ValueError('Patient evidence incomplete/duplicated')
    if (evidence.get('v05_sha256') != V05_SHA or
            evidence.get('stable_patient_key_verified') is not True or
            not evidence.get('stable_patient_key_source')):
        raise ValueError('Patient-key semantics unverified; report hashes are insufficient')
    parent = {uid: uid for uid in expected}
    def root(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid
    seen = {}
    for record in found:
        uid = record['StudyInstanceUID']
        selected = {s['series_uid'] for s in expected[uid][2]}
        if (not record.get('all_headers_consistent') or
                set(record.get('series_uids', [])) != selected or
                record.get('headers_verified', 0) < len(selected)):
            raise ValueError('Within-study patient/series audit incomplete')
        patient = record.get('patient_hash', '')
        if len(patient) != 64 or any(c not in '0123456789abcdef' for c in patient):
            raise ValueError('Missing/invalid patient key')
        for key in [('patient', patient), ('report', expected[uid][1])]:
            if key in seen:
                parent[root(uid)] = root(seen[key])
            else:
                seen[key] = uid
    ownership = {}
    groups = {}
    for uid, (split, _, _) in expected.items():
        group = root(uid)
        if group in ownership and ownership[group] != split:
            raise ValueError('Cross-split patient/report overlap; block, do not silently change splits')
        ownership[group] = split
        groups[uid] = contract_hash({'component': sorted(u for u in expected if root(u) == group)})
    return groups


def paired_metrics(logits, labels, groups, targets=TARGETS, replicates=None):
    """logits axes: resolution [224,336], seed [2026,42], study, condition.

    Resample studies as paired units, clustering every linked patient/report.
    No slices/seeds treated as independent patients. AUC on exact hard labels.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    g = np.asarray(groups, dtype=str)
    if (x.shape != (2, 2, len(y), len(targets)) or y.shape != (len(g), len(targets)) or
            len(targets) != 12 or len(set(targets)) != 12 or not len(y) or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any((y < 0) | (y > 1)) or any(not v for v in g)):
        raise ValueError('Invalid paired predictions/labels/groups')
    unique, inverse = np.unique(g, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('Insufficient independent clusters')
    loss = (np.logaddexp(0., x) - y[None, None] * x).mean(axis=1)
    # Resolution-level probabilities averaged over the SAME two fixed seeds.
    probabilities = (1 / (1 + np.exp(-np.clip(x, -700, 700)))).mean(axis=1)
    delta = loss[1] - loss[0]
    count = RECIPE['bootstrap_replicates'] if replicates is None else replicates
    if count < 100:
        raise ValueError('Insufficient bootstrap replicates')
    rng = np.random.default_rng(RECIPE['bootstrap_seed'])
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    bce = np.empty((count, 2, len(targets)))
    auc = np.full_like(bce, np.nan)
    for b in range(count):
        indices = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        bce[b] = loss[:, indices].mean(axis=1)
        for j in range(len(targets)):
            keep = (y[indices, j] == 0) | (y[indices, j] == 1)
            yy = y[indices, j][keep]
            if len(np.unique(yy)) == 2:
                for arm in range(2):
                    auc[b, arm, j] = roc_auc_score(yy, probabilities[arm, indices, j][keep])
    def interval(a, alpha=.05):
        a = np.asarray(a)
        valid = a[np.isfinite(a)]
        return {'ci': np.quantile(valid, [alpha/2, 1-alpha/2]).tolist() if len(valid) else None,
                'valid_replicates': len(valid), 'total_replicates': count,
                'confidence': 1-alpha}
    per = {}
    for j, target in enumerate(targets):
        keep = (y[:, j] == 0) | (y[:, j] == 1)
        yy = y[keep, j]
        values = [float(roc_auc_score(yy, probabilities[a, keep, j]))
                  if len(np.unique(yy)) == 2 else None for a in range(2)]
        per[target] = {
            'hard_positive': int((yy == 1).sum()), 'hard_negative': int((yy == 0).sum()),
            'soft_or_uncertain_excluded_from_auc': int((~keep).sum()),
            'bce224': float(loss[0, :, j].mean()), 'bce336': float(loss[1, :, j].mean()),
            'bce224_interval': interval(bce[:, 0, j]), 'bce336_interval': interval(bce[:, 1, j]),
            'bce_delta': float(delta[:, j].mean()), 'bce_delta_interval': interval(bce[:, 1, j]-bce[:, 0, j]),
            'bce_delta_simultaneous_interval': interval(bce[:, 1, j]-bce[:, 0, j], .05/len(targets)),
            'auc224': values[0], 'auc336': values[1],
            'auc_delta': values[1]-values[0] if None not in values else None,
            'auc224_interval': interval(auc[:, 0, j]), 'auc336_interval': interval(auc[:, 1, j]),
            'auc_delta_interval': interval(auc[:, 1, j]-auc[:, 0, j]),
            'low_power_warning': int((yy == 1).sum()) < 20 or int((yy == 0).sum()) < 20,
        }
    primary = interval((bce[:, 1]-bce[:, 0]).mean(axis=1))
    seed_deltas = (np.logaddexp(0., x)-y[None, None]*x).mean(axis=(2, 3))
    seed_deltas = (seed_deltas[1]-seed_deltas[0]).tolist()
    gates = {
        'minimum_mean_improvement': float(delta.mean()) <= -RECIPE['minimum_BCE_improvement'],
        'primary_ci95_upper_below_zero': primary['ci'][1] < 0,
        'both_seeds_improve': all(v < 0 for v in seed_deltas),
        'no_condition_bce_regression': all(v['bce_delta'] <= RECIPE['maximum_target_BCE_regression'] for v in per.values()),
        'condition_simultaneous_noninferiority': all(v['bce_delta_simultaneous_interval']['ci'][1] <= RECIPE['maximum_target_BCE_simultaneous_upper'] for v in per.values()),
        'condition_auc_safety': all(v['auc_delta'] is not None and v['auc_delta'] >= -RECIPE['maximum_target_AUC_regression'] for v in per.values()),
        'hard_label_coverage': all(min(v['hard_positive'], v['hard_negative']) >= RECIPE['minimum_hard_positive_and_negative'] for v in per.values()),
        'auc_bootstrap_coverage': all(v['auc_delta_interval']['valid_replicates'] >= .95*count for v in per.values()),
    }
    return {'reference': 'frozen_weak_teacher_not_clinical_or_Kaggle_score',
            'studies': len(y), 'patient_report_clusters': len(unique),
            'mean_soft_bce224': float(loss[0].mean()), 'mean_soft_bce336': float(loss[1].mean()),
            'primary_delta': float(delta.mean()), 'primary_interval': primary,
            'delta_by_seed': dict(zip(map(str, RECIPE['seeds']), seed_deltas)),
            'per_condition': per, 'quality_gates': gates, 'quality_passed': all(gates.values()),
            'macro_auc224': float(np.mean([v['auc224'] for v in per.values()])) if all(v['auc224'] is not None for v in per.values()) else None,
            'macro_auc336': float(np.mean([v['auc336'] for v in per.values()])) if all(v['auc336'] is not None for v in per.values()) else None,
            'regressions': [t for t, v in per.items() if v['bce_delta'] > 0 or v['auc_delta'] is None or v['auc_delta'] < 0],
            'bootstrap_seed': RECIPE['bootstrap_seed'], 'submission_eligible': False}


def prepare(output):
    manifest = load_manifest()
    files = [Path('reports/avance_av035_g04/audit_v2.json'),
             Path('reports/avance_av024_v03_v1/v03_features.npz'),
             Path('reports/avance_av035_g04/v2_output/g04_features.npz'),
             Path('reports/research_20260923/maverick-v3/rsna-knee-restructured-version-3.ipynb'),
             Path('reports/avance_av034_h46_exact/essential/submission.csv')]
    expected = ['0b67b9356f934785f6ea979bfabe12d5372809399f44f5a023a7895b9ae5b922', FEATURE_SHA,
                '3175b1a08d08a80fd371e8c9c10ed1dde2fd4d93b0ed27d1215764dfe45d9948',
                '7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522',
                '7c6dfe8ba6c71d557d2a6b21af8a96bddfeb4b75626ee38a7a8cae44ec80c23d']
    pins = {str(p): digest(p) for p in files}
    for p, h in zip(files, expected):
        if h is not None and pins[str(p)] != h:
            raise ValueError('Baseline/cache changed')
    audit = json.loads(files[0].read_text())
    if audit['status'] != 'PASSED_G04_PREFLIGHT_NOT_MODEL_VALIDATION':
        raise ValueError('G04 not reconciled')
    for p in [Path(__file__), Path('scripts/run_resolution_heads.py'),
              Path('scripts/v02_pilot_runtime.py')]:
        pins[str(p.relative_to(Path.cwd()) if p.is_absolute() else p)] = digest(p)
    contract = {'recipe': RECIPE, 'targets': manifest['targets'], 'pins': pins,
                'sizes': {s: len(manifest['splits'][s]) for s in ('train', 'development', 'confirmation')},
                'status': 'PREREGISTERED_NOT_TRAINED_PATIENT_AND_PIXEL_GATES_PENDING',
                'h46_reference': {'submission_ref': 56696639, 'public_score': .943},
                'no_auto_submit': True, 'confirmation_evaluated': False}
    contract['contract_hash'] = contract_hash(contract)
    freeze(output, contract)
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.output)
    print(json.dumps({'status': value['status'], 'contract_hash': value['contract_hash'],
                      'output_sha256': digest(args.output)}, indent=2))
