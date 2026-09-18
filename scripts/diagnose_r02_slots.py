"""#RSNA #Kaggle #Pesquisa — CPU-only, frozen-checkpoint missing-plane stress."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.assess_v02_baseline import loss, replay
from scripts.freeze_weak_validation import digest, freeze
from scripts.prepare_v03_scale import PLANES, literal

THRESHOLD = 0.01  # Operational priority, not statistical significance.


def stress(features, labels, state, targets):
    """Remove a slot from attention, or zero its vector but retain attention.

    The shared head has no positional term; selecting the two present slots
    is equivalent to applying its boolean presence mask before softmax.
    Neither scenario measures real acquisition noise or incomplete patients.
    """
    x = np.asarray(features)
    y = np.asarray(labels)
    if (x.ndim != 3 or x.shape[1:] != (3, 384) or not len(x)
            or y.shape != (len(x), 12) or len(targets) != 12
            or len(set(targets)) != 12 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any((y < 0) | (y > 1))):
        raise ValueError('Invalid stress inputs')
    intact = replay(x, state)
    base = loss(intact, y).mean(axis=0)
    results = []
    for mode in ['missing_masked', 'zero_present']:
        for i, plane in enumerate(PLANES):
            if mode == 'missing_masked':
                changed = x[:, [j for j in range(3) if j != i], :]
            else:
                changed = x.copy()
                changed[:, i, :] = 0
            per = loss(replay(changed, state), y).mean(axis=0)
            results.append({
                'scenario': mode, 'plane': plane,
                'mean_soft_bce': float(per.mean()),
                'delta_vs_intact': float((per-base).mean()),
                'per_target_delta': dict(zip(targets, (per-base).tolist())),
            })
    return intact, results


def priority(runs):
    if [r['seed'] for r in runs] != [2026, 42]:
        raise ValueError('Expected both frozen seeds')
    expected = [(mode, plane) for mode in ['missing_masked', 'zero_present']
                for plane in PLANES]
    for run in runs:
        if [(s['scenario'], s['plane']) for s in run['scenarios']] != expected:
            raise ValueError('Incomplete stress matrix')
        if not all(np.isfinite(s['delta_vs_intact']) for s in run['scenarios']):
            raise ValueError('Nonfinite stress delta')
    planes = [plane for plane in PLANES if all(
        next(s['delta_vs_intact'] for s in r['scenarios']
             if s['scenario'] == 'missing_masked' and s['plane'] == plane) > THRESHOLD
        for r in runs)]
    return {'threshold': THRESHOLD, 'planes_above_in_both_seeds': planes,
            'prioritize_slot_dropout_trial': bool(planes)}


def diagnose(directory, manifest, audit, build, output):
    """Consume private, already-audited V03 artifacts; never retrain or submit."""
    if output.exists():
        raise FileExistsError(output)
    a = json.loads(audit.read_text())
    r = json.loads((directory/'v03_receipt.json').read_text())
    m = json.loads(manifest.read_text())
    contract = hashlib.sha256(json.dumps(r['spec'], sort_keys=True).encode()).hexdigest()
    if (a['status'] != 'PASSED_V03_AUDIT' or a['selected_reference'] != 'expanded'
            or a['build_sha256'] != digest(build) or r['spec']['manifest'] != m
            or r['spec'] != literal(build.read_text(), 'V03')
            or r['spec']['manifest_sha256'] != digest(manifest)
            or r['status'] != 'COMPLETE_V03_NOT_SUBMISSION'
            or contract != r['contract_hash']
            or any(d['confirmation_evaluated'] or d['submission_eligible'] for d in [a, r])):
        raise ValueError('Require audited expanded V03, unchanged scope and provenance')
    path = directory/'v03_features.npz'
    if digest(path) != r['feature_sha256']:
        raise ValueError('Feature archive changed')
    ids = [row['StudyInstanceUID'] for row in m['development']]
    if len(ids) != 250 or len(set(ids)) != 250:
        raise ValueError('Development identity changed')
    if any([s['plane'] for s in row['series']] != PLANES for row in m['development']):
        raise ValueError('Plane order changed')
    with np.load(path, allow_pickle=False) as data:
        if data['development_ids'].tolist() != ids or str(data['contract_hash']) != contract:
            raise ValueError('Feature IDs/contract changed')
        features = data['development'].copy()
    labels = np.asarray([row['labels'] for row in m['development']], dtype=np.float32)
    runs = []
    torch.set_num_threads(1)
    for seed in [2026, 42]:
        receipts = [s for s in r['seeds'] if (s['arm'], s['seed']) == ('expanded', seed)]
        audited = [s for s in a['results'] if (s['arm'], s['seed']) == ('expanded', seed)]
        if len(receipts) != 1 or len(audited) != 1:
            raise ValueError('Seed inventory changed')
        s, previous = receipts[0], audited[0]
        cp = directory/'expanded'/f'v02_seed{seed}_best.pt'
        pred = directory/'expanded'/f'v02_seed{seed}_development.npz'
        if digest(cp) != s['best_checkpoint_sha256'] or digest(pred) != s['prediction_sha256']:
            raise ValueError('Checkpoint/predictions changed')
        state = torch.load(cp, map_location='cpu', weights_only=True)
        if (state['seed'], state['epoch'], state['fingerprint']) != (seed, s['best_epoch'], s['fingerprint']):
            raise ValueError('Checkpoint identity changed')
        intact, scenarios = stress(features, labels, state['model'], m['targets'])
        with np.load(pred, allow_pickle=False) as data:
            observed = data['best_logits']
            if (data['ids'].tolist() != ids or observed.shape != intact.shape
                    or not np.allclose(intact, observed, atol=1e-4, rtol=1e-5)):
                raise ValueError('Intact replay failed')
            replay_delta = float(np.abs(intact-observed).max())
        bce = float(loss(intact, labels).mean())
        if abs(bce-previous['mean_soft_bce']) > 2e-6:
            raise ValueError('Intact metric changed')
        runs.append({'seed': seed, 'selected_epoch': s['best_epoch'],
                     'checkpoint_sha256': digest(cp), 'intact_soft_bce': bce,
                     'intact_replay_max_delta': replay_delta, 'scenarios': scenarios})
    result = {
        'status': 'COMPLETE_R02_SYNTHETIC_DIAGNOSTIC_NOT_TRAINING',
        'source_sha256': digest(Path(__file__)), 'audit_sha256': digest(audit),
        'manifest_sha256': digest(manifest), 'build_sha256': digest(build),
        'feature_sha256': r['feature_sha256'], 'contract_hash': contract,
        'development_studies': 250, 'planes': PLANES, 'runs': runs,
        'decision': priority(runs), 'confirmation_evaluated': False,
        'submission_eligible': False, 'training_performed': False,
        'limitations': [
            'Synthetic removal on complete studies, not real incomplete-protocol validation.',
            'Zeroing features is not image corruption and is not correct missing-slot handling.',
            'Weak teacher agreement on development used for epoch selection; not clinical or Kaggle score.',
            'Operational threshold, not significance. Dropout training remains untested.',
        ],
    }
    freeze(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['directory', 'manifest', 'audit', 'build', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.directory, args.manifest, args.audit, args.build, args.output)
