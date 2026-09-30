"""#RSNA #Kaggle #Pesquisa — independent NumPy replay and preregistered decision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_m01_ablation import assemble, digest, literal
from scripts.assess_v02_baseline import loss, replay as shared_replay


def replay(features, state, architecture):
    if architecture == 'shared':
        return shared_replay(features, state)
    expected = {'classifier.weight', 'classifier.bias'}
    if architecture == 'target':
        expected |= {'score.0.weight', 'score.0.bias', 'score.2.weight', 'score.2.bias'}
    elif architecture != 'mean':
        raise ValueError('Unknown architecture')
    if set(state) != expected:
        raise ValueError('Unexpected parameters')
    w = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in state.items()}
    if any(not np.isfinite(v).all() for v in w.values()):
        raise ValueError('Nonfinite weights')
    x = np.asarray(features, dtype=np.float64)
    if architecture == 'mean':
        return x.mean(axis=1) @ w['classifier.weight'].T + w['classifier.bias']
    scores = np.tanh(x @ w['score.0.weight'].T + w['score.0.bias']) @ w['score.2.weight'].T + w['score.2.bias']
    attention = np.exp(scores - scores.max(axis=1, keepdims=True))
    attention /= attention.sum(axis=1, keepdims=True)
    pooled = np.einsum('bpc,bpd->bcd', attention, x)
    return (pooled * w['classifier.weight'][None]).sum(axis=-1) + w['classifier.bias']


def decide(results):
    values = {(r['architecture'], r['seed']): r['mean_soft_bce'] for r in results}
    if len(values) != 6 or len(results) != 6 or not np.isfinite(list(values.values())).all():
        raise ValueError('Incomplete result matrix')
    eligible = [a for a in ['mean', 'target']
                if all(values[a, s] < values['shared', s] - 2e-6 for s in [2026, 42])]
    return min(eligible, key=lambda a: (np.mean([values[a, s] for s in [2026, 42]]), a != 'mean')) if eligible else 'shared'


def assess(directory, build, baseline, output):
    source = build.read_text()
    if source != assemble():
        raise ValueError('Build/source changed')
    spec = literal(source, 'M01')
    payload = literal(source, 'V02_BASELINE')
    contract = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    r = json.loads((directory / 'm01_receipt.json').read_text())
    if (r['status'] != 'COMPLETE_M01_NOT_SUBMISSION' or r['contract_hash'] != contract
            or r['spec'] != spec or r['gpu_names'] != ['Tesla T4'] * 2
            or r['confirmation_evaluated'] or r['submission_eligible']):
        raise ValueError('Receipt contract/scope drift')
    expected = [(a, s) for a in spec['architectures'] for s in [2026, 42]]
    if [(s['architecture'], s['seed']) for s in r['seeds']] != expected:
        raise ValueError('Run inventory drift')
    archive = baseline / 'v02_baseline_features.npz'
    if digest(archive) != spec['feature_sha256']:
        raise ValueError('Local audited feature archive changed')
    with np.load(archive, allow_pickle=False) as a:
        features = a['features'][299:]
        ids = a['ids'][299:].tolist()
    dev = payload['splits']['development']
    if ids != [d['StudyInstanceUID'] for d in dev]:
        raise ValueError('Development identity drift')
    labels = np.asarray([d['labels'] for d in dev], dtype=np.float32)
    base_receipt = json.loads((baseline / 'v02_baseline_receipt.json').read_text())
    results = []
    for s in r['seeds']:
        arch, seed = s['architecture'], s['seed']
        history = s['history']
        scores = [h['development']['mean_soft_bce'] for h in history]
        if (s['epochs'] != 20 or [h['epoch'] for h in history] != list(range(1, 21))
                or not np.isfinite(scores).all()
                or any(not np.isfinite(h['train_soft_bce']) or h['train_soft_bce'] < 0 for h in history)):
            raise ValueError('Incomplete/nonfinite training history')
        best = int(np.argmin(scores)) + 1
        if s['best_epoch'] != best or abs(s['best_metrics']['mean_soft_bce'] - scores[best-1]) > 2e-6:
            raise ValueError('Selection rule drift')
        fp = hashlib.sha256(f'{contract}:{arch}:{seed}'.encode()).hexdigest()
        if fp != s['fingerprint']:
            raise ValueError('Training fingerprint drift')
        folder = directory / arch
        prediction = folder / f'v02_seed{seed}_development.npz'
        if digest(prediction) != s['prediction_sha256']:
            raise ValueError('Predictions changed')
        with np.load(prediction, allow_pickle=False) as a:
            if a['ids'].tolist() != ids:
                raise ValueError('Prediction identity drift')
            predictions = {k: a[k+'_logits'].copy() for k in ['best', 'last']}
        deltas = {}
        for kind, epoch in [('best', best), ('last', 20)]:
            checkpoint = folder / f'v02_seed{seed}_{kind}.pt'
            if digest(checkpoint) != s[kind+'_checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if (state['fingerprint'], state['seed'], state['epoch']) != (fp, seed, epoch):
                raise ValueError('Checkpoint metadata drift')
            if kind == 'last' and (state['history'] != history or not state['optimizer']['state']):
                raise ValueError('Incomplete resume state')
            observed = predictions[kind]
            predicted = replay(features, state['model'], arch)
            if observed.shape != labels.shape or not np.allclose(predicted, observed, atol=1e-4, rtol=1e-5):
                raise ValueError('Independent head replay failed')
            deltas[kind] = float(np.max(np.abs(predicted-observed)))
            if abs(float(loss(observed, labels).mean()) - scores[epoch-1]) > 2e-6:
                raise ValueError('Independent loss replay failed')
        if arch == 'shared':
            previous = next(z for z in base_receipt['seeds'] if z['seed'] == seed)
            prior_path = baseline / f'v02_seed{seed}_development.npz'
            if digest(prior_path) != previous['prediction_sha256'] or best != previous['best_epoch']:
                raise ValueError('Baseline reference drift/epoch mismatch')
            with np.load(prior_path, allow_pickle=False) as a:
                for kind in ['best', 'last']:
                    if not np.allclose(predictions[kind], a[kind+'_logits'], atol=1e-4, rtol=1e-5):
                        raise ValueError('Shared control did not reproduce baseline')
            if not np.allclose(scores, [h['development']['mean_soft_bce'] for h in previous['history']], atol=2e-6, rtol=0):
                raise ValueError('Shared control learning curve changed')
        per_target = loss(predictions['best'], labels).mean(axis=0)
        if not np.allclose(per_target, s['best_metrics']['per_target_soft_bce'], atol=2e-6, rtol=0):
            raise ValueError('Per-target metrics changed')
        results.append({'architecture': arch, 'seed': seed, 'selected_epoch': best,
                        'mean_soft_bce': float(per_target.mean()),
                        'last_soft_bce': float(loss(predictions['last'], labels).mean()),
                        'per_target_soft_bce': dict(zip(payload['targets'], per_target.tolist())),
                        'head_replay_max_absolute_deltas': deltas})
    out = {'status': 'PASSED_M01_AUDIT', 'build_sha256': digest(build), 'results': results,
           'selected_reference': decide(results), 'shared_control_reproduced': True,
           'confirmation_evaluated': False, 'submission_eligible': False,
           'seconds': r['seconds'], 'cuda_peak_allocated_bytes': r['cuda_peak_allocated_bytes'],
           'limitations': ['Weak teacher agreement, not clinical or leaderboard performance.',
                          'Development selects epochs/architecture; not an untouched final test.',
                          'Architecture parameter counts and initialization differ.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['directory', 'build', 'baseline', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assess(a.directory, a.build, a.baseline, a.output)
