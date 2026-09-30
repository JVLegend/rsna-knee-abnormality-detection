"""#RSNA #Kaggle #Pesquisa — paired frozen heads; no network/job/submission API.

Runs only on verified features and patient identities, after preregistration.
Fit does not select epochs using development and cannot read confirmation.
The confirmation stage must receive the audited development decision first.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch import nn

from scripts.resolution_comparison import (RECIPE, contract_hash, digest, freeze,
                                         identity_gate, load_manifest, paired_metrics)
from scripts.v02_pilot_runtime import StudyAttention


def read_protocol(path):
    value = json.loads(path.read_text())
    expected = value['contract_hash']
    if contract_hash({k: v for k, v in value.items() if k != 'contract_hash'}) != expected or value['recipe'] != RECIPE:
        raise ValueError('Preregistered contract drift')
    for source, sha in value['pins'].items():
        if digest(Path(source)) != sha:
            raise ValueError('Pinned code/cache/source changed')
    return value


def read_features(archive, receipt, protocol, rows):
    r = json.loads(receipt.read_text())
    if (r.get('status') != 'VERIFIED_G05_PAIRED_FEATURES' or
            r.get('archive_sha256') != digest(archive) or
            r.get('contract_hash') != protocol['contract_hash'] or
            r.get('model_sha256') != RECIPE['model_sha256'] or
            r.get('config_sha256') != RECIPE['config_sha256'] or
            r.get('same_physical_slices_verified') is not True or
            r.get('baseline224_pixel_hashes_verified') is not True or
            r.get('baseline224_feature_parity_verified') is not True or
            r.get('cross_split_exact_and_near_duplicate_audit_passed') is not True):
        raise ValueError('Paired pixel/feature provenance unverified')
    ids = [row['StudyInstanceUID'] for row in rows]
    with np.load(archive, allow_pickle=False) as a:
        if a['ids'].tolist() != ids or str(a['contract_hash']) != protocol['contract_hash']:
            raise ValueError('Cache study order/contract drift')
        features = [a[f'features{size}'].copy() for size in RECIPE['resolutions']]
    for f in features:
        if f.shape != (len(rows), 3, 384) or f.dtype != np.float32 or not np.isfinite(f).all():
            raise ValueError('Invalid feature shape/dtype/values')
    return features, r


def resource_gate(value):
    limit = RECIPE['resource_limits']
    keys = ['total_session_seconds', 'maximum_peak_allocated_bytes', 'forward336_over224']
    if any(not isinstance(value.get(k), (float, int)) or not np.isfinite(value[k]) or value[k] <= 0 for k in keys):
        raise ValueError('Measured whole-job memory/timing receipt missing')
    return {'bounded_job': value['total_session_seconds'] <= limit['gpu_session_seconds'],
            'memory': value['maximum_peak_allocated_bytes'] <= limit['maximum_peak_allocated_bytes'],
            'encoder_latency': value['forward336_over224'] <= limit['maximum_forward336_over224']}


def main(args):
    if args.output.exists():
        raise FileExistsError('Existing run/exposure: reconcile it, never duplicate')
    protocol = read_protocol(args.protocol)
    manifest = load_manifest()
    groups = identity_gate(manifest['splits'], json.loads(args.identity.read_text()))
    stage = args.stage
    if stage == 'confirmation':
        dev = json.loads((args.heads/'decision.json').read_text())
        if (dev['contract_hash'] != protocol['contract_hash'] or
                dev['stage'] != 'development' or dev['decision'] != 'PASS_TO_ONE_SHOT_CONFIRMATION'):
            raise ValueError('Development quality/resource gates did not pass')
    rows = manifest['splits']['train'] + manifest['splits']['development'] if stage == 'development' else manifest['splits']['confirmation']
    features, cache = read_features(args.features, args.feature_receipt, protocol, rows)
    costs = resource_gate(cache['resources'])
    if not all(costs.values()):
        raise ValueError('Feature extraction resource gate failed')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise ValueError('Existing T4 required; this tool does not provision compute')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if stage == 'confirmation' and (args.heads/'confirmation_exposure.json').exists():
        raise FileExistsError('Confirmation already exposed for these heads; no second evaluation')
    args.output.mkdir(parents=True)
    # Write before inference: an interrupted confirmation is exposed, not retryable for tuning.
    freeze(args.output/'exposure.json', {'stage': stage, 'contract_hash': protocol['contract_hash'],
                                         'state': 'STARTED_RECONCILE_BEFORE_ANY_RETRY',
                                         'identity_sha256': digest(args.identity)})
    if stage == 'confirmation':
        freeze(args.heads/'confirmation_exposure.json', {'contract_hash': protocol['contract_hash'],
                                                       'output': str(args.output), 'state': 'STARTED'})
    started = time.perf_counter()
    prediction_rows = manifest['splits'][stage]
    predictions = []
    checkpoints = []
    timings = []
    initial_hashes = {}
    for arm, size in enumerate(RECIPE['resolutions']):
        arm_predictions = []
        f = torch.from_numpy(features[arm]).cuda()
        y = torch.tensor([r['labels'] for r in manifest['splits']['train']], device='cuda') if stage == 'development' else None
        for seed in RECIPE['seeds']:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            head = StudyAttention().cuda()
            initial_hash = contract_hash({k: v.detach().cpu().tolist() for k, v in head.state_dict().items()})
            if seed in initial_hashes and initial_hashes[seed] != initial_hash:
                raise ValueError('Paired initialization mismatch')
            initial_hashes[seed] = initial_hash
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            if stage == 'development':
                optimizer = torch.optim.AdamW(head.parameters(), lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
                x = f[:1000]
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
                for epoch in range(RECIPE['epochs']):
                    if time.perf_counter()-started+cache['resources']['total_session_seconds'] > RECIPE['resource_limits']['gpu_session_seconds']:
                        raise TimeoutError('Whole-session resource budget')
                    head.train()
                    order = torch.randperm(len(x))
                    for begin in range(0, len(x), RECIPE['batch_size']):
                        ix = order[begin:begin+RECIPE['batch_size']].cuda()
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.binary_cross_entropy_with_logits(head(x[ix], mask[ix]), y[ix])
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                            raise ValueError('Nonfinite gradients')
                        optimizer.step()
                checkpoint = args.output/f'head{size}_seed{seed}.pt'
                torch.save({'model': copy.deepcopy(head.state_dict()), 'seed': seed,
                            'epochs': RECIPE['epochs'], 'contract_hash': protocol['contract_hash'],
                            'resolution': size, 'initial_hash': initial_hash}, checkpoint)
                checkpoints.append({'file': checkpoint.name, 'sha256': digest(checkpoint)})
                held = f[1000:]
            else:
                checkpoint = args.heads/f'head{size}_seed{seed}.pt'
                pins = {c['file']: c['sha256'] for c in dev['checkpoints']}
                if digest(checkpoint) != pins[checkpoint.name]:
                    raise ValueError('Promoted head changed')
                state = torch.load(checkpoint, weights_only=True)
                if (state['contract_hash'] != protocol['contract_hash'] or state['seed'] != seed or
                        state['epochs'] != RECIPE['epochs'] or state['resolution'] != size):
                    raise ValueError('Head checkpoint recipe drift')
                head.load_state_dict(state['model'])
                held = f
            head.eval()
            with torch.no_grad():
                logits = head(held, torch.ones(held.shape[:2], dtype=torch.bool, device='cuda'))
                arm_predictions.append(logits.cpu().numpy())
            torch.cuda.synchronize()
            timings.append({'resolution': size, 'seed': seed, 'seconds': time.perf_counter()-tick,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        predictions.append(arm_predictions)
        del f
    predictions = np.asarray(predictions)
    prediction_path = args.output/'predictions.npz'
    np.savez_compressed(prediction_path, logits=predictions, ids=[r['StudyInstanceUID'] for r in prediction_rows],
                        contract_hash=protocol['contract_hash'])
    result = paired_metrics(predictions, [r['labels'] for r in prediction_rows],
                            [groups[r['StudyInstanceUID']] for r in prediction_rows])
    measured = dict(cache['resources'])
    measured['total_session_seconds'] += time.perf_counter()-started
    measured['maximum_peak_allocated_bytes'] = max(measured['maximum_peak_allocated_bytes'], max(t['peak_allocated_bytes'] for t in timings))
    costs = resource_gate(measured)
    passed = result['quality_passed'] and all(costs.values())
    result.update(stage=stage, contract_hash=protocol['contract_hash'], checkpoints=checkpoints,
                  resources=measured, resource_gates=costs, head_timings=timings,
                  prediction_sha256=digest(prediction_path), identity_sha256=digest(args.identity),
                  feature_receipt_sha256=digest(args.feature_receipt),
                  confirmation_evaluated=stage == 'confirmation',
                  decision=('PASS_TO_ONE_SHOT_CONFIRMATION' if stage == 'development' else 'CONFIRMED_FOR_MANUAL_REVIEW') if passed else 'NOT_CONFIRMED_DO_NOT_PROMOTE',
                  submission_eligible=False, no_auto_submit=True)
    freeze(args.output/'decision.json', result)
    print(json.dumps({'decision': result['decision'], 'stage': stage,
                      'primary_delta': result['primary_delta']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['development', 'confirmation'])
    for name in ['protocol', 'identity', 'features', 'feature-receipt', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--heads', type=Path)
    args = parser.parse_args()
    if args.stage == 'confirmation' and args.heads is None:
        parser.error('--heads is required for confirmation; no retraining allowed')
    main(args)
