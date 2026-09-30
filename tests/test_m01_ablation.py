"""#RSNA #Kaggle #Pesquisa — synthetic CPU gates, no real-label training."""
import ast
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from scripts.assess_m01_ablation import decide, replay
from scripts.prepare_m01_ablation import assemble, literal


def runtime():
    source = Path('scripts/m01_ablation_runtime.py').read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.If)]
    scope = {'torch': torch, 'nn': nn, 'np': np, 'hashlib': hashlib}
    exec(compile(tree, '<m01-test>', 'exec'), scope)
    return scope


@pytest.mark.parametrize('name,architecture', [('MeanPool', 'mean'), ('TargetAttention', 'target')])
def test_numpy_replay_and_masks(name, architecture):
    torch.manual_seed(2026)
    scope = runtime()
    head = scope[name]()
    features = torch.randn(5, 3, 384)
    mask = torch.ones(5, 3, dtype=torch.bool)
    predicted = head(features, mask)
    np.testing.assert_allclose(predicted.detach().numpy(), replay(features.numpy(), head.state_dict(), architecture), atol=2e-6)
    mask[:, 2] = False
    changed = features.clone(); changed[:, 2] = 999
    torch.testing.assert_close(head(features, mask), head(changed, mask), atol=1e-6, rtol=0)
    predicted.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match='Empty'):
        head(features, torch.zeros_like(mask))
    with pytest.raises(ValueError, match='Invalid'):
        head(features, mask.float())
    with pytest.raises(ValueError, match='Invalid'):
        head(features, mask[:, :2])
    state = head.state_dict(); state['classifier.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        replay(features.numpy(), state, architecture)


def test_decision_requires_both_seeds_and_margin():
    rows = [{'architecture': a, 'seed': s, 'mean_soft_bce': .6}
            for a in ['shared', 'mean', 'target'] for s in [2026, 42]]
    assert decide(rows) == 'shared'
    rows[2]['mean_soft_bce'] = .5
    assert decide(rows) == 'shared'
    rows[3]['mean_soft_bce'] = .6 - 1e-6
    assert decide(rows) == 'shared'
    rows[3]['mean_soft_bce'] = .55
    assert decide(rows) == 'mean'
    rows[4]['mean_soft_bce'] = .5; rows[5]['mean_soft_bce'] = .55
    assert decide(rows) == 'mean'
    rows[5]['mean_soft_bce'] = .54
    assert decide(rows) == 'target'
    with pytest.raises(ValueError):
        decide(rows[:-1])


def test_feature_gate(tmp_path):
    scope = runtime()
    path = tmp_path / 'features.npz'
    features = np.ones((549, 3, 384), dtype=np.float32)
    rows = [{'StudyInstanceUID': str(i), 'series': [{'image_sha256': f'{i}:{j}'} for j in range(3)]} for i in range(549)]
    np.savez(path, features=features, ids=[r['StudyInstanceUID'] for r in rows], contract_hash='contract',
             image_hashes=[s['image_sha256'] for r in rows for s in r['series']])
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    scope.update(sha=sha, V02_BASELINE={'splits': {'train': rows[:299], 'development': rows[299:]}},
                 M01={'feature_sha256': sha(path), 'baseline_contract': 'contract',
                      'feature_fingerprint': hashlib.sha256(features.tobytes()).hexdigest()})
    assert scope['load_features'](path).shape == (549, 3, 384)
    scope['M01']['baseline_contract'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        scope['load_features'](path)
    scope['M01']['feature_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='archive'):
        scope['load_features'](path)


def test_builder_keeps_exact_training_functions():
    source = assemble()
    spec = literal(source, 'M01')
    assert spec['architectures'] == ['shared', 'mean', 'target']
    assert spec['confirmation_evaluated'] is False
    assert spec['submission_eligible'] is False
    baseline = Path('scripts/v02_baseline_runtime.py').read_text()
    def function(text, name):
        return next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ['train_seed', 'metrics', 'atomic_save']:
        assert function(source, name) == function(baseline, name)
    assert 'submission.csv' not in source
    assert 'rebuild_series' not in source
    with patch('scripts.prepare_m01_ablation.BASE_SHA', 'wrong'):
        with pytest.raises(ValueError, match='source changed'):
            assemble()
